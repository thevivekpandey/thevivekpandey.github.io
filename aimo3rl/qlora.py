"""
QLoRA + GRPO Training Script for AI Math Olympiad (Kaggle)
==========================================================
Strategy  : RL (GRPO) on frozen 4-bit quantized 120B model via LoRA adapters
Hardware  : Single H100 80GB
Libs      : transformers, peft, trl>=0.12, bitsandbytes, torch, datasets
"""

import re
import os
import math
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class ScriptConfig:
    # --- Model ---
    model_path: str = "/kaggle/input/your-model-dir"   # Path to MXFP4 / NF4 weights
    use_flash_attention: bool = False

    # --- LoRA ---
    lora_r: int = 32                   # Rank; 16–64 is a good range for 120B
    lora_alpha: int = 64               # Usually 2× rank
    lora_dropout: float = 0.05
    # Target the attention + feed-forward projections; adjust to your model arch
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # --- Dataset ---
    dataset_path: str = "rl-question-answer.jsonl"   # Local JSONL file
    max_train_samples: Optional[int] = None
    max_prompt_length: int = 512
    max_response_length: int = 1024

    # --- GRPO ---
    num_generations: int = 8          # Rollouts per prompt (G in GRPO)
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8   # Effective batch = 8
    num_train_epochs: int = 1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 0.1
    output_dir: str = "./grpo-math-lora"
    logging_steps: int = 10
    save_steps: int = 100


cfg = ScriptConfig()


# ---------------------------------------------------------------------------
# 2. QUANTIZATION CONFIG
# ---------------------------------------------------------------------------
# The competition provides an MXFP4 checkpoint. bitsandbytes NF4 is used here
# as the standard QLoRA-compatible 4-bit format (same 60GB footprint class).
# If loading a pre-quantized MXFP4 ckpt directly, set load_in_4bit=False and
# pass the model path as-is — the frozen weights are already quantized.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 is optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Adapter compute stays in bf16
    bnb_4bit_use_double_quant=True,     # Nested quantization saves ~0.4 GB extra
)


# ---------------------------------------------------------------------------
# 3. LOAD TOKENIZER & MODEL
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: ScriptConfig):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=True)

    # Most LLMs don't have a pad token; use EOS to avoid shape mismatches
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"     # Required for batch generation in GRPO

    print("Loading model (4-bit)...")
    # Try loading with MXFP4 quantization, fallback to BitsAndBytes if needed
    try:
        # Model is already quantized with MXFP4
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="kernels-community/vllm-flash-attn3" if cfg.use_flash_attention else "eager",
            trust_remote_code=True,
        )
    except (ImportError, OutOfMemoryError) as e:
        print(f"MXFP4 loading failed: {e}")
        print("Falling back to BitsAndBytes NF4 quantization...")
        # Apply BitsAndBytes quantization instead
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        )

    # Required step: freezes base weights, casts norms/embeds to bf16 for stable training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False      # Must disable for gradient checkpointing

    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. ADD LORA ADAPTERS
# ---------------------------------------------------------------------------

def attach_lora(model, cfg: ScriptConfig):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expect something like: trainable: ~0.1-0.3% of total 120B params

    return model


# ---------------------------------------------------------------------------
# 5. DATASET & PROMPT FORMATTING
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert mathematician. "
    "Solve the problem step by step, showing your full reasoning. "
    "Put your final answer inside \\boxed{}."
)

def format_prompt(example: dict, tokenizer) -> dict:
    """Wrap each problem in a chat template the model understands."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["question"]},
    ]
    # apply_chat_template returns the formatted string; do NOT add generation prompt
    # so the model generates the assistant turn during rollout
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "answer": example["answer"]}


def build_dataset(cfg: ScriptConfig, tokenizer) -> Dataset:
    print("Loading dataset...")
    ds = load_dataset("json", data_files=cfg.dataset_path, split="train")

    if cfg.max_train_samples:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))

    ds = ds.map(
        lambda ex: format_prompt(ex, tokenizer),
        remove_columns=ds.column_names,
    )
    return ds


# ---------------------------------------------------------------------------
# 6. REWARD FUNCTIONS
# ---------------------------------------------------------------------------
# GRPO is reward-model-free. We use rule-based signals — ideal for math
# because ground truth answers are available and unambiguous.

def extract_boxed(text: str) -> Optional[str]:
    """Pull the last \\boxed{...} expression from the model output."""
    # Handles nested braces like \boxed{\frac{1}{2}}
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: str) -> str:
    """Light normalization for robust comparison."""
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", "", ans)          # Remove all whitespace
    ans = ans.replace(",", "")             # 1,000 → 1000
    # Attempt numeric normalization
    try:
        return str(float(ans))
    except ValueError:
        return ans


def reward_correctness(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """
    +1.0  if boxed answer exactly matches ground truth (after normalization)
    +0.1  if a boxed answer is present but wrong  (partial credit — tried)
     0.0  if no boxed answer at all
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        predicted = extract_boxed(completion)
        if predicted is None:
            rewards.append(0.0)
        elif normalize_answer(predicted) == normalize_answer(gt):
            rewards.append(1.0)
        else:
            rewards.append(0.1)
    return rewards


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """
    Encourages structured chain-of-thought.
    +0.2  for having a reasoning section before the final answer
    +0.1  for ending with \\boxed{}
    """
    rewards = []
    for completion in completions:
        score = 0.0
        if extract_boxed(completion):
            score += 0.1
        # Heuristic: response should have substantial content before the boxed answer
        boxed_pos = completion.rfind(r"\boxed")
        if boxed_pos > 200:             # At least 200 chars of reasoning first
            score += 0.2
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# 7. GRPO TRAINING
# ---------------------------------------------------------------------------

def build_grpo_config(cfg: ScriptConfig) -> GRPOConfig:
    return GRPOConfig(
        # Generation
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_response_length,

        # Optimizer
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        optim="adamw_torch_fused",       # Faster fused AdamW

        # Precision
        bf16=True,
        fp16=False,

        # Logging & saving
        output_dir=cfg.output_dir,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",                # Change to "wandb" if you want tracking

        # GRPO-specific
        temperature=0.8,                 # Diversity in rollouts; tune if needed
        top_p=0.95,
        beta=0.04,                       # KL penalty coefficient vs reference policy
    )


def main():
    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Attach LoRA adapters (only these will receive gradient updates)
    model = attach_lora(model, cfg)

    # Prepare dataset
    dataset = build_dataset(cfg, tokenizer)
    print(f"Dataset size: {len(dataset)} examples")

    # Build GRPO config
    grpo_config = build_grpo_config(cfg)

    # Reward functions are passed as a list; TRL sums them automatically
    reward_fns = [
        reward_correctness,
        reward_format,
    ]

    # Instantiate trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fns,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save LoRA adapters only (not the frozen 120B base)
    adapter_save_path = os.path.join(cfg.output_dir, "final-adapter")
    trainer.model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    print(f"LoRA adapters saved to {adapter_save_path}")

    # At inference time, load base model + merge adapters:
    #   from peft import PeftModel
    #   model = AutoModelForCausalLM.from_pretrained(base_path, ...)
    #   model = PeftModel.from_pretrained(model, adapter_save_path)


if __name__ == "__main__":
    main()
