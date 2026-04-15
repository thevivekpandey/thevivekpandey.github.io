
import os
import re
import json
import torch
import transformers
import sys
import time
from typing import Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssPreTrainedModel
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from datetime import timedelta

# Bypass initialization since we're loading weights sharded
GptOssPreTrainedModel._init_weights = lambda self, module: None

# 1. ENVIRONMENT SETUP
os.environ['HF_TOKEN'] = '<something>'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/home/ubuntu/tiktoken'

# Monkey-patch to bypass kernel checks
transformers.utils.is_kernels_available = lambda: True
transformers.is_kernels_available = lambda: True

# Bypass MXFP4 "not trainable" check — we're only training LoRA adapters
from transformers.quantizers.quantizer_mxfp4 import Mxfp4HfQuantizer
Mxfp4HfQuantizer.is_trainable = property(lambda self: True)

# ─────────────────────────────────────────────────────────────────────────────
# Harmony encoding: the model requires tiktoken-based harmony tokenization,
# NOT the HF tokenizer's apply_chat_template (which produces wrong tokens).
# ─────────────────────────────────────────────────────────────────────────────
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding,
    SystemContent, ReasoningEffort, Message as HarmonyMessage,
    Role as HarmonyRole, Conversation,
)
from transformers.tokenization_utils_base import BatchEncoding

_harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

_ROLE_MAP = {
    "system": HarmonyRole.SYSTEM,
    "developer": HarmonyRole.SYSTEM,
    "user": HarmonyRole.USER,
    "assistant": HarmonyRole.ASSISTANT,
}

def _harmony_apply_chat_template(
    self, conversation, tokenize=True, add_generation_prompt=False,
    return_dict=False, padding=False, **kwargs
):
    """Replace HF's apply_chat_template with harmony encoding."""
    # Handle single conversation or batch
    if isinstance(conversation[0], dict):
        conversation = [conversation]

    all_ids = []
    max_len = 0
    for messages in conversation:
        # Build harmony conversation
        harmony_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role in ("system", "developer"):
                sc = SystemContent.new().with_model_identity(content).with_reasoning_effort(
                    reasoning_effort=ReasoningEffort.HIGH
                )
                harmony_msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, sc))
            else:
                harmony_msgs.append(
                    HarmonyMessage.from_role_and_content(_ROLE_MAP[role], content)
                )
        conv = Conversation.from_messages(harmony_msgs)
        ids = _harmony_enc.render_conversation_for_completion(conv, HarmonyRole.ASSISTANT)
        all_ids.append(ids)
        max_len = max(max_len, len(ids))

    if not tokenize:
        return [self.decode(ids) for ids in all_ids]

    # Pad if requested
    if padding:
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        attention_masks = []
        padded_ids = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))
        all_ids = padded_ids
    else:
        attention_masks = [[1] * len(ids) for ids in all_ids]

    import torch as _torch
    if return_dict:
        return BatchEncoding({
            "input_ids": _torch.tensor(all_ids),
            "attention_mask": _torch.tensor(attention_masks),
        })
    return all_ids

# MXFP4 quantization can produce NaN logits — sanitize before sampling
_orig_multinomial = torch.multinomial
def _safe_multinomial(input, num_samples, **kwargs):
    input = torch.nan_to_num(input, nan=0.0, posinf=0.0, neginf=0.0)
    # If an entire row is zero after NaN removal, set uniform distribution
    row_sums = input.sum(dim=-1, keepdim=True)
    zero_rows = row_sums == 0
    if zero_rows.any():
        input = input.masked_fill(zero_rows.expand_as(input), 1.0)
    return _orig_multinomial(input, num_samples, **kwargs)
torch.multinomial = _safe_multinomial

# Paths
MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"
DATA_PATH  = "rl-question-answer.jsonl"
OUTPUT_DIR = "./grpo_mxfp4_8gpu_checkpoints"
LORA_OUTPUT_DIR = "./lora_adapters_final"

SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver. "
    "Your goal is to find the correct answer through rigorous mathematical reasoning.\n\n"
    "# Problem-Solving Approach:\n"
    "1. UNDERSTAND: Carefully read the problem. Identify what is given and what needs to be found.\n"
    "2. PLAN: Select the most promising approach and outline key steps.\n"
    "3. EXECUTE: Work through your solution methodically.\n"
    "4. VERIFY: Check your answer.\n\n"
    "# Output Format:\n"
    "The final answer must be a non-negative integer.\n"
    "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n"
    "Think step-by-step and show your complete reasoning process."
)

# <|end|> token ID — the model's actual end-of-turn signal
END_TOKEN_ID = 200007

# ─────────────────────────────────────────────────────────────────────────────
# Reward Function
# ─────────────────────────────────────────────────────────────────────────────
def correctness_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    rewards = []
    for content, gold in zip(responses, answer):
        match = re.search(r"\\boxed\{(\d+)\}", content)
        if match:
            predicted = match.group(1)
            rewards.append(2.0 if str(predicted) == str(gold) else 0.0)
        else:
            rewards.append(0.0)
    return rewards

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Logic
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"--- INITIALIZING 8-GPU RL TRAINING (120B) ---")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)

    # 1. Load Data
    dataset = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try: dataset.append(json.loads(line))
            except: dataset.append(json.loads(line.replace("\\", "\\\\")))
    
    train_dataset = Dataset.from_list([{"prompt": [{"role": "developer", "content": SYSTEM_PROMPT}, {"role": "user", "content": i["question"]}], "answer": i["answer"]} for i in dataset])

    # 2. LoRA config
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. Monkey-patch tokenizer to use harmony encoding instead of HF chat template
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _tok.apply_chat_template = _harmony_apply_chat_template.__get__(_tok, type(_tok))

    # 4. Distributed GRPO Trainer
    trainer = GRPOTrainer(
        processing_class=_tok,
        model=MODEL_PATH,
        reward_funcs=[correctness_reward_fn],
        peft_config=lora_config,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            max_completion_length=4096,
            num_generations=4,
            logging_steps=1,
            deepspeed="ds_config.json",
            bf16=True,
            report_to="none",
            log_completions=True,
            temperature=0.5,
            top_k=50,
            min_p=0.02,
            generation_kwargs={
                # Include <|end|> (200007) so model.generate() stops at end-of-turn
                "eos_token_id": [200002, 199999, 200012, END_TOKEN_ID],
            },
            model_init_kwargs={
                "quantization_config": Mxfp4Config(pre_quantized=True),
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            },
        ),
        train_dataset=train_dataset,
    )

    print(f"[Rank {rank}] STARTING RL LOOP")
    trainer.train()
    
    if rank == 0:
        trainer.model.save_pretrained(LORA_OUTPUT_DIR)

if __name__ == "__main__":
    main()
