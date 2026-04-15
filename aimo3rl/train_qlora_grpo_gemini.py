
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
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssPreTrainedModel
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

# Bypass initialization since we're loading weights sharded
GptOssPreTrainedModel._init_weights = lambda self, module: None

# 1. ENVIRONMENT SETUP
os.environ['HF_TOKEN'] = '<something>'
os.environ['PYTHONUNBUFFERED'] = '1'
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
    Role as HarmonyRole, Conversation, ToolNamespaceConfig,
)
from transformers.tokenization_utils_base import BatchEncoding

_harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

_TOOL_PROMPT = (
    'Use this tool to execute Python code for:\n'
    '- Complex calculations that would be error-prone by hand\n'
    '- Numerical verification of analytical results\n'
    '- Generating examples or testing conjectures\n'
    '- Visualizing problem structure when helpful\n'
    '- Brute-force verification for small cases\n\n'
    'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
    'Always use print() to display results. Write clear, well-commented code.\n\n'
    'Remember: Code should support your mathematical reasoning, not replace it. '
    'Explain what you\'re computing and why before running code.'
)
_TOOL_CONFIG = ToolNamespaceConfig(name='python', description=_TOOL_PROMPT, tools=[])

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
    if isinstance(conversation[0], dict):
        conversation = [conversation]

    all_ids = []
    max_len = 0
    for messages in conversation:
        harmony_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role in ("system", "developer"):
                sc = (
                    SystemContent.new()
                    .with_model_identity(content)
                    .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                    .with_tools(_TOOL_CONFIG)
                )
                harmony_msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, sc))
            else:
                harmony_msgs.append(
                    HarmonyMessage.from_role_and_content(_ROLE_MAP[role], content)
                )
        conv = Conversation.from_messages(harmony_msgs)
        ids = _harmony_enc.render_conversation_for_completion(conv, HarmonyRole.ASSISTANT)
        
        # Surgically append the <thought>\n tokens AFTER the turn is rendered
        # This bypasses the automatic stop-token insertion
        trigger_ids = _tok.encode("<thought>\n", add_special_tokens=False)
        ids = ids + trigger_ids
        
        all_ids.append(ids)
        max_len = max(max_len, len(ids))

    if not tokenize:
        return [self.decode(ids) for ids in all_ids]

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

# Paths
MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"
DATA_PATH  = "rl-question-answer.jsonl"
OUTPUT_DIR = "./grpo_mxfp4_8gpu_checkpoints"
LORA_OUTPUT_DIR = "./lora_adapters_final"

SYSTEM_PROMPT = (
    'You are an elite mathematical problem solver with expertise at the International '
    'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
    'rigorous mathematical reasoning.\n\n'
    '# Problem-Solving Approach:\n'
    '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
    'Identify what is given, what needs to be found, and any constraints.\n'
    '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
    'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
    '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
    '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
    '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
    'alternative methods. Ensure logical consistency throughout.\n\n'
    '# Mathematical Reasoning Principles:\n'
    '- Break complex problems into smaller, manageable sub-problems\n'
    '- Look for patterns, symmetries, and special cases that provide insight\n'
    '- Use concrete examples to build intuition before generalizing\n'
    '- Consider extreme cases and boundary conditions\n'
    '- If stuck, try working backwards from the desired result\n'
    '- Be willing to restart with a different approach if needed\n\n'
    '# Verification Requirements:\n'
    '- Cross-check arithmetic and algebraic manipulations\n'
    '- Verify that your solution satisfies all problem constraints\n'
    '- Test your answer with simple cases or special values when possible\n'
    '- Ensure dimensional consistency and reasonableness of the result\n\n'
    '# Output Format:\n'
    'The final answer must be a non-negative integer between 0 and 99999.\n'
    'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
    'Think step-by-step and show your complete reasoning process. Quality of reasoning '
    'is as important as the final answer.'
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

def format_reward_fn(completions, **kwargs) -> list[float]:
    """Rewards the model for following the <thought> ... </thought> <solution> ... </solution> format."""
    pattern = r"^<thought>\n.*?\n</thought>\n<solution>\n.*?\n</solution>$"
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Logic
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("--- INITIALIZING MODEL-PARALLEL RL TRAINING (120B) ---")

    # Update system prompt to include formatting tags
    fmt_system_prompt = SYSTEM_PROMPT + "\n\nFormat your response as:\n<thought>\n[your reasoning]\n</thought>\n<solution>\n\\boxed{[final answer]}\n</solution>"

    # Debug: hardcoded simple question until basics are working
    train_dataset = Dataset.from_list([
        {"prompt": [
            {"role": "developer", "content": fmt_system_prompt}, 
            {"role": "user", "content": "What is 2 + 2?"}
         ],
         "answer": "4"}
        for _ in range(37)
    ])

    # ─────────────────────────────────────────────────────────────────────────────
    # ROBUST LOADING BRIDGE: Remap 'block.' to 'model.layers.'
    # ─────────────────────────────────────────────────────────────────────────────
    import safetensors.torch
    import transformers.modeling_utils

    def remap_state_dict(state_dict):
        """Surgically splits QKV and renames weights from disk to model structure."""
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k.replace("block.", "model.layers.")
            # 1. SPLIT QKV (Disk: 5120x2880 -> Model: 4096, 512, 512)
            if "attn.qkv.weight" in new_k:
                prefix = new_k.replace("attn.qkv.weight", "self_attn")
                new_sd[f"{prefix}.q_proj.weight"] = v[:4096, :]
                new_sd[f"{prefix}.k_proj.weight"] = v[4096:4096+512, :]
                new_sd[f"{prefix}.v_proj.weight"] = v[4096+512:, :]
                continue
            if "attn.qkv.bias" in new_k:
                prefix = new_k.replace("attn.qkv.bias", "self_attn")
                new_sd[f"{prefix}.q_proj.bias"] = v[:4096]
                new_sd[f"{prefix}.k_proj.bias"] = v[4096:4096+512]
                new_sd[f"{prefix}.v_proj.bias"] = v[4096+512:]
                continue
            # 2. Rename Projections & Experts
            new_k = new_k.replace("attn.out.weight", "self_attn.o_proj.weight")
            new_k = new_k.replace("attn.out.bias", "self_attn.o_proj.bias")
            new_k = new_k.replace("mlp.mlp1_weight", "mlp.experts.gate_up_proj")
            new_k = new_k.replace("mlp.mlp2_weight", "mlp.experts.down_proj")
            new_k = new_k.replace("mlp.mlp1_bias", "mlp.experts.gate_up_proj_bias")
            new_k = new_k.replace("mlp.mlp2_bias", "mlp.experts.down_proj_bias")
            new_k = new_k.replace("mlp.gate.weight", "mlp.router.weight")
            new_k = new_k.replace("mlp.gate.bias", "mlp.router.bias")
            new_k = new_k.replace("attn.norm.scale", "input_layernorm.weight")
            new_k = new_k.replace("mlp.norm.scale", "post_attention_layernorm.weight")
            new_k = new_k.replace("attn.sinks", "self_attn.sinks")
            new_k = new_k.replace("unembedding.weight", "lm_head.weight")
            new_k = new_k.replace("embedding.weight", "model.embed_tokens.weight")
            new_k = new_k.replace("norm.scale", "model.norm.weight")
            new_sd[new_k] = v
        return new_sd

    original_load_file = safetensors.torch.load_file
    def patched_load_file(filename, device="cpu"):
        sd = original_load_file(filename, device=device)
        # Only remap if it looks like an Astral weight shard
        if any("block." in k for k in sd.keys()):
            return remap_state_dict(sd)
        return sd

    safetensors.torch.load_file = patched_load_file
    transformers.modeling_utils.safe_load_file = patched_load_file
    # ─────────────────────────────────────────────────────────────────────────────

    # 2. Load model with device_map="auto" (pipeline parallel across 8 GPUs)
    #    No DeepSpeed — it corrupts MXFP4 quantized weights.
    print("Loading model with device_map='auto'...")
    model = GptOssForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=Mxfp4Config(pre_quantized=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded.")

    # Bypass "is_trainable" check
    if hasattr(model, "hf_quantizer"):
        type(model.hf_quantizer).is_trainable = property(lambda self: True)

    # 3. LoRA config
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Tokenizer with harmony encoding
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _tok.apply_chat_template = _harmony_apply_chat_template.__get__(_tok, type(_tok))

    # 5. GRPO Trainer — single process, no DeepSpeed
    trainer = GRPOTrainer(
        processing_class=_tok,
        model=model,
        reward_funcs=[correctness_reward_fn, format_reward_fn],
        peft_config=lora_config,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            max_completion_length=1024, # Reduced from 4096 for faster iteration
            num_generations=8,
            logging_steps=1,
            bf16=True,
            report_to="none",
            log_completions=True,
            temperature=0.7, # Higher for more diversity in RL
            top_k=50,
            min_p=0.05,
            generation_kwargs={
                # Stop tokens: <|end|>, <|return|>, <|call|>
                "eos_token_id": [200002, 200007, 200012],
            },
        ),
        train_dataset=train_dataset,
    )

    print("STARTING RL LOOP")
    trainer.train()

    model.save_pretrained(LORA_OUTPUT_DIR)

if __name__ == "__main__":
    main()
