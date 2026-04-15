import argparse
import os
import re
import json
import torch
import transformers
import sys
import time
from typing import Optional
from datetime import timedelta
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import trl.generation.vllm_client as trl_vllm_client
import transformers.integrations.mxfp4 as hf_mxfp4
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssPreTrainedModel
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

# Bypass initialization since we're loading weights sharded
GptOssPreTrainedModel._init_weights = lambda self, module: None

# 1. ENVIRONMENT SETUP
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/home/ubuntu/tiktoken'
os.environ["PYTHONPATH"] = "/home/ubuntu/qlora" + (
    ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
)
os.environ["QLORA_PATCH_TRANSFORMERS_MXFP4_DTYPE"] = "1"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"

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
VLLM_SERVER_BASE_URL = "http://127.0.0.1:8000"

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


def patch_vllm_client_init_communicator() -> None:
    """Keep TRL server/client communicator bootstrap on the same host."""
    if getattr(trl_vllm_client.VLLMClient.init_communicator, "_qlora_patched", False):
        return

    def _patched_init_communicator(self, device: torch.device | str | int = 0):
        url = f"{self.base_url}/get_world_size/"
        response = trl_vllm_client.requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1
        self.rank = vllm_world_size

        url = f"{self.base_url}/init_communicator/"
        if trl_vllm_client.is_torch_xpu_available():
            if hasattr(torch.xpu.get_device_properties(device), "uuid"):
                client_device_uuid = str(torch.xpu.get_device_properties(device).uuid)
            else:
                client_device_uuid = "42"
        else:
            client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

        response = self.session.post(
            url,
            json={
                "host": self.host,
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        trl_vllm_client.time.sleep(0.1)

        if trl_vllm_client.is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=self.host, port=self.group_port, world_size=world_size, is_master=(self.rank == 0)
            )
            prefixed_store = trl_vllm_client.c10d.PrefixStore("client2server", store)
            xccl_options = trl_vllm_client.c10d.ProcessGroupXCCL.Options()
            pg = trl_vllm_client.c10d.ProcessGroupXCCL(
                store=prefixed_store,
                rank=self.rank,
                size=world_size,
                options=xccl_options,
            )
            self.communicator = pg
        else:
            pg = trl_vllm_client.StatelessProcessGroup.create(
                host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
            )
            self.communicator = trl_vllm_client.PyNcclCommunicator(pg, device=device)

        trl_vllm_client.atexit.register(self.close_communicator)

    _patched_init_communicator._qlora_patched = True
    trl_vllm_client.VLLMClient.init_communicator = _patched_init_communicator


def patch_transformers_mxfp4_dtype() -> None:
    """Avoid float32 entering the Triton MXFP4 MoE matmul path."""
    if getattr(hf_mxfp4.Mxfp4GptOssExperts.forward, "_qlora_patched", False):
        return

    def _patched_forward(self, hidden_states, routing_data, gather_idx, scatter_idx):
        FnSpecs, FusedActivation, matmul_ogs = (
            hf_mxfp4.triton_kernels_hub.matmul_ogs.FnSpecs,
            hf_mxfp4.triton_kernels_hub.matmul_ogs.FusedActivation,
            hf_mxfp4.triton_kernels_hub.matmul_ogs.matmul_ogs,
        )
        swiglu_fn = hf_mxfp4.triton_kernels_hub.swiglu.swiglu_fn

        with hf_mxfp4.on_device(hidden_states.device):
            hidden_states = hidden_states.to(torch.bfloat16)
            act = FusedActivation(
                FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
                (self.alpha, self.limit),
                2,
            )

            intermediate_cache1 = matmul_ogs(
                hidden_states,
                self.gate_up_proj,
                None,
                routing_data,
                gather_indx=gather_idx,
                precision_config=self.gate_up_proj_precision_config,
                gammas=None,
                fused_activation=act,
            )

            if intermediate_cache1.dtype != torch.bfloat16:
                intermediate_cache1 = intermediate_cache1.to(torch.bfloat16)

            if hasattr(routing_data, "gate_scal") and routing_data.gate_scal is not None:
                routing_data.gate_scal = routing_data.gate_scal.to(torch.bfloat16)

            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                self.down_proj,
                None,
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=self.down_proj_precision_config,
                gammas=routing_data.gate_scal,
            )
        return intermediate_cache3

    _patched_forward._qlora_patched = True
    hf_mxfp4.Mxfp4GptOssExperts.forward = _patched_forward


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


def build_prompt_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_debug_dataset(question: str, answer: str, copies: int = 37) -> Dataset:
    return Dataset.from_list(
        [{"prompt": build_prompt_messages(question), "answer": answer} for _ in range(copies)]
    )


def print_token_preview(tokenizer, token_ids: list[int], label: str, preview_chars: int = 800) -> None:
    print(f"\n=== {label} ===")
    print(f"Token count: {len(token_ids)}")
    print(f"Head token IDs: {token_ids[:24]}")
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(decoded[:preview_chars])
    if len(decoded) > preview_chars:
        print("... [truncated]")


def run_tokenizer_preflight(question: str) -> tuple[AutoTokenizer, list[int]]:
    print("--- TOKENIZER PREFLIGHT ---")
    messages = build_prompt_messages(question)

    base_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_out = base_tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    builtin_ids = base_out["input_ids"]
    if builtin_ids and isinstance(builtin_ids[0], list):
        builtin_ids = builtin_ids[0]
    print_token_preview(base_tok, builtin_ids, "Built-in tokenizer.apply_chat_template")

    print("\nUsing built-in GPT-OSS tokenizer chat template for training/inference preflight.")
    print("Harmony stop token IDs:", _harmony_enc.stop_tokens_for_assistant_actions())
    print("Tokenizer eos_token_id:", base_tok.eos_token_id)
    print("Tokenizer pad_token_id:", base_tok.pad_token_id)
    return base_tok, builtin_ids


def load_hf_model():
    print("\n--- LOADING HF MODEL ---")
    model = GptOssForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=Mxfp4Config(pre_quantized=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if hasattr(model, "hf_quantizer"):
        type(model.hf_quantizer).is_trainable = property(lambda self: True)
    return model


def run_hf_generation_preflight(
    tokenizer: AutoTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    min_p: float,
) -> None:
    model = load_hf_model()
    model.eval()

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print("\n--- HF FORWARD PASS ---")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1]
        nan_count = torch.isnan(logits).sum().item()
        print(f"NaN logits: {nan_count}/{logits.numel()}")
        probs = torch.softmax(logits.float(), dim=-1)
        topk = torch.topk(probs, 10)
        print("Top 10 next-token predictions:")
        for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            token_text = tokenizer.decode([idx], skip_special_tokens=False).replace("\n", "\\n")
            print(f"  id={idx:>6d} prob={prob:.6f} text={token_text!r}")

    print("\n--- HF GENERATION (GREEDY) ---")
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=_harmony_enc.stop_tokens_for_assistant_actions(),
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = generated[0, input_ids.shape[1]:].tolist()
    print(f"Generated token count: {len(completion_ids)}")
    print(f"Completion token IDs head: {completion_ids[:24]}")
    print(tokenizer.decode(completion_ids, skip_special_tokens=False))

    print("\n--- HF GENERATION (SAMPLED) ---")
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            min_p=min_p,
            eos_token_id=_harmony_enc.stop_tokens_for_assistant_actions(),
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = generated[0, input_ids.shape[1]:].tolist()
    print(f"Generated token count: {len(completion_ids)}")
    print(f"Completion token IDs head: {completion_ids[:24]}")
    print(tokenizer.decode(completion_ids, skip_special_tokens=False))


def run_vllm_preflight() -> None:
    print("\n--- vLLM PREFLIGHT ---")
    try:
        import vllm  # noqa: F401
        from openai import OpenAI  # noqa: F401
    except Exception as exc:
        print(f"Skipping vLLM check: dependencies unavailable in this environment ({exc}).")
        return

    print(
        "vLLM dependencies are installed, but this script does not embed a server launcher yet. "
        "Use ../20b-on-astral-bench/main.py as the known-good reference path."
    )


def run_preflight(args) -> None:
    print("=== GPT-OSS GRPO PREFLIGHT ===")
    print(f"Question: {args.question}")
    tok, selected_prompt_ids = run_tokenizer_preflight(args.question)

    if not args.load_model:
        print("\nSkipping HF model load by default. Pass --load-model to run HF inference preflight.")
    else:
        run_hf_generation_preflight(
            tokenizer=tok,
            prompt_ids=selected_prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            min_p=args.min_p,
        )

    run_vllm_preflight()

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Logic
# ─────────────────────────────────────────────────────────────────────────────
def run_training(args):
    print("--- INITIALIZING MODEL-PARALLEL RL TRAINING (120B) ---")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    patch_vllm_client_init_communicator()
    patch_transformers_mxfp4_dtype()

    # 1. Load Data
    # dataset = []
    # with open(DATA_PATH, "r", encoding="utf-8") as f:
    #     for line in f:
    #         try: dataset.append(json.loads(line))
    #         except: dataset.append(json.loads(line.replace("\\", "\\\\")))
    #
    # train_dataset = Dataset.from_list([
    #     {"prompt": [{"role": "developer", "content": SYSTEM_PROMPT}, {"role": "user", "content": i["question"]}],
    #      "answer": i["answer"]}
    #     for i in dataset
    # ])

    # Debug: hardcoded simple question until basics are working
    train_dataset = build_debug_dataset(args.question, args.answer)

    # 2. LoRA config
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Tokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    # 4. GRPO Trainer — distributed training, vLLM server for rollouts
    trainer = GRPOTrainer(
        processing_class=_tok,
        model=MODEL_PATH,
        reward_funcs=[correctness_reward_fn],
        peft_config=lora_config,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            max_completion_length=4096,
            num_generations=4,
            logging_steps=1,
            bf16=True,
            report_to="none",
            log_completions=True,
            temperature=args.temperature,
            top_k=args.top_k,
            min_p=args.min_p,
            use_vllm=True,
            vllm_mode="server",
            vllm_server_base_url=VLLM_SERVER_BASE_URL,
            vllm_server_timeout=240.0,
            vllm_gpu_memory_utilization=0.9,
            vllm_tensor_parallel_size=1,
            generation_kwargs={
                # Stop tokens from harmony: <|return|> and <|call|>
                "stop_token_ids": [200002, 200012],
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "train"], default="preflight")
    parser.add_argument("--question", default="What is 2 + 2?")
    parser.add_argument("--answer", default="4")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--load-model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "preflight":
        run_preflight(args)
    else:
        run_training(args)

if __name__ == "__main__":
    main()
