"""
Compare token sequences between harmony encoding and HF model generation.
Loads the model with device_map='auto' and generates a short completion
to verify the model produces coherent output with the correct tokens.
"""
import os
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/home/ubuntu/tiktoken'
os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN_HERE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import transformers
transformers.utils.is_kernels_available = lambda: True
transformers.is_kernels_available = lambda: True

from transformers import AutoTokenizer, Mxfp4Config
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssPreTrainedModel
GptOssPreTrainedModel._init_weights = lambda self, module: None

from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding,
    SystemContent, ReasoningEffort, Message, Role, Conversation,
)

MODEL_PATH = "/home/ubuntu/models/gpt-oss-120b"

# ── Setup ────────────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Use the same question from the training data
QUESTION = "If \\(X\\) and \\(Y\\) follow a bivariate geometric distribution with \\(E[X]=a\\), \\(E[Y]=b\\), and \\(\\operatorname{Cov}(X,Y)=c\\), and the parameters are \\(a=2\\), \\(b=3\\), and \\(c=4\\), let \\(C=\\operatorname{Cov}(X^{2},Y)\\). What is the remainder when \\(C\\) is divided by 88885?"

SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver. "
    "Your goal is to find the correct answer through rigorous mathematical reasoning.\n\n"
    "# Output Format:\n"
    "The final answer must be a non-negative integer.\n"
    "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n"
    "Think step-by-step."
)

# ── Build harmony token sequence ─────────────────────────────────────────────
sc = SystemContent.new().with_model_identity(SYSTEM_PROMPT).with_reasoning_effort(
    reasoning_effort=ReasoningEffort.HIGH
)
sys_msg = Message.from_role_and_content(Role.SYSTEM, sc)
user_msg = Message.from_role_and_content(Role.USER, QUESTION)
conv = Conversation.from_messages([sys_msg, user_msg])
harmony_ids = harmony_enc.render_conversation_for_completion(conv, Role.ASSISTANT)

stop_ids = harmony_enc.stop_tokens_for_assistant_actions()

print("=" * 80)
print("HARMONY ENCODING")
print("=" * 80)
print(f"Total tokens: {len(harmony_ids)}")
print(f"Stop token IDs: {stop_ids}")
print()

# Decode and show the full prompt as text
harmony_text = tok.decode(harmony_ids)
print("── Decoded prompt text ──")
print(harmony_text)
print()

# Show last 10 tokens (generation prefix)
print("── Last 10 tokens (generation prefix) ──")
for i, tid in enumerate(harmony_ids[-10:]):
    pos = len(harmony_ids) - 10 + i
    print(f"  [{pos:3d}] {tid:>6d}  {repr(tok.decode([tid]))}")

print()
print("=" * 80)
print("LOADING MODEL & GENERATING")
print("=" * 80)

model = GptOssForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=Mxfp4Config(pre_quantized=True),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded.\n")

# Generate with the harmony token IDs
input_ids = torch.tensor([harmony_ids], dtype=torch.long).to(model.device)

print(f"Input shape: {input_ids.shape}")
print(f"Input device: {input_ids.device}")
print()

# Check logits for NaN first
with torch.no_grad():
    out = model(input_ids)
    logits = out.logits[0, -1]
    nan_count = torch.isnan(logits).sum().item()
    print(f"Logits NaN: {nan_count}/{logits.numel()}")
    probs = torch.softmax(logits.float(), dim=-1)
    top10 = torch.topk(probs, 10)
    print("Top 10 next-token predictions:")
    for p, idx in zip(top10.values, top10.indices):
        print(f"  {tok.decode([idx.item()]):20s} (id={idx.item():>6d}) prob={p.item():.4f}")

print()
print("── Generating 200 tokens ──")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        eos_token_id=stop_ids,
    )

generated_ids = output[0][len(harmony_ids):]
generated_text = tok.decode(generated_ids.tolist())
print(f"Generated {len(generated_ids)} tokens:")
print(generated_text)
print()

# Show first 20 generated token IDs
print("── First 20 generated token IDs ──")
for i, tid in enumerate(generated_ids[:20].tolist()):
    print(f"  [{i:3d}] {tid:>6d}  {repr(tok.decode([tid]))}")
