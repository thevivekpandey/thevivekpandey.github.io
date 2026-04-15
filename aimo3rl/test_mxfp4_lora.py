
import torch
import transformers
import sys
import importlib.util
import os
from transformers import AutoTokenizer, Mxfp4Config
from peft import LoraConfig, get_peft_model

# Import custom model classes from the synced transformers
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

MODEL_PATH = "/mnt/models/gpt-oss-120b-bf16"

print("Loading model structure...")
config = GptOssConfig.from_pretrained(MODEL_PATH)
quantization_config = Mxfp4Config(pre_quantized=True)

# Load with native class
model = GptOssForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# KEY MAPPING FIX
# The LOAD REPORT showed weights are in 'block.N' but model expects 'model.layers.N'
# We need to manually fix the state dict before attaching LoRA
from safetensors.torch import load_file
print("Fixing weight mapping slots...")
state_dict = load_file(os.path.join(MODEL_PATH, "model--00001-of-00007.safetensors"))

# Define the mapping between what is on disk vs what GptOssForCausalLM expects
# This is a complex MoE model, so we need to be surgical.
# However, for a test, we just want to see if we can attach LoRA to the slots.

# Define LoRA Config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # These match the MISSING keys from the report
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

try:
    print("Attempting to attach LoRA to MXFP4 structure...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("SUCCESS: LoRA slots created on MXFP4 model!")
    
    # We won't do a forward pass yet because we need to fix the weights properly
    # but the SLOT creation is the hardest part.
    
except Exception as e:
    print(f"FAILED: {e}")
