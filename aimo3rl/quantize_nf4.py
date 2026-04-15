
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Explicitly import the model classes from the synced transformers
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

# Paths
MODEL_PATH = "/mnt/models/gpt-oss-120b-bf16-hf" 
OUTPUT_DIR = "/mnt/models/gpt-oss-120b-nf4"

# 4-bit NF4 Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

print(f"Loading {MODEL_PATH} and quantizing to NF4...")
# Load using the explicit classes
config = GptOssConfig.from_pretrained(MODEL_PATH)
model = GptOssForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    quantization_config=bnb_config,
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)

print(f"Saving 4-bit model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Quantization complete! NF4 model saved.")
