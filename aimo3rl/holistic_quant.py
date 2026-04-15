
import os
import json
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import bitsandbytes.functional as F

# Paths
INPUT_DIR = "/mnt/models/gpt-oss-120b-bf16-hf"
OUTPUT_DIR = "/mnt/models/gpt-oss-120b-nf4"
INDEX_FILE = os.path.join(INPUT_DIR, "model.safetensors.index.json")

def quantize_to_nf4(tensor):
    """Quantizes a bf16 tensor to NF4 format using bitsandbytes."""
    # We use the official bnb functional API
    # 4 corresponds to NF4
    out_tensor, state = F.quantize_4bit(tensor, quant_type="nf4")
    return out_tensor, state

def process_shard(shard_name):
    input_path = os.path.join(INPUT_DIR, shard_name)
    output_path = os.path.join(OUTPUT_DIR, shard_name)
    
    tensors = load_file(input_path)
    output_tensors = {}
    
    for name, tensor in tensors.items():
        # Only quantize linear layer weights (not biases, norms, or embeddings)
        # Usually identified by ending in .weight and having more than 1 dimension
        if name.endswith(".weight") and len(tensor.shape) > 1 and "layernorm" not in name and "norm" not in name:
            # We don't quantize the router in MoE
            if "router" in name:
                output_tensors[name] = tensor
                continue
                
            # Quantize to NF4
            # Note: Standard transformers NF4 loading expects specific metadata
            # For this holistic run, we will save the raw bf16 for now 
            # OR we can use the bitsandbytes save format.
            # To be truly competitive, we want the model to load in FastLanguageModel later.
            output_tensors[name] = tensor # Placeholder for logic selection
        else:
            output_tensors[name] = tensor
            
    save_file(output_tensors, output_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)
    
    shard_files = sorted(set(index["weight_map"].values()))
    
    print(f"Holistic Quantization: Processing {len(shard_files)} shards...")
    
    # Copy configuration files
    for entry in os.listdir(INPUT_DIR):
        if not entry.endswith(".safetensors") and not entry.endswith(".json"):
            import shutil
            src = os.path.join(INPUT_DIR, entry)
            dst = os.path.join(OUTPUT_DIR, entry)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # For a real NF4 comparison, we actually just need to load the bf16 model
    # with `load_in_4bit=True` in a clean script. 
    # The reason it failed before was the overhead of the WHOLE model.
    
    print("DONE (Simulation). Read below for the actual competitive test.")

if __name__ == "__main__":
    main()
