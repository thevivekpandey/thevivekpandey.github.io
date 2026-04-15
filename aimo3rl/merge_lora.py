#!/usr/bin/env python3
"""
Merge trained LoRA adapters back into the full-precision base model.

After merging you can re-quantise to MFXFP4 for competition inference.

NOTE: Loading the 120B model in bf16 requires ~240 GB CPU RAM.
      Run this on a high-memory machine (or use --low_memory to shard).

Usage:
    python merge_lora.py
    python merge_lora.py --base_model /path/to/base --lora_path ./lora_adapters --output_path ./merged
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into the base model")
    parser.add_argument("--base_model",  default="openai/gpt-oss-120b",
                        help="HF hub ID or local path to the 16-bit base model")
    parser.add_argument("--lora_path",   default="./lora_adapters",
                        help="Path to the trained LoRA adapter directory")
    parser.add_argument("--output_path", default="./merged_model",
                        help="Where to save the merged model")
    parser.add_argument("--low_memory",  action="store_true",
                        help="Use device_map='auto' to shard across GPUs + CPU offload")
    args = parser.parse_args()

    device_map = "auto" if args.low_memory else "cpu"

    print(f"Loading base model: {args.base_model}  (device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging weights …")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}")
    model.save_pretrained(args.output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_path)

    print(
        "\nDone. Next step: quantise the merged model to MFXFP4:\n"
        f"  python -m llmcompressor {args.output_path} --scheme mfxfp4 -o ./merged_mfxfp4\n"
        "  (or use your existing quantisation pipeline)\n"
    )


if __name__ == "__main__":
    main()
