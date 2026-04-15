#!/usr/bin/env python3
"""
Shard-by-shard MXFP4 → bf16 dequantization for openai/gpt-oss-120b.

Reads each MXFP4 safetensor shard, dequantizes the MoE expert weights
(gate_up_proj, down_proj) from packed FP4 blocks+scales to bf16, and
saves the result as a new set of safetensor shards.

Non-quantized weights (attention, layernorm, router, embeddings) are
copied through unchanged.

Peak RAM per shard: ~20 GB (well within a 196 GB machine).
Total output size:  ~240 GB bf16 weights.

Usage:
    python dequantize_mxfp4.py
    python dequantize_mxfp4.py --output /mnt/models/gpt-oss-120b-bf16-hf
"""

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

# ── FP4 E2M1 look-up table (same as transformers.integrations.mxfp4) ───────
FP4_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                      dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize packed MXFP4 blocks + E8M0 scales → dense bf16 tensor.

    Args:
        blocks: uint8 tensor  [..., G, B]  (B bytes = 2B nibbles per group)
        scales: uint8 tensor  [..., G]     (one E8M0 exponent per group)

    Returns:
        bf16 tensor  [..., G * B * 2]  then transposed on last two dims.
    """
    blocks = blocks.to(torch.uint8)
    scales = scales.to(torch.int32) - 127  # E8M0 → signed exponent

    lut = torch.tensor(FP4_VALUES, dtype=dtype)

    *prefix, G, B = blocks.shape
    rows = 1
    for d in prefix:
        rows *= d
    rows *= G

    blocks = blocks.reshape(rows, B)
    scales = scales.reshape(rows, 1)

    out = torch.empty(rows, B * 2, dtype=dtype)

    # Low nibble → even columns, high nibble → odd columns
    idx_lo = (blocks & 0x0F).to(torch.int)
    out[:, 0::2] = lut[idx_lo]
    del idx_lo

    idx_hi = (blocks >> 4).to(torch.int)
    out[:, 1::2] = lut[idx_hi]
    del idx_hi

    # Apply per-group scale: value * 2^exponent
    torch.ldexp(out, scales, out=out)

    out = out.reshape(*prefix, G, B * 2).view(*prefix, G * B * 2)
    return out.transpose(-2, -1).contiguous()


def process_shard(shard_path: str, weight_map: dict, output_dir: str,
                  shard_name: str) -> dict:
    """Process one safetensor shard: dequantize MXFP4 weights, pass others through.

    Returns a dict mapping output weight names → output shard filename.
    """
    print(f"  Loading {shard_name} ...")
    tensors = load_file(shard_path)

    output_tensors = {}
    new_weight_map = {}
    out_shard_name = shard_name  # keep same shard names

    # Collect blocks/scales pairs
    blocks_keys = {k for k in tensors if k.endswith("_blocks")}
    scales_keys = {k for k in tensors if k.endswith("_scales")}
    paired = set()

    for bk in sorted(blocks_keys):
        base = bk.rsplit("_blocks", 1)[0]
        sk = base + "_scales"
        if sk in scales_keys:
            # Dequantize this pair
            print(f"    Dequantizing {base} ...")
            blocks_t = tensors[bk]
            scales_t = tensors[sk]
            dequantized = dequantize_mxfp4(blocks_t, scales_t)

            # Output weight name: remove _blocks/_scales suffix
            out_name = base
            output_tensors[out_name] = dequantized
            new_weight_map[out_name] = out_shard_name
            paired.add(bk)
            paired.add(sk)

            mem_mb = dequantized.numel() * 2 / 1e6
            print(f"      {list(blocks_t.shape)} → {list(dequantized.shape)}  ({mem_mb:.0f} MB)")

            del blocks_t, scales_t, dequantized

    # Copy non-quantized tensors as-is
    for name, tensor in tensors.items():
        if name not in paired:
            output_tensors[name] = tensor
            new_weight_map[name] = out_shard_name

    # Save output shard
    out_path = os.path.join(output_dir, out_shard_name)
    print(f"    Saving {out_shard_name} ({len(output_tensors)} tensors) ...")
    save_file(output_tensors, out_path)

    # Free memory
    del tensors, output_tensors
    return new_weight_map


def main():
    parser = argparse.ArgumentParser(description="Dequantize MXFP4 model to bf16")
    parser.add_argument("--repo", default="openai/gpt-oss-120b",
                        help="HuggingFace repo ID")
    parser.add_argument("--output", default="/mnt/models/gpt-oss-120b-bf16-hf",
                        help="Output directory for dequantized model")
    args = parser.parse_args()

    output_dir = args.output
    repo_id = args.repo

    # Check for completion marker
    marker = os.path.join(output_dir, ".dequantized")
    if os.path.exists(marker):
        print(f"Output directory {output_dir} already contains dequantized model. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load MXFP4 model index
    print("Downloading model index ...")
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"Model has {len(weight_map)} weights across {len(shard_files)} shards\n")

    # Process each shard
    full_weight_map = {}
    for i, shard_name in enumerate(shard_files, 1):
        print(f"[{i}/{len(shard_files)}] Processing {shard_name}")
        shard_path = hf_hub_download(repo_id, shard_name)
        shard_map = process_shard(shard_path, weight_map, output_dir, shard_name)
        full_weight_map.update(shard_map)
        print()

    # Write new index
    new_index = {
        "metadata": {"format": "pt", "dequantized_from": "mxfp4"},
        "weight_map": full_weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    print("Wrote model.safetensors.index.json")

    # Copy config (without quantization_config) and tokenizer files
    print("Copying config and tokenizer files ...")
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "generation_config.json", "chat_template.jinja"]:
        src = hf_hub_download(repo_id, fname)
        shutil.copy2(src, os.path.join(output_dir, fname))

    # Write completion marker
    open(marker, "w").close()
    print(f"\nDone! Dequantized bf16 model saved to {output_dir}")
    print(f"Total weights: {len(full_weight_map)}")


if __name__ == "__main__":
    main()
