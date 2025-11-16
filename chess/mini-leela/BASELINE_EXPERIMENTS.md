# Baseline Model Size Experiments

## Goal
Find the smallest model that achieves 94-95% test accuracy on mate-in-1 puzzles.
This becomes your baseline for chess-specific convolution experiments.

## Current Best (Upper Bound)
- **Config:** 10 blocks, 256 channels
- **Params:** ~10M
- **Results:** 98.6% train, 95.3% test
- **File:** supervised_source_dest.py (already completed)

## Experiments to Run

### Config 1: 10 blocks, 192 channels
- **File:** `train_config1_10b_192ch.py`
- **Expected params:** ~6-7M
- **Target:** 94-95% test
- **Strategy:** Reduce channels, keep blocks

### Config 2: 8 blocks, 256 channels
- **File:** `train_config2_8b_256ch.py`
- **Expected params:** ~8-9M
- **Target:** 94-95% test
- **Strategy:** Reduce blocks, keep channels

### Config 3: 8 blocks, 192 channels
- **File:** `train_config3_8b_192ch.py`
- **Expected params:** ~5M (SMALLEST)
- **Target:** 93-94% test
- **Strategy:** Reduce both (most aggressive)

## How to Run

Run each config sequentially (one GPU):

```bash
# Config 1
python train_config1_10b_192ch.py

# Config 2 (after Config 1 finishes)
python train_config2_8b_256ch.py

# Config 3 (after Config 2 finishes)
python train_config3_8b_192ch.py
```

Each will:
- Auto-stop after 10 epochs without improvement
- Save best model checkpoint
- Print final results summary

## Expected Timeline
- Each config: ~30-50 epochs (same as your 10b/256ch run)
- Total time: Run overnight, have results by morning

## Decision Matrix

After completion, choose your baseline:

| If Config 3 gets... | Then... |
|---------------------|---------|
| ≥94% test | Use Config 3 (5M params) - most efficient! |
| 93-94% test | Use Config 2 (8-9M params) - good balance |
| <93% test | Use Config 1 (6-7M params) - or stick with 10b/256ch |

## Next Steps

Once you have your baseline (e.g., "Config 3: 5M params, 94.5% test"):

1. **Chess-specific convolutions** - Can you match 94.5% with only 3M params?
2. **Legal move encoding** - Can you reach 94.5% faster or with less data?
3. **Scale up data** - Can you push baseline to 96%+ with 600K examples?

## Results Summary Table

Fill this in as you complete each config:

| Config | Blocks | Channels | Params | Train Acc | Test Acc | Gap |
|--------|--------|----------|--------|-----------|----------|-----|
| Baseline | 10 | 256 | 10M | 98.6% | 95.3% | 3.3% |
| Config 1 | 10 | 192 | ? | ? | ? | ? |
| Config 2 | 8 | 256 | ? | ? | ? | ? |
| Config 3 | 8 | 192 | ? | ? | ? | ? |

Good luck!
