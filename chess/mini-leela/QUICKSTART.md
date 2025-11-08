# Quick Start - Mini Leela Chess Zero

## What You Have

Three files to help you understand Leela Chess Zero:

1. **mini_leela_complete.py** - Complete implementation (~500 lines)
2. **GUIDE.md** - Comprehensive explanation of every component
3. **demo.py** - Examples showing how to use each part
4. **requirements.txt** - Dependencies

## Installation

```bash
pip install torch numpy python-chess
```

## Run the Full Training Demo

```bash
python mini_leela_complete.py
```

This will:
- Generate 5 self-play games per iteration
- Run 3 training iterations
- Show you the training losses
- Demonstrate the trained network

Expected runtime: 5-30 minutes depending on your hardware.

## Understanding the Code

Read the files in this order:

### 1. Start with GUIDE.md
- Explains the theory behind each component
- Shows diagrams of the architecture
- Provides intuition for why things work

### 2. Then read mini_leela_complete.py
The code is organized in clear sections:
```python
# PART 1: Board Representation (lines 1-100)
# PART 2: Neural Network (lines 101-200)
# PART 3: Move Encoding (lines 201-250)
# PART 4: MCTS (lines 251-400)
# PART 5: Self-Play (lines 401-470)
# PART 6: Training (lines 471-530)
# PART 7: Demo (lines 531-end)
```

### 3. Try demo.py examples
Uncomment each example one at a time:
```python
# Start with simple ones
example_network()      # See the network structure
example_encoding()     # Understand input representation
example_mcts()         # Watch MCTS in action

# Then try full pipeline
example_selfplay()     # Generate one game
example_training()     # Run one training iteration
```

## Key Concepts to Understand

### 1. Why Two Heads?
- **Policy head**: Suggests which moves look promising
- **Value head**: Evaluates how good the position is
- Together they make MCTS efficient

### 2. Why MCTS?
- Raw network policy is just pattern matching
- MCTS uses search to find tactics the network missed
- Training on MCTS results improves the network

### 3. Why Self-Play?
- Network plays against itself at its own level
- Always challenged appropriately
- Creates a curriculum of increasing difficulty

### 4. The Virtuous Cycle
```
Better Network → Better MCTS → Better Training Data → Better Network
```

## Experimentation Ideas

### Change network size
```python
# In mini_leela_complete.py, line ~580
network = ChessNet(
    input_channels=19,
    num_res_blocks=8,      # Try 2, 4, 8, 16
    num_channels=256       # Try 64, 128, 256, 512
)
```

### Adjust MCTS simulations
```python
# More simulations = stronger but slower
num_simulations=100    # Try 50, 100, 200, 400
```

### Training parameters
```python
# In trainer.train_iteration()
num_games=5           # Try 2, 5, 10, 20
batch_size=32         # Try 16, 32, 64, 128
lr=0.001              # Try 0.0001, 0.001, 0.01
```

## Common Questions

**Q: How long to train for decent play?**
A: 100+ iterations with 10+ games each. Full Leela trains on millions of games!

**Q: Why is it so slow?**
A: MCTS requires many neural network evaluations. Use GPU or reduce simulations.

**Q: Can I use this to play against?**
A: Yes! After training, use MCTS.search() to pick moves.

**Q: How do I save/load the network?**
A: 
```python
# Save
torch.save(network.state_dict(), 'model.pth')

# Load
network = ChessNet()
network.load_state_dict(torch.load('model.pth'))
```

**Q: How does this compare to real Leela?**
A: This is simplified for learning:
- Real Leela: 20-40 residual blocks, 256+ channels
- Real Leela: Position history (8 positions not 1)
- Real Leela: Millions of games, professional compute
- This version: Great for understanding the concepts!

## Next Steps

1. ✅ Run mini_leela_complete.py to see it work
2. ✅ Read GUIDE.md to understand the theory
3. ✅ Study the code section by section
4. ✅ Run demo.py examples
5. ✅ Try the experiments above
6. ✅ Add your own improvements!

## Help & Resources

- **AlphaZero paper**: Search "Silver AlphaZero Nature 2018"
- **Leela Chess Zero**: https://lczero.org/
- **Chess programming wiki**: https://www.chessprogramming.org/
- **python-chess docs**: https://python-chess.readthedocs.io/

## File Summary

| File | Purpose | Lines | Read Time |
|------|---------|-------|-----------|
| mini_leela_complete.py | Complete implementation | ~500 | 30-60 min |
| GUIDE.md | Detailed explanations | - | 20-30 min |
| demo.py | Usage examples | ~300 | 15-20 min |
| requirements.txt | Dependencies | 3 | 1 min |

**Total learning time**: 2-3 hours to understand everything!

---

Good luck with your learning! Feel free to experiment and break things - that's how you learn best! 🎉
