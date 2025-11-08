# Mini Leela Chess Zero - Complete Package

## 📦 Package Contents

This package contains everything you need to understand and implement a simplified version of Leela Chess Zero / AlphaZero.

### Core Files

| File | Size | Purpose | Read First? |
|------|------|---------|-------------|
| **QUICKSTART.md** | 5KB | Start here! Quick overview and setup | ✅ 1st |
| **mini_leela_complete.py** | 21KB | Complete implementation in one file | ⭐ Main |
| **GUIDE.md** | 10KB | Detailed explanations of all concepts | 📖 2nd |
| **ARCHITECTURE.md** | 17KB | Visual diagrams and architecture details | 📊 3rd |
| **demo.py** | 9KB | Example usage of each component | 💻 4th |
| **requirements.txt** | 47B | Python dependencies | 📦 - |

## 🎯 Learning Path

### Beginner (Just Getting Started)

1. Read **QUICKSTART.md** (5 minutes)
   - Understand what this package does
   - Learn how to install and run it

2. Run **mini_leela_complete.py** (5-10 minutes)
   ```bash
   python mini_leela_complete.py
   ```
   - Watch it generate games and train
   - See the training losses

3. Read **GUIDE.md** sections 1-3 (20 minutes)
   - Board representation
   - Neural network architecture
   - MCTS algorithm

### Intermediate (Understanding the Details)

4. Study **mini_leela_complete.py** code (45 minutes)
   - Read each section with comments
   - Focus on understanding one component at a time:
     - Part 1: Board Encoding
     - Part 2: Neural Network
     - Part 3: Move Encoding
     - Part 4: MCTS
     - Part 5: Self-Play
     - Part 6: Training

5. Read **ARCHITECTURE.md** (30 minutes)
   - See visual diagrams
   - Understand data flow
   - Compare to real Leela

6. Run **demo.py** examples (20 minutes)
   ```python
   # Uncomment and run each example
   example_network()
   example_encoding()
   example_mcts()
   example_selfplay()
   ```

### Advanced (Experimenting and Extending)

7. Modify and experiment (hours/days)
   - Change network architecture
   - Adjust MCTS parameters
   - Add position history
   - Implement your own improvements

8. Train for longer (hours/days)
   - Run 100+ iterations
   - Save and evaluate models
   - Compare different configurations

## 📖 What Each File Teaches You

### QUICKSTART.md
- Installation instructions
- How to run the code
- What to expect
- Where to start

### mini_leela_complete.py
**The main implementation - teaches you:**
- How to encode chess positions (BoardEncoder class)
- ResNet architecture with policy and value heads (ChessNet class)
- Monte Carlo Tree Search algorithm (MCTS class)
- Self-play game generation (SelfPlayGame class)
- Training loop and loss functions (ChessTrainer class)

**Key concepts demonstrated:**
- Residual connections in neural networks
- UCT algorithm for tree search
- Policy and value predictions
- Self-play reinforcement learning

### GUIDE.md
**Comprehensive explanations covering:**
- Why we need 19 input planes
- How residual blocks work
- The UCT formula and why it balances exploration/exploitation
- Why MCTS improves on raw network predictions
- The training data format
- How the network learns from MCTS results

**Includes:**
- Concrete examples
- Step-by-step walkthroughs
- Intuitive explanations
- Comparison with full Leela

### ARCHITECTURE.md
**Visual reference guide with:**
- System overview diagrams
- Neural network architecture diagram
- MCTS tree structure
- Data flow illustrations
- Training loop visualization
- Size comparisons

**Best for:**
- Visual learners
- Understanding the big picture
- Seeing how components connect

### demo.py
**Hands-on examples showing:**
- How to create and inspect the network
- How to encode chess positions
- How to run MCTS search
- How to generate self-play games
- How to train the network

**Use this to:**
- Test individual components
- Experiment with parameters
- Build your own applications
- Debug and understand behavior

## 🎓 Key Concepts Covered

### 1. Neural Network Architecture
- **Residual blocks**: Why they're crucial for deep networks
- **Policy head**: Predicting move probabilities
- **Value head**: Evaluating positions
- **Shared features**: Both heads use same convolutional layers

### 2. Monte Carlo Tree Search
- **Selection**: UCT formula for balancing exploration/exploitation
- **Expansion**: Using network to create new nodes
- **Backup**: Propagating values through the tree
- **Move selection**: Visit count distribution

### 3. Self-Play Training
- **Data generation**: Playing games against itself
- **Training targets**: MCTS visit counts and game outcomes
- **Loss functions**: Cross-entropy for policy, MSE for value
- **Improvement cycle**: Better network → better MCTS → better data

### 4. Why It Works
- **MCTS improves network**: Finds tactics network misses
- **Network improves MCTS**: Better priors for search
- **Self-play curriculum**: Always challenged at right level
- **Emergent intelligence**: Complex behavior from simple rules

## 🚀 Quick Command Reference

```bash
# Install dependencies
pip install torch numpy python-chess

# Run full training demo (3 iterations)
python mini_leela_complete.py

# Run single example from demo.py
python demo.py
# (remember to uncomment the examples you want)

# Train for longer
# Edit mini_leela_complete.py, change:
# num_iterations = 10  (or more)
# num_games = 10       (or more)
```

## 🔬 Experimentation Guide

### Easy Experiments (change one line)

1. **More training iterations**
   ```python
   # Line ~584 in mini_leela_complete.py
   num_iterations = 10  # instead of 3
   ```

2. **More simulations per move**
   ```python
   # Line ~447 in mini_leela_complete.py
   self.mcts = MCTS(network, device, num_simulations=200)
   ```

3. **Larger batches**
   ```python
   # Line ~592 in mini_leela_complete.py
   trainer.train_iteration(num_games=5, batch_size=64)
   ```

### Medium Experiments (change a few lines)

4. **Bigger network**
   ```python
   # Line ~580 in mini_leela_complete.py
   network = ChessNet(
       input_channels=19,
       num_res_blocks=8,    # was 4
       num_channels=256     # was 128
   )
   ```

5. **Different learning rate**
   ```python
   # Line ~582 in mini_leela_complete.py
   trainer = ChessTrainer(network, device=device, lr=0.0001)
   ```

### Advanced Experiments (add new features)

6. **Add position history**
   - Modify BoardEncoder to accept list of boards
   - Stack 8 positions as input (152 planes instead of 19)

7. **Save and load models**
   ```python
   # Save
   torch.save(network.state_dict(), 'model_iter_100.pth')
   
   # Load
   network.load_state_dict(torch.load('model_iter_100.pth'))
   ```

8. **Evaluation mode**
   - Create a function to play against the trained model
   - Compare different checkpoints

## 💡 Tips for Success

1. **Start small**: Run with default parameters first
2. **Read before running**: Understand what the code does
3. **Experiment incrementally**: Change one thing at a time
4. **Be patient**: Training takes time (real Leela trains for months!)
5. **Use GPU if available**: Will speed up training significantly
6. **Save your models**: Don't lose progress
7. **Compare results**: Keep notes on what works

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` or `num_channels` |
| Too slow | Reduce `num_simulations` or use GPU |
| Not improving | Train longer (100+ iterations) |
| ImportError | Install dependencies: `pip install torch numpy python-chess` |
| Can't run on GPU | Check `torch.cuda.is_available()` |

## 📚 Further Resources

### Papers to Read
1. **AlphaGo Zero** (Silver et al., 2017) - Original self-play paper
2. **AlphaZero** (Silver et al., 2018) - Extended to chess/shogi
3. **MCTS Survey** (Browne et al., 2012) - Comprehensive MCTS overview

### Websites
- Leela Chess Zero: https://lczero.org/
- Chess Programming Wiki: https://www.chessprogramming.org/
- python-chess docs: https://python-chess.readthedocs.io/

### Next Steps
- Implement pondering (thinking on opponent's time)
- Add opening book
- Create a UCI interface
- Train with different evaluation functions
- Experiment with network architecture

## 🎉 Have Fun!

Remember: This is a learning implementation, not a production chess engine. The goal is to understand the concepts behind Leela Chess Zero and AlphaZero.

Don't worry about making it super strong - focus on understanding how it works!

---

**Total Package**: 6 files, ~60KB of code and documentation
**Estimated Learning Time**: 2-4 hours to understand, endless hours to master! 😄

Good luck and enjoy your journey into chess AI! ♟️🤖
