# Mini Leela Chess Zero - Complete Learning Guide

## 📚 What You've Got

A fully functional, simplified version of Leela Chess Zero in a single Python file (~500 lines) that includes:

1. ✅ **Board Encoding** - Converts chess positions to neural network input
2. ✅ **ResNet Neural Network** - With policy and value heads
3. ✅ **MCTS** - Monte Carlo Tree Search 
4. ✅ **Self-Play** - Generates training games
5. ✅ **Training Loop** - Complete pipeline

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch numpy python-chess

# Run the demo (will generate 5 games per iteration, 3 iterations)
python mini_leela_complete.py
```

Expected output:
```
====================================================================
Mini Leela Chess Zero - Training Pipeline
====================================================================

Using device: cpu

Initializing neural network...
Network parameters: 2,234,368

Running 3 training iterations...
...
```

## 🧠 Understanding Each Component

### 1. Board Representation

**Why 19 planes?**

```python
# 12 planes for pieces (6 types × 2 colors)
Planes 0-5:  White pieces (P, N, B, R, Q, K)
Planes 6-11: Black pieces (P, N, B, R, Q, K)

# 7 metadata planes
Plane 12:    Whose turn is it? (1 = white, 0 = black)
Planes 13-16: Castling rights (4 separate binary planes)
Plane 17:    En passant target square
Plane 18:    Fifty-move rule counter (normalized 0-1)
```

**Example**: For the starting position:
- Plane 0 (white pawns): Row 2 is all 1s, rest 0s
- Plane 5 (white king): e1 square = 1, rest 0s
- Plane 12 (turn): All 1s (white to move)

### 2. Neural Network Architecture

```
                    INPUT: 19×8×8
                         ↓
              ┌──────────────────────┐
              │ Conv 3×3, 128 filters│
              │   BatchNorm + ReLU   │
              └──────────────────────┘
                         ↓
         ┌───────────────────────────────┐
         │     Residual Block 1          │
         │  Conv 3×3 → BN → ReLU →       │
         │  Conv 3×3 → BN → Add → ReLU   │
         └───────────────────────────────┘
                         ↓
              (3 more residual blocks)
                         ↓
         ┌───────────────┴───────────────┐
         │                               │
    POLICY HEAD                    VALUE HEAD
         │                               │
    Conv 1×1, 32                   Conv 1×1, 32
    BN + ReLU                      BN + ReLU
    Flatten                        Flatten
    FC → 4096                      FC(256) + ReLU
         │                         FC(1) + Tanh
         │                               │
    Move Probs                    Position Value
    (64×64 from-to)                  [-1, +1]
```

**Key Insight**: Both heads share the same feature extractor (the residual blocks). This is efficient and helps the network learn features useful for both tasks.

### 3. Monte Carlo Tree Search (MCTS)

**The UCT Formula** (Upper Confidence bounds applied to Trees):

```
Score(node) = Q(node) + c_puct × P(node) × √(N_parent) / (1 + N_node)
              \_____/   \___________________________________________/
              Exploitation              Exploration

Where:
- Q(node) = average value from simulations through this node
- P(node) = prior probability from policy network
- N = visit count
- c_puct = exploration constant (typically 1.0-2.0)
```

**Why this formula works**:
- **High Q** (good position) → higher score
- **High P** (network thinks it's good) → higher score
- **Low N** (unvisited) → higher score (exploration bonus)
- As visits increase, exploration bonus decreases → exploitation takes over

**One MCTS Simulation**:

```
1. SELECT: Start at root, go down tree picking highest UCT score
   ┌─────┐
   │Root │ → Pick child with highest UCT
   └─────┘
      ↓
   ┌─────┐
   │Child│ → Continue until leaf
   └─────┘

2. EXPAND: At leaf, run neural network
   - Get policy (move probabilities)
   - Get value (position evaluation)
   - Create children for all legal moves

3. BACKUP: Propagate value up the tree
   - Increment visit counts
   - Add value to running sum
   - Flip value sign at each level (opponent's perspective)
```

After 100 simulations, pick the move with most visits.

### 4. Why MCTS Beats Raw Network

**Example scenario**:

```python
Position: White to move

Network raw policy says:
  - Knight f3: 40% (looks natural)
  - Pawn e4: 35% (central pawn)
  - Bishop c4: 25% (develop piece)

But MCTS discovers (by searching):
  - Knight f3 leads to positions valued at -0.3 (losing!)
  - Pawn e4 leads to positions valued at +0.5 (winning!)

MCTS will heavily visit the e4 branch, so final distribution:
  - Knight f3: 10 visits
  - Pawn e4: 85 visits  ← This becomes the move!
  - Bishop c4: 5 visits
```

**Key point**: We train on the MCTS visit distribution (85% e4), NOT the raw network policy (35% e4). This is how the network learns to be better!

### 5. Self-Play Training Loop

```python
# Simplified pseudocode

for iteration in range(num_iterations):
    
    # Phase 1: Generate games
    games = []
    for _ in range(num_games):
        game_data = []
        board = starting_position
        
        while not game_over:
            # Use MCTS with current network
            visit_distribution = MCTS(board, network)
            
            # Store training example
            game_data.append((
                board_state,
                visit_distribution,  # This is the "improved" policy
                placeholder_value
            ))
            
            # Sample and make move
            move = sample(visit_distribution)
            board.push(move)
        
        # Fill in actual game outcome
        outcome = get_result(board)  # +1, 0, or -1
        for i, (state, policy, _) in enumerate(game_data):
            value = outcome if i % 2 == 0 else -outcome
            games.append((state, policy, value))
    
    # Phase 2: Train network
    for batch in shuffle(games):
        loss = policy_loss(network_policy, mcts_policy) + \
               value_loss(network_value, game_outcome)
        optimize(loss)
```

**Why this works**:
1. Network gets better at predicting what MCTS would choose
2. Better network → better MCTS → better training data
3. Virtuous cycle of improvement!

## 🔬 Experimentation Guide

### Experiment 1: Network Size

```python
# Tiny network (fast, weak)
network = ChessNet(num_res_blocks=2, num_channels=64)

# Medium network (balanced)
network = ChessNet(num_res_blocks=4, num_channels=128)

# Large network (slow, strong)
network = ChessNet(num_res_blocks=8, num_channels=256)
```

### Experiment 2: MCTS Simulations

```python
# Fast but weaker
mcts = MCTS(network, num_simulations=50)

# Balanced
mcts = MCTS(network, num_simulations=100)

# Slow but stronger
mcts = MCTS(network, num_simulations=400)
```

### Experiment 3: Temperature (Exploration)

```python
# High temperature (more exploration, more random)
self_play = SelfPlayGame(network, temperature=1.5)

# Low temperature (more exploitation, more deterministic)
self_play = SelfPlayGame(network, temperature=0.5)

# Greedy (always pick best)
self_play = SelfPlayGame(network, temperature=0.0)
```

Typical schedule: Start with temperature=1.0 for first 10 moves, then temperature=0.1 for rest of game.

## 📊 Expected Learning Curve

| Iterations | Strength | What it learned |
|------------|----------|-----------------|
| 0-10 | Random | Basic piece values, obvious captures |
| 10-50 | Beginner | Simple tactics (forks, pins), avoid blunders |
| 50-200 | Novice | Opening principles, 2-3 move combinations |
| 200-1000 | Intermediate | Complex tactics, basic strategy |
| 1000+ | Advanced | Positional understanding, deep tactics |

**Note**: Full Leela trains on *millions* of games with *millions* of positions. This is just a learning implementation!

## 🐛 Common Issues

**Issue**: "Out of memory"
- **Fix**: Reduce `num_channels` or `batch_size`

**Issue**: "Training is very slow"
- **Fix**: Reduce `num_simulations` or use GPU
- Check if CUDA available: `torch.cuda.is_available()`

**Issue**: "Network not improving"
- **Fix**: 
  - Train for more iterations (need at least 50-100)
  - Increase `num_games` per iteration
  - Check learning rate (try 0.01 or 0.0001)

## 🎯 Key Takeaways

1. **Policy head alone isn't enough**: MCTS search is essential for finding good moves

2. **Value head enables search**: Without position evaluation, MCTS would need to play to the end every time

3. **Self-play creates curriculum**: The network plays against itself at its own level, always challenged appropriately

4. **Visit counts > raw policy**: Training target is MCTS improved policy, not network's raw output

5. **Shared features help**: Policy and value heads share the same convolutional layers

## 📚 Further Reading

- **AlphaZero paper**: "Mastering Chess and Shogi by Self-Play" (Silver et al., 2017)
- **AlphaGo Zero paper**: "Mastering the game of Go without human knowledge" (Silver et al., 2017)
- **Leela Chess Zero**: https://lczero.org/ and https://github.com/LeelaChessZero/lc0
- **MCTS Survey**: "A Survey of Monte Carlo Tree Search Methods" (Browne et al., 2012)

## 🔧 Next Steps

1. **Run the code** and watch it train
2. **Read through the implementation** section by section
3. **Try the experiments** above
4. **Add improvements**:
   - Position history (last 8 positions)
   - Better move encoding (separate promotion/castling)
   - Opening book to skip early learning
   - Resignation threshold (don't play hopeless games)
   - Dirichlet noise for exploration

Good luck with your learning journey! 🎉
