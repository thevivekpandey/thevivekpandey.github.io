# Mini Leela Architecture - Visual Guide

## Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    SELF-PLAY PHASE                           │  │
│  │                                                              │  │
│  │  Start Position                                             │  │
│  │       │                                                      │  │
│  │       ↓                                                      │  │
│  │  ┌─────────┐         ┌──────────────────────┐              │  │
│  │  │ Encode  │────────→│   Neural Network     │              │  │
│  │  │ Board   │         │  ┌──────────────┐    │              │  │
│  │  └─────────┘         │  │ Res Blocks   │    │              │  │
│  │       │              │  └───┬──────┬───┘    │              │  │
│  │       │              │      │      │         │              │  │
│  │       ↓              │  ┌───▼──┐ ┌▼────┐    │              │  │
│  │  ┌─────────┐         │  │Policy│ │Value│    │              │  │
│  │  │  MCTS   │←────────│  │ Head │ │ Head│    │              │  │
│  │  │ Search  │         │  └──────┘ └─────┘    │              │  │
│  │  └────┬────┘         └──────────────────────┘              │  │
│  │       │                                                      │  │
│  │       ↓                                                      │  │
│  │  Select Move                                                │  │
│  │       │                                                      │  │
│  │       ↓                                                      │  │
│  │  Store: (board, visit_distribution, outcome)                │  │
│  │       │                                                      │  │
│  │       ↓                                                      │  │
│  │  Make Move ──→ Game Over? ──No──→ (loop back)              │  │
│  │                     │                                        │  │
│  │                    Yes                                       │  │
│  │                     ↓                                        │  │
│  │              Training Data                                   │  │
│  └──────────────────────┼───────────────────────────────────────┘  │
│                         │                                           │
│                         ↓                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     TRAINING PHASE                           │  │
│  │                                                              │  │
│  │  All Games' Data                                            │  │
│  │       │                                                      │  │
│  │       ↓                                                      │  │
│  │  Shuffle & Batch                                            │  │
│  │       │                                                      │  │
│  │       ↓                                                      │  │
│  │  ┌──────────────────────────────────────────┐              │  │
│  │  │  Loss = Policy Loss + Value Loss         │              │  │
│  │  │                                           │              │  │
│  │  │  Policy: CrossEntropy(predicted,         │              │  │
│  │  │                       mcts_visits)        │              │  │
│  │  │                                           │              │  │
│  │  │  Value: MSE(predicted, game_outcome)     │              │  │
│  │  └───────────────┬──────────────────────────┘              │  │
│  │                  │                                          │  │
│  │                  ↓                                          │  │
│  │         Backprop & Update Network                          │  │
│  │                                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Network Improved! → Next Iteration                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Neural Network Architecture Detail

```
INPUT: Chess Board (19 channels × 8×8)
│
├─ Channel 0: White Pawns      [0,0,0,0,0,0,0,0]
├─ Channel 1: White Knights    [0,0,0,0,0,0,0,0]
├─ ...                          [0,0,0,0,0,0,0,0]
├─ Channel 11: Black Kings      [0,0,0,0,0,0,0,0]
├─ Channel 12: Turn             [1,1,1,1,1,1,1,1]
├─ Channel 13-16: Castling      [1,1,1,1,1,1,1,1]
├─ Channel 17: En Passant       [0,0,0,0,0,0,0,0]
└─ Channel 18: 50-move counter  [0,0,0,0,0,0,0,0]
│
↓
┌─────────────────────────────────────────┐
│  Initial Convolution (3×3, 128 filters) │
│  BatchNorm + ReLU                       │
└─────────────────┬───────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────┐
│         Residual Block 1                │
│                                         │
│  Input                                  │
│    │                                    │
│    ├──→ Conv 3×3 → BN → ReLU           │
│    │         ↓                          │
│    │    Conv 3×3 → BN                   │
│    │         ↓                          │
│    └────────ADD ──→ ReLU ──→ Output    │
│                                         │
└─────────────────┬───────────────────────┘
                  │
                  ↓
      (Blocks 2, 3, 4 - same structure)
                  │
                  ↓
    ┌─────────────┴──────────────┐
    │                            │
    ↓                            ↓
┌─────────────────┐    ┌──────────────────┐
│   POLICY HEAD   │    │   VALUE HEAD     │
└─────────────────┘    └──────────────────┘
│                      │
│ Conv 1×1 (32)        │ Conv 1×1 (32)
│ BatchNorm + ReLU     │ BatchNorm + ReLU
│ Flatten              │ Flatten
│ FC → 4096            │ FC → 256 + ReLU
│                      │ FC → 1 + Tanh
↓                      ↓
Move Probabilities     Position Value
(one for each          [-1 = Black wins,
 from-to square)        +1 = White wins]
```

## MCTS Tree Structure

```
                        ROOT (Starting Position)
                        Visits: 100
                        Value: 0.0
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    Move: e4            Move: d4             Move: Nf3
    Prior: 0.35         Prior: 0.30          Prior: 0.25
    Visits: 45          Visits: 38           Visits: 17
    Value: +0.2         Value: +0.1          Value: -0.1
        │                    │                    
    ┌───┴───┐           ┌───┴───┐           
    │       │           │       │           
  e7-e5   Nf6         d7-d5   c5          
  V:22    V:23        V:19    V:19        
  Q:+0.3  Q:+0.1      Q:+0.2  Q:0.0       

UCT Score Calculation for each node:
Score = Q (average value) + c_puct × P (prior) × √(N_parent) / (1 + N_node)

Example for e4 node:
Score = 0.2 + 1.0 × 0.35 × √100 / (1 + 45)
      = 0.2 + 0.35 × 10 / 46
      = 0.2 + 0.076
      = 0.276
```

## Data Flow During One MCTS Simulation

```
Step 1: SELECT
    Start at root, pick child with highest UCT score
    ROOT (N=99) 
      → e4 (N=44, Q=0.2, P=0.35, UCT=0.28) ✓ SELECTED
      → d4 (N=38, Q=0.1, P=0.30, UCT=0.25)
      → Nf3 (N=17, Q=-0.1, P=0.25, UCT=0.31)
    
    Continue down tree until reaching a leaf...

Step 2: EXPAND
    At leaf node (position after e4):
    ┌──────────────────────┐
    │  Neural Network      │
    │  Input: position     │
    │  Output:             │
    │   - Policy: [0.4 e5, │
    │              0.3 Nf6,│
    │              ...]     │
    │   - Value: +0.15     │
    └──────────────────────┘
    
    Create child nodes with these priors

Step 3: BACKUP
    Propagate value up the tree:
    
    Leaf (e4-e5): V = +0.15 (from network)
                  N = 0 → 1
    
    Parent (e4):  V = +0.15 (opponent's view = -0.15)
                  N = 44 → 45
                  Q = (0.2×44 + (-0.15)) / 45 = 0.195
    
    Root:         V = -0.15 (back to original side = +0.15)
                  N = 99 → 100
                  Q = (0.0×99 + 0.15) / 100 = 0.0015

After 100 simulations, pick move with most visits (e4 with 45 visits)
```

## Training Data Format

```
One Game Example:
Position 0: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
           ↓
    ┌──────────────────────────────────────────────────────┐
    │ Board State (19×8×8 tensor)                         │
    ├──────────────────────────────────────────────────────┤
    │ Policy Target (4096 values):                        │
    │   e4: 0.45, d4: 0.38, Nf3: 0.17, others: 0.00      │
    │   (visit count distribution from MCTS)              │
    ├──────────────────────────────────────────────────────┤
    │ Value Target: +1.0                                  │
    │   (white won this game, so from white's view = +1)  │
    └──────────────────────────────────────────────────────┘

Position 1: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
           ↓
    ┌──────────────────────────────────────────────────────┐
    │ Board State (19×8×8 tensor)                         │
    ├──────────────────────────────────────────────────────┤
    │ Policy Target (4096 values):                        │
    │   e5: 0.52, Nf6: 0.31, c5: 0.17, others: 0.00      │
    ├──────────────────────────────────────────────────────┤
    │ Value Target: -1.0                                  │
    │   (white won, but black's turn, so from black = -1) │
    └──────────────────────────────────────────────────────┘

... (more positions until game end)

Network learns to:
1. Predict the MCTS visit distribution (policy target)
2. Predict the game outcome (value target)
```

## Why It Works: The Learning Loop

```
Iteration 1: Random Network
  → MCTS finds some tactics the random network missed
  → Training data contains these tactics
  → Network learns basic patterns

Iteration 2: Slightly Better Network
  → MCTS finds more sophisticated tactics
  → Training data quality improves
  → Network learns more complex patterns

Iteration 100: Strong Network
  → Network has internalized many patterns
  → MCTS still finds improvements
  → Continuous refinement

The key insight: MCTS discovers better moves than the network's
raw policy, and we train the network to predict what MCTS found!
```

## Size Comparison: Mini vs Real Leela

```
┌─────────────────┬──────────────┬──────────────┐
│  Component      │  Mini Leela  │  Real Leela  │
├─────────────────┼──────────────┼──────────────┤
│ Res Blocks      │      4       │    20-40     │
│ Channels        │     128      │   256-512    │
│ Parameters      │    2.2M      │   80M-300M   │
│ Input Planes    │     19       │   112-128    │
│ Position History│      1       │      8       │
│ MCTS Sims/Move  │    100       │   800-1600   │
│ Training Games  │    10-100    │  Millions    │
│ Training Time   │   Minutes    │   Months     │
└─────────────────┴──────────────┴──────────────┘

Despite being much smaller, Mini Leela demonstrates
all the core concepts and learning principles!
```

## Performance Expectations

```
Training Progress Over Time:
                                            Real Leela
    Strength                                    ↗
        │                                     ↗
        │                                   ↗
 Master │                                 ↗
        │                               ↗
        │                             ↗
Advanced│                           ↗
        │                         ↗
        │                       ↗
Intermed│                     ↗
        │                   ↗   ← Mini Leela after
        │                 ↗       1000 iterations
        │               ↗
Beginner│             ↗
        │           ↗
        │         ↗
 Random │━━━━━━━━
        └────────────────────────────────────→
          0   10   50  100  500  1000  Millions
                 Training Games

Mini Leela is for learning the concepts, not for
achieving superhuman performance!
```
