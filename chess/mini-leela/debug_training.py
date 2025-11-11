"""
Debug script to understand what's happening during training
"""
import chess
import torch
import numpy as np

from mini_leela_complete_fixed import ChessNet, BoardEncoder, MoveEncoder
from mate_in_1_positions import TRAINING_POSITIONS

# Initialize
encoder = BoardEncoder()
move_encoder = MoveEncoder()

# Create a fresh network
network = ChessNet()
network.eval()

print("="*70)
print("DEBUGGING: What does the network see?")
print("="*70)

# Test on the first training position
fen, solution_uci, description = TRAINING_POSITIONS[0]
board = chess.Board(fen)
solution_move = chess.Move.from_uci(solution_uci)
solution_idx = move_encoder.move_to_index(solution_move)

print(f"\nPosition: {description}")
print(f"FEN: {fen}")
print(f"Solution: {solution_uci} = {board.san(solution_move)}")
print(f"Solution index: {solution_idx}")

# Encode board
board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0)

# Get network output
with torch.no_grad():
    policy_logits, value = network(board_tensor)

print(f"\nNetwork value prediction: {value.item():.4f}")
print(f"(Should learn to predict +1.0 for mate-in-1)")

# Get policy distribution
policy_probs = torch.softmax(policy_logits, dim=1)[0]

# Rank all legal moves
legal_moves = list(board.legal_moves)
move_scores = []
for move in legal_moves:
    idx = move_encoder.move_to_index(move)
    prob = policy_probs[idx].item()
    move_scores.append((move, prob, idx))

move_scores.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 10 moves by network policy:")
for i, (move, prob, idx) in enumerate(move_scores[:10], 1):
    is_solution = (idx == solution_idx)
    marker = " ← SOLUTION!" if is_solution else ""
    print(f"  {i}. {board.san(move)}: {prob:.6f}{marker}")

# Find solution rank
solution_rank = None
for i, (move, prob, idx) in enumerate(move_scores, 1):
    if idx == solution_idx:
        solution_rank = i
        break

print(f"\nSolution ranked: #{solution_rank}/{len(legal_moves)} legal moves")

# Now let's check what a training target looks like
print("\n" + "="*70)
print("DEBUGGING: What does a training target look like?")
print("="*70)

# Simulate what MCTS might produce (visit counts -> policy target)
# Let's say MCTS visited solution 100 times, other move 50 times
print("\nExample: If MCTS visit counts are:")
print("  Solution move: 100 visits")
print("  Other move: 50 visits")
print("  Total: 150 visits")

policy_target = np.zeros(4096, dtype=np.float32)
policy_target[solution_idx] = 100.0 / 150.0  # 0.667
policy_target[42] = 50.0 / 150.0  # 0.333 for some other move

print(f"\nPolicy target has {(policy_target > 0).sum()} non-zero entries")
print(f"Policy target at solution index [{solution_idx}]: {policy_target[solution_idx]:.4f}")
print(f"Sum of policy target: {policy_target.sum():.4f} (should be 1.0)")

# Check loss computation
policy_targets_tensor = torch.FloatTensor(policy_target).unsqueeze(0)
policy_loss = -(policy_targets_tensor * torch.nn.functional.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
print(f"\nPolicy loss with this target: {policy_loss.item():.4f}")

value_target = torch.FloatTensor([[1.0]])
value_loss = torch.nn.functional.mse_loss(value, value_target)
print(f"Value loss (target=1.0): {value_loss.item():.4f}")

print("\n" + "="*70)
print("Next: Check actual MCTS visit counts during training")
print("="*70)
print("\nRun mini_leela_mate_in_1.py with verbose=True to see MCTS visit counts")
print("Look for: Does MCTS visit the solution move often?")
