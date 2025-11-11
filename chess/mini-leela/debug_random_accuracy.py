"""
Debug: Why is random network getting 24% accuracy?
"""
import chess
import torch
import numpy as np
from collections import Counter

from mini_leela_complete_fixed import ChessNet, BoardEncoder, MoveEncoder
from mate_in_1_positions import TRAINING_POSITIONS

encoder = BoardEncoder()
move_encoder = MoveEncoder()
network = ChessNet()
network.eval()

print("="*70)
print("Debugging Random Network Accuracy")
print("="*70)

# Analyze the dataset
num_legal_moves = []
solution_to_squares = []
solution_from_squares = []

for fen, solution_uci, desc in TRAINING_POSITIONS[:1000]:  # Check first 1000
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    num_legal_moves.append(len(legal_moves))

    solution_move = chess.Move.from_uci(solution_uci)
    solution_to_squares.append(chess.square_name(solution_move.to_square))
    solution_from_squares.append(chess.square_name(solution_move.from_square))

print(f"\nDataset Statistics (first 1000 puzzles):")
print(f"Average legal moves per position: {np.mean(num_legal_moves):.1f}")
print(f"Min legal moves: {min(num_legal_moves)}")
print(f"Max legal moves: {max(num_legal_moves)}")
print(f"Expected random accuracy: {100.0 / np.mean(num_legal_moves):.1f}%")

print(f"\nTop 10 destination squares for solutions:")
to_square_counts = Counter(solution_to_squares)
for square, count in to_square_counts.most_common(10):
    print(f"  {square}: {count} ({100*count/len(solution_to_squares):.1f}%)")

print(f"\nTop 10 source squares for solutions:")
from_square_counts = Counter(solution_from_squares)
for square, count in from_square_counts.most_common(10):
    print(f"  {square}: {count} ({100*count/len(solution_from_squares):.1f}%)")

# Now test what the random network predicts
print("\n" + "="*70)
print("Random Network Behavior")
print("="*70)

correct = 0
predicted_to_squares = []
predicted_from_squares = []

for fen, solution_uci, desc in TRAINING_POSITIONS[:1000]:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    solution_idx = move_encoder.move_to_index(solution_move)

    board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = network(board_tensor)

    # Get top predicted move among legal moves
    legal_moves = list(board.legal_moves)
    legal_indices = [move_encoder.move_to_index(m) for m in legal_moves]
    legal_logits = policy_logits[0, legal_indices]
    best_legal_idx = legal_indices[legal_logits.argmax().item()]

    # Find which move this corresponds to
    for move in legal_moves:
        if move_encoder.move_to_index(move) == best_legal_idx:
            predicted_to_squares.append(chess.square_name(move.to_square))
            predicted_from_squares.append(chess.square_name(move.from_square))
            break

    if best_legal_idx == solution_idx:
        correct += 1

print(f"\nRandom network accuracy: {correct}/{len(TRAINING_POSITIONS[:1000])} = {100*correct/1000:.1f}%")

print(f"\nTop 10 destination squares PREDICTED by random network:")
pred_to_counts = Counter(predicted_to_squares)
for square, count in pred_to_counts.most_common(10):
    print(f"  {square}: {count} ({100*count/len(predicted_to_squares):.1f}%)")

print(f"\nTop 10 source squares PREDICTED by random network:")
pred_from_counts = Counter(predicted_from_squares)
for square, count in pred_from_counts.most_common(10):
    print(f"  {square}: {count} ({100*count/len(predicted_from_squares):.1f}%)")

# Check if there's overlap
print("\n" + "="*70)
print("Analysis")
print("="*70)

# Check if network bias matches solution bias
print("\nDo network predictions match solution patterns?")
top_solution_squares = set([sq for sq, _ in to_square_counts.most_common(5)])
top_predicted_squares = set([sq for sq, _ in pred_to_counts.most_common(5)])
overlap = top_solution_squares & top_predicted_squares
print(f"Top 5 solution squares: {sorted(top_solution_squares)}")
print(f"Top 5 predicted squares: {sorted(top_predicted_squares)}")
print(f"Overlap: {sorted(overlap)}")

if len(overlap) >= 3:
    print("\n⚠️  EXPLANATION: Network initialization is biased toward common mate squares!")
    print("   The random network happens to favor moves to squares that are")
    print("   frequently the correct answer in mate-in-1 puzzles.")
else:
    print("\n⚠️  Need further investigation - the bias source is unclear.")
