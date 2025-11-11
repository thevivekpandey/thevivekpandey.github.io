"""
Supervised learning baseline - Train directly on correct moves
This removes MCTS from the equation to test if the network architecture can learn at all
"""
import chess
import torch
import numpy as np
import random
from datetime import datetime

from mini_leela_complete_fixed import ChessNet, BoardEncoder, MoveEncoder, ChessTrainer
from mate_in_1_positions import TRAINING_POSITIONS

print("="*70)
print("Supervised Learning Baseline")
print("="*70)
print("\nThis trains DIRECTLY on correct moves (no MCTS)")
print("If this works, the network architecture is fine.")
print("If this fails, there's a deeper issue.\n")

# Setup
encoder = BoardEncoder()
move_encoder = MoveEncoder()
network = ChessNet()
trainer = ChessTrainer(network, device='cpu', lr=0.01)

# Prepare supervised training data
supervised_data = []
for fen, solution_uci, description in TRAINING_POSITIONS:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    solution_idx = move_encoder.move_to_index(solution_move)

    # Board state
    board_state = encoder.encode_board(board)

    # Policy target: 1.0 for correct move, 0.0 for all others
    policy_target = np.zeros(4096, dtype=np.float32)
    policy_target[solution_idx] = 1.0

    # Value target: +1.0 (winning position)
    value_target = 1.0

    supervised_data.append((board_state, policy_target, value_target))

print(f"Training set: {len(supervised_data)} positions")

# Training loop
num_epochs = 100
batch_size = 10

#Test before network training

network.eval()
correct = 0
for fen, solution_uci, description in TRAINING_POSITIONS:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    solution_idx = move_encoder.move_to_index(solution_move)

    board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = network(board_tensor)

    # Get top predicted move
    legal_moves = list(board.legal_moves)
    legal_indices = [move_encoder.move_to_index(m) for m in legal_moves]
    legal_logits = policy_logits[0, legal_indices]
    best_legal_idx = legal_indices[legal_logits.argmax().item()]

    if best_legal_idx == solution_idx:
        correct += 1

accuracy = correct / len(TRAINING_POSITIONS)
print(f"Before training start:  Accuracy={correct}/{len(TRAINING_POSITIONS)} ({accuracy*100:.1f}%)")

# Create timestamp for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Training run ID: {timestamp}\n")

for epoch in range(1, num_epochs + 1):
    # Shuffle data
    random.shuffle(supervised_data)

    # Train
    network.train()
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    num_batches = 0

    for i in range(0, len(supervised_data), batch_size):
        batch = supervised_data[i:i+batch_size]
        policy_loss, value_loss = trainer.train_on_batch(batch)
        epoch_policy_loss += policy_loss
        epoch_value_loss += value_loss
        num_batches += 1

    avg_policy_loss = epoch_policy_loss / num_batches
    avg_value_loss = epoch_value_loss / num_batches

    # Evaluate accuracy every 10 epochs
    if epoch % 1 == 0:
        network.eval()
        correct = 0

        for fen, solution_uci, description in TRAINING_POSITIONS:
            board = chess.Board(fen)
            solution_move = chess.Move.from_uci(solution_uci)
            solution_idx = move_encoder.move_to_index(solution_move)

            board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0)

            with torch.no_grad():
                policy_logits, value = network(board_tensor)

            # Get top predicted move
            legal_moves = list(board.legal_moves)
            legal_indices = [move_encoder.move_to_index(m) for m in legal_moves]
            legal_logits = policy_logits[0, legal_indices]
            best_legal_idx = legal_indices[legal_logits.argmax().item()]

            if best_legal_idx == solution_idx:
                correct += 1

        accuracy = correct / len(TRAINING_POSITIONS)
        print(f"Epoch {epoch:3d}: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}, Accuracy={correct}/{len(TRAINING_POSITIONS)} ({accuracy*100:.1f}%)")

        # Save model every epoch
        model_path = f"supervised_model_epoch{epoch:03d}_{timestamp}.pth"
        torch.save({
            'model_state_dict': network.state_dict(),
            'network_config': {
                'input_channels': 19,
                'num_res_blocks': 4,
                'num_channels': 128
            },
            'epoch': epoch,
            'accuracy': accuracy,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'training_size': len(TRAINING_POSITIONS)
        }, model_path)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

if correct == len(TRAINING_POSITIONS):
    print("✓ Network CAN learn! 100% accuracy on supervised learning.")
    print("  Problem is likely with MCTS not exploring correctly.")
elif correct >= len(TRAINING_POSITIONS) * 0.8:
    print("✓ Network mostly works but needs tuning.")
else:
    print("✗ Network struggles even with supervised learning.")
    print("  Issue might be: network size, learning rate, or implementation bug.")

print(f"\n💾 All models saved with timestamp: {timestamp}")
print(f"   Example: supervised_model_epoch001_{timestamp}.pth")
print(f"   Total epochs saved: {num_epochs}")
