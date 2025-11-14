"""
Supervised learning with regularization (Dropout + Weight Decay)
To combat overfitting on the mate-in-1 dataset
"""
import sys
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from torchinfo import summary

from mini_leela_complete_fixed import BoardEncoder, MoveEncoder
from mini_leela_complete_with_dropout import ChessNetWithDropout
from mate_in_1_positions import TRAINING_POSITIONS, TEST_POSITIONS

# Enable multi-core CPU usage
torch.set_num_threads(8)

print("="*70)
print("Supervised Learning - LARGER MODEL")
print("="*70)
print("\nModel: 4 ResNet blocks, 128 channels (~4.6M parameters)")
print("Regularization techniques:")
print("  1. Dropout (10% in ResNet blocks and FC layers)")
print("  2. Weight Decay (L2 regularization)")
print("  3. Early stopping based on test accuracy\n")

# Setup
encoder = BoardEncoder()
move_encoder = MoveEncoder()

# REGULARIZATION SETTINGS
DROPOUT_RATE = 0.1        # 10% dropout (reduced from 30% - we were underfitting!)
WEIGHT_DECAY = 5e-5       # L2 regularization (reduced from 1e-4)
LEARNING_RATE = 0.01
BATCH_SIZE = 32           # Larger batch size for better generalization

# Create network WITH dropout (LARGER MODEL to combat underfitting)
network = ChessNetWithDropout(
    input_channels=19,
    num_res_blocks=4,  # Increased from 2 to 4
    num_channels=128,   # Increased from 64 to 128
    dropout=DROPOUT_RATE
)

# Optimizer with weight decay
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Print model summary
print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
summary(network, input_size=(1, 19, 8, 8),
        col_names=["output_size", "num_params", "trainable"],
        depth=3)

print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
print(f"Dropout rate: {DROPOUT_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}\n")
print("Exiting now")
sys.exit(1)

# Calculate and display average ratings
train_ratings = [rating for _, _, rating in TRAINING_POSITIONS]
test_ratings = [rating for _, _, rating in TEST_POSITIONS]
avg_train_rating = sum(train_ratings) / len(train_ratings)
avg_test_rating = sum(test_ratings) / len(test_ratings)

print("Dataset Statistics:")
print(f"  Training puzzles: {len(TRAINING_POSITIONS)}")
print(f"  Average training rating: {avg_train_rating:.1f}")
print(f"  Test puzzles: {len(TEST_POSITIONS)}")
print(f"  Average test rating: {avg_test_rating:.1f}\n")

# Prepare training data
train_data = []
for fen, solution_uci, rating in TRAINING_POSITIONS:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    solution_idx = move_encoder.move_to_index(solution_move)

    board_state = encoder.encode_board(board)
    policy_target = np.zeros(4096, dtype=np.float32)
    policy_target[solution_idx] = 1.0
    value_target = 1.0

    train_data.append((board_state, policy_target, value_target))

# Training function with proper loss calculation
def train_epoch(data, network, optimizer, batch_size):
    network.train()
    random.shuffle(data)

    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        states = torch.FloatTensor(np.array([d[0] for d in batch]))
        policy_targets = torch.FloatTensor(np.array([d[1] for d in batch]))
        value_targets = torch.FloatTensor(np.array([d[2] for d in batch])).unsqueeze(1)

        policy_logits, values = network(states)

        # Proper cross-entropy for soft targets
        policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        total_loss = policy_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_policy_loss += policy_loss.item()
        epoch_value_loss += value_loss.item()
        num_batches += 1

    return epoch_policy_loss / num_batches, epoch_value_loss / num_batches

# Evaluation function
def evaluate(positions, network):
    network.eval()
    correct = 0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    with torch.no_grad():
        for fen, solution_uci, rating in positions:
            board = chess.Board(fen)
            solution_move = chess.Move.from_uci(solution_uci)
            solution_idx = move_encoder.move_to_index(solution_move)

            board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0)
            policy_logits, values = network(board_tensor)

            # Calculate loss
            policy_target = torch.zeros(1, 4096)
            policy_target[0, solution_idx] = 1.0
            value_target = torch.FloatTensor([[1.0]])

            policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, value_target)

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            # Get top predicted move among legal moves
            legal_moves = list(board.legal_moves)
            legal_indices = [move_encoder.move_to_index(m) for m in legal_moves]
            legal_logits = policy_logits[0, legal_indices]
            best_legal_idx = legal_indices[legal_logits.argmax().item()]

            if best_legal_idx == solution_idx:
                correct += 1

    avg_policy_loss = total_policy_loss / len(positions)
    avg_value_loss = total_value_loss / len(positions)
    accuracy = correct / len(positions)

    return correct, len(positions), accuracy, avg_policy_loss, avg_value_loss

# Initial evaluation
#print("Evaluating before training...")
#train_correct, train_total, train_acc, train_policy_loss, train_value_loss = evaluate(TRAINING_POSITIONS, network)
#test_correct, test_total, test_acc, test_policy_loss, test_value_loss = evaluate(TEST_POSITIONS, network)
#print(f"Before training:")
#print(f"  Train: {train_correct}/{train_total} ({train_acc*100:.1f}%) - Loss: {train_policy_loss:.4f}")
#print(f"  Test:  {test_correct}/{test_total} ({test_acc*100:.1f}%) - Loss: {test_policy_loss:.4f}")

# Training loop with early stopping
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\nTraining run ID: {timestamp}\n")

num_epochs = 100
best_test_acc = 0.0
best_test_epoch = 0
patience = 10  # Stop if no improvement for 10 epochs
epochs_without_improvement = 0

print(f"{'Epoch':<6} {'Train Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Status'}")
print("-" * 80)

for epoch in range(1, num_epochs + 1):
    # Train
    policy_loss, value_loss = train_epoch(train_data, network, optimizer, BATCH_SIZE)

    # Evaluate
    train_correct, train_total, train_acc, train_policy_loss, train_value_loss = evaluate(TRAINING_POSITIONS, network)
    test_correct, test_total, test_acc, test_policy_loss, test_value_loss = evaluate(TEST_POSITIONS, network)

    overfit_gap = train_acc - test_acc

    # Track best test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_epoch = epoch
        epochs_without_improvement = 0
        status = "✓ BEST"
    else:
        epochs_without_improvement += 1
        status = ""

    # Print progress
    train_str = f"{train_correct}/{train_total}"
    test_str = f"{test_correct}/{test_total}"
    print(f"{epoch:<6} {policy_loss:.4f}       "
          f"{test_policy_loss:.4f}       "
          f"{train_str:>6} ({train_acc*100:5.1f}%)  "
          f"{test_str:>6} ({test_acc*100:5.1f}%)  "
          f"{status}")

    # Save model
    model_path = f"regularized_model_epoch{epoch:03d}_{timestamp}.pth"
    torch.save({
        'model_state_dict': network.state_dict(),
        'network_config': {
            'input_channels': 19,
            'num_res_blocks': 4,
            'num_channels': 128,
            'dropout': DROPOUT_RATE
        },
        'epoch': epoch,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'best_test_acc': best_test_acc,
        'best_test_epoch': best_test_epoch
    }, model_path)

    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
        print(f"   Best test accuracy: {best_test_acc*100:.2f}% at epoch {best_test_epoch}")
        break

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nBest Test Accuracy: {best_test_acc*100:.2f}% at epoch {best_test_epoch}")
print(f"Final Train Accuracy: {train_acc*100:.2f}%")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"Final Overfitting Gap: {(train_acc - test_acc)*100:+.2f}%")

if test_acc >= 0.90:
    print("\n🎉 Excellent generalization! (90%+ test accuracy)")
elif test_acc >= 0.75:
    print("\n✓ Good generalization (75-90% test accuracy)")
elif test_acc >= 0.50:
    print("\n⚠️  Moderate generalization (50-75% test accuracy)")
else:
    print("\n⚠️  Poor generalization (<50% test accuracy)")

print(f"\n💾 Models saved: regularized_model_epoch*_{timestamp}.pth")
print(f"   Best model: epoch {best_test_epoch}")
