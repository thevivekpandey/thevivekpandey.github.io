"""
Supervised learning with SOURCE/DEST move encoding
Much more efficient architecture - saves 4M parameters!
Reads training data directly from mate_in_1_processed.csv
"""
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
from datetime import datetime
from torchinfo import summary

from mini_leela_complete_fixed import BoardEncoder
from chess_net_source_dest import ChessNetSourceDest

# Enable multi-core CPU usage
torch.set_num_threads(8)

# Device configuration - works on both CPU (Mac) and GPU (Databricks)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ============================================================================
# DATA LOADING CONFIGURATION
# ============================================================================
CSV_FILE = "mate_in_1_processed.csv"
TRAIN_SIZE = 200000  # Number of training examples to use
TEST_SIZE = 20000    # Number of test examples to use

print("="*70)
print("Supervised Learning - SOURCE/DEST ENCODING")
print("="*70)
print("\nModel: 10 ResNet blocks, 256 channels (~10-12M params)")
print("Move encoding: Separate source (64) + dest (64) predictions")
print("  Instead of wasteful 1024→4096 (4.2M params)")
print("  Using efficient source/dest encoding")
print("\nTraining Configuration:")
print("  Learning Rate: 0.001 (reduced from 0.01)")
print("  Dropout: 0.0 (removed - was limiting capacity)")
print("  Weight Decay: 5e-5 (light regularization)")
print("  Early stopping: 10 epoch patience\n")

# Setup
encoder = BoardEncoder()

# Load data from CSV
print(f"Loading data from {CSV_FILE}...")
all_positions = []
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fen = row['fen']
        answer = row['answer']
        rating = int(row['rating'])
        all_positions.append((fen, answer, rating))

print(f"Loaded {len(all_positions):,} positions from CSV")

# Shuffle and split into train/test
random.shuffle(all_positions)
total_needed = TRAIN_SIZE + TEST_SIZE

if len(all_positions) < total_needed:
    print(f"WARNING: Only {len(all_positions):,} positions available, but {total_needed:,} requested")
    print(f"Using all available data")
    # Adjust sizes proportionally
    ratio = len(all_positions) / total_needed
    TRAIN_SIZE = int(TRAIN_SIZE * ratio)
    TEST_SIZE = len(all_positions) - TRAIN_SIZE

TRAINING_POSITIONS = all_positions[:TRAIN_SIZE]
TEST_POSITIONS = all_positions[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

print(f"Using {TRAIN_SIZE:,} training positions and {TEST_SIZE:,} test positions\n")

# REGULARIZATION SETTINGS
DROPOUT_RATE = 0.0         # Removed dropout - was preventing learning
WEIGHT_DECAY = 5e-5        # Keep light weight decay
LEARNING_RATE = 0.001      # Reduced from 0.01 - previous LR was too high!
BATCH_SIZE = 32

# Create network with SOURCE/DEST encoding
network = ChessNetSourceDest(
    input_channels=19,
    num_res_blocks=10,  # Increased from 8 to 10
    num_channels=256,    # Increased from 128 to 256
    dropout=DROPOUT_RATE
).to(device)  # Move model to GPU if available

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

# Prepare training data - SOURCE/DEST encoding
train_data = []
for fen, solution_uci, rating in TRAINING_POSITIONS:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)

    board_state = encoder.encode_board(board)

    # Create source and dest targets (one-hot encoded)
    source_target = np.zeros(64, dtype=np.float32)
    source_target[solution_move.from_square] = 1.0

    dest_target = np.zeros(64, dtype=np.float32)
    dest_target[solution_move.to_square] = 1.0

    value_target = 1.0

    train_data.append((board_state, source_target, dest_target, value_target))


# Training function with proper loss calculation for source/dest
def train_epoch(data, network, optimizer, batch_size, device):
    network.train()
    random.shuffle(data)

    epoch_source_loss = 0.0
    epoch_dest_loss = 0.0
    epoch_value_loss = 0.0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        states = torch.FloatTensor(np.array([d[0] for d in batch])).to(device)
        source_targets = torch.FloatTensor(np.array([d[1] for d in batch])).to(device)
        dest_targets = torch.FloatTensor(np.array([d[2] for d in batch])).to(device)
        value_targets = torch.FloatTensor(np.array([d[3] for d in batch])).unsqueeze(1).to(device)

        source_logits, dest_logits, values = network(states)

        # Cross-entropy for source and dest
        source_loss = -(source_targets * F.log_softmax(source_logits, dim=1)).sum(dim=1).mean()
        dest_loss = -(dest_targets * F.log_softmax(dest_logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)

        total_loss = source_loss + dest_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_source_loss += source_loss.item()
        epoch_dest_loss += dest_loss.item()
        epoch_value_loss += value_loss.item()
        num_batches += 1

    return (epoch_source_loss / num_batches,
            epoch_dest_loss / num_batches,
            epoch_value_loss / num_batches)


# Evaluation function for source/dest
def evaluate(positions, network, device):
    network.eval()
    correct = 0
    total_source_loss = 0.0
    total_dest_loss = 0.0
    total_value_loss = 0.0

    with torch.no_grad():
        for fen, solution_uci, rating in positions:
            board = chess.Board(fen)
            solution_move = chess.Move.from_uci(solution_uci)

            board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0).to(device)
            source_logits, dest_logits, values = network(board_tensor)

            # Calculate loss
            source_target = torch.zeros(1, 64).to(device)
            source_target[0, solution_move.from_square] = 1.0

            dest_target = torch.zeros(1, 64).to(device)
            dest_target[0, solution_move.to_square] = 1.0

            value_target = torch.FloatTensor([[1.0]]).to(device)

            source_loss = -(source_target * F.log_softmax(source_logits, dim=1)).sum(dim=1).mean()
            dest_loss = -(dest_target * F.log_softmax(dest_logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, value_target)

            total_source_loss += source_loss.item()
            total_dest_loss += dest_loss.item()
            total_value_loss += value_loss.item()

            # Get top predicted move among legal moves
            legal_moves = list(board.legal_moves)

            # Score each legal move by source_prob * dest_prob
            best_score = -float('inf')
            best_move = None

            source_probs = F.softmax(source_logits[0], dim=0)
            dest_probs = F.softmax(dest_logits[0], dim=0)

            for move in legal_moves:
                score = source_probs[move.from_square].item() * dest_probs[move.to_square].item()
                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move and best_move.from_square == solution_move.from_square and \
               best_move.to_square == solution_move.to_square:
                correct += 1

    avg_source_loss = total_source_loss / len(positions)
    avg_dest_loss = total_dest_loss / len(positions)
    avg_value_loss = total_value_loss / len(positions)
    accuracy = correct / len(positions)

    return correct, len(positions), accuracy, avg_source_loss, avg_dest_loss, avg_value_loss


# Training loop with early stopping
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\nTraining run ID: {timestamp}\n")

num_epochs = 100
best_test_acc = 0.0
best_test_epoch = 0
patience = 10  # Stop if no improvement for 10 epochs
epochs_without_improvement = 0

print(f"{'Epoch':<6} {'Policy Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Status'}")
print("-" * 80)

for epoch in range(1, num_epochs + 1):
    # Train
    source_loss, dest_loss, value_loss = train_epoch(train_data, network, optimizer, BATCH_SIZE, device)
    policy_loss = source_loss + dest_loss  # Combined policy loss for display

    # Evaluate
    train_correct, train_total, train_acc, train_src_loss, train_dst_loss, train_val_loss = evaluate(TRAINING_POSITIONS, network, device)
    test_correct, test_total, test_acc, test_src_loss, test_dst_loss, test_val_loss = evaluate(TEST_POSITIONS, network, device)

    test_policy_loss = test_src_loss + test_dst_loss

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
    model_path = f"source_dest_model_epoch{epoch:03d}_{timestamp}.pth"
    torch.save({
        'model_state_dict': network.state_dict(),
        'network_config': {
            'input_channels': 19,
            'num_res_blocks': 10,
            'num_channels': 256,
            'dropout': DROPOUT_RATE
        },
        'epoch': epoch,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'source_loss': source_loss,
        'dest_loss': dest_loss,
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

print(f"\n💾 Models saved: source_dest_model_epoch*_{timestamp}.pth")
print(f"   Best model: epoch {best_test_epoch}")
