"""
Configuration 2: 8 blocks, 256 channels
Expected: ~8-9M parameters
Target: 94-95% test accuracy
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# DATA LOADING
CSV_FILE = "mate_in_1_processed.csv"
TRAIN_SIZE = 200000
TEST_SIZE = 20000

print("="*70)
print("CONFIG 2: 8 ResNet blocks, 256 channels")
print("="*70)
print("\nExpected: ~8-9M parameters")
print("Target: 94-95% test accuracy")
print("\nTraining Configuration:")
print("  Learning Rate: 0.001")
print("  Dropout: 0.0")
print("  Weight Decay: 5e-5\n")

# Setup
encoder = BoardEncoder()

# Load data
print(f"Loading data from {CSV_FILE}...")
all_positions = []
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_positions.append((row['fen'], row['answer'], int(row['rating'])))

print(f"Loaded {len(all_positions):,} positions")
random.shuffle(all_positions)
TRAINING_POSITIONS = all_positions[:TRAIN_SIZE]
TEST_POSITIONS = all_positions[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
print(f"Using {TRAIN_SIZE:,} training and {TEST_SIZE:,} test positions\n")

# MODEL CONFIGURATION
DROPOUT_RATE = 0.0
WEIGHT_DECAY = 5e-5
LEARNING_RATE = 0.001
BATCH_SIZE = 32

network = ChessNetSourceDest(
    input_channels=19,
    num_res_blocks=8,
    num_channels=256,
    dropout=DROPOUT_RATE
).to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Model summary
print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
summary(network, input_size=(1, 19, 8, 8),
        col_names=["output_size", "num_params", "trainable"],
        depth=3)

print(f"\n{'='*70}")
print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
print(f"Dropout: {DROPOUT_RATE}, Weight decay: {WEIGHT_DECAY}")
print(f"Learning rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}\n")

# Prepare training data
train_data = []
for fen, solution_uci, rating in TRAINING_POSITIONS:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    board_state = encoder.encode_board(board)

    source_target = np.zeros(64, dtype=np.float32)
    source_target[solution_move.from_square] = 1.0

    dest_target = np.zeros(64, dtype=np.float32)
    dest_target[solution_move.to_square] = 1.0

    train_data.append((board_state, source_target, dest_target, 1.0))

# Training function
def train_epoch(data, network, optimizer, batch_size, device):
    network.train()
    random.shuffle(data)

    epoch_source_loss = 0.0
    epoch_dest_loss = 0.0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        states = torch.FloatTensor(np.array([d[0] for d in batch])).to(device)
        source_targets = torch.FloatTensor(np.array([d[1] for d in batch])).to(device)
        dest_targets = torch.FloatTensor(np.array([d[2] for d in batch])).to(device)
        value_targets = torch.FloatTensor(np.array([d[3] for d in batch])).unsqueeze(1).to(device)

        source_logits, dest_logits, values = network(states)

        source_loss = -(source_targets * F.log_softmax(source_logits, dim=1)).sum(dim=1).mean()
        dest_loss = -(dest_targets * F.log_softmax(dest_logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)

        total_loss = source_loss + dest_loss + value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_source_loss += source_loss.item()
        epoch_dest_loss += dest_loss.item()
        num_batches += 1

    return epoch_source_loss / num_batches, epoch_dest_loss / num_batches

# Evaluation function
def evaluate(positions, network, device):
    network.eval()
    correct = 0
    total_source_loss = 0.0
    total_dest_loss = 0.0

    with torch.no_grad():
        for fen, solution_uci, rating in positions:
            board = chess.Board(fen)
            solution_move = chess.Move.from_uci(solution_uci)

            board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0).to(device)
            source_logits, dest_logits, values = network(board_tensor)

            source_target = torch.zeros(1, 64).to(device)
            source_target[0, solution_move.from_square] = 1.0

            dest_target = torch.zeros(1, 64).to(device)
            dest_target[0, solution_move.to_square] = 1.0

            source_loss = -(source_target * F.log_softmax(source_logits, dim=1)).sum(dim=1).mean()
            dest_loss = -(dest_target * F.log_softmax(dest_logits, dim=1)).sum(dim=1).mean()

            total_source_loss += source_loss.item()
            total_dest_loss += dest_loss.item()

            source_probs = F.softmax(source_logits[0], dim=0)
            dest_probs = F.softmax(dest_logits[0], dim=0)

            best_score = -float('inf')
            best_move = None

            for move in board.legal_moves:
                score = source_probs[move.from_square].item() * dest_probs[move.to_square].item()
                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move and best_move.from_square == solution_move.from_square and \
               best_move.to_square == solution_move.to_square:
                correct += 1

    accuracy = correct / len(positions)
    return correct, len(positions), accuracy, total_source_loss / len(positions), total_dest_loss / len(positions)

# Training loop
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Training run ID: {timestamp}\n")

num_epochs = 100
best_test_acc = 0.0
best_test_epoch = 0
patience = 10
epochs_without_improvement = 0

print(f"{'Epoch':<6} {'Policy Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Status'}")
print("-" * 80)

for epoch in range(1, num_epochs + 1):
    source_loss, dest_loss = train_epoch(train_data, network, optimizer, BATCH_SIZE, device)
    policy_loss = source_loss + dest_loss

    train_correct, train_total, train_acc, train_src_loss, train_dst_loss = evaluate(TRAINING_POSITIONS, network, device)
    test_correct, test_total, test_acc, test_src_loss, test_dst_loss = evaluate(TEST_POSITIONS, network, device)

    test_policy_loss = test_src_loss + test_dst_loss

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_epoch = epoch
        epochs_without_improvement = 0
        status = "✓ BEST"
    else:
        epochs_without_improvement += 1
        status = ""

    train_str = f"{train_correct}/{train_total}"
    test_str = f"{test_correct}/{test_total}"
    print(f"{epoch:<6} {policy_loss:.4f}       "
          f"{test_policy_loss:.4f}       "
          f"{train_str:>6} ({train_acc*100:5.1f}%)  "
          f"{test_str:>6} ({test_acc*100:5.1f}%)  "
          f"{status}")

    # Save best model
    if status:
        model_path = f"config2_8b_256ch_best_{timestamp}.pth"
        torch.save({
            'model_state_dict': network.state_dict(),
            'network_config': {
                'input_channels': 19,
                'num_res_blocks': 8,
                'num_channels': 256,
                'dropout': DROPOUT_RATE
            },
            'epoch': epoch,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }, model_path)

    if epochs_without_improvement >= patience:
        print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
        print(f"   Best test accuracy: {best_test_acc*100:.2f}% at epoch {best_test_epoch}")
        break

print("\n" + "="*70)
print("RESULTS - CONFIG 2 (8 blocks, 256 channels)")
print("="*70)
print(f"Best Test Accuracy: {best_test_acc*100:.2f}% at epoch {best_test_epoch}")
print(f"Final Train Accuracy: {train_acc*100:.2f}%")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"Gap: {(train_acc - test_acc)*100:+.2f}%")
