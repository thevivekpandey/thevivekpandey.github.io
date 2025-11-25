"""
Training on Stockfish Best Moves (with filtered data)
Configuration: 8 ResNet blocks, 256 channels
Target: Best move prediction (source + destination)

This combines:
- Good data: Filtered positions with clear advantages (0.5-8 pawns)
- Good objective: Predict Stockfish's best move
"""
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
import os
import sys
import tempfile
from datetime import datetime
from torchinfo import summary

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from mini_leela_complete_fixed import BoardEncoder
from chess_net_source_dest import ChessNetSourceDest

# S3 Configuration
MODEL_SAVE_PATH = 's3://adhoc-query-data/vivek.pandey/mcts/'

# Enable multi-core CPU usage
torch.set_num_threads(8)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")


def save_model_to_s3(model_dict, s3_path):
    """Save model to S3 using Databricks dbutils"""
    try:
        # Check if running in Databricks
        try:
            from pyspark.dbutils import DBUtils
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            dbutils = DBUtils(spark)
            use_dbutils = True
        except:
            use_dbutils = False

        if use_dbutils:
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                torch.save(model_dict, tmp_file.name)
                tmp_path = tmp_file.name

            # Copy to S3 using dbutils
            dbutils.fs.cp(f"file://{tmp_path}", s3_path)

            # Clean up temp file
            os.remove(tmp_path)
            print(f"✓ Model saved to S3: {s3_path}")
        else:
            # Fallback: try using boto3 (if available)
            import boto3

            # Parse S3 path
            s3_path_clean = s3_path.replace('s3://', '')
            bucket = s3_path_clean.split('/')[0]
            key = '/'.join(s3_path_clean.split('/')[1:])

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                torch.save(model_dict, tmp_file.name)
                tmp_path = tmp_file.name

            # Upload to S3
            s3_client = boto3.client('s3')
            s3_client.upload_file(tmp_path, bucket, key)

            # Clean up
            os.remove(tmp_path)
            print(f"✓ Model saved to S3: {s3_path}")

    except Exception as e:
        print(f"⚠ Warning: Failed to save to S3: {e}")
        print(f"  Saving locally instead...")
        # Fallback to local save
        local_path = s3_path.split('/')[-1]
        torch.save(model_dict, local_path)
        print(f"✓ Model saved locally: {local_path}")


# DATA LOADING
TRAIN_CSV_FILE = "training_data_04.csv"
VALIDATION_CSV_FILE = "test_data_04.csv"

print("="*70)
print("TRAINING ON GAMES AND PUZZLES")
print("CONFIG: 8 ResNet blocks, 256 channels")
print("="*70)
print("\nTraining Targets:")
print("  - Policy: Stockfish best move (source + destination)")
print("  - Value: Position evaluation")
print("\nModel Selection: Based on COMBINED policy + value loss")
print("  (Ensures both move prediction AND position evaluation are good)")
print("\nTraining Configuration:")
print("  Learning Rate: 0.0005")
print("  Dropout: 0.2")
print("  Weight Decay: 1e-4")
print(f"\nModel Save Path: {MODEL_SAVE_PATH}")
print()

# Setup
encoder = BoardEncoder()

# Load training data
print(f"Loading training data from {TRAIN_CSV_FILE}...")
training_positions = []
with open(TRAIN_CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        training_positions.append((row['fen'], row['answer'], float(row['stockfish_eval'])))

print(f"Loaded {len(training_positions):,} training positions")

# Load validation data
print(f"Loading validation data from {VALIDATION_CSV_FILE}...")
validation_positions = []
with open(VALIDATION_CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        validation_positions.append((row['fen'], row['answer'], float(row['stockfish_eval'])))

print(f"Loaded {len(validation_positions):,} validation positions\n")

# MODEL CONFIGURATION
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
VALUE_LOSS_WEIGHT = 3.0  # Scale up value loss to balance with policy loss

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
print(f"Learning rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")
print(f"Value loss weight: {VALUE_LOSS_WEIGHT}\n")

# Prepare training data
print("Preparing training data...")
train_data = []
for fen, solution_uci, stockfish_eval in training_positions:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    board_state = encoder.encode_board(board)

    source_target = np.zeros(64, dtype=np.float32)
    source_target[solution_move.from_square] = 1.0

    dest_target = np.zeros(64, dtype=np.float32)
    dest_target[solution_move.to_square] = 1.0

    # Normalize stockfish eval to [-1, 1] range using tanh
    # Evals are in range 0.5-8.0, divide by 10 to get ~0.05-0.8, then tanh
    normalized_value = np.tanh(stockfish_eval / 10.0)

    train_data.append((board_state, source_target, dest_target, normalized_value))

print(f"Training data prepared: {len(train_data):,} positions\n")


# Training function
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

        source_loss = -(source_targets * F.log_softmax(source_logits, dim=1)).sum(dim=1).mean()
        dest_loss = -(dest_targets * F.log_softmax(dest_logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)

        total_loss = source_loss + dest_loss + VALUE_LOSS_WEIGHT * value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_source_loss += source_loss.item()
        epoch_dest_loss += dest_loss.item()
        epoch_value_loss += value_loss.item()
        num_batches += 1

    return epoch_source_loss / num_batches, epoch_dest_loss / num_batches, epoch_value_loss / num_batches


# Evaluation function
def evaluate(positions, network, device):
    network.eval()
    correct = 0
    total_source_loss = 0.0
    total_dest_loss = 0.0
    total_value_loss = 0.0

    with torch.no_grad():
        for fen, solution_uci, stockfish_eval in positions:
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

            # Add value loss computation
            normalized_value_target = np.tanh(stockfish_eval / 10.0)
            value_target = torch.FloatTensor([[normalized_value_target]]).to(device)
            value_loss = F.mse_loss(values, value_target)

            total_source_loss += source_loss.item()
            total_dest_loss += dest_loss.item()
            total_value_loss += value_loss.item()

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
    avg_source_loss = total_source_loss / len(positions)
    avg_dest_loss = total_dest_loss / len(positions)
    avg_value_loss = total_value_loss / len(positions)

    return correct, len(positions), accuracy, avg_source_loss, avg_dest_loss, avg_value_loss


# Training loop
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Training run ID: {timestamp}\n")

num_epochs = 100
best_val_loss = float('inf')  # Track combined validation loss (lower is better)
best_val_epoch = 0
patience = 10
epochs_without_improvement = 0

print(f"{'Epoch':<6} {'Train Policy':<13} {'Train Value':<13} {'Val Policy':<12} {'Val Value':<12} {'Total Val Loss':<15} {'Train Acc':<20} {'Val Acc':<20} {'Status'}")
print("-" * 140)

for epoch in range(1, num_epochs + 1):
    # Train for one epoch - returns average losses
    train_src_loss, train_dst_loss, train_value_loss = train_epoch(train_data, network, optimizer, BATCH_SIZE, device)
    train_policy_loss = train_src_loss + train_dst_loss
    train_combined_loss = train_policy_loss + VALUE_LOSS_WEIGHT * train_value_loss

    # Evaluate on a sample of training data (to save time) - for accuracy tracking
    train_sample = random.sample(training_positions, min(5000, len(training_positions)))
    train_correct, train_total, train_acc, _, _, _ = evaluate(train_sample, network, device)

    # Evaluate on full validation set - this determines best model
    val_correct, val_total, val_acc, val_src_loss, val_dst_loss, val_value_loss = evaluate(validation_positions, network, device)

    val_policy_loss = val_src_loss + val_dst_loss
    val_combined_loss = val_policy_loss + VALUE_LOSS_WEIGHT * val_value_loss  # Combined loss for model selection (weighted)

    # Select best model based on COMBINED loss (policy + value)
    # Lower combined loss = better at BOTH move prediction AND position evaluation
    if val_combined_loss < best_val_loss:
        best_val_loss = val_combined_loss
        best_val_epoch = epoch
        best_val_acc = val_acc  # Store for reporting only (NOT used for selection)
        epochs_without_improvement = 0
        status = "✓ BEST"
    else:
        epochs_without_improvement += 1
        status = ""

    train_str = f"{train_correct}/{train_total}"
    val_str = f"{val_correct}/{val_total}"
    print(f"{epoch:<6} {train_policy_loss:.4f}        "
          f"{train_value_loss:.4f}        "
          f"{val_policy_loss:.4f}       "
          f"{val_value_loss:.4f}       "
          f"{val_combined_loss:.4f}          "
          f"{train_str:>10} ({train_acc*100:5.1f}%)  "
          f"{val_str:>10} ({val_acc*100:5.1f}%)  "
          f"{status}")

    # Save best model
    if status:
        model_filename = f"games_and_puzzles_{epoch:03d}_{timestamp}.pth"
        model_dict = {
            'model_state_dict': network.state_dict(),
            'network_config': {
                'input_channels': 19,
                'num_res_blocks': 8,
                'num_channels': 256,
                'dropout': DROPOUT_RATE
            },
            'epoch': epoch,
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'validation_policy_loss': val_policy_loss,
            'validation_value_loss': val_value_loss,
            'validation_combined_loss': val_combined_loss,
            'timestamp': timestamp,
        }

        # Construct full S3 path
        s3_path = MODEL_SAVE_PATH.rstrip('/') + '/' + model_filename

        # Save to S3 (with fallback to local)
        print(f"         → ", end='')
        save_model_to_s3(model_dict, s3_path)

    if epochs_without_improvement >= patience:
        print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
        print(f"   Best validation combined loss: {best_val_loss:.4f} at epoch {best_val_epoch}")
        print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
        break

print("\n" + "="*70)
print("TRAINING COMPLETE - Games and Puzzles Training")
print("="*70)
print(f"Best Epoch: {best_val_epoch}")
print(f"  Validation Combined Loss: {best_val_loss:.4f}")
print(f"  Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"\nFinal Epoch: {epoch}")
print(f"  Train Accuracy: {train_acc*100:.2f}% (sampled)")
print(f"  Validation Accuracy: {val_acc*100:.2f}%")
print(f"  Gap: {(train_acc - val_acc)*100:+.2f}%")
print(f"  Policy Loss: {val_policy_loss:.4f}")
print(f"  Value Loss: {val_value_loss:.4f}")
print(f"\nBest model saved as: games_and_puzzles_{best_val_epoch:03d}_{timestamp}.pth")
print("="*70)
