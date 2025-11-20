"""
Training on Stockfish Evaluations
Configuration: 8 ResNet blocks, 256 channels
Target: Position evaluation (centipawns)
"""
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
import os
import tempfile
from datetime import datetime
from torchinfo import summary

from mini_leela_complete_fixed import BoardEncoder
from chess_net_value import ChessNetValue

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
    """
    Save model to S3 using Databricks dbutils

    Args:
        model_dict: Dictionary containing model state and metadata
        s3_path: Full S3 path (e.g., s3://bucket/path/model.pth)
    """
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
TRAIN_CSV_FILE = "training_data_with_evals.csv"
VALIDATION_CSV_FILE = "validation_data_with_evals.csv"

print("="*70)
print("STOCKFISH EVALUATION TRAINING")
print("CONFIG: 8 ResNet blocks, 256 channels")
print("="*70)
print("\nExpected: ~8-9M parameters")
print("Target: Position evaluation (centipawns)")
print("\nTraining Configuration:")
print("  Learning Rate: 0.001")
print("  Dropout: 0.1")
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
        # Stockfish eval is in centipawns, already normalized
        # Format: fen, answer (unused), stockfish_eval, rating (unused)
        training_positions.append((row['fen'], float(row['stockfish_eval'])))

print(f"Loaded {len(training_positions):,} training positions")

# For validation, we'll use mate-in-1 positions with evaluation +/-10900 (forced mate)
print(f"Loading validation data from {VALIDATION_CSV_FILE}...")
validation_positions = []
with open(VALIDATION_CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Mate-in-1 positions should have high evaluation
        # We'll assign +10900 for white to move, -10900 for black to move
        board = chess.Board(row['fen'])
        eval_score = 10900.0 if board.turn == chess.WHITE else -10900.0
        validation_positions.append((row['fen'], eval_score))

print(f"Loaded {len(validation_positions):,} validation positions\n")

# MODEL CONFIGURATION
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.001
BATCH_SIZE = 64

network = ChessNetValue(
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
print("Preparing training data...")
train_data = []
for fen, eval_score in training_positions:
    board = chess.Board(fen)
    board_state = encoder.encode_board(board)
    # Normalize eval score (divide by 100 to get it in pawn units)
    normalized_eval = eval_score / 100.0
    train_data.append((board_state, normalized_eval))

print(f"Training data prepared: {len(train_data):,} positions\n")

# Training function
def train_epoch(data, network, optimizer, batch_size, device):
    network.train()
    random.shuffle(data)

    epoch_loss = 0.0
    num_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        states = torch.FloatTensor(np.array([d[0] for d in batch])).to(device)
        eval_targets = torch.FloatTensor(np.array([d[1] for d in batch])).unsqueeze(1).to(device)

        values = network(states)

        # Use Huber loss (less sensitive to outliers than MSE)
        loss = F.smooth_l1_loss(values, eval_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches

# Evaluation function
def evaluate(positions, network, device):
    network.eval()
    total_loss = 0.0
    total_mae = 0.0  # Mean Absolute Error

    with torch.no_grad():
        for fen, eval_score in positions:
            board = chess.Board(fen)
            board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0).to(device)

            predicted_value = network(board_tensor)

            # Normalize eval score
            normalized_eval = eval_score / 100.0
            eval_target = torch.FloatTensor([[normalized_eval]]).to(device)

            loss = F.smooth_l1_loss(predicted_value, eval_target)
            mae = torch.abs(predicted_value - eval_target).item()

            total_loss += loss.item()
            total_mae += mae

    avg_loss = total_loss / len(positions)
    avg_mae = total_mae / len(positions)

    return avg_loss, avg_mae

# Training loop
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Training run ID: {timestamp}\n")

num_epochs = 100
best_val_loss = float('inf')
best_val_epoch = 0
patience = 15
epochs_without_improvement = 0

print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val MAE':<12} {'Status'}")
print("-" * 70)

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(train_data, network, optimizer, BATCH_SIZE, device)

    # Evaluate on a sample of training data (to save time)
    train_sample = random.sample(training_positions, min(5000, len(training_positions)))
    train_loss_eval, train_mae = evaluate(train_sample, network, device)

    # Evaluate on full validation set
    val_loss, val_mae = evaluate(validation_positions, network, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch
        epochs_without_improvement = 0
        status = "✓ BEST"
    else:
        epochs_without_improvement += 1
        status = ""

    print(f"{epoch:<6} {train_loss:.6f}     "
          f"{val_loss:.6f}     "
          f"{val_mae:.4f}       "
          f"{status}")

    # Save best model
    if status:
        model_filename = f"stockfish_eval_best_{timestamp}.pth"
        model_dict = {
            'model_state_dict': network.state_dict(),
            'network_config': {
                'input_channels': 19,
                'num_res_blocks': 8,
                'num_channels': 256,
                'dropout': DROPOUT_RATE
            },
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'validation_mae': val_mae,
            'timestamp': timestamp,
        }

        # Construct full S3 path
        s3_path = MODEL_SAVE_PATH.rstrip('/') + '/' + model_filename

        # Save to S3 (with fallback to local)
        print(f"         → ", end='')
        save_model_to_s3(model_dict, s3_path)

    if epochs_without_improvement >= patience:
        print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
        print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_val_epoch}")
        break

print("\n" + "="*70)
print("TRAINING COMPLETE - Stockfish Evaluation Model")
print("="*70)
print(f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_val_epoch}")
print(f"Best Validation MAE: {val_mae:.4f} pawns")
print(f"\nBest model saved as: stockfish_eval_best_{timestamp}.pth")
print("="*70)
