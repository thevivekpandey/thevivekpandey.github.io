"""
Evaluate multiple epoch models on test set
Load models from epoch 1 to N and check test accuracy progression
"""
import chess
import torch
import glob
import re
from collections import defaultdict

from mini_leela_complete_fixed import ChessNet, BoardEncoder, MoveEncoder
from mate_in_1_positions import TRAINING_POSITIONS, TEST_POSITIONS

print("="*70)
print("Evaluating Epoch Models on Test Set")
print("="*70)

# Get model pattern from user
import sys
if len(sys.argv) > 1:
    pattern = sys.argv[1]
else:
    # Default pattern - user can override
    pattern = "supervised_model_epoch*.pth"

print(f"\nSearching for models matching: {pattern}")

# Find all model files
model_files = glob.glob(pattern)
if not model_files:
    print(f"❌ No models found matching pattern: {pattern}")
    print("\nUsage: python evaluate_epoch_models.py 'supervised_model_epoch*_TIMESTAMP.pth'")
    sys.exit(1)

# Extract epoch numbers and sort
epoch_models = []
for model_file in model_files:
    match = re.search(r'epoch(\d+)', model_file)
    if match:
        epoch_num = int(match.group(1))
        epoch_models.append((epoch_num, model_file))

epoch_models.sort(key=lambda x: x[0])
print(f"Found {len(epoch_models)} models (epochs {epoch_models[0][0]} to {epoch_models[-1][0]})")

# Setup
encoder = BoardEncoder()
move_encoder = MoveEncoder()
device = 'cpu'

def evaluate_model(model_path, positions, description=""):
    """Evaluate a model on a set of positions"""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['network_config']

    network = ChessNet(
        input_channels=config['input_channels'],
        num_res_blocks=config['num_res_blocks'],
        num_channels=config['num_channels']
    )
    network.load_state_dict(checkpoint['model_state_dict'])
    network.to(device)
    network.eval()

    # Evaluate
    correct = 0
    for fen, solution_uci, _ in positions:
        board = chess.Board(fen)
        solution_move = chess.Move.from_uci(solution_uci)
        solution_idx = move_encoder.move_to_index(solution_move)

        board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = network(board_tensor)

        # Get top predicted move among legal moves
        legal_moves = list(board.legal_moves)
        legal_indices = [move_encoder.move_to_index(m) for m in legal_moves]
        legal_logits = policy_logits[0, legal_indices]
        best_legal_idx = legal_indices[legal_logits.argmax().item()]

        if best_legal_idx == solution_idx:
            correct += 1

    accuracy = correct / len(positions)
    return correct, len(positions), accuracy

# Evaluate all models
print("\n" + "="*70)
print("Evaluating on Training and Test Sets")
print("="*70)
print(f"Training set size: {len(TRAINING_POSITIONS)}")
print(f"Test set size: {len(TEST_POSITIONS)}")
print()

results = []

print(f"{'Epoch':<6} {'Train Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
print("-" * 70)

for epoch_num, model_path in epoch_models:
    # Evaluate on training set
    train_correct, train_total, train_acc = evaluate_model(model_path, TRAINING_POSITIONS, "train")

    # Evaluate on test set
    test_correct, test_total, test_acc = evaluate_model(model_path, TEST_POSITIONS, "test")

    # Calculate overfitting gap
    overfit_gap = train_acc - test_acc

    results.append({
        'epoch': epoch_num,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'overfit_gap': overfit_gap,
        'train_correct': train_correct,
        'test_correct': test_correct
    })

    # Format output
    train_str = f"{train_correct}/{train_total} ({train_acc*100:.1f}%)"
    test_str = f"{test_correct}/{test_total} ({test_acc*100:.1f}%)"
    overfit_str = f"{overfit_gap*100:+.1f}%"

    print(f"{epoch_num:<6} {train_str:<12} {test_str:<12} {overfit_str:<12}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

best_test_epoch = max(results, key=lambda x: x['test_acc'])
best_train_epoch = max(results, key=lambda x: x['train_acc'])
least_overfit_epoch = min(results, key=lambda x: x['overfit_gap'])

print(f"\nBest Test Accuracy:")
print(f"  Epoch {best_test_epoch['epoch']}: {best_test_epoch['test_acc']*100:.2f}% "
      f"({best_test_epoch['test_correct']}/{len(TEST_POSITIONS)})")

print(f"\nBest Training Accuracy:")
print(f"  Epoch {best_train_epoch['epoch']}: {best_train_epoch['train_acc']*100:.2f}% "
      f"({best_train_epoch['train_correct']}/{len(TRAINING_POSITIONS)})")

print(f"\nLeast Overfitting:")
print(f"  Epoch {least_overfit_epoch['epoch']}: Gap = {least_overfit_epoch['overfit_gap']*100:+.2f}%")

# Check for overfitting trend
if len(results) >= 5:
    last_5_overfit = [r['overfit_gap'] for r in results[-5:]]
    avg_recent_overfit = sum(last_5_overfit) / len(last_5_overfit)

    print(f"\nRecent Overfitting Trend (last 5 epochs):")
    print(f"  Average gap: {avg_recent_overfit*100:+.2f}%")

    if avg_recent_overfit > 0.05:
        print("  ⚠️  Model may be overfitting - test accuracy not improving")
    elif avg_recent_overfit < 0:
        print("  ⚠️  Test accuracy > train accuracy (unusual - check data)")
    else:
        print("  ✓ Good generalization - train and test accuracy similar")

# Final accuracy analysis
final = results[-1]
print(f"\nFinal Model (Epoch {final['epoch']}):")
print(f"  Training: {final['train_acc']*100:.2f}%")
print(f"  Test:     {final['test_acc']*100:.2f}%")
print(f"  Gap:      {final['overfit_gap']*100:+.2f}%")

if final['test_acc'] >= 0.95:
    print("\n🎉 Excellent! Model generalizes very well (95%+ test accuracy)")
elif final['test_acc'] >= 0.85:
    print("\n✓ Good! Model generalizes well (85%+ test accuracy)")
elif final['test_acc'] >= 0.70:
    print("\n⚠️  Moderate generalization (70-85% test accuracy)")
else:
    print("\n⚠️  Poor generalization (<70% test accuracy)")

print("\n" + "="*70)
