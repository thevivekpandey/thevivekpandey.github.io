"""
Validate that all positions in mate_in_1_positions.py are correct:
1. The FEN is valid
2. The solution move is legal
3. The solution move results in checkmate
4. It's actually mate-in-1 (no other moves lead to immediate mate)
"""
import chess
from mate_in_1_positions import TRAINING_POSITIONS, TEST_POSITIONS

def validate_position(fen, solution_uci, description, index, dataset_name):
    """Validate a single position. Returns (is_valid, error_message)"""
    try:
        board = chess.Board(fen)
    except Exception as e:
        return False, f"Invalid FEN: {e}"

    # Check if solution move is legal
    try:
        solution_move = chess.Move.from_uci(solution_uci)
    except Exception as e:
        return False, f"Invalid UCI move '{solution_uci}': {e}"

    if solution_move not in board.legal_moves:
        return False, f"Move {solution_uci} is not legal in this position"

    # Check if the move leads to checkmate
    board.push(solution_move)
    if not board.is_checkmate():
        board.pop()
        return False, f"Move {solution_uci} does not result in checkmate"

    board.pop()

    # Check if it's truly mate-in-1 (only one move leads to mate)
    mate_moves = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mate_moves.append(move.uci())
        board.pop()

    if len(mate_moves) == 0:
        return False, f"No moves lead to checkmate!"

    if len(mate_moves) > 1 and solution_uci not in mate_moves:
        return False, f"Solution {solution_uci} not in mate moves: {mate_moves}"

    # Note if there are multiple mate-in-1 moves
    if len(mate_moves) > 1:
        return True, f"WARNING: Multiple mate-in-1 moves found: {mate_moves}"

    return True, "OK"

def validate_dataset(positions, dataset_name):
    """Validate all positions in a dataset"""
    print(f"\n{'='*70}")
    print(f"Validating {dataset_name} ({len(positions)} positions)")
    print(f"{'='*70}\n")

    valid_count = 0
    invalid_count = 0
    warning_count = 0
    errors = []

    for i, (fen, solution_uci, description) in enumerate(positions):
        is_valid, message = validate_position(fen, solution_uci, description, i, dataset_name)

        if is_valid:
            if "WARNING" in message:
                warning_count += 1
                if warning_count <= 5:  # Only print first 5 warnings
                    print(f"⚠️  {dataset_name}[{i}]: {message}")
                    print(f"    FEN: {fen}")
                    print(f"    Solution: {solution_uci}")
                    print()
            else:
                valid_count += 1
        else:
            invalid_count += 1
            errors.append((i, fen, solution_uci, description, message))
            if invalid_count <= 10:  # Only print first 10 errors
                print(f"❌ {dataset_name}[{i}]: {message}")
                print(f"    FEN: {fen}")
                print(f"    Solution: {solution_uci}")
                print(f"    Description: {description}")
                print()

    # Summary
    print(f"\n{'-'*70}")
    print(f"SUMMARY for {dataset_name}:")
    print(f"  ✓ Valid: {valid_count}/{len(positions)} ({valid_count/len(positions)*100:.1f}%)")
    print(f"  ⚠️  Warnings (multiple mates): {warning_count}/{len(positions)} ({warning_count/len(positions)*100:.1f}%)")
    print(f"  ❌ Invalid: {invalid_count}/{len(positions)} ({invalid_count/len(positions)*100:.1f}%)")

    if invalid_count > 10:
        print(f"\n  (Showing first 10 errors, {invalid_count - 10} more errors not shown)")

    return valid_count, invalid_count, warning_count, errors

# Validate both datasets
print("Starting validation...")
print(f"This may take a minute...\n")

train_valid, train_invalid, train_warnings, train_errors = validate_dataset(TRAINING_POSITIONS, "TRAINING")
test_valid, test_invalid, test_warnings, test_errors = validate_dataset(TEST_POSITIONS, "TEST")

# Overall summary
print(f"\n{'='*70}")
print("OVERALL RESULTS")
print(f"{'='*70}")
print(f"Training: {train_valid} valid, {train_warnings} warnings, {train_invalid} invalid")
print(f"Test:     {test_valid} valid, {test_warnings} warnings, {test_invalid} invalid")
print(f"Total:    {train_valid + test_valid} valid out of {len(TRAINING_POSITIONS) + len(TEST_POSITIONS)}")

if train_invalid > 0 or test_invalid > 0:
    print(f"\n⚠️  CRITICAL: Found {train_invalid + test_invalid} invalid positions!")
    print("   These positions have incorrect labels and should be removed.")
else:
    print(f"\n✓ All positions are valid!")

if train_warnings > 0 or test_warnings > 0:
    print(f"\n⚠️  Note: {train_warnings + test_warnings} positions have multiple mate-in-1 solutions.")
    print("   This is OK, but the model needs to predict the specific solution given.")
