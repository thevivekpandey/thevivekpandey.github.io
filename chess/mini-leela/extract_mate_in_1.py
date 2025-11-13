"""
Extract mate-in-1 puzzles from mate_in_1.csv (pre-filtered Lichess puzzles)
Creates TRAINING_POSITIONS (10K) and TEST_POSITIONS (4K)
"""
import csv
import chess
import random

print("="*70)
print("Processing mate-in-1 puzzles from mate_in_1.csv")
print("="*70)
print()

mate_in_1_puzzles = []

# Read the CSV file
print("Reading mate_in_1.csv...")
with open('mate_in_1.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)

    for i, row in enumerate(reader):
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i:,} rows, validated {len(mate_in_1_puzzles):,} puzzles...")

        # CSV format: PuzzleId, FEN, Moves, Rating, Popularity, NbPlays, Themes, GameUrl, OpeningTags
        if len(row) < 4:
            continue

        puzzle_id = row[0]
        fen = row[1]
        moves = row[2]
        rating = row[3]

        if not fen or not moves:
            continue

        # Extract solution move (first move in the sequence)
        solution_uci = moves.split()[0] if moves else ''
        if not solution_uci:
            continue

        # Validate the position
        try:
            board = chess.Board(fen)
            solution_move = chess.Move.from_uci(solution_uci)

            # Verify it's legal
            if solution_move not in board.legal_moves:
                if len(mate_in_1_puzzles) < 5:  # Debug first few
                    print(f"DEBUG: Illegal move {solution_uci} for {fen}")
                continue

            # Verify it leads to checkmate
            board.push(solution_move)
            if not board.is_checkmate():
                if len(mate_in_1_puzzles) < 5:  # Debug first few
                    print(f"DEBUG: Not checkmate - {solution_uci} for {fen}")
                continue
            board.pop()

            # Add to collection
            description = f"Lichess #{puzzle_id} (Rating: {rating})"
            try:
                rating_int = int(rating)
            except:
                rating_int = 1500
            mate_in_1_puzzles.append((fen, solution_uci, description, rating_int))

            # Print milestone
            if len(mate_in_1_puzzles) in [1000, 5000, 10000, 14000, 15000]:
                print(f"✓ Validated {len(mate_in_1_puzzles):,} puzzles...")

            # Stop once we have enough
            if len(mate_in_1_puzzles) >= 15000:
                print(f"✓ Collected enough puzzles! Stopping at {len(mate_in_1_puzzles):,}")
                break

        except Exception as e:
            # Skip invalid positions
            if len(mate_in_1_puzzles) < 5:  # Debug first few
                print(f"DEBUG: Exception {e} for FEN: {fen}, Move: {solution_uci}")
            continue

print(f"\n{'='*70}")
print(f"Successfully validated {len(mate_in_1_puzzles):,} mate-in-1 puzzles")
print(f"{'='*70}\n")

if len(mate_in_1_puzzles) < 14000:
    print(f"⚠️  WARNING: Only found {len(mate_in_1_puzzles):,} puzzles, need at least 14,000")
    if len(mate_in_1_puzzles) >= 1000:
        print("Continuing with what we have...\n")
    else:
        print("Not enough puzzles to create datasets!")
        exit(1)

# Sort by rating for good distribution, then shuffle
mate_in_1_puzzles.sort(key=lambda x: x[3])
random.seed(42)
random.shuffle(mate_in_1_puzzles)

# Split into train (10K) and test (4K)
num_train = min(10000, len(mate_in_1_puzzles))
num_test = min(4000, len(mate_in_1_puzzles) - num_train)

training_positions = [(fen, uci, desc) for fen, uci, desc, _ in mate_in_1_puzzles[:num_train]]
test_positions = [(fen, uci, desc) for fen, uci, desc, _ in mate_in_1_puzzles[num_train:num_train+num_test]]

print(f"Created datasets:")
print(f"  Training: {len(training_positions):,} positions")
print(f"  Test:     {len(test_positions):,} positions\n")

# Write to Python file
print("Writing to mate_in_1_positions.py...")
with open('mate_in_1_positions.py', 'w', encoding='utf-8') as f:
    f.write('"""\n')
    f.write('Mate-in-1 chess positions from Lichess puzzle database\n')
    f.write('Format: (FEN, solution_UCI, description)\n')
    f.write(f'Total: {len(training_positions)} training + {len(test_positions)} test positions\n')
    f.write('"""\n\n')

    f.write('TRAINING_POSITIONS = [\n')
    for fen, uci, desc in training_positions:
        fen_escaped = fen.replace("'", "\\'")
        desc_escaped = desc.replace("'", "\\'")
        f.write(f"    ('{fen_escaped}', '{uci}', '{desc_escaped}'),\n")
    f.write(']\n\n')

    f.write('TEST_POSITIONS = [\n')
    for fen, uci, desc in test_positions:
        fen_escaped = fen.replace("'", "\\'")
        desc_escaped = desc.replace("'", "\\'")
        f.write(f"    ('{fen_escaped}', '{uci}', '{desc_escaped}'),\n")
    f.write(']\n')

print(f"✓ Successfully created mate_in_1_positions.py!")
print(f"  Total puzzles: {len(training_positions) + len(test_positions):,}\n")
print(f"Next step: Validate the positions:")
print(f"  python3 validate_positions.py")
