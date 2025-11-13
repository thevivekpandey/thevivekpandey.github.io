"""
Validate mate-in-1 positions from mate_in_1.csv
Only keeps positions where:
1. It's white to move
2. White has at least one move that delivers checkmate
Writes validated positions to mate_in_1_validated.csv
"""
import csv
import chess

print("="*70)
print("Validating mate-in-1 puzzles from mate_in_1.csv")
print("="*70)
print()

valid_count = 0
invalid_count = 0
not_white_to_move = 0
no_mate_found = 0

print("Reading and validating mate_in_1.csv...")

with open('mate_in_1.csv', 'r', encoding='utf-8') as infile, \
     open('mate_in_1_validated.csv', 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i:,} rows - Valid: {valid_count:,}, Invalid: {invalid_count:,}")

        # CSV format: PuzzleId, FEN, Moves, Rating, ...
        if len(row) < 3:
            invalid_count += 1
            continue

        puzzle_id = row[0]
        fen = row[1]
        moves = row[2]

        if not fen:
            invalid_count += 1
            continue

        try:
            board = chess.Board(fen)

            # Check if it's white to move
            if not board.turn:  # board.turn is True for white, False for black
                not_white_to_move += 1
                invalid_count += 1
                continue

            # Check if white has a mate-in-1 move
            has_mate_in_1 = False
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    has_mate_in_1 = True
                    board.pop()
                    break
                board.pop()

            if has_mate_in_1:
                # This is a valid mate-in-1 position for white
                writer.writerow(row)
                valid_count += 1

                # Print milestones
                if valid_count in [1000, 5000, 10000, 15000, 20000]:
                    print(f"✓ Found {valid_count:,} valid positions...")
            else:
                no_mate_found += 1
                invalid_count += 1

        except Exception as e:
            # Invalid FEN or other error
            invalid_count += 1
            continue

print(f"\n{'='*70}")
print(f"Validation complete!")
print(f"{'='*70}")
print(f"Valid positions (white to move, mate-in-1): {valid_count:,}")
print(f"Invalid positions: {invalid_count:,}")
print(f"  - Not white to move: {not_white_to_move:,}")
print(f"  - No mate-in-1 found: {no_mate_found:,}")
print(f"  - Other errors: {invalid_count - not_white_to_move - no_mate_found:,}")
print(f"\nOutput written to: mate_in_1_validated.csv")
print(f"{'='*70}\n")
