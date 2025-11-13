import csv

def generate_positions_file(input_csv, output_py, num_training=10000, num_test=4000):
    """
    Generate mate_in_1_positions.py from the processed CSV file.
    """
    positions = []

    # Read all positions from CSV
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append((row['fen'], row['answer'], row['rating']))

    print(f"Read {len(positions)} positions from {input_csv}")

    # Split into training and test sets
    training_positions = positions[:num_training]
    test_positions = positions[num_training:num_training + num_test]

    print(f"Training positions: {len(training_positions)}")
    print(f"Test positions: {len(test_positions)}")

    # Generate the Python file
    with open(output_py, 'w') as f:
        # Write header
        f.write('"""\n')
        f.write('Mate-in-1 positions from Lichess puzzle database\n')
        f.write(f'{num_training} training + {num_test} test puzzles\n')
        f.write('Sampled uniformly across all difficulty ratings for challenging learning\n')
        f.write('"""\n\n')

        # Write training positions
        f.write(f'# {num_training} training positions\n')
        f.write('# Format: (fen, answer_move, rating)\n')
        f.write('TRAINING_POSITIONS = [\n')
        for fen, move, rating in training_positions:
            f.write(f"    ('{fen}', '{move}', {rating}),\n")
        f.write(']\n\n')

        # Write test positions
        f.write(f'# {num_test} test positions\n')
        f.write('# Format: (fen, answer_move, rating)\n')
        f.write('TEST_POSITIONS = [\n')
        for fen, move, rating in test_positions:
            f.write(f"    ('{fen}', '{move}', {rating}),\n")
        f.write(']\n\n')

        # Write helper functions
        f.write('def get_training_fens():\n')
        f.write('    """Return list of training position FENs"""\n')
        f.write('    return [pos[0] for pos in TRAINING_POSITIONS]\n\n')

        f.write('def get_test_fens():\n')
        f.write('    """Return list of test position FENs"""\n')
        f.write('    return [pos[0] for pos in TEST_POSITIONS]\n\n')

        f.write('def validate_positions():\n')
        f.write('    """Validate that all positions have the correct format"""\n')
        f.write('    import chess\n')
        f.write('    for pos_list in [TRAINING_POSITIONS, TEST_POSITIONS]:\n')
        f.write('        for fen, move, rating in pos_list:\n')
        f.write('            try:\n')
        f.write('                board = chess.Board(fen)\n')
        f.write('                board.parse_san(move)\n')
        f.write('            except Exception as e:\n')
        f.write('                print(f"Invalid position (rating {rating}): {e}")\n')
        f.write('                return False\n')
        f.write('    return True\n')

    print(f"Generated {output_py}")

if __name__ == "__main__":
    input_csv = "mate_in_1_processed.csv"
    output_py = "mate_in_1_positions.py"

    generate_positions_file(input_csv, output_py, num_training=200000, num_test=20000)
    print("Done!")
