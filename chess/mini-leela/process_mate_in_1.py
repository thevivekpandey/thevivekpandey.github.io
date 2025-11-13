import chess
import csv

def process_mate_in_1_puzzles(input_file, output_file):
    """
    Process mate in 1 puzzles from the input CSV file.
    For each puzzle:
    1. Parse the FEN position
    2. Apply the first move to get the actual puzzle position
    3. Extract the second move as the answer
    4. Extract the rating (first number in the CSV)
    5. Write to output CSV
    """
    processed_count = 0
    error_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['fen', 'answer', 'rating'])

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the line - format is: id,fen,moves,rating,num2,num3,num4,tags,url,opening
                parts = line.strip().split(',')
                if len(parts) < 4:
                    continue

                puzzle_id = parts[0]
                fen = parts[1]
                moves_str = parts[2]
                rating = parts[3]  # First number is the rating

                # Parse the moves (format: "move1 move2")
                moves = moves_str.strip().split()
                if len(moves) < 2:
                    print(f"Line {line_num}: Not enough moves - {moves_str}")
                    error_count += 1
                    continue

                first_move = moves[0]
                answer_move = moves[1]

                # Create a board from the FEN
                board = chess.Board(fen)

                # Apply the first move
                try:
                    move = board.parse_san(first_move)
                    board.push(move)
                except ValueError:
                    # Try UCI format if SAN fails
                    try:
                        move = chess.Move.from_uci(first_move)
                        board.push(move)
                    except ValueError:
                        print(f"Line {line_num}: Invalid first move {first_move} for position {fen}")
                        error_count += 1
                        continue

                # Get the FEN after the first move (this is the puzzle position)
                puzzle_fen = board.fen()

                # Write to output
                writer.writerow([puzzle_fen, answer_move, rating])
                processed_count += 1

                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count} puzzles...")

            except Exception as e:
                print(f"Line {line_num}: Error - {e}")
                error_count += 1
                continue

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} puzzles")
    print(f"Errors: {error_count}")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "mate_in_1.csv"
    output_file = "mate_in_1_processed.csv"

    print("Processing mate in 1 puzzles...")
    process_mate_in_1_puzzles(input_file, output_file)
