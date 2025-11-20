#!/usr/bin/env python3
"""
Generate training data from high-quality chess games
Each training example: FEN position -> move played by strong player

Usage: python generate_training_data.py --pgn trimmed_lichess_db_standard_rated_2024-11.pgn --output training_data.csv
"""
import chess
import chess.pgn
import csv
import argparse
from tqdm import tqdm


def generate_training_data_from_pgn(pgn_file, output_file, max_games=None,
                                     skip_opening_moves=0, skip_endgame_moves=0):
    """
    Extract training positions from PGN games.
    Each example is (FEN position, move played by strong player)

    Args:
        pgn_file: Path to PGN file with games
        output_file: Output CSV file
        max_games: Maximum games to process (None = all games)
        skip_opening_moves: Skip first N moves (default: 0)
        skip_endgame_moves: Skip last N moves (default: 0)
    """

    print(f"Loading games from {pgn_file}...")
    print(f"Output will be written to {output_file}")

    # Output CSV with headers
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fen', 'move', 'white_elo', 'black_elo', 'side_to_move'])

        games_processed = 0
        positions_collected = 0

        with open(pgn_file) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                if max_games and games_processed >= max_games:
                    break

                # Get ELO ratings from headers
                headers = game.headers
                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                except:
                    white_elo = 0
                    black_elo = 0

                # Get all moves in the game
                moves = list(game.mainline_moves())

                # Skip very short games (less than 10 moves total)
                if len(moves) < 10:
                    games_processed += 1
                    continue

                # Generate training examples from this game
                board = game.board()

                for move_num, move in enumerate(moves):
                    # Skip opening moves (may be from book)
                    if move_num < skip_opening_moves:
                        board.push(move)
                        continue

                    # Skip endgame moves (may have time pressure blunders)
                    if move_num >= len(moves) - skip_endgame_moves:
                        break

                    # Record the position before the move
                    fen = board.fen()
                    move_uci = move.uci()
                    side = 'white' if board.turn == chess.WHITE else 'black'

                    # Write training example
                    writer.writerow([
                        fen,
                        move_uci,
                        white_elo,
                        black_elo,
                        side
                    ])
                    positions_collected += 1

                    # Make the move for next iteration
                    board.push(move)

                games_processed += 1

                # Progress update
                if games_processed % 1000 == 0:
                    print(f"Processed {games_processed} games, collected {positions_collected} positions")

        print(f"\nDone! Collected {positions_collected} training positions from {games_processed} games")
        print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate chess training data from PGN games')
    parser.add_argument('--pgn', required=True, help='Input PGN file')
    parser.add_argument('--output', default='training_data.csv', help='Output CSV file')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum games to process (default: all)')
    parser.add_argument('--skip-opening', type=int, default=5,
                       help='Skip first N moves of each game (default: 0)')
    parser.add_argument('--skip-endgame', type=int, default=0,
                       help='Skip last N moves of each game (default: 0)')

    args = parser.parse_args()

    generate_training_data_from_pgn(
        args.pgn,
        args.output,
        max_games=args.max_games,
        skip_opening_moves=args.skip_opening,
        skip_endgame_moves=args.skip_endgame
    )


if __name__ == "__main__":
    main()
