"""
Play a match between two neural network engines
Compare different models or training checkpoints

Usage: python engine_vs_engine.py <model1_path> <model2_path> --games 50 --time 1.0
"""
import chess
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from encoder import BoardEncoder
from chess_net_source_dest import ChessNetSourceDest


class NeuralEngine:
    """Neural network chess engine"""

    def __init__(self, model_path, name=None):
        self.model_path = model_path
        self.name = name or Path(model_path).stem
        self.encoder = BoardEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['network_config']

        self.network = ChessNetSourceDest(
            input_channels=config['input_channels'],
            num_res_blocks=config['num_res_blocks'],
            num_channels=config['num_channels'],
            dropout=config.get('dropout', 0.0)
        ).to(self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()

        # Store model info
        self.epoch = checkpoint.get('epoch', 'N/A')
        self.val_accuracy = checkpoint.get('validation_accuracy', None)

    def evaluate_position(self, board):
        """Evaluate a position using the neural network"""
        with torch.no_grad():
            board_tensor = torch.FloatTensor(
                self.encoder.encode_board(board)
            ).unsqueeze(0).to(self.device)

            _, _, value = self.network(board_tensor)
            return value.item()

    def get_move(self, board):
        """Get best move using policy + value-based evaluation"""
        with torch.no_grad():
            # Get policy probabilities for current position
            board_tensor = torch.FloatTensor(
                self.encoder.encode_board(board)
            ).unsqueeze(0).to(self.device)

            source_logits, dest_logits, _ = self.network(board_tensor)

            # Get probabilities
            source_probs = F.softmax(source_logits[0], dim=0).cpu().numpy()
            dest_probs = F.softmax(dest_logits[0], dim=0).cpu().numpy()

            # Evaluate each legal move by looking at resulting position
            best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
            best_move = None

            for move in board.legal_moves:
                # Policy prior for this move
                policy_prob = source_probs[move.from_square] * dest_probs[move.to_square]

                # Make the move
                board.push(move)

                # Evaluate resulting position
                if board.is_game_over():
                    # Terminal position
                    result = board.result()
                    if result == "1-0":
                        eval_score = 10.0  # White wins
                    elif result == "0-1":
                        eval_score = -10.0  # Black wins
                    else:
                        eval_score = 0.0  # Draw
                else:
                    eval_score = self.evaluate_position(board)

                # Undo the move
                board.pop()

                # Combine policy and value: use value as primary, policy as tiebreaker
                # Value is in range [-1, 1] (from tanh), so add small policy bonus
                combined_score = eval_score + 0.1 * policy_prob

                # Update best move
                if board.turn == chess.WHITE:
                    if combined_score > best_score:
                        best_score = combined_score
                        best_move = move
                else:
                    if combined_score < best_score:
                        best_score = combined_score
                        best_move = move

            return best_move


def play_game(engine1, engine2, engine1_plays_white, game_log=None):
    """Play one game between two engines"""
    board = chess.Board()
    moves = []
    san_moves = []

    while not board.is_game_over():
        if len(list(board.legal_moves)) == 0:
            break

        # Check for draw conditions
        if board.can_claim_draw():
            return 0.5, moves, "draw", san_moves

        # Who's turn?
        white_to_move = board.turn == chess.WHITE
        engine1_turn = (engine1_plays_white and white_to_move) or \
                      (not engine1_plays_white and not white_to_move)

        if engine1_turn:
            # Engine 1's turn
            move = engine1.get_move(board)
            if move is None:
                # No legal move found (shouldn't happen)
                return 0.0 if engine1_plays_white else 1.0, moves, "illegal", san_moves
        else:
            # Engine 2's turn
            move = engine2.get_move(board)
            if move is None:
                # No legal move found (shouldn't happen)
                return 1.0 if engine1_plays_white else 0.0, moves, "illegal", san_moves

        # Get SAN notation before pushing the move
        san = board.san(move)
        san_moves.append(san)

        board.push(move)
        moves.append(move)

        # Limit game length to prevent infinite games
        if len(moves) >= 200:
            return 0.5, moves, "draw (move limit)", san_moves

    # Determine result
    result = board.result()

    if result == "1-0":  # White wins
        score = 1.0 if engine1_plays_white else 0.0
        outcome = "win" if engine1_plays_white else "loss"
    elif result == "0-1":  # Black wins
        score = 0.0 if engine1_plays_white else 1.0
        outcome = "loss" if engine1_plays_white else "win"
    else:  # Draw
        score = 0.5
        outcome = "draw"

    return score, moves, outcome, san_moves


def play_match(model1_path, model2_path, num_games=50, name1=None, name2=None):
    """Play a match between two neural engines"""

    print("=" * 80)
    print("ENGINE vs ENGINE MATCH")
    print("=" * 80)

    # Load engines
    print("Loading engines...")
    engine1 = NeuralEngine(model1_path, name1)
    engine2 = NeuralEngine(model2_path, name2)
    print(f"✓ Engine 1 loaded: {engine1.name}")
    if engine1.val_accuracy is not None:
        print(f"  Epoch {engine1.epoch}, Val Accuracy: {engine1.val_accuracy*100:.2f}%")
    print(f"✓ Engine 2 loaded: {engine2.name}")
    if engine2.val_accuracy is not None:
        print(f"  Epoch {engine2.epoch}, Val Accuracy: {engine2.val_accuracy*100:.2f}%")
    print()

    print(f"Games to play: {num_games}")
    print()

    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"engine_vs_engine_{timestamp}.txt"
    game_log = open(log_file, 'w')
    game_log.write(f"Engine vs Engine Match Log\n")
    game_log.write(f"{'='*80}\n")
    game_log.write(f"Engine 1: {engine1.name}\n")
    if engine1.val_accuracy is not None:
        game_log.write(f"  Epoch {engine1.epoch}, Val Accuracy: {engine1.val_accuracy*100:.2f}%\n")
    game_log.write(f"Engine 2: {engine2.name}\n")
    if engine2.val_accuracy is not None:
        game_log.write(f"  Epoch {engine2.epoch}, Val Accuracy: {engine2.val_accuracy*100:.2f}%\n")
    game_log.write(f"Games: {num_games}\n")
    game_log.write(f"{'='*80}\n\n")

    # Play match
    engine1_wins = 0
    engine2_wins = 0
    draws = 0

    start_time = time.time()

    for game_num in range(1, num_games + 1):
        # Alternate colors
        engine1_plays_white = (game_num % 2 == 1)

        white_name = engine1.name if engine1_plays_white else engine2.name
        black_name = engine2.name if engine1_plays_white else engine1.name

        print(f"Game {game_num}/{num_games} (White: {white_name[:20]}, Black: {black_name[:20]})... ",
              end="", flush=True)

        # Log game header
        game_log.write(f"\n{'='*80}\n")
        game_log.write(f"GAME {game_num}/{num_games}\n")
        game_log.write(f"White: {white_name}\n")
        game_log.write(f"Black: {black_name}\n")
        game_log.write(f"{'='*80}\n\n")

        score, moves, outcome, san_moves = play_game(
            engine1,
            engine2,
            engine1_plays_white
        )

        # Format moves in standard chess notation
        move_text = ""
        for i in range(0, len(san_moves), 2):
            move_num = (i // 2) + 1
            white_move = san_moves[i]
            black_move = san_moves[i+1] if i+1 < len(san_moves) else ""
            move_text += f"{move_num}. {white_move} {black_move} "
            if move_num % 5 == 0:  # Line break every 5 moves
                move_text += "\n"

        game_log.write(f"{move_text.strip()}\n\n")

        # Determine winner
        if "draw" in outcome:
            draws += 1
            result_str = "= DRAW"
            winner = "Draw"
        elif outcome == "win":
            engine1_wins += 1
            result_str = f"✓ {engine1.name[:20]} WINS"
            winner = engine1.name
        else:  # loss
            engine2_wins += 1
            result_str = f"✓ {engine2.name[:20]} WINS"
            winner = engine2.name

        game_log.write(f"Result: {outcome.upper()} - {winner} ({len(moves)} moves)\n")

        print(f"{result_str} ({len(moves)} moves)")

        # Show progress every 10 games
        if game_num % 10 == 0:
            engine1_score = engine1_wins + 0.5 * draws
            percentage = (engine1_score / game_num) * 100
            print(f"  Progress: {engine1.name[:30]}: {engine1_wins}W-{engine2_wins}L-{draws}D ({percentage:.1f}%)")
            print()

    elapsed = time.time() - start_time

    # Close log file
    game_log.write(f"\n{'='*80}\n")
    game_log.write(f"FINAL RESULTS\n")
    game_log.write(f"{'='*80}\n")
    game_log.write(f"{engine1.name}: {engine1_wins}W - {engine2_wins}L - {draws}D\n")
    game_log.write(f"Score: {engine1_wins + 0.5 * draws}/{num_games} ({(engine1_wins + 0.5 * draws) / num_games * 100:.1f}%)\n")
    game_log.close()

    # Calculate results
    engine1_score = engine1_wins + 0.5 * draws
    engine2_score = engine2_wins + 0.5 * draws
    engine1_percentage = (engine1_score / num_games) * 100
    engine2_percentage = (engine2_score / num_games) * 100

    # Calculate ELO difference
    import math
    win_ratio = engine1_score / num_games
    if win_ratio >= 0.99:
        win_ratio = 0.99
    elif win_ratio <= 0.01:
        win_ratio = 0.01

    elo_diff = 400 * math.log10(win_ratio / (1 - win_ratio))

    # Print results
    print()
    print("=" * 80)
    print("MATCH RESULTS")
    print("=" * 80)
    print(f"Games played: {num_games}")
    print()
    print(f"{engine1.name}:")
    print(f"  Record: {engine1_wins}W - {engine2_wins}L - {draws}D")
    print(f"  Score: {engine1_score}/{num_games} ({engine1_percentage:.1f}%)")
    print()
    print(f"{engine2.name}:")
    print(f"  Record: {engine2_wins}W - {engine1_wins}L - {draws}D")
    print(f"  Score: {engine2_score}/{num_games} ({engine2_percentage:.1f}%)")
    print()
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print()

    # Determine winner
    if engine1_wins > engine2_wins:
        print(f"🏆 WINNER: {engine1.name}")
        print(f"   Estimated ELO advantage: +{elo_diff:.0f}")
    elif engine2_wins > engine1_wins:
        print(f"🏆 WINNER: {engine2.name}")
        print(f"   Estimated ELO advantage: {elo_diff:.0f}")
    else:
        print("🤝 TIE - Both engines are evenly matched!")

    print()

    # Interpretation
    if engine1_percentage >= 75:
        print(f"✓ {engine1.name} is significantly stronger")
    elif engine1_percentage >= 60:
        print(f"✓ {engine1.name} is stronger")
    elif engine1_percentage >= 55:
        print(f"✓ {engine1.name} is slightly stronger")
    elif engine1_percentage >= 45:
        print("≈ Evenly matched!")
    elif engine1_percentage >= 40:
        print(f"✓ {engine2.name} is slightly stronger")
    elif engine1_percentage >= 25:
        print(f"✓ {engine2.name} is stronger")
    else:
        print(f"✓ {engine2.name} is significantly stronger")

    print()
    print("=" * 80)

    # Save results
    results_file = f"engine_vs_engine_results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Engine 1: {model1_path}\n")
        f.write(f"Engine 2: {model2_path}\n")
        f.write(f"Games: {num_games}\n")
        f.write(f"\n{engine1.name} Result: {engine1_wins}W - {engine2_wins}L - {draws}D\n")
        f.write(f"{engine1.name} Score: {engine1_percentage:.1f}%\n")
        f.write(f"\n{engine2.name} Result: {engine2_wins}W - {engine1_wins}L - {draws}D\n")
        f.write(f"{engine2.name} Score: {engine2_percentage:.1f}%\n")
        f.write(f"\nELO Difference: {elo_diff:+.0f} (in favor of {engine1.name if elo_diff > 0 else engine2.name})\n")

    print(f"Results saved to: {results_file}")
    print(f"Game log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='Play match between two neural engines')
    parser.add_argument('model1', help='Path to first model file (.pth)')
    parser.add_argument('model2', help='Path to second model file (.pth)')
    parser.add_argument('--games', type=int, default=50, help='Number of games to play (default: 50)')
    parser.add_argument('--name1', type=str, default=None, help='Name for first engine (default: filename)')
    parser.add_argument('--name2', type=str, default=None, help='Name for second engine (default: filename)')

    args = parser.parse_args()

    play_match(args.model1, args.model2, args.games, args.name1, args.name2)


if __name__ == "__main__":
    main()
