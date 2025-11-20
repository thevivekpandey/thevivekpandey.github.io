"""
Play a match between your neural network engine and Stockfish
No cutechess-cli needed - pure Python!

Usage: python play_match.py <model_path> --games 50 --skill 5
"""
import chess
import chess.engine
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path

from mini_leela_complete_fixed import BoardEncoder
from chess_net_source_dest import ChessNetSourceDest


class NeuralEngine:
    """Your neural network chess engine"""

    def __init__(self, model_path):
        self.encoder = BoardEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['network_config']

        self.network = ChessNetSourceDest(
            input_channels=config['input_channels'],
            num_res_blocks=config['num_res_blocks'],
            num_channels=config['num_channels'],
            dropout=config['dropout']
        ).to(self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()

    def get_move(self, board):
        """Get best move from neural network"""
        with torch.no_grad():
            board_tensor = torch.FloatTensor(
                self.encoder.encode_board(board)
            ).unsqueeze(0).to(self.device)

            source_logits, dest_logits, values = self.network(board_tensor)

            # Get probabilities
            source_probs = F.softmax(source_logits[0], dim=0).cpu().numpy()
            dest_probs = F.softmax(dest_logits[0], dim=0).cpu().numpy()

            # Score legal moves
            best_score = -float('inf')
            best_move = None

            for move in board.legal_moves:
                score = source_probs[move.from_square] * dest_probs[move.to_square]
                if score > best_score:
                    best_score = score
                    best_move = move

            return best_move


def play_game(neural_engine, stockfish_engine, neural_plays_white, time_limit=1.0, game_log=None):
    """Play one game"""
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
        neural_turn = (neural_plays_white and white_to_move) or \
                     (not neural_plays_white and not white_to_move)

        if neural_turn:
            # Neural engine's turn
            move = neural_engine.get_move(board)
            if move is None:
                # No legal move found (shouldn't happen)
                return 0.0 if neural_plays_white else 1.0, moves, "illegal", san_moves
            engine_name = "Neural"
        else:
            # Stockfish's turn
            result = stockfish_engine.play(board, chess.engine.Limit(time=time_limit))
            move = result.move
            engine_name = "Stockfish"

        # Get SAN notation before pushing the move
        san = board.san(move)
        san_moves.append(san)

        board.push(move)
        moves.append(move)

    # Determine result
    result = board.result()

    if result == "1-0":  # White wins
        score = 1.0 if neural_plays_white else 0.0
        outcome = "win" if neural_plays_white else "loss"
    elif result == "0-1":  # Black wins
        score = 0.0 if neural_plays_white else 1.0
        outcome = "loss" if neural_plays_white else "win"
    else:  # Draw
        score = 0.5
        outcome = "draw"

    return score, moves, outcome, san_moves


def play_match(model_path, num_games=50, stockfish_skill=5, time_limit=1.0):
    """Play a match and calculate ELO"""

    print("=" * 70)
    print("CHESS ENGINE MATCH")
    print("=" * 70)
    print(f"Model: {Path(model_path).name}")
    print(f"Opponent: Stockfish (Skill Level {stockfish_skill})")
    print(f"Games: {num_games}")
    print(f"Time per move: {time_limit}s")
    print()

    # Stockfish skill to approximate ELO
    skill_elo = {
        0: 1350, 1: 1400, 2: 1450, 3: 1500, 4: 1525,
        5: 1550, 6: 1600, 7: 1650, 8: 1700, 9: 1750,
        10: 1800, 11: 1900, 12: 2000, 13: 2100, 14: 2150,
        15: 2200, 16: 2300, 17: 2400, 18: 2500, 19: 2650,
        20: 2850
    }
    opponent_elo = skill_elo.get(stockfish_skill, 1500)
    print(f"Stockfish approximate ELO: {opponent_elo}")
    print()

    # Load engines
    print("Loading engines...")
    neural_engine = NeuralEngine(model_path)

    # Start Stockfish
    stockfish_path = "/opt/homebrew/bin/stockfish"  # Adjust if needed
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    stockfish_engine.configure({"Skill Level": stockfish_skill})

    print("Engines loaded!")
    print()

    # Create log file
    log_file = f"match_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    game_log = open(log_file, 'w')
    game_log.write(f"Chess Match Log\n")
    game_log.write(f"{'='*70}\n")
    game_log.write(f"Model: {Path(model_path).name}\n")
    game_log.write(f"Opponent: Stockfish (Skill Level {stockfish_skill}, ~{opponent_elo} ELO)\n")
    game_log.write(f"Games: {num_games}\n")
    game_log.write(f"Time per move: {time_limit}s\n")
    game_log.write(f"{'='*70}\n\n")

    # Play match
    wins = 0
    losses = 0
    draws = 0

    start_time = time.time()

    for game_num in range(1, num_games + 1):
        # Alternate colors
        neural_plays_white = (game_num % 2 == 1)
        color_str = "White" if neural_plays_white else "Black"

        print(f"Game {game_num}/{num_games} (Neural plays {color_str})... ", end="", flush=True)

        # Log game header
        game_log.write(f"\n{'='*70}\n")
        game_log.write(f"GAME {game_num}/{num_games}\n")
        game_log.write(f"White: {'Neural' if neural_plays_white else 'Stockfish'}\n")
        game_log.write(f"Black: {'Stockfish' if neural_plays_white else 'Neural'}\n")
        game_log.write(f"{'='*70}\n\n")

        score, moves, outcome, san_moves = play_game(
            neural_engine,
            stockfish_engine,
            neural_plays_white,
            time_limit
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
        game_log.write(f"Result: {outcome.upper()} ({len(moves)} moves)\n")

        if outcome == "win":
            wins += 1
            result_str = "✓ WIN"
        elif outcome == "loss":
            losses += 1
            result_str = "✗ LOSS"
        else:
            draws += 1
            result_str = "= DRAW"

        print(f"{result_str} ({len(moves)} moves)")

        # Show progress
        if game_num % 10 == 0:
            current_score = wins + 0.5 * draws
            percentage = (current_score / game_num) * 100
            print(f"  Progress: {wins}W-{losses}L-{draws}D ({percentage:.1f}%)")
            print()

    elapsed = time.time() - start_time

    # Close log file
    game_log.write(f"\n{'='*70}\n")
    game_log.write(f"FINAL RESULTS\n")
    game_log.write(f"{'='*70}\n")
    game_log.write(f"Record: {wins}W - {losses}L - {draws}D\n")
    game_log.write(f"Score: {wins + 0.5 * draws}/{num_games} ({(wins + 0.5 * draws) / num_games * 100:.1f}%)\n")
    game_log.close()

    # Close Stockfish
    stockfish_engine.quit()

    # Calculate results
    total_score = wins + 0.5 * draws
    score_percentage = (total_score / num_games) * 100

    # Calculate ELO
    import math
    win_ratio = total_score / num_games
    if win_ratio >= 0.99:
        win_ratio = 0.99
    elif win_ratio <= 0.01:
        win_ratio = 0.01

    elo_diff = 400 * math.log10(win_ratio / (1 - win_ratio))
    estimated_elo = opponent_elo + elo_diff

    # Print results
    print()
    print("=" * 70)
    print("MATCH RESULTS")
    print("=" * 70)
    print(f"Games played: {num_games}")
    print(f"Record: {wins}W - {losses}L - {draws}D")
    print(f"Score: {total_score}/{num_games} ({score_percentage:.1f}%)")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print()
    print(f"Opponent ELO: {opponent_elo}")
    print(f"Your estimated ELO: {estimated_elo:.0f}")
    print()

    # Interpretation
    if score_percentage >= 75:
        print("🎉 Excellent! You're significantly stronger!")
        print(f"   → Try Stockfish Skill {min(20, stockfish_skill + 5)}")
    elif score_percentage >= 60:
        print("✓ Good! You're stronger than opponent")
        print(f"   → Try Stockfish Skill {min(20, stockfish_skill + 3)}")
    elif score_percentage >= 55:
        print("✓ Slightly stronger")
    elif score_percentage >= 45:
        print("≈ Evenly matched - good test opponent!")
    elif score_percentage >= 40:
        print("⚠ Slightly weaker")
    elif score_percentage >= 25:
        print("⚠ Weaker than opponent")
        print(f"   → Try Stockfish Skill {max(0, stockfish_skill - 3)}")
    else:
        print("⚠⚠ Much weaker")
        print(f"   → Try Stockfish Skill {max(0, stockfish_skill - 5)}")

    print()
    print("=" * 70)

    # Save results
    results_file = f"match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Opponent: Stockfish Skill {stockfish_skill} (~{opponent_elo} ELO)\n")
        f.write(f"Games: {num_games}\n")
        f.write(f"Result: {wins}W - {losses}L - {draws}D\n")
        f.write(f"Score: {score_percentage:.1f}%\n")
        f.write(f"Estimated ELO: {estimated_elo:.0f}\n")

    print(f"Results saved to: {results_file}")
    print(f"Game log saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description='Play match between neural engine and Stockfish')
    parser.add_argument('model', help='Path to model file (.pth)')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--skill', type=int, default=0, help='Stockfish skill level (0-20)')
    parser.add_argument('--time', type=float, default=5.0, help='Time per move (seconds)')

    args = parser.parse_args()

    play_match(args.model, args.games, args.skill, args.time)


if __name__ == "__main__":
    main()
