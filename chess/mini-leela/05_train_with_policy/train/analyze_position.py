"""
Interactive Chess Position Analyzer
Analyzes FEN positions using your neural network engine

Usage: python analyze_position.py <model_path>
Example: python analyze_position.py games_and_puzzles_005_20251124_080951.pth
"""
import chess
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from encoder import BoardEncoder
from chess_net_source_dest import ChessNetSourceDest


class PositionAnalyzer:
    """Neural network chess position analyzer"""

    def __init__(self, model_path):
        self.encoder = BoardEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['network_config']

        self.network = ChessNetSourceDest(
            input_channels=config['input_channels'],
            num_res_blocks=config['num_res_blocks'],
            num_channels=config['num_channels'],
            dropout=config.get('dropout', 0.0)
        ).to(self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()

        print(f"✓ Model loaded successfully!")
        print(f"  Network: {config['num_res_blocks']} ResNet blocks, {config['num_channels']} channels")
        print(f"  Device: {self.device}")
        if 'epoch' in checkpoint:
            print(f"  Trained epochs: {checkpoint['epoch']}")
        if 'validation_accuracy' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['validation_accuracy']*100:.1f}%")
        print()

    def evaluate_position(self, board):
        """Evaluate a position (raw value from network)"""
        with torch.no_grad():
            board_tensor = torch.FloatTensor(
                self.encoder.encode_board(board)
            ).unsqueeze(0).to(self.device)

            _, _, value = self.network(board_tensor)
            return value.item()

    def get_best_move_simple(self, board):
        """Get best move using only policy network (no search)"""
        with torch.no_grad():
            board_tensor = torch.FloatTensor(
                self.encoder.encode_board(board)
            ).unsqueeze(0).to(self.device)

            source_logits, dest_logits, value = self.network(board_tensor)

            # Get probabilities
            source_probs = F.softmax(source_logits[0], dim=0).cpu().numpy()
            dest_probs = F.softmax(dest_logits[0], dim=0).cpu().numpy()

            # Find best legal move by policy probability
            best_score = -float('inf')
            best_move = None

            for move in board.legal_moves:
                policy_prob = source_probs[move.from_square] * dest_probs[move.to_square]
                if policy_prob > best_score:
                    best_score = policy_prob
                    best_move = move

            return best_move, best_score, value.item()

    def get_best_move_with_search(self, board):
        """Get best move using policy + 1-ply value search (like play_match.py)"""
        with torch.no_grad():
            # Get policy probabilities for current position
            board_tensor = torch.FloatTensor(
                self.encoder.encode_board(board)
            ).unsqueeze(0).to(self.device)

            source_logits, dest_logits, current_value = self.network(board_tensor)

            # Get probabilities
            source_probs = F.softmax(source_logits[0], dim=0).cpu().numpy()
            dest_probs = F.softmax(dest_logits[0], dim=0).cpu().numpy()

            # Evaluate each legal move by looking at resulting position
            best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
            best_move = None
            best_policy_prob = 0

            move_evaluations = []

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
                combined_score = eval_score + 0.1 * policy_prob

                move_evaluations.append({
                    'move': move,
                    'policy_prob': policy_prob,
                    'eval': eval_score,
                    'combined': combined_score
                })

                # Update best move
                if board.turn == chess.WHITE:
                    if combined_score > best_score:
                        best_score = combined_score
                        best_move = move
                        best_policy_prob = policy_prob
                else:
                    if combined_score < best_score:
                        best_score = combined_score
                        best_move = move
                        best_policy_prob = policy_prob

            # Sort moves by combined score (descending for white, ascending for black)
            if board.turn == chess.WHITE:
                move_evaluations.sort(key=lambda x: x['combined'], reverse=True)
            else:
                move_evaluations.sort(key=lambda x: x['combined'])

            return best_move, best_policy_prob, current_value.item(), move_evaluations[:5]

    def analyze_position(self, fen, show_top_moves=True):
        """Analyze a position given its FEN"""
        try:
            board = chess.Board(fen)
        except Exception as e:
            return f"Error: Invalid FEN - {e}"

        # Get position evaluation
        current_eval = self.evaluate_position(board)

        # Get best move with search
        best_move, policy_prob, _, top_moves = self.get_best_move_with_search(board)

        # Format output
        result = []
        result.append("=" * 70)
        result.append("POSITION ANALYSIS")
        result.append("=" * 70)
        result.append(f"FEN: {fen}")
        result.append(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'} to move")
        result.append(f"Legal moves: {board.legal_moves.count()}")
        result.append("")
        result.append(f"Current Position Evaluation: {current_eval:+.3f}")
        result.append(f"  (Range: -1.0 = Black winning, +1.0 = White winning)")
        result.append("")

        if best_move:
            result.append(f"Best Move: {board.san(best_move)} ({best_move.uci()})")
            result.append(f"  Policy Probability: {policy_prob:.4f}")

            if show_top_moves and len(top_moves) > 0:
                result.append("")
                result.append("Top 5 Candidate Moves:")
                result.append("-" * 70)
                result.append(f"{'Rank':<6} {'Move':<10} {'UCI':<8} {'Policy':<12} {'Eval':<12} {'Combined':<10}")
                result.append("-" * 70)

                for i, move_data in enumerate(top_moves[:5], 1):
                    move = move_data['move']
                    san = board.san(move)
                    uci = move.uci()
                    policy = move_data['policy_prob']
                    eval_score = move_data['eval']
                    combined = move_data['combined']

                    result.append(f"{i:<6} {san:<10} {uci:<8} {policy:>10.4f}  {eval_score:>+10.3f}  {combined:>+9.3f}")
        else:
            result.append("No legal moves available!")

        # Show ASCII board
        result.append("")
        result.append("Board Position:")
        result.append(str(board))
        result.append("=" * 70)

        return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(
        description='Interactive chess position analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_position.py model.pth

Then enter FEN positions when prompted, or use special commands:
  - 'startpos' or 'start' for starting position
  - 'quit' or 'exit' or 'q' to exit
  - Empty line shows help
        """
    )
    parser.add_argument('model', help='Path to model file (.pth)')
    parser.add_argument('--no-top-moves', action='store_true',
                       help='Only show best move, not top 5 candidates')

    args = parser.parse_args()

    # Load analyzer
    try:
        analyzer = PositionAnalyzer(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    print("Interactive Chess Position Analyzer")
    print("=" * 70)
    print("Enter FEN positions to analyze, or:")
    print("  'startpos' or 'start' - Starting position")
    print("  'quit' or 'exit' or 'q' - Exit program")
    print("  Empty line - Show this help")
    print("=" * 70)
    print()

    # Pre-defined positions for quick testing
    STARTING_POS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    while True:
        try:
            # Get FEN from user
            fen_input = input("\nEnter FEN position: ").strip()

            # Handle special commands
            if not fen_input:
                print("\nCommands:")
                print("  'startpos' or 'start' - Starting position")
                print("  'quit' or 'exit' or 'q' - Exit program")
                continue

            if fen_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if fen_input.lower() in ['startpos', 'start']:
                fen_input = STARTING_POS
                print(f"Using starting position")

            # Analyze position
            print()
            result = analyzer.analyze_position(fen_input, show_top_moves=not args.no_top_moves)
            print(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
