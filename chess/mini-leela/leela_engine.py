"""
Mini Leela Chess Engine
A chess engine that uses the trained Mini Leela model to play chess.
"""

import chess
import torch
import numpy as np
from typing import Optional
from mini_leela_complete import ChessNet, BoardEncoder, MoveEncoder, MCTS


class Engine:
    """
    Chess engine that loads a trained Mini Leela model and generates moves.

    Usage:
        engine = Engine("mini_leela_model.pth")
        board = chess.Board()
        move = engine.get_move(board)
        board.push(move)
    """

    def __init__(self, model_path: str = "mini_leela_model.pth",
                 device: Optional[str] = None,
                 num_simulations: int = 100):
        """
        Initialize the engine by loading a trained model.

        Args:
            model_path: Path to the saved model file (.pth)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            num_simulations: Number of MCTS simulations per move (higher = stronger but slower)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_simulations = num_simulations

        # Load the model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract network configuration
        config = checkpoint['network_config']
        self.network = ChessNet(
            input_channels=config['input_channels'],
            num_res_blocks=config['num_res_blocks'],
            num_channels=config['num_channels']
        )

        # Load model weights
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.to(self.device)
        self.network.eval()

        # Initialize MCTS
        self.mcts = MCTS(self.network, self.device, num_simulations=self.num_simulations)

        print(f"Model loaded successfully on {self.device}")
        print(f"MCTS simulations per move: {self.num_simulations}")

    def get_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        """
        Get the best move for the given board position.

        Args:
            board: The current chess board position
            temperature: Controls move selection:
                         0.0 = always pick most visited move (deterministic, strongest)
                         1.0 = sample proportional to visit counts (stochastic)
                         >1.0 = more random exploration

        Returns:
            The selected chess move

        Raises:
            ValueError: If the board is in a game-over state (no legal moves)
        """
        if board.is_game_over():
            raise ValueError("Cannot get move for a board in game-over state")

        # Run MCTS to get visit counts for each legal move
        visit_counts, _ = self.mcts.search(board)

        if not visit_counts:
            raise ValueError("MCTS returned no moves (this should not happen)")

        # Select move based on temperature
        if temperature == 0.0:
            # Greedy: pick most visited move
            best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
            return best_move
        else:
            # Stochastic: sample proportional to visit_count^(1/temperature)
            moves = list(visit_counts.keys())
            counts = np.array([visit_counts[m] for m in moves])

            # Apply temperature
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()

            # Sample move
            selected_move = np.random.choice(moves, p=probs)
            return selected_move

    def get_move_with_info(self, board: chess.Board, temperature: float = 0.0,
                          top_n: int = 5) -> tuple[chess.Move, dict]:
        """
        Get the best move along with detailed information about the search.

        Args:
            board: The current chess board position
            temperature: Controls move selection (see get_move)
            top_n: Number of top moves to include in info

        Returns:
            Tuple of (selected_move, info_dict) where info_dict contains:
                - 'move': The selected move
                - 'total_simulations': Total number of MCTS simulations
                - 'top_moves': List of (move, visit_count, visit_percentage) tuples
        """
        if board.is_game_over():
            raise ValueError("Cannot get move for a board in game-over state")

        # Run MCTS
        visit_counts, root = self.mcts.search(board)

        # Select move
        if temperature == 0.0:
            best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
        else:
            moves = list(visit_counts.keys())
            counts = np.array([visit_counts[m] for m in moves])
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()
            best_move = np.random.choice(moves, p=probs)

        # Prepare info
        total_visits = sum(visit_counts.values())
        sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_moves = [
            (move, count, 100 * count / total_visits)
            for move, count in sorted_moves
        ]

        info = {
            'move': best_move,
            'total_simulations': total_visits,
            'top_moves': top_moves
        }

        return best_move, info


def demo():
    """Demonstrate the engine playing a few moves from the starting position"""
    print("=" * 70)
    print("Mini Leela Chess Engine Demo")
    print("=" * 70)

    # Initialize engine
    engine = Engine("mini_leela_model.pth", num_simulations=100)

    # Create starting position
    board = chess.Board()

    # Play 5 moves
    num_moves = 5
    print(f"\nPlaying {num_moves} moves from starting position...\n")

    for i in range(num_moves):
        if board.is_game_over():
            print("Game over!")
            break

        print(f"Move {i+1}:")
        print(f"Position: {board.fen()}")

        # Get move with info
        move, info = engine.get_move_with_info(board, temperature=0.0, top_n=3)

        print(f"Selected move: {move}")
        print(f"Top 3 moves:")
        for m, count, pct in info['top_moves']:
            print(f"  {m}: {count} visits ({pct:.1f}%)")

        # Make the move
        board.push(move)
        print()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
