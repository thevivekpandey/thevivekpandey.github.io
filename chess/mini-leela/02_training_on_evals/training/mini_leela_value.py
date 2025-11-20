"""
Minimal Leela Chess Zero Implementation - VALUE PREDICTION VERSION
Modified to use position evaluation instead of policy prediction

This implementation uses:
1. Board representation (bitboard-like encoding)
2. ResNet neural network with VALUE head only
3. Position evaluation for move selection
4. Can be used with stockfish eval trained models

Key difference from original:
- No policy head - uses value prediction to evaluate positions
- Moves are selected by evaluating resulting positions
"""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


# ============================================================================
# PART 1: BOARD REPRESENTATION (unchanged)
# ============================================================================

class BoardEncoder:
    """
    Encode chess board as neural network input.
    Uses 12 planes for pieces (6 piece types × 2 colors) + metadata planes
    """

    @staticmethod
    def encode_board(board: chess.Board) -> np.ndarray:
        """
        Encode board position as 19×8×8 tensor:
        - 12 planes for pieces (P,N,B,R,Q,K for white and black)
        - 1 plane for turn (all 1s if white to move, all 0s if black)
        - 4 planes for castling rights
        - 1 plane for en passant
        - 1 plane for fifty-move counter (normalized)

        Returns: numpy array of shape (19, 8, 8)
        """
        planes = np.zeros((19, 8, 8), dtype=np.float32)

        # Piece planes (planes 0-11)
        piece_idx = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                plane = piece_idx[(piece.piece_type, piece.color)]
                planes[plane, rank, file] = 1

        # Turn plane (plane 12)
        if board.turn == chess.WHITE:
            planes[12, :, :] = 1

        # Castling rights (planes 13-16)
        planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

        # En passant (plane 17)
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            planes[17, rank, file] = 1

        # Fifty-move counter (plane 18)
        planes[18, :, :] = board.halfmove_clock / 100.0

        return planes


# ============================================================================
# PART 2: VALUE-BASED MOVE SELECTION
# ============================================================================

class ValueBasedPlayer:
    """
    Chess player that evaluates positions using a value network.
    Selects moves by evaluating resulting positions.
    """

    def __init__(self, network, device: str = 'cpu'):
        """
        Args:
            network: ChessNetValue model that predicts position evaluation
            device: 'cpu' or 'cuda'
        """
        self.network = network
        self.device = device
        self.encoder = BoardEncoder()
        self.network.eval()

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a position using the neural network.

        Args:
            board: Chess board to evaluate

        Returns:
            Evaluation in centipawns (positive = white advantage)
        """
        board_tensor = torch.FloatTensor(self.encoder.encode_board(board)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.network(board_tensor)
            # Model outputs normalized value (in pawn units), convert to centipawns
            return value.item() * 100.0

    def select_move(self, board: chess.Board, depth: int = 1) -> Tuple[chess.Move, float]:
        """
        Select best move by evaluating resulting positions.

        Args:
            board: Current board position
            depth: Search depth (1 = evaluate all immediate moves)

        Returns:
            (best_move, evaluation)
        """
        if board.is_game_over():
            return None, 0.0

        legal_moves = list(board.legal_moves)
        best_move = None
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')

        for move in legal_moves:
            # Make move
            board.push(move)

            # Evaluate resulting position
            if board.is_game_over():
                # Terminal position
                result = board.result()
                if result == "1-0":
                    eval_score = 10000.0
                elif result == "0-1":
                    eval_score = -10000.0
                else:
                    eval_score = 0.0
            else:
                eval_score = self.evaluate_position(board)

            # Undo move
            board.pop()

            # Update best move
            if board.turn == chess.WHITE:
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move

        return best_move, best_eval

    def select_move_with_search(self, board: chess.Board, depth: int = 2) -> Tuple[chess.Move, float]:
        """
        Select best move with minimax search.

        Args:
            board: Current board position
            depth: Search depth

        Returns:
            (best_move, evaluation)
        """
        def minimax(board: chess.Board, depth: int, alpha: float, beta: float,
                   maximizing: bool) -> float:
            # Base case
            if depth == 0 or board.is_game_over():
                if board.is_game_over():
                    result = board.result()
                    if result == "1-0":
                        return 10000.0
                    elif result == "0-1":
                        return -10000.0
                    else:
                        return 0.0
                return self.evaluate_position(board)

            if maximizing:
                max_eval = -float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval_score = minimax(board, depth - 1, alpha, beta, False)
                    board.pop()
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval_score = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                return min_eval

        # Root level move selection
        legal_moves = list(board.legal_moves)
        best_move = None
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')

        maximizing = (board.turn == chess.WHITE)

        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, -float('inf'), float('inf'), not maximizing)
            board.pop()

            if maximizing:
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move

        return best_move, best_eval


# ============================================================================
# DEMONSTRATION USAGE
# ============================================================================

def play_game_demo(network, device='cpu', max_moves=100, search_depth=1):
    """
    Play a demo game using the value network.

    Args:
        network: Trained ChessNetValue model
        device: 'cpu' or 'cuda'
        max_moves: Maximum number of moves before stopping
        search_depth: How deep to search (1 = greedy, 2+ = minimax)
    """
    board = chess.Board()
    player = ValueBasedPlayer(network, device)

    print("Starting position:")
    print(board)
    print()

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        # Select and make move
        if search_depth == 1:
            move, eval_score = player.select_move(board)
        else:
            move, eval_score = player.select_move_with_search(board, depth=search_depth)

        if move is None:
            break

        move_count += 1
        if board.turn == chess.WHITE:
            print(f"{(move_count + 1) // 2}. {board.san(move)}", end=" ")
        else:
            print(f"{board.san(move)}")

        print(f"  [Eval: {eval_score/100:.2f} pawns]")

        board.push(move)

    print("\nFinal position:")
    print(board)
    print(f"\nResult: {board.result()}")


if __name__ == "__main__":
    print("This module provides value-based move selection for chess.")
    print("Load a trained ChessNetValue model and use ValueBasedPlayer to play.")
    print("\nExample usage:")
    print("  from chess_net_value import ChessNetValue")
    print("  from mini_leela_value import ValueBasedPlayer, play_game_demo")
    print("")
    print("  # Load model")
    print("  network = ChessNetValue()")
    print("  network.load_state_dict(torch.load('model.pth')['model_state_dict'])")
    print("")
    print("  # Play a game")
    print("  play_game_demo(network, search_depth=2)")
