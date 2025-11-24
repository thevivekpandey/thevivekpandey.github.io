import chess
import numpy as np
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

