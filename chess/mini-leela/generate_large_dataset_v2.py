"""
Generate a large dataset of verified mate-in-1 positions
Simpler approach with manual position building
"""
import chess

def verify_mate_in_1(fen, move_uci):
    """Verify a position is actually mate-in-1"""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return False
        board.push(move)
        return board.is_checkmate()
    except:
        return False

verified_positions = []

# Manually curated positions with known patterns
positions_to_test = [
    # Back rank mates - Ra8#
    ('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8#'),
    ('7k/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v2'),
    ('5k2/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v3'),
    ('4k3/4pppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v4'),
    ('6k1/6pp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v5'),
    ('7k/6pp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v6'),
    ('7k/7p/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v7'),
    ('7k/5p1p/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v8'),
    ('7k/6p1/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v9'),
    ('7k/5pp1/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v10'),

    # Back rank with Re8#
    ('5rk1/6pp/8/8/8/8/8/4R1K1 w - - 0 1', 'e1e8', 'Back rank Re8#'),
    ('r4rk1/6pp/8/8/8/8/8/4R1K1 w - - 0 1', 'e1e8', 'Back rank Re8# v2'),
    ('r5k1/6pp/8/8/8/8/8/4R1K1 w - - 0 1', 'e1e8', 'Back rank Re8# v3'),

    # Queen mates on h7
    ('7k/6Q1/6K1/8/8/8/8/8 w - - 0 1', 'g7h7', 'Qh7#'),
    ('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1', 'f7h7', 'Qh7# v2'),
    ('7k/4Q3/6K1/8/8/8/8/8 w - - 0 1', 'e7h7', 'Qh7# v3'),
    ('7k/8/6KQ/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# v4'),
    ('7k/8/5K1Q/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# v5'),
    ('7k/6Q1/5K2/8/8/8/8/8 w - - 0 1', 'g7h7', 'Qh7# v6'),
    ('7k/6Q1/7K/8/8/8/8/8 w - - 0 1', 'g7h7', 'Qh7# v7'),
    ('7k/7p/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# with pawn'),
    ('7k/5p2/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# with pawn v2'),
    ('7k/5p1p/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# with pawns'),
    ('7k/6p1/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# with pawn v3'),

    # Rook mates on h7/h8
    ('7k/6R1/6K1/8/8/8/8/8 w - - 0 1', 'g7h7', 'Rh7#'),
    ('7k/7R/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8#'),
    ('7k/8/6RK/8/8/8/8/8 w - - 0 1', 'g6g8', 'Rg8#'),
    ('7k/6pp/6RK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Rh7# with pawns'),
    ('7k/7R/5K2/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8# v2'),
    ('7k/6pR/5K2/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8# with pawn'),

    # Smothered mate
    ('6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7#'),
    ('r6k/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v2'),
    ('6kr/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v3'),
    ('5rkr/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v4'),
    ('5r1k/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v5'),

    # Corner mates
    ('k7/p7/K1R5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Rc8# corner'),
    ('k7/pp6/K1R5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Rc8# corner v2'),
    ('k7/p7/KR6/8/8/8/8/8 w - - 0 1', 'b6b8', 'Rb8# corner'),
    ('k7/p7/K1Q5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Qc8# corner'),

    # Scholar's mate
    ('r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1', 'f3f7', "Scholar's mate"),

    # More Queen mates
    ('6k1/5Qpp/6K1/8/8/8/8/8 w - - 0 1', 'f7g8', 'Qg8#'),
    ('5rk1/5Qpp/6K1/8/8/8/8/8 w - - 0 1', 'f7f8', 'Qxf8#'),
    ('6k1/5Q1p/6K1/8/8/8/8/8 w - - 0 1', 'f7g8', 'Qg8# v2'),
    ('6k1/4Q1pp/6K1/8/8/8/8/8 w - - 0 1', 'e7g8', 'Qg8# v3'),

    # More variations - build systematically
]

# Add many more back rank variations
for king_file_idx in range(5, 8):  # e, f, g, h files = 4,5,6,7
    king_square = chess.SQUARES[king_file_idx + 56]  # rank 8
    king_file_name = chess.square_name(king_square)

    for pawn_pattern_idx in range(1, 16):  # Various pawn configurations
        board = chess.Board()
        board.clear()

        # Place kings
        board.set_piece_at(king_square, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))

        # Place rook
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))

        # Place pawns based on pattern
        if pawn_pattern_idx & 1:
            board.set_piece_at(chess.F7, chess.Piece(chess.PAWN, chess.BLACK))
        if pawn_pattern_idx & 2:
            board.set_piece_at(chess.G7, chess.Piece(chess.PAWN, chess.BLACK))
        if pawn_pattern_idx & 4:
            board.set_piece_at(chess.H7, chess.Piece(chess.PAWN, chess.BLACK))
        if pawn_pattern_idx & 8:
            board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.BLACK))

        board.turn = chess.WHITE
        fen = board.fen()

        if verify_mate_in_1(fen, 'a1a8'):
            desc = f'Back rank Ra8# K{king_file_name} pawns#{pawn_pattern_idx}'
            positions_to_test.append((fen, 'a1a8', desc))

# Add Queen + King mating patterns on h-file
for queen_file_idx in range(4, 8):  # e, f, g, h
    for queen_rank_idx in range(5, 8):  # ranks 6, 7, 8
        queen_square = chess.SQUARES[queen_file_idx + queen_rank_idx * 8]
        queen_sq_name = chess.square_name(queen_square)

        for king_file_idx in range(4, 8):
            for king_rank_idx in range(4, 7):  # ranks 5, 6, 7
                king_square = chess.SQUARES[king_file_idx + king_rank_idx * 8]

                if queen_square == king_square:
                    continue

                board = chess.Board()
                board.clear()

                # Black king on h8
                board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))

                # White queen
                board.set_piece_at(queen_square, chess.Piece(chess.QUEEN, chess.WHITE))

                # White king
                board.set_piece_at(king_square, chess.Piece(chess.KING, chess.WHITE))

                board.turn = chess.WHITE
                fen = board.fen()

                # Try mating on h7 or h8
                for target in ['h7', 'h8']:
                    move = queen_sq_name + target
                    if verify_mate_in_1(fen, move):
                        desc = f'Queen {move}'
                        positions_to_test.append((fen, move, desc))

print("Verifying positions...")
for fen, move, desc in positions_to_test:
    if verify_mate_in_1(fen, move):
        verified_positions.append((fen, move, desc))

# Remove duplicates
unique_positions = []
seen_fens = set()
for fen, move, desc in verified_positions:
    if fen not in seen_fens:
        unique_positions.append((fen, move, desc))
        seen_fens.add(fen)

print(f"Generated {len(unique_positions)} unique verified mate-in-1 positions")

# Split into training and test
if len(unique_positions) >= 140:
    training_positions = unique_positions[:100]
    test_positions = unique_positions[100:140]
else:
    split_idx = int(len(unique_positions) * 0.7)
    training_positions = unique_positions[:split_idx]
    test_positions = unique_positions[split_idx:]

print(f"Training: {len(training_positions)} positions")
print(f"Test: {len(test_positions)} positions")

# Save to file
with open('mate_in_1_positions_large.py', 'w') as f:
    f.write('"""\nLarge collection of mate-in-1 positions\n"""\n\n')
    f.write(f"# {len(training_positions)} training positions\n")
    f.write("TRAINING_POSITIONS = [\n")
    for fen, move, desc in training_positions:
        f.write(f"    ('{fen}', '{move}', '{desc}'),\n")
    f.write("]\n\n")
    f.write(f"# {len(test_positions)} test positions\n")
    f.write("TEST_POSITIONS = [\n")
    for fen, move, desc in test_positions:
        f.write(f"    ('{fen}', '{move}', '{desc}'),\n")
    f.write("]\n\n")
    f.write("""
def get_training_fens():
    return [pos[0] for pos in TRAINING_POSITIONS]

def get_test_fens():
    return [pos[0] for pos in TEST_POSITIONS]

def get_all_fens():
    return get_training_fens() + get_test_fens()

def get_solution(fen):
    for pos_list in [TRAINING_POSITIONS, TEST_POSITIONS]:
        for pos_fen, solution, desc in pos_list:
            if pos_fen == fen:
                return solution, desc
    return None, None
""")

print(f"\nSaved to mate_in_1_positions_large.py")
print(f"You can now replace mate_in_1_positions.py with this file")
