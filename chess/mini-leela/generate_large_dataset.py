"""
Generate a large dataset of verified mate-in-1 positions
Target: 100+ training positions, 40+ test positions
"""
import chess
import itertools

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

# Pattern generators
verified_positions = []

# ============================================================================
# PATTERN 1: Back rank mates with Rook on a1, King on various files
# ============================================================================
print("Generating back rank mates...")
for king_file in ['e', 'f', 'g', 'h']:
    for pawn_setup in [
        f'{king_file}pp',
        f'{chr(ord(king_file)-1)}pp' if king_file > 'e' else 'epp',
        f'{chr(ord(king_file)-1)}p{king_file}p' if king_file > 'e' else 'efp',
        f'{king_file}p{chr(ord(king_file)+1)}p' if king_file < 'h' else 'fgp',
    ]:
        fen = f'{king_file}5k1/{pawn_setup:>8}/8/8/8/8/8/R5K1 w - - 0 1'
        if verify_mate_in_1(fen, 'a1a8'):
            verified_positions.append((fen, 'a1a8', f'Back rank Ra8# (K{king_file}8, pawns:{pawn_setup})'))

# ============================================================================
# PATTERN 2: Queen mates on h-file with King on h8
# ============================================================================
print("Generating Queen mates on h-file...")
for queen_square in ['g7', 'f7', 'e7', 'd7', 'g6', 'f6', 'e6', 'h6', 'h7']:
    for king_support in ['g6', 'f6', 'f5', 'g5', 'h6']:
        if queen_square == king_support:
            continue
        fen = f'7k/8/8/8/8/8/8/6K1 w - - 0 1'
        # Build position with Queen and King
        board = chess.Board('7k/8/8/8/8/8/8/6K1 w - - 0 1')

        queen_sq = chess.parse_square(queen_square)
        king_sq = chess.parse_square(king_support)

        # Remove white king, place pieces, then place king back
        board.remove_piece_at(chess.G1)
        board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))

        fen = board.fen()

        # Try h7 and h8 as target squares
        for target in ['h7', 'h8']:
            move = queen_square + target
            if verify_mate_in_1(fen, move):
                verified_positions.append((fen, move, f'Queen {move} mate'))
                break

# ============================================================================
# PATTERN 3: Rook mates on h-file with King on h8
# ============================================================================
print("Generating Rook mates on h-file...")
for rook_square in ['g7', 'f7', 'e7', 'h7', 'g6', 'f6', 'h6']:
    for king_support in ['g6', 'f6', 'f5', 'g5', 'h6', 'g7']:
        if rook_square == king_support:
            continue
        board = chess.Board('7k/8/8/8/8/8/8/6K1 w - - 0 1')

        rook_sq = chess.parse_square(rook_square)
        king_sq = chess.parse_square(king_support)

        board.remove_piece_at(chess.G1)
        board.set_piece_at(rook_sq, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))

        fen = board.fen()

        for target in ['h7', 'h8']:
            move = rook_square + target
            if verify_mate_in_1(fen, move):
                verified_positions.append((fen, move, f'Rook {move} mate'))
                break

# ============================================================================
# PATTERN 4: Queen mates on g8 with King on various squares
# ============================================================================
print("Generating Queen mates on g8...")
base_fens = ['6k1/8/8/8/8/8/8/6K1 w - - 0 1', '7k/8/8/8/8/8/8/6K1 w - - 0 1']
for fen_base in base_fens:
    for queen_sq_str in ['f7', 'e7', 'd7', 'f8', 'g7', 'h7']:
        board = chess.Board(fen_base)
        queen_sq = chess.parse_square(queen_sq_str)
        board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))

        fen = board.fen()
        move = queen_sq_str + 'g8'
        if verify_mate_in_1(fen, move):
            verified_positions.append((fen, move, f'Queen {move} mate'))

# ============================================================================
# PATTERN 5: Smothered mate with Knight
# ============================================================================
print("Generating Knight smothered mates...")
knight_patterns = [
    ('6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7#'),
    ('r6k/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v2'),
    ('6kr/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v3'),
    ('5rkr/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v4'),
    ('5r1k/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered Nf7# v5'),
]
for fen, move, desc in knight_patterns:
    if verify_mate_in_1(fen, move):
        verified_positions.append((fen, move, desc))

# ============================================================================
# PATTERN 6: Rook mates on a8 with King on various squares (different from pattern 1)
# ============================================================================
print("Generating more back rank variations...")
for king_file in ['d', 'e', 'f', 'g', 'h']:
    for rook_start in ['a1', 'b1', 'c1', 'a2', 'b2']:
        fen = f'{king_file}5k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1'
        board = chess.Board(fen)

        # Move rook to start position
        if rook_start != 'a1':
            board.remove_piece_at(chess.A1)
            rook_sq = chess.parse_square(rook_start)
            board.set_piece_at(rook_sq, chess.Piece(chess.ROOK, chess.WHITE))
            fen = board.fen()

        move = rook_start + 'a8'
        if verify_mate_in_1(fen, move):
            verified_positions.append((fen, move, f'Rook {move} mate'))

# ============================================================================
# PATTERN 7: Queen mates from various squares to a8 (corner mate)
# ============================================================================
print("Generating corner mates...")
corner_patterns = [
    ('k7/p7/K1R5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Rc8# corner'),
    ('k7/p7/KR6/8/8/8/8/8 w - - 0 1', 'b6b8', 'Rb8# corner'),
    ('k7/p7/K1Q5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Qc8# corner'),
    ('k7/pp6/K1R5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Rc8# corner v2'),
]
for fen, move, desc in corner_patterns:
    if verify_mate_in_1(fen, move):
        verified_positions.append((fen, move, desc))

# ============================================================================
# PATTERN 8: Scholar's mate and similar
# ============================================================================
print("Adding Scholar's mate...")
if verify_mate_in_1('r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1', 'f3f7'):
    verified_positions.append(('r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1',
                              'f3f7', "Scholar's mate"))

# ============================================================================
# PATTERN 9: More Queen mates with pawns
# ============================================================================
print("Generating Queen mates with pawn shields...")
for king_file in ['g', 'h']:
    for pawn_setup_idx in range(8):
        # Binary representation for pawn presence
        f7 = bool(pawn_setup_idx & 1)
        g7 = bool(pawn_setup_idx & 2)
        h7 = bool(pawn_setup_idx & 4)

        if not any([f7, g7, h7]):  # Skip if no pawns
            continue

        board = chess.Board('6k1/8/8/8/8/8/8/6K1 w - - 0 1')

        if f7:
            board.set_piece_at(chess.F7, chess.Piece(chess.PAWN, chess.BLACK))
        if g7:
            board.set_piece_at(chess.G7, chess.Piece(chess.PAWN, chess.BLACK))
        if h7:
            board.set_piece_at(chess.H7, chess.Piece(chess.PAWN, chess.BLACK))

        # Try Queen on various squares
        for queen_sq_str in ['f8', 'g8', 'h8', 'f6', 'g6', 'h6']:
            test_board = board.copy()
            queen_sq = chess.parse_square(queen_sq_str)
            test_board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))

            # Try King on supporting squares
            for king_sq_str in ['f6', 'g5', 'h6', 'f5', 'g4']:
                final_board = test_board.copy()
                king_sq = chess.parse_square(king_sq_str)
                final_board.remove_piece_at(chess.G1)
                final_board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))

                fen = final_board.fen()

                # Try various target squares
                for target in ['f8', 'g8', 'h8', 'g7', 'h7']:
                    if queen_sq_str == target:
                        continue
                    move = queen_sq_str + target
                    if verify_mate_in_1(fen, move):
                        verified_positions.append((fen, move, f'Queen {move} with pawns'))
                        break

print(f"\nGenerated {len(verified_positions)} verified mate-in-1 positions")

# Remove duplicates
unique_positions = []
seen_fens = set()
for fen, move, desc in verified_positions:
    if fen not in seen_fens:
        unique_positions.append((fen, move, desc))
        seen_fens.add(fen)

print(f"After removing duplicates: {len(unique_positions)} positions")

# Split into training and test
training_size = min(100, int(len(unique_positions) * 0.7))
test_size = min(40, len(unique_positions) - training_size)

training_positions = unique_positions[:training_size]
test_positions = unique_positions[training_size:training_size + test_size]

print(f"\nTraining: {len(training_positions)} positions")
print(f"Test: {len(test_positions)} positions")

# Generate Python code
print("\n" + "="*70)
print("Paste this into mate_in_1_positions.py:")
print("="*70)
print()
print("TRAINING_POSITIONS = [")
for fen, move, desc in training_positions:
    print(f"    ('{fen}', '{move}', '{desc}'),")
print("]")
print()
print("TEST_POSITIONS = [")
for fen, move, desc in test_positions:
    print(f"    ('{fen}', '{move}', '{desc}'),")
print("]")
