"""Helper script to generate and verify mate-in-1 positions"""
import chess

# Collection of known good mate-in-1 positions from various sources
candidate_positions = [
    # Scholar's mate (verified working)
    ('r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1', 'f3f7', 'Scholar\'s mate'),

    # Simple back rank mates
    ('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8#'),
    ('7k/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v2'),
    ('6k1/6pp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v3'),
    ('5k2/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v4'),
    ('4k3/4ppp1/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Back rank Ra8# v5'),

    # Queen mates
    ('7k/6Q1/6K1/8/8/8/8/8 w - - 0 1', 'g7h7', 'Queen h7#'),
    ('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1', 'f7h7', 'Queen h7# v2'),
    ('7k/4Q3/6K1/8/8/8/8/8 w - - 0 1', 'e7h7', 'Queen h7# v3'),
    ('6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1', 'f7g8', 'Queen g8#'),
    ('6k1/4Q3/6K1/8/8/8/8/8 w - - 0 1', 'e7g8', 'Queen g8# v2'),
    ('5k2/4Q3/5K2/8/8/8/8/8 w - - 0 1', 'e7f8', 'Queen f8#'),
    ('5k2/5Q2/5K2/8/8/8/8/8 w - - 0 1', 'f7f8', 'Queen f8# v2'),

    # Rook mates
    ('7k/6R1/6K1/8/8/8/8/8 w - - 0 1', 'g7h7', 'Rook h7#'),
    ('7k/5R2/6K1/8/8/8/8/8 w - - 0 1', 'f7h7', 'Rook h7# v2'),
    ('6k1/6R1/6K1/8/8/8/8/8 w - - 0 1', 'g7g8', 'Rook g8#'),
    ('6k1/5R2/6K1/8/8/8/8/8 w - - 0 1', 'f7g8', 'Rook g8# v2'),
    ('5k2/5R2/5K2/8/8/8/8/8 w - - 0 1', 'f7f8', 'Rook f8#'),

    # Knight mates
    ('6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Smothered mate Nf7#'),
    ('r6k/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Knight f7# v2'),
    ('6kr/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Knight f7# v3'),
    ('5rkr/6pp/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Knight f7# v4'),

    # More Queen + King mates (different king positions on h-file)
    ('7k/7Q/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Qh8#'),
    ('7k/6pQ/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Qh8# with pawn'),
    ('6k1/6pQ/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Qh8# king on g8'),
    ('7k/8/6KQ/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# from h6'),
    ('7k/8/5K1Q/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# K on f6'),

    # More Rook + King mates (similar patterns)
    ('7k/7R/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8#'),
    ('7k/6pR/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8# with pawn'),
    ('6k1/6pR/6K1/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8# king on g8'),
    ('7k/8/6KR/8/8/8/8/8 w - - 0 1', 'h6h7', 'Rh7# from h6'),
    ('7k/8/5K1R/8/8/8/8/8 w - - 0 1', 'h6h7', 'Rh7# K on f6'),

    # Back rank variations (King at different files)
    ('5k2/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# king on f8'),
    ('4k3/4pppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# king on e8'),
    ('3k4/3ppppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# king on d8'),

    # Simple corner mates
    ('k7/p7/KQ6/8/8/8/8/8 w - - 0 1', 'b6b8', 'Qb8# corner'),
    ('k7/p7/K1R5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Rc8# corner'),

    # More h-file mates with slight variations
    ('7k/6pp/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# with 2 pawns'),
    ('7k/7p/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# with 1 pawn'),
    ('7k/6pp/6RK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Rh7# with 2 pawns'),
    ('7k/7p/6RK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Rh7# with 1 pawn'),

    # More back rank mates with different pawn structures
    ('6k1/6pp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# with 2 pawns'),
    ('6k1/7p/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# with 1 pawn'),
    ('7k/6p1/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# g7 pawn'),
    ('7k/7p/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# h7 pawn'),
    ('7k/8/6pp/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# pawns on h6/g6'),

    # More Queen mates
    ('7k/7p/5K1Q/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# different pawn'),
    ('7k/6p1/5K1Q/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# g7 pawn'),

    # More Rook mates with support
    ('7k/7p/5K1R/8/8/8/8/8 w - - 0 1', 'h6h7', 'Rh7# different pawn'),
    ('7k/6p1/5K1R/8/8/8/8/8 w - - 0 1', 'h6h7', 'Rh7# g7 pawn'),

    # ===== ADDITIONAL 10 TEST POSITIONS (slightly different patterns) =====
    # More back rank mates with different file positions
    ('5k2/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# TEST king f8'),
    ('5rk1/6pp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# TEST with rook f8'),
    ('6k1/5p1p/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# TEST f7+h7 pawns'),

    # Queen mates on different squares with king support
    ('7k/6p1/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# TEST g7 pawn'),
    ('7k/5p2/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# TEST f7 pawn'),
    ('7k/7Q/5K2/8/8/8/8/8 w - - 0 1', 'h7h8', 'Qh8# TEST from h7'),
    ('7k/6pQ/5K2/8/8/8/8/8 w - - 0 1', 'h7h8', 'Qh8# TEST h7 with K on f6'),

    # Rook mates with slight variations
    ('7k/6pp/6RK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Rh7# TEST 2 pawns'),
    ('7k/7R/5K2/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8# TEST from h7'),
    ('7k/6pR/5K2/8/8/8/8/8 w - - 0 1', 'h7h8', 'Rh8# TEST h7 with pawn'),

    # More test positions - using exact patterns from training but with tiny variations
    ('7k/6pp/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# TEST 2 pawns'),
    ('7k/5p1p/8/8/8/8/8/R5K1 w - - 0 1', 'a1a8', 'Ra8# TEST f7+h7'),
    ('7k/6Q1/5K2/8/8/8/8/8 w - - 0 1', 'g7h7', 'Qh7# TEST K on f6'),
    ('7k/5p1p/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# TEST f7+h7 pawns'),
    ('7k/8/5K1Q/8/8/8/8/8 w - - 0 1', 'h6h7', 'Qh7# TEST K on f6'),
    ('k7/pp6/K1R5/8/8/8/8/8 w - - 0 1', 'c6c8', 'Rc8# TEST 2 pawns'),
    ('r6k/7p/7N/8/8/8/8/6K1 w - - 0 1', 'h6f7', 'Nf7# TEST rook a8'),
    ('7k/4p1p1/6QK/8/8/8/8/8 w - - 0 1', 'g6h7', 'Qh7# TEST e7+g7'),
    ('7k/6Q1/7K/8/8/8/8/8 w - - 0 1', 'g7h7', 'Qh7# TEST K on h6'),
]

verified_training = []
verified_test = []

print("Verifying positions...\n")

for fen, move_uci, desc in candidate_positions:
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)

        if move not in board.legal_moves:
            print(f"❌ {desc}: Move not legal")
            continue

        board.push(move)
        if not board.is_checkmate():
            print(f"❌ {desc}: Not checkmate")
            continue

        print(f"✓ {desc}")

        # Add to appropriate list (first 10 to training, rest to test)
        if len(verified_training) < 10:
            verified_training.append((fen, move_uci, desc))
        elif len(verified_test) < 10:
            verified_test.append((fen, move_uci, desc))

    except Exception as e:
        print(f"❌ {desc}: Error - {e}")

print(f"\nVerified {len(verified_training)} training positions")
print(f"Verified {len(verified_test)} test positions")

# Print Python code for the positions
print("\n" + "="*70)
print("TRAINING_POSITIONS = [")
for fen, move, desc in verified_training:
    print(f"    ('{fen}', '{move}', '{desc}'),")
print("]")

print("\nTEST_POSITIONS = [")
for fen, move, desc in verified_test:
    print(f"    ('{fen}', '{move}', '{desc}'),")
print("]")
