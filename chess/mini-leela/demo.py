"""
Quick Start Demo - How to use the Mini Leela components

This shows how to use each component independently.
You can uncomment sections to test them after installing dependencies.
"""

# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
"""
1. Install dependencies:
   pip install torch numpy python-chess

2. Run this script:
   python demo.py

3. Or import and use components:
   from mini_leela_complete import ChessNet, MCTS, SelfPlayGame
"""

# ============================================================================
# EXAMPLE 1: Create and inspect the neural network
# ============================================================================

def example_network():
    """Create and examine the neural network"""
    from mini_leela_complete import ChessNet
    
    # Create network
    network = ChessNet(
        input_channels=19,
        num_res_blocks=4,
        num_channels=128
    )
    
    print("=" * 70)
    print("NEURAL NETWORK")
    print("=" * 70)
    print(f"Total parameters: {sum(p.numel() for p in network.parameters()):,}")
    print("\nNetwork structure:")
    print(network)
    
    # Test with dummy input
    import torch
    dummy_input = torch.randn(1, 19, 8, 8)  # Batch of 1 position
    policy_logits, value = network(dummy_input)
    
    print(f"\nOutput shapes:")
    print(f"  Policy logits: {policy_logits.shape} (4096 possible moves)")
    print(f"  Value: {value.shape} (scalar position evaluation)")
    print(f"  Value range: {value.item():.3f} (should be in [-1, 1])")


# ============================================================================
# EXAMPLE 2: Encode a chess position
# ============================================================================

def example_encoding():
    """Show how board encoding works"""
    from mini_leela_complete import BoardEncoder
    import chess
    
    print("\n" + "=" * 70)
    print("BOARD ENCODING")
    print("=" * 70)
    
    # Create a position
    board = chess.Board()
    encoder = BoardEncoder()
    
    # Encode it
    encoded = encoder.encode_board(board)
    
    print(f"\nStarting position encoded as: {encoded.shape} tensor")
    print(f"  - 12 planes for pieces")
    print(f"  - 7 planes for metadata (turn, castling, ep, 50-move)")
    
    # Show which squares have white pawns
    print(f"\nWhite pawns (plane 0):")
    print(encoded[0])
    
    print(f"\nWhite king (plane 5):")
    print(encoded[5])
    
    print(f"\nTurn indicator (plane 12) - all 1s means white to move:")
    print(encoded[12, 0, :])  # Just show first row


# ============================================================================
# EXAMPLE 3: Run MCTS on a position
# ============================================================================

def example_mcts():
    """Demonstrate MCTS search"""
    from mini_leela_complete import ChessNet, MCTS
    import chess
    
    print("\n" + "=" * 70)
    print("MONTE CARLO TREE SEARCH")
    print("=" * 70)
    
    # Create network and MCTS
    network = ChessNet()
    mcts = MCTS(network, num_simulations=50)  # Use fewer for demo
    
    # Search from starting position
    board = chess.Board()
    print(f"\nPosition:\n{board}\n")
    print("Running 50 MCTS simulations...")
    
    visit_counts, root = mcts.search(board)
    
    print(f"\nTop 5 moves by visit count:")
    sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    total_visits = sum(visit_counts.values())
    
    for move, count in sorted_moves:
        percentage = 100 * count / total_visits
        print(f"  {move:6s}: {count:3d} visits ({percentage:5.1f}%)")
    
    # Best move
    best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
    print(f"\nBest move: {best_move}")


# ============================================================================
# EXAMPLE 4: Play one self-play game
# ============================================================================

def example_selfplay():
    """Generate one self-play game"""
    from mini_leela_complete import ChessNet, SelfPlayGame
    
    print("\n" + "=" * 70)
    print("SELF-PLAY GAME")
    print("=" * 70)
    
    network = ChessNet()
    self_play = SelfPlayGame(
        network,
        num_simulations=30,  # Fewer for demo
        temperature=1.0
    )
    
    print("Playing one game (this may take a minute)...")
    training_data = self_play.play_game()
    
    print(f"\nGame complete!")
    print(f"  Total positions: {len(training_data)}")
    print(f"  Each position has: (state, policy_target, value_target)")
    print(f"\nFirst position value target: {training_data[0][2]:.2f}")
    print(f"Last position value target: {training_data[-1][2]:.2f}")


# ============================================================================
# EXAMPLE 5: Training iteration
# ============================================================================

def example_training():
    """Run one training iteration"""
    from mini_leela_complete import ChessNet, ChessTrainer
    
    print("\n" + "=" * 70)
    print("TRAINING ITERATION")
    print("=" * 70)
    
    network = ChessNet()
    trainer = ChessTrainer(network, lr=0.001)
    
    print("Running 1 training iteration (2 games)...")
    print("This will take several minutes...\n")
    
    policy_loss, value_loss = trainer.train_iteration(
        num_games=2,
        batch_size=32
    )
    
    print(f"\nTraining complete!")
    print(f"  Policy loss: {policy_loss:.4f}")
    print(f"  Value loss: {value_loss:.4f}")


# ============================================================================
# MAIN - Run all examples
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   MINI LEELA CHESS ZERO - DEMO                       ║
╚══════════════════════════════════════════════════════════════════════╝

This script demonstrates each component of the implementation.
Uncomment the examples you want to run.
    """)
    
    try:
        # Uncomment to run each example:
        
        # example_network()
        # example_encoding()
        # example_mcts()
        # example_selfplay()
        # example_training()
        
        print("\n" + "=" * 70)
        print("Uncomment the examples in demo.py to run them!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install dependencies first:")
        print("  pip install torch numpy python-chess")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# BONUS: Interactive position analysis
# ============================================================================

def interactive_analysis():
    """
    Analyze any position you want.
    
    Usage:
        from demo import interactive_analysis
        interactive_analysis()
    """
    from mini_leela_complete import ChessNet, MCTS
    import chess
    
    network = ChessNet()
    mcts = MCTS(network, num_simulations=100)
    
    print("\n" + "=" * 70)
    print("INTERACTIVE POSITION ANALYSIS")
    print("=" * 70)
    print("\nEnter a FEN string (or press Enter for starting position):")
    
    fen = input("> ").strip()
    if not fen:
        fen = chess.STARTING_FEN
    
    try:
        board = chess.Board(fen)
        print(f"\nPosition:\n{board}\n")
        
        print("Running MCTS analysis...")
        visit_counts, root = mcts.search(board)
        
        print(f"\nTop 10 moves:")
        sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        total_visits = sum(visit_counts.values())
        
        for i, (move, count) in enumerate(sorted_moves, 1):
            percentage = 100 * count / total_visits
            print(f"  {i:2d}. {move:6s}: {count:3d} visits ({percentage:5.1f}%)")
        
        best_move = sorted_moves[0][0]
        print(f"\n✓ Best move: {best_move}")
        
    except Exception as e:
        print(f"Error: {e}")
