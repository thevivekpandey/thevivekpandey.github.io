"""
Test harness for evaluating mate-in-1 performance

This script loads a trained model and tests whether it can find mate-in-1
in both training positions (seen during training) and test positions (unseen).
"""

import chess
import torch
import argparse
from typing import Tuple, List

from mini_leela_complete_fixed import ChessNet, MCTS, MoveEncoder
from mate_in_1_positions import (
    TRAINING_POSITIONS, TEST_POSITIONS, get_solution
)


def test_position(board: chess.Board, solution_uci: str, mcts: MCTS,
                  move_encoder: MoveEncoder, verbose: bool = False) -> Tuple[bool, str, int]:
    """
    Test if the model finds the correct mate-in-1 move

    Args:
        board: Chess board in the test position
        solution_uci: The correct move in UCI format
        mcts: MCTS searcher
        move_encoder: Move encoder
        verbose: Print detailed info

    Returns:
        (found_mate, chosen_move_san, rank_of_solution)
        - found_mate: True if the top move is the solution
        - chosen_move_san: The move chosen by the model (in SAN)
        - rank_of_solution: Where the solution appears in the ranking (1=best, 2=second, etc.)
    """
    # Run MCTS search
    visit_counts, root = mcts.search(board)

    # Sort moves by visit count
    sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)

    # Find the chosen move (most visited)
    chosen_move = sorted_moves[0][0]
    chosen_move_san = board.san(chosen_move)

    # Convert solution to move object
    solution_move = chess.Move.from_uci(solution_uci)
    solution_san = board.san(solution_move)

    # Check if the chosen move is the solution
    found_mate = (move_encoder.move_to_index(chosen_move) ==
                  move_encoder.move_to_index(solution_move))

    # Find rank of solution
    rank_of_solution = None
    for rank, (move, count) in enumerate(sorted_moves, 1):
        if move_encoder.move_to_index(move) == move_encoder.move_to_index(solution_move):
            rank_of_solution = rank
            break

    if verbose:
        print(f"\n  Top 5 moves:")
        for rank, (move, count) in enumerate(sorted_moves[:5], 1):
            is_solution = (move_encoder.move_to_index(move) ==
                         move_encoder.move_to_index(solution_move))
            marker = " ← SOLUTION" if is_solution else ""
            percentage = 100 * count / sum(visit_counts.values())
            print(f"    {rank}. {board.san(move)}: {count} visits ({percentage:.1f}%){marker}")

        result_marker = "✓" if found_mate else "❌"
        print(f"\n  {result_marker} Chosen: {chosen_move_san}, Solution: {solution_san}")
        if not found_mate:
            print(f"  Solution ranked #{rank_of_solution}")

    return found_mate, chosen_move_san, rank_of_solution


def evaluate_model(model_path: str, num_simulations: int = 100, verbose: bool = True):
    """
    Evaluate a trained model on training and test positions

    Args:
        model_path: Path to the saved model checkpoint
        num_simulations: Number of MCTS simulations per position
        verbose: Print detailed results
    """
    print("=" * 70)
    print("Mate-in-1 Model Evaluation")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['network_config']
    network = ChessNet(
        input_channels=config['input_channels'],
        num_res_blocks=config['num_res_blocks'],
        num_channels=config['num_channels']
    )
    network.load_state_dict(checkpoint['model_state_dict'])
    network.to(device)
    network.eval()

    iteration = checkpoint.get('iteration', 'unknown')
    print(f"Model from iteration: {iteration}")
    print(f"Device: {device}")
    print(f"MCTS simulations: {num_simulations}\n")

    # Initialize MCTS
    mcts = MCTS(network, device, num_simulations=num_simulations)
    move_encoder = MoveEncoder()

    # Test on training positions
    print("=" * 70)
    print("TRAINING POSITIONS (seen during training)")
    print("=" * 70)

    training_results = []
    for i, (fen, solution_uci, description) in enumerate(TRAINING_POSITIONS, 1):
        if verbose:
            print(f"\n{i}. {description}")
            print(f"   FEN: {fen}")

        board = chess.Board(fen)
        found, chosen, rank = test_position(board, solution_uci, mcts, move_encoder, verbose)
        training_results.append((found, chosen, rank, description))

    # Test on test positions
    print("\n" + "=" * 70)
    print("TEST POSITIONS (unseen, held out)")
    print("=" * 70)

    test_results = []
    for i, (fen, solution_uci, description) in enumerate(TEST_POSITIONS, 1):
        if verbose:
            print(f"\n{i}. {description}")
            print(f"   FEN: {fen}")

        board = chess.Board(fen)
        found, chosen, rank = test_position(board, solution_uci, mcts, move_encoder, verbose)
        test_results.append((found, chosen, rank, description))

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    training_success = sum(1 for found, _, _, _ in training_results if found)
    test_success = sum(1 for found, _, _, _ in test_results if found)

    training_total = len(training_results)
    test_total = len(test_results)

    print(f"\nTraining Set Performance:")
    print(f"  Found mate-in-1: {training_success}/{training_total} ({100*training_success/training_total:.1f}%)")

    print(f"\nTest Set Performance:")
    print(f"  Found mate-in-1: {test_success}/{test_total} ({100*test_success/test_total:.1f}%)")

    # Detailed failures
    training_failures = [(desc, rank) for found, _, rank, desc in training_results if not found]
    test_failures = [(desc, rank) for found, _, rank, desc in test_results if not found]

    if training_failures:
        print(f"\nTraining failures:")
        for desc, rank in training_failures:
            print(f"  ❌ {desc} (solution ranked #{rank})")

    if test_failures:
        print(f"\nTest failures:")
        for desc, rank in test_failures:
            print(f"  ❌ {desc} (solution ranked #{rank})")

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if training_success == training_total and test_success == test_total:
        print("🎉 Perfect! Model finds all mates in both training and test sets.")
        print("   The model has learned the concept of checkmate and generalizes well!")
    elif training_success == training_total and test_success > 0:
        print(f"✓ Good! Model finds all training mates and {test_success}/{test_total} test mates.")
        print("  The model is learning but needs more training to fully generalize.")
    elif training_success > training_total // 2:
        print(f"⚠ Partial learning. Model finds {training_success}/{training_total} training mates.")
        print("  The model is starting to learn but needs more training iterations.")
    else:
        print(f"❌ Poor performance. Model only finds {training_success}/{training_total} training mates.")
        print("  The model may need:")
        print("  - More training iterations")
        print("  - More MCTS simulations")
        print("  - Verification that the training is working correctly")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate mate-in-1 model performance")
    parser.add_argument("model_path", help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--simulations", type=int, default=100,
                       help="Number of MCTS simulations (default: 100)")
    parser.add_argument("--quiet", action="store_true",
                       help="Only show summary, not detailed results")

    args = parser.parse_args()

    evaluate_model(args.model_path, num_simulations=args.simulations,
                  verbose=not args.quiet)


if __name__ == "__main__":
    main()
