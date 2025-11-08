"""
Verify that training is actually happening by comparing models from different iterations
"""

import chess
import numpy as np
import torch
import glob
import os
from mini_leela_complete import ChessNet, BoardEncoder, MCTS

def load_model(model_path):
    """Load a model checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    network = ChessNet(
        input_channels=checkpoint['network_config']['input_channels'],
        num_res_blocks=checkpoint['network_config']['num_res_blocks'],
        num_channels=checkpoint['network_config']['num_channels']
    )
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    return network, checkpoint['iteration']

def compare_models_on_position(model1, model2, board):
    """Compare two models' outputs on the same position"""
    encoder = BoardEncoder()
    board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0)

    with torch.no_grad():
        policy1, value1 = model1(board_tensor)
        policy2, value2 = model2(board_tensor)

        # Get top 5 moves for each model
        legal_moves = list(board.legal_moves)
        move_scores1 = {}
        move_scores2 = {}

        for move in legal_moves:
            idx = move.from_square * 64 + move.to_square
            move_scores1[move] = policy1[0, idx].item()
            move_scores2[move] = policy2[0, idx].item()

        top5_model1 = sorted(move_scores1.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_model2 = sorted(move_scores2.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'value1': value1.item(),
            'value2': value2.item(),
            'value_diff': abs(value1.item() - value2.item()),
            'top5_model1': top5_model1,
            'top5_model2': top5_model2,
            'policy_correlation': np.corrcoef(
                [move_scores1[m] for m in legal_moves],
                [move_scores2[m] for m in legal_moves]
            )[0, 1]
        }

def play_game_between_models(model1, model2, num_simulations=100):
    """Play a game between two models and return the result"""
    board = chess.Board()
    mcts1 = MCTS(model1, 'cpu', num_simulations)
    mcts2 = MCTS(model2, 'cpu', num_simulations)

    move_count = 0
    max_moves = 150  # Prevent infinite games

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            mcts = mcts1
        else:
            mcts = mcts2

        visit_counts, _ = mcts.search(board)
        move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
        board.push(move)
        move_count += 1

    if move_count >= max_moves:
        return "1/2-1/2 (max moves)"

    return board.result()

def main():
    print("="*70)
    print("Training Verification Script")
    print("="*70)

    # Find all model files
    model_files = sorted(glob.glob("mini_leela_model_iter*.pth"))

    if len(model_files) < 2:
        print("\nERROR: Need at least 2 model files to compare!")
        return

    print(f"\nFound {len(model_files)} model files")

    # Load first and latest models
    first_model_path = model_files[0]
    last_model_path = model_files[-1]

    print(f"\nLoading models:")
    print(f"  First: {first_model_path}")
    print(f"  Latest: {last_model_path}")

    model1, iter1 = load_model(first_model_path)
    model2, iter2 = load_model(last_model_path)

    print(f"\nComparing iteration {iter1} vs iteration {iter2}")

    # Test on several positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),  # After 1.e4 e5 2.Nf3 Nc6
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3"),  # Two knights defense
    ]

    print("\n" + "="*70)
    print("TEST 1: Output Comparison on Different Positions")
    print("="*70)

    total_value_diff = 0
    total_policy_corr = 0

    for i, board in enumerate(test_positions):
        print(f"\nPosition {i+1}: {board.fen()[:50]}...")
        results = compare_models_on_position(model1, model2, board)

        print(f"  Value Iter{iter1}: {results['value1']:+.4f}")
        print(f"  Value Iter{iter2}: {results['value2']:+.4f}")
        print(f"  Value Difference: {results['value_diff']:.4f}")
        print(f"  Policy Correlation: {results['policy_correlation']:.4f}")

        print(f"\n  Top moves (Iter{iter1}):")
        for move, score in results['top5_model1']:
            print(f"    {move}: {score:.4f}")

        print(f"  Top moves (Iter{iter2}):")
        for move, score in results['top5_model2']:
            print(f"    {move}: {score:.4f}")

        total_value_diff += results['value_diff']
        total_policy_corr += results['policy_correlation']

    avg_value_diff = total_value_diff / len(test_positions)
    avg_policy_corr = total_policy_corr / len(test_positions)

    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"  Average Value Difference: {avg_value_diff:.4f}")
    print(f"  Average Policy Correlation: {avg_policy_corr:.4f}")
    print("="*70)

    print("\nINTERPRETATION:")
    if avg_value_diff < 0.01:
        print("  ⚠️  WARNING: Value predictions are nearly identical!")
        print("      This suggests the value head is not learning.")
    else:
        print("  ✓ Value predictions are different between iterations.")

    if avg_policy_corr > 0.98:
        print("  ⚠️  WARNING: Policy outputs are highly correlated!")
        print("      This suggests the policy head is not learning much.")
    else:
        print("  ✓ Policy outputs have changed between iterations.")

    # Play games with different MCTS simulation counts
    print("\n" + "="*70)
    print("TEST 2: Play Games with Different MCTS Simulations")
    print("="*70)

    for num_sims in [10, 50, 200]:
        print(f"\nPlaying with {num_sims} simulations per move...")
        result = play_game_between_models(model2, model1, num_sims)
        print(f"  Result (Iter{iter2} as White vs Iter{iter1} as Black): {result}")

    print("\n" + "="*70)
    print("TEST 3: Checking Weight Changes")
    print("="*70)

    # Check if weights have actually changed
    param_diffs = []
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        diff = torch.abs(p1 - p2).mean().item()
        param_diffs.append((name1, diff))

    print("\nLargest weight changes:")
    sorted_diffs = sorted(param_diffs, key=lambda x: x[1], reverse=True)[:10]
    for name, diff in sorted_diffs:
        print(f"  {name}: {diff:.6f}")

    if sorted_diffs[0][1] < 1e-6:
        print("\n  ⚠️  WARNING: Weights have barely changed!")
        print("      Training may not be happening correctly.")
    else:
        print("\n  ✓ Weights have changed significantly.")

    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
1. Fix the loss function bug (use KL divergence for policy loss)
2. Increase MCTS simulations during self-play (try 100-200)
3. Generate more games per iteration (try 20-50)
4. Add a replay buffer to retain training data
5. Try playing with more simulations (500+) for evaluation
6. Check training loss curves - are they decreasing?
7. Consider temperature scheduling (start with 1.0, reduce over time)
    """)

if __name__ == "__main__":
    main()
