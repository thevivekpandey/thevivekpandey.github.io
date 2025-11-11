"""
Mini Leela - Mate-in-1 Training
Specialized training script for debugging with mate-in-1 positions

This script trains the model on mate-in-1 positions to verify learning is working.
"""

import chess
import numpy as np
import torch
import random
from datetime import datetime
from typing import List, Tuple

# Import from the FIXED implementation (corrected policy loss)
from mini_leela_complete_fixed import (
    BoardEncoder, ChessNet, MoveEncoder, MCTS, ChessTrainer
)

# Import mate-in-1 positions
from mate_in_1_positions import get_training_fens, get_solution


class MateIn1SelfPlay:
    """Generate self-play games starting from mate-in-1 positions"""

    def __init__(self, network: ChessNet, device: str = 'cpu',
                 num_simulations: int = 100, temperature: float = 1.0):
        self.mcts = MCTS(network, device, num_simulations)
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.temperature = temperature
        self.training_fens = get_training_fens()

    def play_game(self, verbose=True) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], str, str]:
        """
        Play a self-play game starting from a random mate-in-1 position.

        Returns:
            training_data: List of (board_state, policy_target, value_target)
            starting_fen: The FEN position used
            first_move: The first move made (in SAN notation)
        """
        # Randomly select a training position
        fen = random.choice(self.training_fens)
        solution_uci, description = get_solution(fen)

        board = chess.Board(fen)
        training_data = []

        if verbose:
            print(f"\nStarting position: {description}")
            print(f"Solution: {solution_uci} ({board.san(chess.Move.from_uci(solution_uci))})")
            print(f"Position: {fen}\n")

        move_count = 0
        first_move_san = None

        while not board.is_game_over() and move_count < 10:  # Limit moves to prevent long games
            # Run MCTS
            visit_counts, root = self.mcts.search(board)

            # Log the top moves considered
            if verbose and move_count == 0:
                print("Top 5 moves by visit count:")
                sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for move, count in sorted_moves:
                    is_solution = (self.move_encoder.move_to_index(move) ==
                                 self.move_encoder.move_to_index(chess.Move.from_uci(solution_uci)))
                    marker = " ← SOLUTION!" if is_solution else ""
                    print(f"  {board.san(move)}: {count} visits ({100*count/sum(visit_counts.values()):.1f}%){marker}")

            # Create policy target (visit count distribution)
            policy_target = np.zeros(4096, dtype=np.float32)

            if self.temperature > 0:
                # Sample move proportional to visit_count^(1/temperature)
                moves = list(visit_counts.keys())
                counts = np.array([visit_counts[m] for m in moves])

                if self.temperature != 1.0:
                    counts = counts ** (1.0 / self.temperature)

                probs = counts / counts.sum()

                # Set policy targets
                for move, prob in zip(moves, probs):
                    idx = self.move_encoder.move_to_index(move)
                    policy_target[idx] = prob

                # Sample move
                move = np.random.choice(moves, p=probs)
            else:
                # Greedy: pick most visited
                move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
                idx = self.move_encoder.move_to_index(move)
                policy_target[idx] = 1.0

            # Store training example - ONLY for the initial mate-in-1 position
            # We don't want to train on random positions after a wrong first move
            if move_count == 0:
                board_state = self.encoder.encode_board(board)
                training_data.append((board_state, policy_target, 0.0))

            # Track first move
            if move_count == 0:
                first_move_san = board.san(move)
                if verbose:
                    print(f"\nChose: {first_move_san}")

            # Make move
            if verbose:
                if move_count % 2 == 0:
                    print(f"{(move_count // 2) + 1}. ", end='')
                print(f"{board.san(move)} ", end='')

            board.push(move)
            move_count += 1

        if verbose:
            print()

        # For mate-in-1 positions, we KNOW the value is +1.0 (winning position)
        # Don't rely on game outcome since if MCTS misses mate, the game might not end in a win
        # All our training positions are White to move with mate in 1, so value = +1.0
        game_value = 1.0

        if verbose:
            result = board.result()
            print(f"Result: {result} (value for mate-in-1 position: {game_value})\n")

        # Set value targets - since we only train on move 0, all positions have value +1.0
        final_data = []
        for i, (state, policy, _) in enumerate(training_data):
            # All positions are from move 0 (White to move), so value = +1.0
            value = game_value
            final_data.append((state, policy, value))

        return final_data, fen, first_move_san


def train_mate_in_1(num_iterations: int = 20, games_per_iteration: int = 5,
                    num_simulations: int = 100, save_every: int = 5):
    """
    Train on mate-in-1 positions with detailed logging

    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Number of games to play per iteration
        num_simulations: Number of MCTS simulations per move
        save_every: Save model every N iterations
    """
    print("=" * 70)
    print("Mini Leela - Mate-in-1 Training")
    print("=" * 70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Training positions: {len(get_training_fens())}")
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"MCTS simulations: {num_simulations}\n")

    # Create network
    network = ChessNet(input_channels=19, num_res_blocks=4, num_channels=128)
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}\n")

    # Create trainer
    trainer = ChessTrainer(network, device=device, lr=0.001)

    # Training loop
    for iteration in range(1, num_iterations + 1):
        print(f"{'='*70}")
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"{'='*70}")

        # Generate self-play games
        network.eval()
        self_play = MateIn1SelfPlay(network, device, num_simulations=num_simulations)

        all_data = []
        for game_num in range(games_per_iteration):
            #print(f"\n--- Game {game_num + 1}/{games_per_iteration} ---")
            game_data, fen, first_move = self_play.play_game(verbose=False)
            all_data.extend(game_data)
            #print(f"Collected {len(game_data)} training positions")

        print(f"\nTotal positions this iteration: {len(all_data)}")

        # Train on data
        network.train()
        random.shuffle(all_data)

        batch_size = 32
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
            policy_loss, value_loss = trainer.train_on_batch(batch)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            num_batches += 1

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        print(f"\n📊 Training Loss - Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")

        # Save model periodically
        if iteration % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"mate_in_1_model_iter{iteration:03d}_{timestamp}.pth"
            print(f"\n💾 Saving model to {model_path}...")
            torch.save({
                'model_state_dict': network.state_dict(),
                'network_config': {
                    'input_channels': 19,
                    'num_res_blocks': 4,
                    'num_channels': 128
                },
                'iteration': iteration,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss
            }, model_path)
            print("✓ Model saved!")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    return network


if __name__ == "__main__":
    # Train the model
    trained_network = train_mate_in_1(
        num_iterations=1000,
        games_per_iteration=5,
        num_simulations=250,  # Reduced for faster training
        save_every=5
    )

    print("\n✓ Training finished! Use test_mate_in_1.py to evaluate the model.")
