"""
Hybrid Training: Supervised pre-training + RL fine-tuning

Step 1: Train network with supervised learning on correct moves
Step 2: Use trained network for MCTS-based RL training
"""
import chess
import torch
import numpy as np
import random
from datetime import datetime

from mini_leela_complete_fixed import ChessNet, BoardEncoder, MoveEncoder, MCTS, ChessTrainer
from mate_in_1_positions import get_training_fens, get_solution, TRAINING_POSITIONS

print("="*70)
print("Hybrid Training: Supervised → Reinforcement Learning")
print("="*70)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

encoder = BoardEncoder()
move_encoder = MoveEncoder()
network = ChessNet()
trainer = ChessTrainer(network, device=device, lr=0.01)

# ============================================================================
# PHASE 1: SUPERVISED PRE-TRAINING
# ============================================================================
print("="*70)
print("PHASE 1: Supervised Pre-training")
print("="*70)

# Prepare supervised data
supervised_data = []
for fen, solution_uci, description in TRAINING_POSITIONS:
    board = chess.Board(fen)
    solution_move = chess.Move.from_uci(solution_uci)
    solution_idx = move_encoder.move_to_index(solution_move)

    board_state = encoder.encode_board(board)
    policy_target = np.zeros(4096, dtype=np.float32)
    policy_target[solution_idx] = 1.0
    value_target = 1.0

    supervised_data.append((board_state, policy_target, value_target))

print(f"Training on {len(supervised_data)} positions with direct supervision\n")

# Train supervised
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    random.shuffle(supervised_data)
    network.train()

    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    num_batches = 0

    for i in range(0, len(supervised_data), 10):
        batch = supervised_data[i:i+10]
        policy_loss, value_loss = trainer.train_on_batch(batch)
        epoch_policy_loss += policy_loss
        epoch_value_loss += value_loss
        num_batches += 1

    if epoch % 5 == 0:
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches

        # Test accuracy
        network.eval()
        correct = 0
        for fen, solution_uci, _ in TRAINING_POSITIONS:
            board = chess.Board(fen)
            solution_idx = move_encoder.move_to_index(chess.Move.from_uci(solution_uci))
            board_tensor = torch.FloatTensor(encoder.encode_board(board)).unsqueeze(0).to(device)

            with torch.no_grad():
                policy_logits, _ = network(board_tensor)

            legal_moves = list(board.legal_moves)
            legal_indices = [move_encoder.move_to_index(m) for m in legal_moves]
            legal_logits = policy_logits[0, legal_indices]
            best_idx = legal_indices[legal_logits.argmax().item()]

            if best_idx == solution_idx:
                correct += 1

        print(f"Epoch {epoch:2d}: Loss={avg_policy_loss:.4f}, {avg_value_loss:.4f}, Accuracy={correct}/10")

print("\n✓ Supervised pre-training complete!")

# Save pre-trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pretrain_path = f"pretrained_supervised_{timestamp}.pth"
torch.save({
    'model_state_dict': network.state_dict(),
    'network_config': {'input_channels': 19, 'num_res_blocks': 4, 'num_channels': 128},
    'phase': 'supervised_pretrain'
}, pretrain_path)
print(f"Saved pre-trained model: {pretrain_path}\n")

# ============================================================================
# PHASE 2: REINFORCEMENT LEARNING FINE-TUNING
# ============================================================================
print("="*70)
print("PHASE 2: RL Fine-tuning with MCTS")
print("="*70)
print("Now the network should guide MCTS better!\n")

# Reduce learning rate for fine-tuning
trainer.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

# RL training loop
num_rl_iterations = 50
games_per_iteration = 5
num_simulations = 250

for iteration in range(1, num_rl_iterations + 1):
    print(f"RL Iteration {iteration}/{num_rl_iterations}")

    # Generate self-play data
    network.eval()
    mcts = MCTS(network, device, num_simulations=num_simulations)

    rl_data = []
    for game_num in range(games_per_iteration):
        # Random position
        fen = random.choice(get_training_fens())
        solution_uci, _ = get_solution(fen)
        board = chess.Board(fen)

        # MCTS search
        visit_counts, _ = mcts.search(board)

        # Create policy target from visit counts
        policy_target = np.zeros(4096, dtype=np.float32)
        total_visits = sum(visit_counts.values())
        for move, count in visit_counts.items():
            idx = move_encoder.move_to_index(move)
            policy_target[idx] = count / total_visits

        # Store
        board_state = encoder.encode_board(board)
        rl_data.append((board_state, policy_target, 1.0))

    # Train on RL data
    network.train()
    random.shuffle(rl_data)

    rl_policy_loss = 0.0
    rl_value_loss = 0.0
    num_batches = 0

    for i in range(0, len(rl_data), 5):
        batch = rl_data[i:i+5]
        policy_loss, value_loss = trainer.train_on_batch(batch)
        rl_policy_loss += policy_loss
        rl_value_loss += value_loss
        num_batches += 1

    if iteration % 10 == 0:
        avg_policy = rl_policy_loss / num_batches
        avg_value = rl_value_loss / num_batches

        # Test accuracy
        network.eval()
        correct = 0
        for fen, solution_uci, _ in TRAINING_POSITIONS:
            board = chess.Board(fen)
            solution_idx = move_encoder.move_to_index(chess.Move.from_uci(solution_uci))

            # Use MCTS for testing
            mcts = MCTS(network, device, num_simulations=100)
            visit_counts, _ = mcts.search(board)
            best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
            best_idx = move_encoder.move_to_index(best_move)

            if best_idx == solution_idx:
                correct += 1

        print(f"  Iter {iteration}: Loss={avg_policy:.4f}, {avg_value:.4f}, MCTS Accuracy={correct}/10")

        # Save checkpoint
        if iteration % 25 == 0:
            rl_path = f"hybrid_rl_iter{iteration:03d}_{timestamp}.pth"
            torch.save({
                'model_state_dict': network.state_dict(),
                'network_config': {'input_channels': 19, 'num_res_blocks': 4, 'num_channels': 128},
                'iteration': iteration,
                'phase': 'rl_finetuning'
            }, rl_path)
            print(f"  Saved: {rl_path}")

print("\n" + "="*70)
print("Hybrid Training Complete!")
print("="*70)
print(f"\nFinal models saved. Test with test_mate_in_1.py")
