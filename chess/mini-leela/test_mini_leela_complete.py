from mini_leela_complete import MCTS, ChessNet
import argparse
import chess
import torch
import numpy as np

model_path = "Nov8_run/mini_leela_model_iter001_20251107_210721.pth"
checkpoint = torch.load(model_path, map_location='cpu')

# Extract network configuration
config = checkpoint['network_config']
network = ChessNet(
    input_channels=config['input_channels'],
    num_res_blocks=config['num_res_blocks'],
    num_channels=config['num_channels']
)

# Load model weights
network.load_state_dict(checkpoint['model_state_dict'])
network.to('cpu')
network.eval()

# Initialize MCTS
mcts = MCTS(network, 'cpu', num_simulations=10)

fen = 'r1bqkbnr/1ppp1ppp/p1n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1'
board = chess.Board(fen)

training_data = []
while not board.is_game_over():
    # Run MCTS
    visit_counts, _ = mcts.search(board)
    
    # Create policy target (visit count distribution)
    policy_target = np.zeros(4096, dtype=np.float32)
    
    # Greedy: pick most visited
    move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
    idx = mcts.move_encoder.move_to_index(move)
    policy_target[idx] = 1.0
    
    # Store training example (we'll fill in the value target at the end)
    board_state = mcts.encoder.encode_board(board)
    training_data.append((board_state, policy_target, 0.0))
    
    # Make move
    board.push(move)

# Fill in game outcome for all positions
result = board.result()
if result == "1-0":
    game_value = 1.0
elif result == "0-1":
    game_value = -1.0
else:
    game_value = 0.0

final_data = []
for i, (state, policy, _) in enumerate(training_data):
    value = game_value if i % 2 == 0 else -game_value
    final_data.append((state, policy, value))

#Overall - final data is list of - sir these are the board positions you had
#these were your policy vectors for each position, and this was the result
#print(final_data)
