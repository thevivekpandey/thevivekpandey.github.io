"""
Minimal Leela Chess Zero Implementation
A simplified but complete chess training pipeline with ResNet + MCTS

This implementation includes:
1. Board representation (bitboard-like encoding)
2. ResNet neural network with policy and value heads
3. Monte Carlo Tree Search (MCTS)
4. Self-play game generation
5. Training loop

Simplified aspects:
- Uses only current position (not 8 positions)
- Smaller network (4 ResNet blocks instead of 20+)
- Simpler input encoding
- Fewer MCTS simulations
"""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
from typing import List, Tuple, Dict
import copy
from datetime import datetime


# ============================================================================
# PART 1: BOARD REPRESENTATION
# ============================================================================

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


# ============================================================================
# PART 2: NEURAL NETWORK (ResNet with Policy and Value Heads)
# ============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block: Conv-BN-ReLU-Conv-BN-Add-ReLU"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ChessNet(nn.Module):
    """
    ResNet-based neural network with policy and value heads.
    Similar to AlphaZero/Leela architecture but smaller.
    """
    
    def __init__(self, input_channels: int = 19, num_res_blocks: int = 4, num_channels: int = 128):
        super().__init__()
        
        # Initial convolutional block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # Output: 64*64 = 4096 for all from-to square combinations
        # (oversized but simple; legal move masking handles invalid moves)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 19, 8, 8) board representation
        
        Returns:
            policy_logits: (batch, 4096) move probabilities
            value: (batch, 1) position evaluation
        """
        # Shared representation
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy_logits, value


# ============================================================================
# PART 3: MOVE ENCODING/DECODING
# ============================================================================

class MoveEncoder:
    """Convert between chess.Move and network output indices"""
    
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """Convert move to index in policy output (from_square * 64 + to_square)"""
        return move.from_square * 64 + move.to_square
    
    @staticmethod
    def index_to_move(index: int, board: chess.Board) -> chess.Move:
        """Convert policy index to move (with promotion handling)"""
        from_square = index // 64
        to_square = index % 64
        
        # Handle promotions (default to queen)
        move = chess.Move(from_square, to_square)
        if board.piece_at(from_square) and board.piece_at(from_square).piece_type == chess.PAWN:
            if chess.square_rank(to_square) in [0, 7]:
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        return move


# ============================================================================
# PART 4: MONTE CARLO TREE SEARCH
# ============================================================================

class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # P(s,a) from policy network
        
        self.children = {}  # Dict[move] -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def uct_score(self, parent_visit_count: int, c_puct: float = 1.0):
        """
        Upper Confidence Bound for Trees with prior (PUCT algorithm)
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        exploration = c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.value() + exploration


class MCTS:
    """Monte Carlo Tree Search using neural network for evaluation"""
    
    def __init__(self, network: ChessNet, device: str = 'cpu', num_simulations: int = 100):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
    
    def search(self, board: chess.Board) -> Tuple[Dict[chess.Move, int], MCTSNode]:
        """
        Run MCTS from the given board position.
        
        Returns:
            visit_counts: Dict mapping moves to their visit counts
            root: The root node (for debugging)
        """
        root = MCTSNode(board.copy())
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Extract visit counts for each move
        visit_counts = {}
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count
        
        return visit_counts, root
    
    def _simulate(self, node: MCTSNode):
        """Single MCTS simulation: select, expand, evaluate, backpropagate"""
        # 1. Select: traverse tree to a leaf
        path = [node]
        while not node.is_leaf() and not node.board.is_game_over():
            node = self._select_child(node)
            path.append(node)
        
        # 2. Expand and Evaluate
        if node.board.is_game_over():
            # Terminal node
            result = node.board.result()
            if result == "1-0":
                value = 1.0
            elif result == "0-1":
                value = -1.0
            else:
                value = 0.0
        else:
            # Expand leaf node and get network evaluation
            value = self._expand_and_evaluate(node)
        
        # 3. Backpropagate
        self._backpropagate(path, value)
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCT score"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.uct_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand node with network policy and return value estimate"""
        board = node.board
        
        # Get network predictions
        board_tensor = torch.FloatTensor(self.encoder.encode_board(board)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.network(board_tensor)
            policy_logits = policy_logits.cpu().numpy()[0]
            value = value.cpu().item()
        
        # Mask illegal moves and create probability distribution
        legal_moves = list(board.legal_moves)
        move_probs = {}
        
        for move in legal_moves:
            idx = self.move_encoder.move_to_index(move)
            move_probs[move] = policy_logits[idx]
        
        # Softmax over legal moves
        max_logit = max(move_probs.values())
        move_probs = {m: np.exp(p - max_logit) for m, p in move_probs.items()}
        total = sum(move_probs.values())
        move_probs = {m: p / total for m, p in move_probs.items()}
        
        # Create children
        for move, prob in move_probs.items():
            new_board = board.copy()
            new_board.push(move)
            node.children[move] = MCTSNode(new_board, parent=node, move=move, prior=prob)
        
        return value
    
    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Update statistics along the path"""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip perspective for opponent


# ============================================================================
# PART 5: SELF-PLAY
# ============================================================================

class SelfPlayGame:
    """Generate self-play games for training"""
    
    def __init__(self, network: ChessNet, device: str = 'cpu', 
                 num_simulations: int = 100, temperature: float = 1.0):
        self.mcts = MCTS(network, device, num_simulations)
        self.encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.temperature = temperature
    
    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play a self-play game.
        
        Returns:
            training_data: List of (board_state, policy_target, value_target)
        """
        board = chess.Board()
        training_data = []
        
        while not board.is_game_over():
            # Run MCTS
            visit_counts, _ = self.mcts.search(board)
            
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
            
            # Store training example (we'll fill in the value target at the end)
            board_state = self.encoder.encode_board(board)
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
        
        # Set value targets (flip perspective each move)
        final_data = []
        for i, (state, policy, _) in enumerate(training_data):
            value = game_value if i % 2 == 0 else -game_value
            final_data.append((state, policy, value))
        
        #Overall - final data is list of - sir these are the board positions you had
        #these were your policy vectors for each position, and this was the result
        return final_data


# ============================================================================
# PART 6: TRAINING LOOP
# ============================================================================

class ChessTrainer:
    """Training pipeline for the chess network"""
    
    def __init__(self, network: ChessNet, device: str = 'cpu', lr: float = 0.001):
        self.network = network.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    def train_on_batch(self, batch_data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Train network on a batch of self-play data.
        
        Args:
            batch_data: List of (board_state, policy_target, value_target)
        """
        if len(batch_data) == 0:
            return 0.0, 0.0
        
        # Prepare batch
        states = torch.FloatTensor(np.array([d[0] for d in batch_data])).to(self.device)
        policy_targets = torch.FloatTensor(np.array([d[1] for d in batch_data])).to(self.device)
        value_targets = torch.FloatTensor(np.array([d[2] for d in batch_data])).unsqueeze(1).to(self.device)
        
        # Forward pass
        policy_logits, values = self.network(states)
        
        # Compute losses
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        value_loss = F.mse_loss(values, value_targets)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train_iteration(self, num_games: int = 10, batch_size: int = 32):
        """
        Run one training iteration: generate self-play games and train.
        
        Args:
            num_games: Number of self-play games to generate
            batch_size: Batch size for training
        """
        print(f"Generating {num_games} self-play games...")
        
        self.network.eval()
        self_play = SelfPlayGame(self.network, self.device, num_simulations=50)
        
        all_data = []
        for i in range(num_games):
            game_data = self_play.play_game()
            all_data.extend(game_data)
            print(f"Game {i+1}/{num_games}: {len(game_data)} positions")
        
        print(f"Total training positions: {len(all_data)}")
        
        # Train on data
        self.network.train()
        random.shuffle(all_data)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
            policy_loss, value_loss = self.train_on_batch(batch)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            num_batches += 1
        
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        print(f"Training complete - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
        
        return avg_policy_loss, avg_value_loss


# ============================================================================
# PART 7: MAIN - DEMONSTRATION
# ============================================================================

def main():
    """Demonstrate the full pipeline"""
    print("=" * 70)
    print("Mini Leela Chess Zero - Training Pipeline")
    print("=" * 70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create network
    print("\nInitializing neural network...")
    network = ChessNet(input_channels=19, num_res_blocks=4, num_channels=128)
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Create trainer
    trainer = ChessTrainer(network, device=device, lr=0.001)
    
    # Run training iterations
    num_iterations = 100
    print(f"\nRunning {num_iterations} training iterations...")

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"{'='*70}")

        trainer.train_iteration(num_games=5, batch_size=32)

        # Save the model after each iteration with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"mini_leela_model_iter{iteration:03d}_{timestamp}.pth"
        print(f"\nSaving model to {model_path}...")
        torch.save({
            'model_state_dict': network.state_dict(),
            'network_config': {
                'input_channels': 19,
                'num_res_blocks': 4,
                'num_channels': 128
            },
            'iteration': iteration
        }, model_path)
        print(f"Model saved successfully!")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    # Demonstrate the trained network
    print("\nDemonstrating trained network on starting position...")
    board = chess.Board()
    
    mcts = MCTS(network, device=device, num_simulations=100)
    visit_counts, root = mcts.search(board)
    
    print("\nTop 5 moves by visit count:")
    sorted_moves = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for move, count in sorted_moves:
        print(f"  {move}: {count} visits ({100*count/sum(visit_counts.values()):.1f}%)")
    
    print("\n" + "="*70)
    print("Done! This network can now be used for:")
    print("  - Playing chess games")
    print("  - Further self-play training")
    print("  - Analysis and evaluation")
    print("="*70)


if __name__ == "__main__":
    main()
