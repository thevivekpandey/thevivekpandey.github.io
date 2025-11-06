import random 
import chess
import math
from copy import deepcopy

class MCTSNode:
    """Node in the Monte Carlo Tree Search."""
    
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()  # Chess board state
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = []
        self.wins = 0
        self.visits = 0
        self.evals = []
        self.untried_moves = list(board.legal_moves)
        random.shuffle(self.untried_moves)
    
    def is_fully_expanded(self):
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Check if this is a terminal node (game over)."""
        return self.board.is_game_over()
    
    def best_child(self, exploration_weight=1.414):
        """
        Select the best child using UCB1 (Upper Confidence Bound) formula.
        UCB1 = (wins/visits) + exploration_weight * sqrt(ln(parent_visits)/visits)
        """
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits) + 
                exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        )
    
    def expand(self):
        """Expand the tree by trying an untried move."""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node
    
    def simulate(self):
        """
        Simulate a random game from this position.
        Returns 1 for win, 0 for loss, 0.5 for draw.
        """
        simulation_board = self.board.copy()
        move_count = 0
        max_moves = 20  # Limit simulation length
        #print('simulation starts')
        
        #print(self.parent.board.san(self.move), end=' ')
        while not simulation_board.is_game_over() and move_count < max_moves:
            legal_moves = list(simulation_board.legal_moves)
            if not legal_moves:
                break
            
            # Use weighted random selection favoring captures and checks
            #print([simulation_board.san(m) for m in legal_moves])
            move = self.select_simulation_move(simulation_board, legal_moves)
            #print(simulation_board.san(move), end=' ')
            simulation_board.push(move)
            move_count += 1
        
        # Return result from perspective of player who made the move to this node
        score = self.evaluate_position(simulation_board)
        return score
    
    def select_simulation_move(self, board, legal_moves):
        """
        Select a move during simulation with some intelligence.
        Prioritize: checks > captures > other moves
        """
        # Categorize moves
        checks = []
        captures = []
        other_moves = []
        
        for move in legal_moves:
            board.push(move)
            if board.is_check():
                checks.append(move)
            board.pop()
            
            if board.is_capture(move):
                captures.append(move)
            else:
                other_moves.append(move)
        
        # Weighted random selection
        if checks and random.random() < 0.5:
            sim_move = random.choice(checks)
        elif captures and random.random() < 0.3:
            sim_move = random.choice(captures)
        else:
            sim_move = random.choice(legal_moves)
        #print('sim move I return is', board.san(sim_move))
        return sim_move
    
    def evaluate_position(self, board):
        """
        Evaluate the final position.
        Returns value from perspective of the player who moved to THIS node.
        """
        if board.is_checkmate():
            # If it's checkmate, the player whose turn it is lost
            # We want to return value from perspective of player who moved to this node
            if board.turn == self.board.turn:
                # Same player to move = we lost
                return 0
            else:
                # Opponent to move = we won
                return 1
        elif board.is_stalemate() or board.is_insufficient_material() or \
             board.is_fifty_moves() or board.is_repetition():
            return 0.5
        else:
            # Game exceeded move limit - evaluate material
            return self.evaluate_material(board)
    
    def evaluate_material(self, board):
        """
        Simple material evaluation for non-terminal positions.
        Returns 0.5 + small adjustment based on material difference.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == self.board.turn:
                    score += value
                else:
                    score -= value
        
        # Normalize to 0-1 range
        return 0.5 + (score / 100.0)
    
    def backpropagate(self, result):
        """Update this node and all ancestors with the simulation result."""
        self.visits += 1
        self.wins += result
        self.evals.append(result)
        if self.parent:
            # Flip result for parent (zero-sum game)
            self.parent.backpropagate(1 - result)


