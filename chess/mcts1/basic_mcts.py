import random
import chess
import time
from mcts_node import MCTSNode

class Engine():
   def get_legal_moves(self, board):
      """Get all legal moves for the current position."""
      return list(board.legal_moves)
   
   def mcts_search(self, board, time_limit, max_iterations):
       root = MCTSNode(board)
       start_time = time.time()
       iterations = 0
       print(board)
   
       while time.time() - start_time < time_limit and iterations < max_iterations:
           node = root
           
           # Selection: Navigate down the tree using UCB1
           while not node.is_terminal() and node.is_fully_expanded():
               node = node.best_child()
           
           # Expansion: Add a new child node if possible
           if not node.is_terminal() and not node.is_fully_expanded():
               node = node.expand()
               #print('will simul from here')
               #print(node.board)
           
           # Simulation: Play out a random game
           result = node.simulate()
           
           # Backpropagation: Update all nodes in the path
           node.backpropagate(1 - result)
           
           iterations += 1
       
       if not root.children:
           # Fallback to random if no children
           return random.choice(list(board.legal_moves))
       
       #best_child = max(root.children, key=lambda c: c.visits)
       best_child = max(root.children, key=lambda c: c.wins/c.visits)
       for c in root.children:
          for i in range(len(c.evals)):
             c.evals[i] = round(c.evals[i], 2)
          print(root.board.san(c.move),"w = ", round(c.wins, 2), "v = ", c.visits, 'evals=', c.evals)
      
       print(f"MCTS completed {iterations} iterations in {time.time() - start_time:.2f}s")
       print(f"Best move: {best_child.move} (visits: {best_child.visits}, win rate: {best_child.wins/best_child.visits:.2%})")
       
       return best_child.move

   def get_move(self, board):
       time_limit = 2.0
       max_iterations = 1000
       return self.mcts_search(board, time_limit, max_iterations)

if __name__ == '__main__':
   e = Engine()
   board = chess.Board('rnbqkbnr/1ppp1ppp/p7/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1')
   move = e.get_move(board)
   print(board.san(move))

   
