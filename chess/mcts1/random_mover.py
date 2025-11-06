import random
import chess

class Engine():
   def get_legal_moves(self, board):
      """Get all legal moves for the current position."""
      return list(board.legal_moves)
   
   def choose_random_move(self, board):
      """Choose a random legal move."""
      legal_moves = self.get_legal_moves(board)
      if legal_moves:
         return random.choice(legal_moves)
      return None
      
if __name__ == '__main__':
   e = Engine()
   board = chess.Board('rnbqkbnr/1ppp1ppp/p7/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1')
   move = e.choose_random_move(board)
   print(board.san(move))
   
