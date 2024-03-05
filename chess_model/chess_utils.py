import chess
# import random
# import pickle
# import numpy as np

def piece_symbol(piece):
  # Map piece type ID to corresponding symbol
  piece_symbols = {
      chess.PAWN: 'p',
      chess.KNIGHT: 'n',
      chess.BISHOP: 'b',
      chess.ROOK: 'r',
      chess.QUEEN: 'q',
      chess.KING: 'k',
  }
  return piece_symbols[piece.piece_type]

def state_to_string(board):
  # Example: convert board to a string representation (replace with your desired method)
  return board.fen()

def action_to_uci(action, board):
  # Convert action index to UCI move string based on legal moves
  legal_moves = list(board.legal_moves)
  return legal_moves[action].uci()

def uci_to_action(move, board):
  # Convert UCI move string to action index based on legal moves
  legal_moves = list(board.legal_moves)
  for i, m in enumerate(legal_moves):
    if m.uci() == move:
      return i

def evaluate_board(board):
  # Assign values to different pieces
  piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 200}
  
  score = 0
  for square, piece in board.piece_map().items():
    # Access piece type and color
    aPiece_symbol = piece_symbol(piece)
    color = piece.color
    # print(piece_type)
    # Add or subtract piece value based on color
    piece_score = piece_values[aPiece_symbol] if color == chess.WHITE else -piece_values[aPiece_symbol]
    score += piece_score
  
  # Add bonus for king safety (implement later)
  # ...
  
  return score

def make_random_move(board):
  # Choose a legal move randomly
  legal_moves = list(board.legal_moves)
  return random.choice(legal_moves)