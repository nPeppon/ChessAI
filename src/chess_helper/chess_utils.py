import chess
# import random
# import pickle
import numpy as np


pieces = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
colours = [chess.WHITE,chess.BLACK]

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

def board_to_vec(board: Board):
  return fen_to_vec(board.fen())

def fen_to_vec(fen: str):
    """
    Convert the given FEN string to a feature vector representation of the board state.
    
    Args:
        fen (str): The FEN string representing the board state.
    
    Returns:
        np.array: The feature vector representation of the board state composed as such:
        - for each color, for each piece type and for each square, 1 if the piece is present, 0 otherwise
        - 1 if the side to move is white, 0 if the side to move is black
        - for each color, castling right, 1 if the castling right is allowed, 0 otherwise, King side first and Queen side second
        - en passant square: 64 elements, 1 if the en passant square is allowed, 0 otherwise
        - the number of half moves since the last capture
        - the number of full moves
        
    """
    posFen = fen.split()[0]
    board = chess.BaseBoard(posFen)
    l = []

    for colour in colours:
        for piece in pieces:
            v = np.zeros(64)
            for i in list(board.pieces(piece,colour)):
                v[i] = 1
            print('Piece: ' + str(piece))
            print('colour: ' + str(colour))
            print(v)
            l.append(v)
    # Side to move - 1 = white, 0 = black
    active_color = 1 if fen.split()[1] == 'w' else 0
    l.append(active_color)
    
    # Castling rights - 0 = not allowed, 1 = allowed 
    castling_sides = ['K', 'Q', 'k', 'q']
    castling_rights_vec = [is_substring(side, fen.split()[2]) for side in castling_sides]
    l.extend(castling_rights_vec)
    #Possible en-passant
    en_passant = np.zeros(64)
    if (fen.split()[3] != '-'):
      en_passant_target = chess.SQUARES[chess.SQUARE_NAMES[fen.split()[3]]]
      en_passant[en_passant_square] = 1
    l.extend(en_passant)
    #Halfmove clock
    halfmove_clock = int(fen.split()[4])
    l.append(halfmove_clock)
    #Fullmove number
    fullmove_number = int(fen.split()[5])
    l.append(fullmove_number)
    
    l = np.concatenate(l)
    return l

def is_substring(string1, string2):
  return 1 if string1 in string2 else

def vec_to_fen(feature_vector):
  """
  Convert the given feature vector representation of the board state to a FEN string.
  
  Args:
    feature_vector (np.array): The feature vector representation of the board state.
  
  Returns:
    str: The FEN string representing the board state.
  """
  # Extract board state from feature vector
  board_state = feature_vector[:-71]  # Exclude side to move, castling rights, en passant, halfmove clock, and fullmove number
  
  # Initialize empty FEN string components
  fen_board = ''
  fen_side_to_move = 'w' if int(feature_vector[-8]) == 1 else 'b'
  fen_castling_rights = ''
  fen_en_passant = '-'
  fen_halfmove_clock = str(int(feature_vector[-2]))
  fen_fullmove_number = str(int(feature_vector[-1]))
  
  # Map indices of feature vector to corresponding piece types and colors
  indices = np.reshape(range(12 * 2 * 64), (12, 2, 64))
  
  for piece_idx in range(12):
    for color_idx in range(2):
      piece_type = piece_types[piece_idx]
      color = colors[color_idx]
      # Extract presence of each piece on the board
      piece_presence = board_state[indices[piece_idx][color_idx]].reshape((8, 8))
      for rank in range(7, -1, -1):
        empty_count = 0
        for file in range(8):
          if piece_presence[rank][file] == 0:
            empty_count += 1
          else:
            if empty_count > 0:
              fen_board += str(empty_count)
              empty_count = 0
            fen_board += piece_type if color == 'w' else piece_type.lower()
        if empty_count > 0:
          fen_board += str(empty_count)
        if rank > 0:
          fen_board += '/'
  
  # Construct FEN string for castling rights
  for side_idx in range(4):
    if int(feature_vector[-7 + side_idx]) == 1:
      fen_castling_rights += castling_rights[side_idx]
  
  # Construct FEN string for en passant square
  for square_idx in range(64):
    if int(feature_vector[-8][square_idx]) == 1:
      fen_en_passant = chess.SQUARE_NAMES[square_idx]
      break
  
  # Construct complete FEN string
  fen = ' '.join([fen_board, fen_side_to_move, fen_castling_rights or '-', fen_en_passant, fen_halfmove_clock, fen_fullmove_number])
  
  return fen

