import chess
# import random
# import pickle
import numpy as np
from typing import Tuple


CHESS_PIECES = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
CHESS_COLOURS = [chess.WHITE,chess.BLACK]
CHESS_NUM_ACTIONS = 4672

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

def board_to_vec(board: chess.Board):
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

    for colour in CHESS_COLOURS:
        for piece in CHESS_PIECES:
            v = np.zeros(64)
            for i in list(board.pieces(piece,colour)):
                v[i] = 1
            l.append(v)
    # Side to move - 1 = white, 0 = black
    active_color = 1 if fen.split()[1] == 'w' else 0
    l.append([active_color])
    
    # Castling rights - 0 = not allowed, 1 = allowed 
    castling_sides = ['K', 'Q', 'k', 'q']
    castling_rights_vec = [is_substring(side, fen.split()[2]) for side in castling_sides]
    l.append(castling_rights_vec)
    #Possible en-passant
    en_passant = np.zeros(64)
    if (fen.split()[3] != '-'):
      try:
        en_passant_target = chess.parse_square(fen.split()[3])
        en_passant[en_passant_target] = 1
      except:
        print(fen.split()[3])
        raise 'En passant square not found in chess.SQUARES'
    l.append(en_passant)
    #Halfmove clock
    halfmove_clock = int(fen.split()[4])
    l.append([halfmove_clock])
    #Fullmove number
    fullmove_number = int(fen.split()[5])
    l.append([fullmove_number])
    
    l = np.concatenate(l)
    # print('Vec:' + str(l))
    # print('Vec shape:' + str(l.shape))
    return l

def is_substring(string1, string2):
  return 1 if string1 in string2 else 0

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

def vec_to_board(feature_vector):
  """
  Convert the given feature vector representation of the board state to a chess.Board object.
  
  Args:
    feature_vector (np.array): The feature vector representation of the board state.
  
  Returns:
    chess.Board: The board state represented by the feature vector.
  """
  fen = vec_to_fen(feature_vector)
  return chess.Board(fen)

def action_index_to_move(action_index: int) -> chess.Move:
  """
  Convert the given action index to a UCI move string.
  
  Args:
    action_index (int): The action index in the vector of all possible actions of 8*8*(8*7 +8 + 9) = 4672 described as such:
    - for each square 8*8 = 64 there are (8*7 +8 + 9) planes of possible moves are
    - 7 squares in each direction {N, NE, E, SE, S, SW, W, NW} + 8 knight moves
    - 9 pawn promotions: 3 promotions (queen promotion is defulat) for 3 possible captures
  
  Returns:
    chess.Move: The chess move string corresponding to the given action index.
  """
  # Each 73 numbers, it repeats for a new square
  
  # 8*7+8+9 = 73
  directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
  tot_plane_of_execution = 73
  starting_square_index = action_index // tot_plane_of_execution
  plane_of_execution = action_index % tot_plane_of_execution
  starting_square = chess.SQUARE_NAMES[starting_square_index]
  chess.Square()
  move = chess.Move.null()
  promotion = None
  target_index = -1
  if plane_of_execution < 8*7:
    # Move in the direction of the plane
    direction = plane_of_execution // 7
    distance = plane_of_execution % 7 + 1
    if directions[direction] == 'N': # up
      target_index = starting_square_index + 8*distance
    if directions[direction] == 'NE': # up-rigth
      target_index = starting_square_index + 8*distance + distance
    if directions[direction] == 'NW': # up-left
      target_index = starting_square_index + 8*distance - distance
    if directions[direction] == 'E': # right
      target_index = starting_square_index + distance
    if directions[direction] == 'W': # left
      target_index = starting_square_index - distance
    if directions[direction] == 'S': # down
      target_index = starting_square_index - 8*distance
    if directions[direction] == 'SE': # down-right
      target_index = starting_square_index - 8*distance + distance
    if directions[direction] == 'SW': # down-left
      target_index = starting_square_index - 8*distance - distance
    promotion = chess.QUEEN
  elif plane_of_execution < 8*7 + 8:
    # Knight move
    knight_move_index = (plane_of_execution  - 8*7)
    rank_diff = 0
    file_diff = 0
    if knight_move_index == 0:
      rank_diff = 2
      file_diff = 1
    elif knight_move_index == 1:
      rank_diff = 1
      file_diff = 2
    elif knight_move_index == 2:
      rank_diff = -1
      file_diff = 2
    elif knight_move_index == 3:
      rank_diff = -2
      file_diff = 1
    elif knight_move_index == 4:
      rank_diff = -2
      file_diff = -1
    elif knight_move_index == 5:
      rank_diff = -1
      file_diff = -2
    elif knight_move_index == 6:
      rank_diff = 1
      file_diff = -2
    elif knight_move_index == 7:
      rank_diff = 2
      file_diff = -1
    target_index = starting_square_index + 8*rank_diff + file_diff
  elif plane_of_execution < 8*7 + 8 + 9:
    # Pawn promotion
    pawn_move = plane_of_execution - 8*7 - 8
    promotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promotion = promotions[pawn_move % 3]
    if pawn_move // 3 == 0: #forward one
      if starting_square_index >= 8 and starting_square_index < 16: #2nd rank
        target_index = starting_square_index - 8
      elif starting_square_index >= 48 and starting_square_index < 53: # 7th rank
        target_index = starting_square_index + 8
    if pawn_move // 3 == 1: # capture on the right
      if starting_square_index >= 8 and starting_square_index < 16: #2nd rank
        target_index = starting_square_index - 8 - 1
      elif starting_square_index >= 48 and starting_square_index < 53: # 7th rank
        target_index = starting_square_index + 8 + 1
    if pawn_move // 3 == 2: # capture on the left
      if starting_square_index >= 8 and starting_square_index < 16: #2nd rank
        target_index = starting_square_index - 8 + 1
      elif starting_square_index >= 48 and starting_square_index < 53: # 7th rank
        target_index = starting_square_index + 8 - 1
  if target_index < 64 and target_index >= 0:
    target_square = chess.SQUARE_NAMES[target_index]
    move = chess.Move(chess.parse_square(starting_square), chess.parse_square(target_square), promotion)
  return move

def is_legal_move(board: chess.Board, move: chess.Move):
  return True if move in board.legal_moves else False

def get_legal_move_if_possible(board: chess.Board, move: chess.Move) -> Tuple[chess.Move, bool]:
  if move.uci() == '0000':
    return (move, False)
  if is_legal_move(board, move):
    return (move, True)
  # try with no promotion
  move = chess.Move(move.from_square, move.to_square, promotion=None)
  return (move, True) if is_legal_move(board, move) else (chess.Move.null(), False)

# fen = '8/k6K/8/8/8/8/1p6/8 b - - 0 1'
# board = chess.Board(fen)
# print(board)
# for move in board.legal_moves:
#   print(move.uci())
  
# board.push_uci('b2b1')
# move = chess.Move(chess.parse_square('a7'),chess.parse_square('a8'), chess.QUEEN )
# move = chess.Move.null()
# move,_ = get_legal_move_if_possible(board, move)
# board.push(move)
# print(board)