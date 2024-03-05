import chess
from chess import Board, Square
import random
import pickle
import pygame
import qlearning_trainer
from chess_model import chess_utils
from typing import Tuple, List
import numpy as np

# Define colors
WHITE = (210, 210, 210)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)  # Color for highlighting valid moves
SQUARESIZE = 80  # Size of each chess square

# Load pieces images (replace with your image paths)
piece_images = {'p': pygame.image.load('data\\b_pawn.png'),
                'n': pygame.image.load('data\\b_knight.png'),
                'b': pygame.image.load('data\\b_bishop.png'),
                'r': pygame.image.load('data\\b_rook.png'),
                'q': pygame.image.load('data\\b_queen.png'),
                'k': pygame.image.load('data\\b_king.png'),
                'P': pygame.image.load('data\\w_pawn.png'),
                'N': pygame.image.load('data\\w_knight.png'),
                'B': pygame.image.load('data\\w_bishop.png'),
                'R': pygame.image.load('data\\w_rook.png'),
                'Q': pygame.image.load('data\\w_queen.png'),
                'K': pygame.image.load('data\\w_king.png')}

def choose_color_screen():
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Choose Your Color")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 36)
    text_white = font.render("Press W to play as White", True, (255, 255, 255))
    text_black = font.render("Press B to play as Black", True, (255, 255, 255))

    screen.fill((0, 0, 0))
    screen.blit(text_white, (50, 50))
    screen.blit(text_black, (50, 100))
    pygame.display.flip()

    color_chosen = False
    player_color = None

    while not color_chosen:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    player_color = chess.WHITE
                    color_chosen = True
                elif event.key == pygame.K_b:
                    player_color = chess.BLACK
                    color_chosen = True

        clock.tick(30)

    pygame.quit()
    return player_color

def draw_board(screen, board, player_color):
  for row in range(8):
    for col in range(8):
      if player_color == chess.WHITE:
        square = chess.square(col, 7 - row)
      else:
        square = chess.square(7 - col, row)
            
      if player_color == chess.WHITE:
        square_name = chess.square_name(square)
      else:
        square_name = chess.square_name(chess.square(7 - chess.square_file(square), 7 - chess.square_rank(square)))

      color = WHITE if (row + col) % 2 == 0 else GREY  
      pygame.draw.rect(screen, color, pygame.Rect(col * SQUARESIZE, row * SQUARESIZE, SQUARESIZE, SQUARESIZE))
      piece = board.piece_at(square)

      if piece is not None:
        piece_image = piece_images[piece.symbol()]
        scaled_piece_image = pygame.transform.scale(piece_image, (SQUARESIZE, SQUARESIZE))
        screen.blit(scaled_piece_image, (col * SQUARESIZE, row * SQUARESIZE))

def draw_valid_moves(screen, board: Board, selected_square: Square, player_color: chess.Color):
  if selected_square is not None:
    for move in board.legal_moves:
      if move.from_square == selected_square:
        # print(move.to_square)
        to_col, to_row = convert_square_to_coord(move.to_square, player_color)
        pygame.draw.rect(screen, GREEN, pygame.Rect(to_col * SQUARESIZE, to_row * SQUARESIZE, SQUARESIZE, SQUARESIZE), width=3)


def convert_square_to_coord(square, player_color):
  if player_color == chess.WHITE:
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
  else:
    row = chess.square_rank(square)
    col = 7 - chess.square_file(square)
  return (col, row)

def convert_click_to_square(x, y, player_color):
  if player_color == chess.WHITE:
    index = 63 - ( x * 8 + (7 - y) ) # Mirrored from y
  else:
    index = 63 - ( (7 - x) * 8 + y)
  square = chess.parse_square(chess.SQUARE_NAMES[index])
  return chess.parse_square(chess.SQUARE_NAMES[index])


def select_square(x, y, player_color):
  return convert_click_to_square(x// SQUARESIZE, y// SQUARESIZE, player_color)

def do_ai_move(board: Board, q_table: dict):
  # AI's move
  state = chess_utils.state_to_string(board)
  if state not in q_table:
    print("Encountered unknown state. Playing random move.")
    move = random.choice(list(board.legal_moves))
    move = move.uci()
  else:
    # Choose the action with the highest Q-value
    action = np.argmax(q_table[state])
    move = chess_utils.action_to_uci(action, board)
  print(f"AI move: {move}")
  board.push_uci(move)

def play_against_ai(board, q_table):
  player_color = choose_color_screen()
  pygame.init()
  screen = pygame.display.set_mode((8 * SQUARESIZE, 8 * SQUARESIZE))
  pygame.display.set_caption("Chess")
  clock = pygame.time.Clock()
        
  selected_square = None
  selected_piece_moves = []

  game_over = False
  if player_color == chess.BLACK:
    do_ai_move(board, q_table)
  while not game_over:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        game_over = True

      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
          pygame.quit()
          return
      elif event.type == pygame.MOUSEBUTTONDOWN:

        row, col = pygame.mouse.get_pos()
        clicked_square = select_square(col, row, player_color)

        if selected_square is None:
          if board.piece_at(clicked_square) is not None:
            selected_square = clicked_square
            selected_piece_moves = [move for move in board.legal_moves if move.from_square == selected_square]

        else:
          if clicked_square == selected_square:
            selected_square = None
            selected_piece_moves = []
          else:
            if clicked_square in [move.to_square for move in selected_piece_moves]:
              board.push([move for move in selected_piece_moves if move.to_square == clicked_square][0])
              if not board.is_game_over():
                do_ai_move(board, q_table)
            selected_square = None
            selected_piece_moves = []

      elif event.type == pygame.MOUSEBUTTONUP:
        
        row, col = pygame.mouse.get_pos()
        released_square = select_square(col, row, player_color)

        if selected_square is not None and released_square != selected_square:
          if released_square in [move.to_square for move in selected_piece_moves]:
            board.push([move for move in selected_piece_moves if move.to_square == released_square][0])
            if not board.is_game_over():
              do_ai_move(board, q_table)
          selected_square = None
          selected_piece_moves = []

    draw_board(screen, board, player_color)
    draw_valid_moves(screen, board, selected_square, player_color)
    pygame.display.flip()
    clock.tick(60)  # Limit framerate to 60fps
    # Check for game over after each move
    if board.is_game_over():
      game_over = True

  pygame.quit()
  print(f"Winner: {board.outcome()}")


if __name__ == "__main__":
  # Load Q-table (replace with your Q-table filename)
  q_table = qlearning_trainer.load_q_table("chess_model\\q_table.dat")

  # Play against the AI
  board = chess.Board()
  play_against_ai(board, q_table)