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

def draw_board(screen, board):
  for row in range(8):
    for col in range(8):
      color = WHITE if (row + col) % 2 == 0 else GREY
      pygame.draw.rect(screen, color, pygame.Rect(col * SQUARESIZE, row * SQUARESIZE, SQUARESIZE, SQUARESIZE))
      piece = board.piece_at(chess.square(col, row))

      try:
        if piece is not None:
          piece_image = piece_images[piece.symbol()]
          # Scale up the size of the piece image
          scaled_piece_image = pygame.transform.scale(piece_image, (SQUARESIZE, SQUARESIZE))
          screen.blit(scaled_piece_image, (col * SQUARESIZE, row * SQUARESIZE))
      except FileNotFoundError:
        print(f"Error loading image for piece: {piece.symbol()}. File not found at {piece_images[piece.symbol()]}")


def draw_valid_moves(screen, board: Board, selected_square: Square):
  if selected_square is not None:
    for move in board.legal_moves:
      if move.from_square == selected_square:
        # print(move.to_square)
        to_row = move.to_square // 8  # Calculate row index of the destination square
        to_col = move.to_square % 8  # Calculate column index of the destination square
        pygame.draw.rect(screen, GREEN, pygame.Rect(to_col * SQUARESIZE, to_row * SQUARESIZE, SQUARESIZE, SQUARESIZE), width=3)

def select_square(positions: [Square, Tuple[int, int]], x: int, y: int) -> Square:
  # Convert mouse click coordinates to chessboard square
  row = y // SQUARESIZE
  col = x // SQUARESIZE
  # print('Selecting square:' + str(col) + ', ' + str(row))
  return chess.square(row,col)
  # return positions.get(chess.square(col, row))

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
  pygame.init()
  screen = pygame.display.set_mode((8 * SQUARESIZE, 8 * SQUARESIZE))
  pygame.display.set_caption("Chess")
  clock = pygame.time.Clock()
        
  selected_square = None
  selected_piece_moves = []
  positions = {}
  for i in range(8):
    for j in range(8):
      positions[chess.square(j, i)] = (j * SQUARESIZE, i * SQUARESIZE)

  game_over = False

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
        clicked_square = select_square(positions, col, row)

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
        released_square = select_square(positions, col, row)

        if selected_square is not None and released_square != selected_square:
          if released_square in [move.to_square for move in selected_piece_moves]:
            board.push([move for move in selected_piece_moves if move.to_square == released_square][0])
            if not board.is_game_over():
              do_ai_move(board, q_table)
          selected_square = None
          selected_piece_moves = []

    draw_board(screen, board)
    draw_valid_moves(screen, board, selected_square)
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