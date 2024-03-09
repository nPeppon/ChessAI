import chess
from chess import Board, Square
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from typing import Tuple, List
from chessai.models import base_bot, qlearning_bot
from chessai.gui.draw_gui import draw_game, select_square, select_bot, choose_color_screen, create_screen

BOTS = [base_bot.BaseBot(), qlearning_bot.QlearningBot()]

def play_against_ai(board):
  bot = select_bot(BOTS)
  player_color = choose_color_screen()
  screen = create_screen()
  clock = pygame.time.Clock()
      
  selected_square = None
  selected_piece_moves = []
  last_uci_move = None
  game_over = False
  if player_color == chess.BLACK:
    last_uci_move = bot.choose_move(board)
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
                last_uci_move = bot.choose_move(board)
            selected_square = None
            selected_piece_moves = []

      elif event.type == pygame.MOUSEBUTTONUP:
        
        row, col = pygame.mouse.get_pos()
        released_square = select_square(col, row, player_color)

        if selected_square is not None and released_square != selected_square:
          if released_square in [move.to_square for move in selected_piece_moves]:
            board.push([move for move in selected_piece_moves if move.to_square == released_square][0])
            if not board.is_game_over():
              do_ai_move(board,  )
          selected_square = None
          selected_piece_moves = []

    draw_game(screen, board, player_color, selected_square, last_uci_move)
    clock.tick(60)  # Limit framerate to 60fps
    # Check for game over after each move
    if board.is_game_over():
      game_over = True

  not_quit = True
  while game_over and not_quit:
    # Draw the board and the outcome bar
    draw_game(screen, board, player_color, selected_square, last_uci_move)
    clock.tick(60)  # Limit framerate to 60fps
    for event in pygame.event.get():
      if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
        not_quit = False

  pygame.quit()
  print(f"Winner: {board.outcome()}")
  print(f"Winner: {board.result()}")


if __name__ == "__main__":
  # Play against the AI
  board = chess.Board()
  play_against_ai(board)
  