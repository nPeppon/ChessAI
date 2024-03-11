import chess
from chess import Board, Square
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from typing import Tuple, List
from chessai.models import base_bot, qlearning_bot, ppo_bot
from chessai.gui.draw_gui import draw_game, select_square, select_bot, choose_color_screen, create_screen, select_promotion_piece

BOTS = [base_bot.BaseBot, qlearning_bot.QlearningBot, ppo_bot.PpoBot]

def select_move(available_moves: List[chess.Move], screen) -> chess.Move:
  move = chess.Move.null()
  if len(available_moves) > 1:
    promotions_type = [move.promotion for move in available_moves if move.promotion is not None]
    piece_type = select_promotion_piece(screen, promotions_type)
    if piece_type is not None:
      move = [move for move in available_moves if move.promotion == piece_type][0]
  elif len(available_moves) == 1:
    move = available_moves[0]
  return move

def play_against_ai(board):
  bot = select_bot(BOTS)
  player_color = choose_color_screen()
  screen = create_screen()
  clock = pygame.time.Clock()
      
  selected_square = None
  selected_piece_moves = []
  last_uci_move = None
  game_over = False
  if player_color == chess.BLACK and board.turn == chess.WHITE or player_color == chess.WHITE and board.turn == chess.BLACK:
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
              possible_moves = [move for move in selected_piece_moves if move.to_square == clicked_square]
              move = select_move(possible_moves, screen)
              if move.uci() != '0000':
                print('My move: ' + move.uci())
                board.push(move)
                if not board.is_game_over():
                  last_uci_move = bot.choose_move(board)
            selected_square = None
            selected_piece_moves = []

      elif event.type == pygame.MOUSEBUTTONUP:
        
        row, col = pygame.mouse.get_pos()
        released_square = select_square(col, row, player_color)

        if selected_square is not None and released_square != selected_square:
          if released_square in [move.to_square for move in selected_piece_moves]:
            possible_moves = [move for move in selected_piece_moves if move.to_square == released_square]
            move = select_move(possible_moves, screen)
            if move.uci() != '0000':
              print('My move: ' + move.uci())
              board.push(move)
              if not board.is_game_over():
                last_uci_move = bot.choose_move(board)
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
      if event.type == KEYDOWN and event.key == K_r:
          board.reset()
          play_against_ai(board)
          not_quit = False
      if event.type == KEYDOWN and event.key == K_t:
        not_quit = False
        # TODO: add method for each bot that uses this match to train the bot itself

  pygame.quit()
  print(f"Winner: {board.outcome()}")
  print(f"Winner: {board.result()}")


if __name__ == "__main__":
  # Play against the AI
  # board = chess.Board(fen = '8/k6K/8/8/8/8/1p6/8 b - - 0 1')
  board = chess.Board()
  play_against_ai(board)
  