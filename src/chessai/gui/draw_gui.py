import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import chess
from chess import Board, Square
from chessai.models import base_bot, qlearning_bot
from typing import List, Type

# Define colors
WHITE = (210, 210, 210)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)  # Color for highlighting valid moves
VIBRANT_BLUE = (0, 0, 255) # Color for highlighting latest move
SQUARESIZE = 80  # Size of each chess square
OUTCOME_BAR_HEIGHT = 50
LIGHT_VIOLET = (204, 153, 255)
VIOLET = (138, 43, 226)

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

def create_screen():
  pygame.init()
  # Define a new constant for the height of the outcome bar
  screen = pygame.display.set_mode((8 * SQUARESIZE, 8 * SQUARESIZE + OUTCOME_BAR_HEIGHT))
  pygame.display.set_caption("Chess")
  return screen

def select_bot(bots: List[Type[base_bot.BaseBot]]) -> base_bot.BaseBot:
  """
  Selects a bot based on user input using Pygame events. Returns the selected bot.
  """
  pygame.init()
  # Set up some constants
  WIDTH, HEIGHT = 640, 480
  FONT_SIZE = 32

  screen = pygame.display.set_mode((WIDTH, HEIGHT))
  font = pygame.font.Font(None, FONT_SIZE)

  running = True
  while running:
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              running = False
          elif event.type == pygame.KEYDOWN:
              if pygame.K_1 <= event.key <= pygame.K_9:
                  index = event.key - pygame.K_1
                  if index < len(bots):
                      return bots[index]()

      screen.fill((0, 0, 0))
      for i, bot in enumerate(bots):
          text = font.render(f"{i+1}. {bot.name}", True, (255, 255, 255))  # replace bot.name with how you get the bot's name
          screen.blit(text, (10, i * FONT_SIZE))

      pygame.display.flip()

  pygame.quit()

def draw_text(text, color, x, y):
  text_surface = font.render(text, True, color)
  text_rect = text_surface.get_rect()
  text_rect.center = (x, y)
  screen.blit(text_surface, text_rect)

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
        
def draw_last_move(screen, board: Board, latest_uci_move: str, player_color: chess.Color):
  if latest_uci_move is not None and len(latest_uci_move) == 4:
    squares = [chess.parse_square(latest_uci_move[:2]), chess.parse_square(latest_uci_move[2:])]
    for square in squares:
      to_col, to_row = convert_square_to_coord(square, player_color)
      pygame.draw.rect(screen, VIBRANT_BLUE, pygame.Rect(to_col * SQUARESIZE, to_row * SQUARESIZE, SQUARESIZE, SQUARESIZE), width=3)

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

def draw_outcome(screen, board: Board):
  outcome = board.outcome()
  if outcome is not None:
    outcome_str = str(outcome)
    start = outcome_str.find("Termination.") + len("Termination.")
    end = outcome_str.find(":", start)
    termination = outcome_str[start:end]
    result = board.result()
    # Render the outcome and result to the outcome bar
    font = pygame.font.Font(None, 32)
    outcome_text = font.render(f"Outcome: {termination}", True, (255, 255, 255))
    result_text = font.render(f"Result: {result}", True, (255, 255, 255))
    # Create a new surface for the outcome bar
    outcome_bar = pygame.Surface((SQUARESIZE * 8, OUTCOME_BAR_HEIGHT))
    outcome_bar.fill((0, 0, 0))  # Clear the outcome bar
    outcome_bar.blit(outcome_text, (10, 10))
    outcome_bar.blit(result_text, (10, 30))
    screen.blit(outcome_bar, (0, 8*SQUARESIZE))

def select_promotion_piece(screen, promotions_types: List[chess.PieceType]):
    # Calculate the size of the overlay
    overlay_width = SQUARESIZE * len(promotions_types)
    overlay_height = SQUARESIZE
    # Create an overlay
    overlay = pygame.Surface((overlay_width, overlay_height))
    overlay.fill(LIGHT_VIOLET)
    # Draw a border around the overlay
    border_color = (138, 43, 226)  # RGB color for violet
    border_thickness = 5  # Thickness of the border in pixels
    pygame.draw.rect(overlay, border_color, overlay.get_rect(), border_thickness)
    # overlay.set_alpha(128)  # Make the overlay semi-transparent
    # Position the overlay in the center of the screen
    overlay_pos = ((screen.get_width() - overlay_width) // 2, (screen.get_height() - overlay_height) // 2)
    screen.blit(overlay, overlay_pos)
    # Fill the overlay with a grey color
    offset = overlay_pos[0]
    for piece_type in promotions_types:
      piece_image = piece_images[chess.piece_symbol(piece_type)]
      scaled_piece_image = pygame.transform.scale(piece_image, (SQUARESIZE, SQUARESIZE))
      screen.blit(scaled_piece_image, (offset, overlay_pos[1]))
      offset += SQUARESIZE

    # Update the display
    pygame.display.flip()

    # Wait for the player to click on a piece
    while True:
      for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
          x, y = pygame.mouse.get_pos()
          # Adjust the mouse position to account for the position of the overlay
          x -= overlay_pos[0]
          y -= overlay_pos[1]
          # Check if the click was within the bounds of the overlay
          if 0 <= x < overlay_width and 0 <= y < overlay_height:
              # Calculate the index of the clicked square
              index = x // SQUARESIZE
              return promotions_types[index]
          else:
            return None
      pygame.time.wait(10)

    # If the player didn't click on any piece, return None
    return None
  

def draw_game(screen, board: Board, player_color: chess.Color, selected_square: Square, latest_uci_move: str):
  draw_board(screen, board, player_color)
  draw_last_move(screen, board, latest_uci_move, player_color)
  draw_valid_moves(screen, board, selected_square, player_color)
  draw_outcome(screen, board)
  pygame.display.flip()