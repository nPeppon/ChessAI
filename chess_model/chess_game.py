import chess
import random


def make_random_move(board):
  # Choose a legal move randomly
  legal_moves = list(board.legal_moves)
  return random.choice(legal_moves)

def self_play(board):
  while not board.is_game_over():
    # Alternate between players (Black and White)
    player = board.turn

    # Make a move for the current player
    move = make_random_move(board)
    board.push(move)

    # Print the move and updated board
    print(f"Player {player}: {move.uci()}")
    # print(board)

# Create a chessboard
board = chess.Board()
# print(board)
# Start self-play
self_play(board)

# Print the final board and winner
print(f"Winner: {board.outcome()}")
