import chess
import random
import pickle
import numpy as np
from chess_model import chess_utils

def eps_greedy_action(q_table, state, eps):
  # Implement exploration-exploitation strategy using epsilon-greedy
  if random.random() < eps:
    return random.randint(0, len(q_table[state]) - 1)  # Explore random action
  else:
    return np.argmax(q_table[state])  # Exploit: choose action with highest Q-value

def load_q_table(filename):
  # Load Q-table from a file if it exists
  try:
    with open(filename, 'rb') as f:
      return pickle.load(f)
  except FileNotFoundError:
    return {}

def save_q_table(q_table, filename):
  # Save Q-table to a file
  with open(filename, 'wb') as f:
    pickle.dump(q_table, f)

def self_play_qlearning(iBoard, iQ_table, iLearning_rate, iDiscount_factor, iEpsilon):
  while not iBoard.is_game_over():
    state = chess_utils.state_to_string(iBoard)

    # Load Q-table if not already loaded
    if state not in iQ_table:
      iQ_table[state] = [0 for _ in range(len(list(iBoard.legal_moves)))]  # Initialize Q-values for all actions

    action = eps_greedy_action(iQ_table, state, iEpsilon)  # Adjust epsilon for exploration-exploitation balance
    move = chess_utils.action_to_uci(action, iBoard)
    iBoard.push_uci(move)

    next_state = chess_utils.state_to_string(iBoard)
    reward = chess_utils.evaluate_board(iBoard)  # Update reward function based on your needs

    # Update Q-table using the Bellman equation
    max_q_next = max(iQ_table[next_state]) if next_state in iQ_table else 0
    iQ_table[state][action] += iLearning_rate * (reward + iDiscount_factor * max_q_next - iQ_table[state][action])
    iQ_table[state][action] += iLearning_rate * (reward + iDiscount_factor * max_q_next - iQ_table[state][action])

  return iQ_table



if __name__ == "__main__":  
  # Initialize learning parameters
  learning_rate = 0.1
  discount_factor = 0.9
  espilon = 0.2

  # Load or initialize Q-table
  q_table = load_q_table("chess_model\\q_table.dat")
  # print(q_table)

  # Run multiple games in a loop
  total_games = 10  # Adjust the number of games for extended training
  completed_games = 0
  for game_number in range(total_games):
    board = chess.Board()
    q_table = self_play_qlearning(board, q_table, learning_rate, discount_factor, espilon)
    completed_games += 1

    # Print progress every 5% of games completed
    if total_games //40 > 0 and completed_games % (total_games // 40) == 0:
      progress = (completed_games / total_games) * 100
      print(f"Training progress: {progress}% completed")
      save_q_table(q_table, "chess_model\\q_table.dat")

  # Save Q-table only once at the end
  save_q_table(q_table, "chess_model\\q_table.dat")

  # print(q_table)
  # Print final message
  print("Training complete. You can play against the AI or continue training.")
