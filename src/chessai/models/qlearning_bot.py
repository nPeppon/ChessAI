import pickle
import chess
import numpy as np
from .base_bot import BaseBot
from chessai.chess_helper import chess_utils
import random


class QlearningBot(BaseBot):
    name = "Q-learning Bot"

    def __init__(self, filename: str = 'q_table_simple.dat'):
        self.name = "Q-learning Bot"
        self.q_table = load_q_table("..\\..\\..\\data\\" + filename)

    def choose_move(self, board: chess.Board) -> str:
        # AI's move
        state = chess_utils.state_to_string(board)
        if state not in self.q_table:
            print("Encountered unknown state. Playing random move.")
            move = random.choice(list(board.legal_moves))
            move = move.uci()
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(self.q_table[state])
            move = chess_utils.action_to_uci(action, board)
        print(f"AI move: {move}")
        board.push_uci(move)
        return move


def load_q_table(filename):
    # Load Q-table from a file if it exists
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
