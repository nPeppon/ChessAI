import chess
from chessai.chess_helper import chess_utils
import random

class BaseBot:
    def __init__(self):
        self.name = "Base Bot: Random Moves"

    def choose_move(self, board: chess.Board) -> str:
        state = chess_utils.state_to_string(board)
        move = random.choice(list(board.legal_moves))
        move = move.uci()
        print(f"AI move: {move}")
        board.push_uci(move)
        return move