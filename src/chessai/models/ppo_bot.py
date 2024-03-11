import chess
from .base_bot import BaseBot
from chessai.chess_helper import chess_utils
from chessai.models.ppo_model import PolicyNetwork
import numpy as np

class PpoBot(BaseBot):
    
    name = "PPO Bot"
    def __init__(self, filename:str = 'policy_network.weights.h5'):
        self.name = "PPO Bot"
        # Load the weights of the policy network
        self.policy_network = PolicyNetwork.create(chess_utils.CHESS_NUM_ACTIONS)  # Replace with your function to create the policy network
        self.policy_network.load_weights('data\\ppo_model\\' + filename)

    def choose_move(self, board: chess.Board) -> str:
        # AI's move
        # Convert the board state to a suitable format for the policy network
        state_vec = chess_utils.board_to_vec(board)
        # Compute the probabilities of the actions
        action_probs = self.policy_network(np.expand_dims(state_vec, axis=0)).numpy().squeeze()

        # Choose the valid action with the highest probability
        valid_action = False
        move = None
        while not valid_action:
            action = np.argmax(action_probs)
            move, valid_action = chess_utils.get_legal_move_if_possible(board, chess_utils.action_index_to_move(action))
            if not valid_action:
                action_probs[action] = 0

        print(f"AI move: {move}")
        board.push_uci(move.uci())
        return move.uci()