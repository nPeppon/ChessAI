import chess
import numpy as np
import tensorflow as tf

import sys
import os
# append the path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from chess_model import chess_utils
# import chess_utils

# Define Policy Network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Define Value Network
class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation=None)  # Linear activation for value estimation

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# PPO Update Function
def ppo_update(policy_network, value_network, optimizer, states, actions, advantages, old_probs, rewards, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    with tf.GradientTape() as tape:
        new_probs = policy_network(states)
        action_masks = tf.one_hot(actions, depth=num_actions)
        ratio = tf.exp(tf.reduce_sum(tf.math.log(new_probs + 1e-10) * action_masks, axis=1) - tf.math.log(old_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
        surrogate = tf.minimum(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -tf.reduce_mean(surrogate)
        
        value_estimate = value_network(states)
        value_loss = tf.reduce_mean(tf.square(value_estimate - rewards))

        entropy_loss = -tf.reduce_mean(tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1))

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    gradients = tape.gradient(total_loss, policy_network.trainable_variables + value_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables + value_network.trainable_variables))



def encode_fen(fen:str):
    """
    Function to encode the given FEN string into a list of piece indices.
    :param fen: A string representing a FEN (Forsyth-Edwards Notation) chess position such as 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    :return: A list of piece indices based on the FEN string.
    """
    pieces = fen.split()[0]
    print(pieces)
    encoding_map = {'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5,
                      'P': 6, 'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, '.': 12, ' ': 13, '-': 14, '/': 15}	
    encoding = [encoding_map[piece] for piece in pieces]
    print(encoding)
    return encoding

# Function to collect trajectory by self-playing
def collect_trajectory(policy_network, board, max_moves=100):
    states = []
    actions = []
    rewards = []
    old_probs = []

    for _ in range(max_moves):
        # Convert board to input state
        state = chess_utils.state_to_string(board)
        print(fenToVec(state))
        # Sample action from policy network
        action_probs = policy_network(np.expand_dims(encode_fen(state), axis=0)).numpy().squeeze()
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # Store old probabilities
        old_probs.append(action_probs[action])

        # Execute action
        print('Action:', action)
        move = chess.Move.from_uci(chess_utils.action_to_uci(action, board))
        board.push(move)

        # Append data to trajectory
        states.append(state)
        actions.append(action)

        # Check for game end
        if board.is_game_over():
            break

    rewards = get_rewards(board, states)
    return states, actions, rewards, old_probs

# Helper function to get reward from the game outcome
def get_rewards(board, states):
    rewards = []
    winner = board.result()
    for i in range(len(states)):
        reward = 0.5
        if winner == '1-0': # White wins
            reward = 1 if i % 2 == 0 else -1
        elif winner == '0-1': # Black wins
            reward = -1 if i % 2 == 0 else 1
        rewards.append(reward)
    return rewards

# Initialize policy and value networks
policy_network = PolicyNetwork(num_actions=64)
value_network = ValueNetwork()

# Check if the files exist
policy_weights_file = "PPO_model\\policy_network_weights.h5"
value_weights_file = "PPO_model\\value_network_weights.h5"

if os.path.exists(policy_weights_file) and os.path.exists(value_weights_file):
    # Load the weights only if both files exist
    policy_network.load_weights(policy_weights_file)
    value_network.load_weights(value_weights_file)
    print("Weights loaded successfully.")
else:
    print("Weights files not found. Please train the model first.")

# Initialize optimizer
optimizer = tf.keras.optimizers.Adam()

# Main training loop
num_episodes = 5
for episode in range(num_episodes):
    board = chess.Board()  # Initialize new game
    states, actions, rewards, old_probs = collect_trajectory(policy_network, board)
    advantages = np.zeros_like(rewards, dtype=np.float32)  # Placeholder for advantages (for now)
    ppo_update(policy_network, value_network, optimizer, states, actions, advantages, old_probs, rewards)
    print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {np.sum(rewards)}")

# Save the weights of the policy and value networks
policy_network.save_weights(policy_weights_file)
value_network.save_weights(value_weights_file)
