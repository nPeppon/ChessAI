import chess
import numpy as np
import tensorflow as tf
from chessai.chess_helper import chess_utils
import os
from typing import List

# Define Policy Network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(839,))
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(inputs)
        return self.dense3(x)
    
    @classmethod
    def create(cls, num_actions):
        # Create an instance of the PolicyNetwork class
        policy_network = cls(num_actions)  # Pass num_actions to the constructor
        return policy_network

# Define Value Network
class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(839,))
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation=None)  # Linear activation for value estimation

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# PPO Update Function
def ppo_update(policy_network, value_network, optimizer, states: List[np.array], actions: List[int], advantages, old_probs: List[float], rewards: List[float], clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    with tf.GradientTape() as tape:
        new_probs = tf.convert_to_tensor([policy_network(np.expand_dims(state_vec, axis=0)).numpy().squeeze()[action] for state_vec,action in zip(states, actions)])
        old_probs = tf.convert_to_tensor(old_probs)
        ratios = tf.exp(tf.math.log(new_probs + 1e-10) - tf.math.log(old_probs + 1e-10))
        clipped_ratios = tf.clip_by_value(ratios, 1 - clip_ratio, 1 + clip_ratio)
        surrogates = tf.minimum(ratios * advantages, clipped_ratios * advantages)
        policy_loss = -tf.reduce_mean(surrogates)
        
        value_estimates = tf.convert_to_tensor([value_network(np.expand_dims(state, axis=0)) for state in states])
        value_loss = tf.reduce_mean(tf.square(value_estimates - rewards))

        entropy_loss = -tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-10))

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    gradients = tape.gradient(total_loss, policy_network.trainable_variables + value_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables + value_network.trainable_variables))

# Function to collect trajectory by self-playing
def collect_trajectory(policy_network, board, max_moves=200):
    states = []
    actions = []
    rewards = []
    old_probs = []

    for _ in range(max_moves):
        # Convert board to input state
        state_vec = chess_utils.board_to_vec(board)
        # Sample action from policy network
        action_probs = policy_network(np.expand_dims(state_vec, axis=0)).numpy().squeeze()
        
        valid_action = False
        while not valid_action:
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            move, valid_action = chess_utils.get_legal_move_if_possible(board, chess_utils.action_index_to_move(action))

        # print('Move found: ' + str(move))
        # Store old probabilities
        old_probs.append(action_probs[action])

        # Execute action
        move = chess.Move.from_uci(move.uci())
        board.push(move)

        # Append data to trajectory
        states.append(state_vec)
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


if __name__ == "__main__":
    # Initialize policy and value networks
    policy_network = PolicyNetwork(num_actions=chess_utils.CHESS_NUM_ACTIONS)
    value_network = ValueNetwork()

    # Check if the files exist
    policy_weights_file = "data\\ppo_model\\policy_network.weights.h5"
    value_weights_file = "data\\ppo_model\\value_network.weights.h5"

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
    num_episodes = 10000
    for episode in range(num_episodes):
        board = chess.Board()  # Initialize new game
        states, actions, rewards, old_probs = collect_trajectory(policy_network, board)
        advantages = np.zeros_like(rewards, dtype=np.float32)  # Placeholder for advantages (for now)
        ppo_update(policy_network, value_network, optimizer, states, actions, advantages, old_probs, rewards)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {np.sum(rewards)}")
        if (episode + 1) % 100 == 0:
            policy_network.save_weights(policy_weights_file)
            value_network.save_weights(value_weights_file)
            

    # Save the weights of the policy and value networks
    policy_network.save_weights(policy_weights_file)
    value_network.save_weights(value_weights_file)
