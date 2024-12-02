import chess
import numpy as np
import tensorflow as tf
from chessai.chess_helper import chess_utils
import os
from typing import List


# Define hyperparameters
learning_rate = 0.0003
gamma = 0.99
clip_epsilon = 0.2
batch_size = 64
num_epochs = 10
lam = 0.95


class PolicyNetwork(tf.keras.Model):
    '''
    Policy Network class
    '''

    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(839,))
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
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


class ValueNetwork(tf.keras.Model):
    '''
    Value Network class
    '''

    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.dense1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(839,))
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')  # Linear activation for value estimation

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


def ppo_update(policy_network, value_network, optimizer, states: List[np.array], actions: List[int], advantages, old_probs: List[float], rewards: List[float], clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
    '''
    PPO update function

    args:
        policy_network: Policy network model
        value_network: Value network model
        optimizer: Optimizer
        states: List of states
        actions: List of actions
        advantages: List of advantages
        old_probs: List of old probabilities
        rewards: List of rewards
        clip_ratio: Clipping ratio
        value_coef: Value coefficient
        entropy_coef: Entropy coefficient
    '''
    with tf.GradientTape() as tape:
        # Compute policy loss
        action_probs = policy_network(np.array(states))
        action_probs = tf.gather_nd(action_probs, list(zip(range(len(actions)), actions)))
        ratios = tf.exp(tf.math.log(action_probs) - tf.math.log(old_probs))
        clipped_ratios = tf.clip_by_value(ratios, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))

        value_estimates = tf.convert_to_tensor([value_network(np.expand_dims(state, axis=0)) for state in states])
        value_loss = tf.reduce_mean(tf.square(value_estimates - rewards))

        entropy_loss = -tf.reduce_mean(action_probs * tf.math.log(action_probs + 1e-10))

        # Normalize the loss components
        policy_loss = policy_loss / tf.reduce_mean(tf.abs(policy_loss) + 1e-10)
        value_loss = value_loss / tf.reduce_mean(tf.abs(value_loss) + 1e-10)
        entropy_loss = entropy_loss / tf.reduce_mean(tf.abs(entropy_loss) + 1e-10)

        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

        # Add penalty for invalid actions
        invalid_action_penalty = compute_invalid_action_penalty(policy_network, states)
        invalid_action_penalty_weight = 0.1
        total_loss += invalid_action_penalty_weight * invalid_action_penalty

        normalized_rewards = normalize_rewards(rewards)
        reward_scale = tf.reduce_mean(tf.abs(normalized_rewards))
        reward_scale = tf.cast(reward_scale, total_loss.dtype)  # Ensure the same type
        print(f"Total Loss: {total_loss}, Reward Scale: {
              reward_scale}, invalid_action_penalty = {invalid_action_penalty}")
        reward_weight = 5
        total_loss = tf.maximum(total_loss - reward_weight * reward_scale, 0)  # Ensure non-negative loss

    gradients = tape.gradient(total_loss, policy_network.trainable_variables + value_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables + value_network.trainable_variables))
    return total_loss


def compute_invalid_action_penalty(policy_network, states):
    '''
    Compute penalty for invalid actions
    '''
    penalty = 0
    for state in states:
        action_probs = policy_network(np.expand_dims(state, axis=0)).numpy().squeeze()
        valid_action_indices = [chess_utils.move_to_action_index(move) for move in board.legal_moves]
        invalid_action_indices = [i for i in range(len(action_probs)) if i not in valid_action_indices]
        penalty += np.sum(action_probs[invalid_action_indices])
    return penalty


def collect_trajectory(policy_network, board, max_moves=200):
    '''
    Collect trajectory by self-playing
    '''
    states = []
    actions = []
    rewards = []
    old_probs = []

    for i in range(max_moves):
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

        # if i == max_moves - 1:
        #     print('Max moves reached')
    rewards = get_rewards(board, states)
    return states, actions, rewards, old_probs


def get_rewards(board, states):
    '''
    Get rewards for the trajectory, from the game outcome
    '''
    rewards = []
    winner = board.result()
    outcome = board.outcome()
    match_length = len(states)

    rewards = [0] * match_length
    if outcome is not None:
        rewards = []
        match_length_scale = 200/len(states)
        positive_reward = 1000 * match_length_scale
        negative_reward = -500 * match_length_scale
        for i in range(match_length):
            reward = 0
            if outcome in [chess.Termination.STALEMATE, chess.Termination.INSUFFICIENT_MATERIAL]:  # Draw
                reward = positive_reward / 5
            elif winner == '1-0':  # White wins
                reward = positive_reward if i % 2 == 0 else negative_reward
            elif winner == '0-1':  # Black wins
                reward = negative_reward if i % 2 == 0 else positive_reward
            rewards.append(reward)

    # Apply discount factor to future rewards
    # gamma = 0.99
    # discounted_rewards = []
    # cumulative_reward = 0
    # for reward in reversed(rewards):
    #     cumulative_reward = reward + gamma * cumulative_reward
    #     discounted_rewards.insert(0, cumulative_reward)

    return rewards


def normalize_rewards(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-10)
    return normalized_rewards


def compute_returns(rewards, gamma):
    returns = []
    discounted_sum = 0
    for reward in reversed(rewards):
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return returns


def compute_value_estimates(states, value_network):
    value_estimates = [value_network(np.expand_dims(state, axis=0)).numpy().squeeze() for state in states]
    return value_estimates


def compute_advantages_and_returns(rewards, value_estimates, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * value_estimates[t + 1] - \
            value_estimates[t] if t < len(rewards) - 1 else rewards[t] - value_estimates[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + value for adv, value in zip(advantages, value_estimates)]
    return advantages, returns


if __name__ == "__main__":
    # Initialize policy and value networks
    policy_network = PolicyNetwork(num_actions=chess_utils.CHESS_NUM_ACTIONS)
    value_network = ValueNetwork()

    # Check if the files exist
    policy_weights_file = "data\\ppo_model\\policy_network.weights.h5"
    value_weights_file = "data\\ppo_model\\value_network.weights.h5"
    loss_file = "data\\ppo_model\\loss.csv"
    if os.path.exists(policy_weights_file) and os.path.exists(value_weights_file):
        # Load the weights only if both files exist
        policy_network.load_weights(policy_weights_file)
        value_network.load_weights(value_weights_file)
        print("Weights loaded successfully.")
    else:
        print("Weights files not found. Please train the model first.")

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Main training loop
    num_episodes = 10000
    for episode in range(num_episodes):
        board = chess.Board()  # Initialize new game

        states, actions, rewards, old_probs = collect_trajectory(policy_network, board)
        value_estimates = compute_value_estimates(states, value_network)
        advantages, returns = compute_advantages_and_returns(rewards, value_estimates, gamma, lam)
        # advantages = np.zeros_like(rewards, dtype=np.float32)  # Placeholder for advantages (for now)
        episode_loss = ppo_update(policy_network, value_network, optimizer,
                                  states, actions, advantages, old_probs, rewards)
        print(f"Episode {episode + 1}/{num_episodes}: Total Reward = {np.sum(rewards)
                                                                      } - Outcome = {board.outcome()} - Loss = {episode_loss}")
        # if board.outcome() is None:
        #     print(board)
        with open(loss_file, 'a') as f:
            f.write(f"{episode + 1},{episode_loss}\n")
        if (episode + 1) % 100 == 0:
            policy_network.save_weights(policy_weights_file)
            value_network.save_weights(value_weights_file)

    # Save the weights of the policy and value networks
    policy_network.save_weights(policy_weights_file)
    value_network.save_weights(value_weights_file)
