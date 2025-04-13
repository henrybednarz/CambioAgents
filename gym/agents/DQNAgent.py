import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from gym.agent import Agent
from gym.util import CambioState


class DQNAgent(Agent):
    def __init__(self, num, state_size=11, action_size=4, memory_size=2000,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.01, policy_path=None):
        super().__init__(num)
        self.state_size = state_size  # Number of state features
        self.action_size = action_size  # Number of possible actions
        self.memory = deque(maxlen=memory_size)  # Experience replay buffer
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.learning_rate = learning_rate
        self.training = True

        self.invalid_actions = set()

        # Optimization: Batch-related tracking
        self.update_frequency = 4  # Only update every N steps
        self.replay_batch_size = 64  # Larger batch size for efficiency
        self.step_counter = 0  # Track steps for batch updates
        self.target_update_frequency = 1000  # Update target network less frequently

        # Configure TensorFlow for faster inference
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)

        # Initialize neural network model
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network for stability
        self.update_target_model()

        # State tracking
        self.last_state = None
        self.last_action = None
        self.current_state = None
        self.current_action = None

        # Load model if path provided
        if policy_path:
            self.load_model(policy_path)

    def _build_model(self):
        """Build a neural network to approximate Q-function"""
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def clear_memory(self):
        self.memory.clear()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on epsilon-greedy policy, avoiding known invalid actions"""
        if self.training and np.random.rand() <= self.epsilon:
            # Random action but avoid known invalid actions
            valid_actions = [a for a in range(self.action_size) if a not in self.invalid_actions]
            if not valid_actions:  # Reset if all actions have been tried and found invalid
                self.invalid_actions = set()
                valid_actions = list(range(self.action_size))
            return random.choice(valid_actions)

        # Get Q-values from model
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

        # Mask out invalid actions with very negative values
        for invalid_action in self.invalid_actions:
            if 0 <= invalid_action < len(act_values):
                act_values[invalid_action] = float('-inf')

        # If all actions have been tried and found invalid, reset
        if all(v == float('-inf') for v in act_values):
            self.invalid_actions = set()
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

        return np.argmax(act_values)

    def replay(self, batch_size=None):
        """Train on random batch from memory - optimized version"""
        # Use default batch size if none provided
        if batch_size is None:
            batch_size = self.replay_batch_size

        # Skip replay if not enough samples or not on update frequency
        if len(self.memory) < batch_size:
            return

        # Increment step counter and check if update is needed
        self.step_counter += 1
        if self.step_counter % self.update_frequency != 0:
            return

        # Sample a larger batch for efficiency
        minibatch = random.sample(self.memory, batch_size)

        # Prepare batch data for vectorized operations
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        # Extract batch data
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Get current Q values for all states in batch
        current_q_values = self.model.predict(states, verbose=0)

        # Get next Q values from target model for all next states
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q values for the actions taken
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i, actions[i]] = rewards[i]
            else:
                current_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train model on entire batch at once
        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=batch_size)

        # Update target model occasionally
        if self.step_counter % self.target_update_frequency == 0:
            self.update_target_model()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def prompt_action(self, known_hands, hand, game_state, discard, turn_count):
        """Choose an action based on current state, avoiding known invalid actions"""
        self.current_state = (known_hands, hand, game_state, discard, turn_count)

        # Reset invalid actions for new state
        self.invalid_actions = set()

        state = self._preprocess_action_state(known_hands, hand, game_state, discard, turn_count)
        self.last_state = state

        action = self.act(state)
        self.last_action = action

        return action

    def prompt_callback(self, known_hands, action):
        """Handle callback actions with validation"""
        player_size = len(known_hands[self.player_id])
        opp_size = len(known_hands[(self.player_id + 1) % 2])

        if action == 0:
            return 0
        elif action == 1:  # Peek own
            # Choose the card with the least information
            for i in range(player_size):
                if known_hands[self.player_id][i] is None:
                    return i
            return 0
        elif action == 2:  # Peek other
            for i in range(opp_size):
                if known_hands[(self.player_id + 1) % 2][i] is None:
                    return i
            return 0
        elif action == 3 or action == 4:  # Swap actions
            my_cards = known_hands[self.player_id]
            opp_cards = known_hands[(self.player_id + 1) % 2]

            # Find my lowest value known card
            my_lowest_idx = 0
            my_lowest_val = float('inf')
            for i in range(len(my_cards)):
                if my_cards[i] is not None and my_cards[i][1] < my_lowest_val:
                    my_lowest_val = my_cards[i][1]
                    my_lowest_idx = i

            # Find opponent's highest value known card
            opp_highest_idx = 0
            opp_highest_val = -float('inf')

            for i in range(len(opp_cards)):
                if opp_cards[i] is not None and opp_cards[i][1] > opp_highest_val:
                    opp_highest_val = opp_cards[i][1]
                    opp_highest_idx = i

            # Make sure indices are valid
            my_lowest_idx = max(0, min(my_lowest_idx, player_size - 1))
            opp_highest_idx = max(0, min(opp_highest_idx, opp_size - 1))

            return my_lowest_idx, opp_highest_idx

        return 0

    def pass_state(self, state, hand, reward, done):
        """Process the result of last action and learn, handling invalid moves"""
        if not self.training or self.last_state is None:
            return

        known_state, hand, game_state, discard, turn_count = self.current_state
        next_state = self._preprocess_action_state(known_state, hand, game_state, discard, turn_count)

        # If reward is very negative, this was likely an invalid move
        if reward < -5:
            # Mark the action as invalid for the current state
            self.invalid_actions.add(self.last_action)

        self.remember(self.last_state, self.last_action, reward, next_state, done)
        self.replay()

        if done:
            self.update_target_model()
            # Clear invalid action tracking on new episode
            self.invalid_actions = set()
        else:
            self.last_state = next_state

        # If done, reset state tracking
        if done:
            self.last_state = None
            self.last_action = None

    def save_model(self, path):
        """Save model weights"""
        self.model.save_weights(path)

    def load_model(self, path):
        """Load model weights"""
        try:
            self.model.load_weights(path)
            self.update_target_model()
        except:
            print(f"Could not load model")

    def _preprocess_action_state(self, known_hands, hand, game_state, discard, turn_count):
        """Convert game state to a normalized feature vector"""
        state = []

        for i in range(3):  # Assuming 3 cards per player (HAND_SIZE = 3)
            if i < len(known_hands[self.player_id]):
                card = known_hands[self.player_id][i]
                state.append(card[1] / 13.0 if card else -1)
            else:
                state.append(-1)  # Padding if hand size is smaller

            # Opponent's hand - 3 cards
        for i in range(3):
            if i < len(known_hands[(self.player_id + 1) % 2]):
                card = known_hands[(self.player_id + 1) % 2][i]
                state.append(card[1] / 13.0 if card else -1)
            else:
                state.append(-1)

                # Current hand
        state.append(hand[1] / 13.0)  # Normalized card value

        # Game state features
        state.append(1.0 if game_state == CambioState.CALLED or game_state == CambioState.LAST_TURN else 0.0)

        my_score = sum(card[1] for card in known_hands[self.player_id] if card is not None)
        opp_score = sum(card[1] for card in known_hands[(self.player_id + 1) % 2] if card is not None)

        unknown_count_me = sum(1 for card in known_hands[self.player_id] if card is None)
        unknown_count_opp = sum(1 for card in known_hands[(self.player_id + 1) % 2] if card is None)
        my_score += unknown_count_me * 7.0  # Assume middle value of 7 for unknown cards
        opp_score += unknown_count_opp * 7.0

        relative_score = (opp_score - my_score) / 21.0  # Normalize by max possible hand value
        state.append(relative_score)

        # Turn count (normalized)
        state.append(min(turn_count / 30.0, 1.0))  # Assuming max 30 turns

        # Top discard card
        if discard:
            state.append(discard[1] / 13.0)
        else:
            state.append(0.0)

        return np.array(state)