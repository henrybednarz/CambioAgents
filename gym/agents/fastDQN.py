import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from gym.agent import Agent
from gym.util import CambioState


class FastDQNAgent(Agent):
    def __init__(self, num, state_size=11, action_size=5, memory_size=2000,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.01, policy_path=None):
        super().__init__(num)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.training = True

        # Create a simpler model
        self.model = self._build_model()

        # Track state/action
        self.last_state = None
        self.last_action = None
        self.current_state = None

        # Keep track of invalid actions to avoid them
        self.invalid_actions = set()

        # Update and batch parameters
        self.batch_size = 64
        self.update_every = 8  # Only update every N steps to save computation
        self.step_counter = 0

        # Load model if path provided
        if policy_path:
            self.load_model(policy_path)

    def _build_model(self):
        """Build a simple neural network model"""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def clear_memory(self):
        """Reset memory"""
        self.memory.clear()
        self.invalid_actions.clear()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if self.training and np.random.rand() <= self.epsilon:
            # Random action that isn't marked as invalid
            valid_actions = [a for a in range(self.action_size) if a not in self.invalid_actions]
            if not valid_actions:  # Reset if all actions invalid
                self.invalid_actions.clear()
                valid_actions = list(range(self.action_size))
            return random.choice(valid_actions)

        # Get Q-values from model
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

        # Avoid invalid actions
        for invalid_action in self.invalid_actions:
            if 0 <= invalid_action < len(act_values):
                act_values[invalid_action] = float('-inf')

        # If all actions invalid, reset
        if np.max(act_values) == float('-inf'):
            self.invalid_actions.clear()
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

        return np.argmax(act_values)

    def replay(self):
        """Train on random batch from memory"""
        # Only update periodically to reduce computation
        self.step_counter += 1
        if self.step_counter % self.update_every != 0:
            return

        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch
        minibatch = random.sample(self.memory, self.batch_size)

        # Process all samples in batch
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state

            # Get current Q-values
            target = self.model.predict(state.reshape(1, -1), verbose=0)[0]

            if done:
                target[action] = reward
            else:
                # Get Q-values for next state
                next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(next_q_values)

            targets[i] = target

        # Train in single batch
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def prompt_action(self, known_hands, hand, game_state, discard, turn_count):
        """Choose an action based on current state"""
        self.current_state = (known_hands, hand, game_state, discard, turn_count)

        # Reset invalid actions for new state
        self.invalid_actions.clear()

        # Process state
        state = self._preprocess_state(known_hands, hand, game_state, discard, turn_count)
        self.last_state = state

        # Choose action
        action = self.act(state)
        self.last_action = action

        return action

    def prompt_callback(self, known_hands, action):
        """Handle callback actions with simple heuristics"""
        player_id = self.player_id
        opponent_id = (player_id + 1) % 2

        # Handle different action types
        if action == 0:  # No action
            return 0

        elif action == 1:  # Peek own
            # Choose first unknown card
            for i in range(len(known_hands[player_id])):
                if known_hands[player_id][i] is None:
                    return i
            return 0

        elif action == 2:  # Peek other
            # Choose first unknown opponent card
            for i in range(len(known_hands[opponent_id])):
                if known_hands[opponent_id][i] is None:
                    return i
            return 0

        elif action == 3 or action == 4:  # Swap actions
            # Default strategy - swap my highest known card with opponent's lowest
            my_highest_idx = 0
            my_highest_val = -1

            for i, card in enumerate(known_hands[player_id]):
                if card is not None and card[1] > my_highest_val:
                    my_highest_val = card[1]
                    my_highest_idx = i

            opp_lowest_idx = 0
            opp_lowest_val = 14  # Higher than any card

            for i, card in enumerate(known_hands[opponent_id]):
                if card is not None and card[1] < opp_lowest_val:
                    opp_lowest_val = card[1]
                    opp_lowest_idx = i

            # If no known cards, use first position
            return my_highest_idx, opp_lowest_idx

        return 0

    def pass_state(self, state, hand, reward, done):
        """Process the result of the last action and learn"""
        if not self.training or self.last_state is None:
            return

        # Get next state
        known_state, hand, game_state, discard, turn_count = self.current_state
        next_state = self._preprocess_state(known_state, hand, game_state, discard, turn_count)

        # If negative reward, mark action as invalid
        if reward < -5:
            self.invalid_actions.add(self.last_action)

        # Remember experience
        self.remember(self.last_state, self.last_action, reward, next_state, done)

        # Learn periodically
        self.replay()

        # Update for next step
        if done:
            self.last_state = None
            self.last_action = None
            self.invalid_actions.clear()
        else:
            self.last_state = next_state

    def save_model(self, path):
        """Save model weights"""
        self.model.save_weights(path)

    def load_model(self, path):
        """Load model weights"""
        try:
            self.model.load_weights(path)
        except Exception as e:
            print(f"Could not load model: {e}")

    def _preprocess_state(self, known_hands, hand, game_state, discard, turn_count):
        """Convert game state to a feature vector"""
        state = np.ones(self.state_size) * -1  # Default to -1 (unknown)

        player_id = self.player_id
        opponent_id = (player_id + 1) % 2

        # Process my hand (first 3 positions)
        for i in range(min(3, len(known_hands[player_id]))):
            card = known_hands[player_id][i]
            if card is not None:
                state[i] = card[1] / 13.0  # Normalize card value

        # Process opponent's hand (next 3 positions)
        for i in range(min(3, len(known_hands[opponent_id]))):
            card = known_hands[opponent_id][i]
            if card is not None:
                state[i + 3] = card[1] / 13.0

        # Current card
        state[6] = hand[1] / 13.0

        # Game state
        state[7] = 1.0 if game_state == CambioState.CALLED or game_state == CambioState.LAST_TURN else 0.0

        # Relative score
        my_score = sum(card[1] for card in known_hands[player_id] if card is not None)
        opp_score = sum(card[1] for card in known_hands[opponent_id] if card is not None)

        # Unknown cards count
        unknown_me = sum(1 for card in known_hands[player_id] if card is None)
        unknown_opp = sum(1 for card in known_hands[opponent_id] if card is None)

        # Assume average value for unknown cards
        my_score += unknown_me * 7.0
        opp_score += unknown_opp * 7.0

        # Normalize score difference
        state[8] = (opp_score - my_score) / 21.0

        # Turn count
        state[9] = min(turn_count / 30.0, 1.0)

        # Top discard card
        if discard:
            state[10] = discard[1] / 13.0

        return state