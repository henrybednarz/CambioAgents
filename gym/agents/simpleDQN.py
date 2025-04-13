import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random


class SimpleDQNAgent:
    """
    A simplified Deep Q-Network (DQN) agent.
    Designed for the Cambio card game environment.
    """

    def __init__(
            self,
            state_size,
            action_size=20,  # Default to 20 for Cambio environment
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            batch_size=32,
            train_frequency=10  # Only train every 10 steps
    ):
        # Validate action_size
        if action_size <= 0:
            raise ValueError("action_size must be positive")

        # Basic parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.step_count = 0

        # Memory buffer (simplified)
        self.memory = []

        # Build a simple model
        self.model = self._build_model()

    def _build_model(self):
        """Build a simple neural network model."""
        model = Sequential()
        # Layer with more units to handle the complexity of the game
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

        # Keep memory size reasonable
        if len(self.memory) > 10000:
            self.memory.pop(0)

        # Increment step counter
        self.step_count += 1

    def should_train(self):
        """Determine if the agent should train on this step."""
        return (self.step_count % self.train_frequency == 0 and
                len(self.memory) >= self.batch_size)

    def act(self, state, training=True):
        """Choose an action using epsilon-greedy policy."""
        # Explore: choose random action
        if training and np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploit: choose best action based on model prediction
        # Flatten and normalize the observation for the neural network
        flat_state = self._flatten_state(state)
        act_values = self.model.predict(np.array([flat_state]), verbose=0)
        return np.argmax(act_values[0])

    def _flatten_state(self, state):
        """
        Convert the dictionary observation from the Cambio environment
        into a flat array for the neural network.
        """
        if isinstance(state, dict):
            # For Cambio environment
            flat_state = np.concatenate([
                state['known_player_cards'].flatten(),
                state['known_opponent_cards'].flatten(),
                state['hand'].flatten(),
                state['discard'].flatten(),
                [state['game_state']],
                state['open_action'].flatten(),
                state['turn_count'].flatten()
            ])
            return flat_state
        else:
            # For other environments or direct array input
            return state

    def replay(self):
        """Train the network using random samples from memory."""
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return 0.0

        # Randomly sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare arrays for batch processing
        states = []
        targets_full = []

        # Process each experience in batch
        for state, action, reward, next_state, done in minibatch:
            # Make sure action is within bounds
            if action >= self.action_size:
                # Skip this sample or clip the action
                action = action % self.action_size  # Clip to valid range

            # Flatten states
            flat_state = self._flatten_state(state)
            flat_next_state = self._flatten_state(next_state)

            # Get current Q values
            target = self.model.predict(np.array([flat_state]), verbose=0)[0]

            # Update target Q value for the action taken
            if done:
                target[action] = reward
            else:
                # Get Q values for next state
                next_q_values = self.model.predict(np.array([flat_next_state]), verbose=0)[0]
                # Update with Bellman equation
                target[action] = reward + self.gamma * np.max(next_q_values)

            # Store for batch training
            states.append(flat_state)
            targets_full.append(target)

        # Convert to numpy arrays
        states_array = np.array(states)
        targets_array = np.array(targets_full)

        # Train model in one batch
        history = self.model.fit(
            states_array,
            targets_array,
            epochs=1,
            verbose=0,
            batch_size=self.batch_size
        )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return average loss
        return history.history['loss'][0] if history.history['loss'] else 0.0

    def select_action(self, env):
        """Interface for the Cambio environment."""
        # Get observation
        obs = env._get_observation()
        # Choose action using policy
        return self.act(obs)

    def prompt_action(self, known_hands, hand, game_state, discard, turn_count):
        """Simple interface to match the expected signature in the environment."""
        # Create a simplified observation
        obs = {
            'known_player_cards': np.array([card[1] if card else -1 for card in known_hands[0]], dtype=np.int8),
            'known_opponent_cards': np.array([card[1] if card else -1 for card in known_hands[1]], dtype=np.int8),
            'hand': np.array([hand[1] if hand else -1], dtype=np.int8),
            'discard': np.array([discard[1] if discard else -1], dtype=np.int8),
            'game_state': 0 if game_state is None else game_state.value,
            'open_action': np.array([-1], dtype=np.int8),  # No open action
            'turn_count': np.array([turn_count], dtype=np.int32)
        }
        return self.act(obs)

    def prompt_callback(self, known_hands, open_action):
        """Handle callback actions."""
        # For peek own card (open_action 1)
        if open_action == 1:
            # Find card indices that we don't know yet
            unknown_indices = [i for i, card in enumerate(known_hands[0]) if card is None]
            if unknown_indices:
                return random.choice(unknown_indices)
            return 0  # Default to first card if all known

        # For peek opponent card (open_action 2)
        elif open_action == 2:
            # Find opponent card indices that we don't know yet
            unknown_indices = [i for i, card in enumerate(known_hands[1]) if card is None]
            if unknown_indices:
                return random.choice(unknown_indices)
            return 0  # Default to first card if all known

        # For swap blind or swap and look (open_action 3 or 4)
        elif open_action == 3 or open_action == 4:
            # Choose a random card from our hand and opponent's hand
            player_idx = random.randint(0, len(known_hands[0]) - 1)
            opponent_idx = random.randint(0, len(known_hands[1]) - 1)
            return (player_idx, opponent_idx)

        # Default return
        return 0

    def load(self, name):
        """Load model weights from file."""
        self.model.load_weights(name)

    def save(self, name):
        """Save model weights to file."""
        self.model.save_weights(name)


# Example usage for Cambio environment
if __name__ == "__main__":
    # Creating an agent for the Cambio environment
    # The state size is calculated based on the flattened observation
    # 3 (known player cards) + 3 (known opponent cards) + 1 (hand) + 1 (discard) + 1 (game state) + 1 (open action) + 1 (turn count) = 11
    state_size = 11
    action_size = 20  # Cambio environment has 20 possible actions

    agent = SimpleDQNAgent(state_size=state_size, action_size=action_size)
    print("Agent created successfully!")