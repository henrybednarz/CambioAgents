import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from gym.agent import Agent
from gym.util import CambioState


class OptimizedDQNAgent(Agent):
    def __init__(self, num, state_size=11, action_size=4, memory_size=10000,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, policy_path=None):
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

        # Track invalid actions to avoid them
        self.invalid_actions = set()

        # Optimization parameters
        self.update_frequency = 8  # Update less frequently
        self.replay_batch_size = 128  # Larger batch size
        self.step_counter = 0
        self.target_update_frequency = 1000

        # Training indicators
        self.train_iterations = 0
        self.total_rewards = 0

        # Tensorflow optimization
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Use mixed precision for faster training if GPU available
        if physical_devices:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # Create models with optimizations
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Create prediction cache to avoid redundant computations
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000

        # State tracking
        self.last_state = None
        self.last_action = None
        self.current_state = None
        self.current_action = None

        # For faster vectorized operations
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []
        self.buffer_size = 0
        self.max_buffer_size = 256  # Accumulate this many before batch update

        # Load model if path provided
        if policy_path:
            self.load_model(policy_path)

    def _build_model(self):
        """Build a more efficient neural network with BatchNormalization"""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])

        # Use a more efficient optimizer configuration
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def clear_memory(self):
        """Reset memory and cache"""
        self.memory.clear()
        self.prediction_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []
        self.buffer_size = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in buffer and potentially trigger batch training"""
        # Add to immediate buffer for batch updates
        self.states_buffer.append(state)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)
        self.next_states_buffer.append(next_state)
        self.dones_buffer.append(done)
        self.buffer_size += 1

        # Also add to long-term memory
        self.memory.append((state, action, reward, next_state, done))

        # If buffer is full, perform a batch update
        if self.buffer_size >= self.max_buffer_size:
            self._batch_update()

    def _batch_update(self):
        """Process accumulated experiences in buffer"""
        if self.buffer_size == 0:
            return

        # Convert buffers to numpy arrays for faster processing
        states = np.array(self.states_buffer)
        actions = np.array(self.actions_buffer)
        rewards = np.array(self.rewards_buffer)
        next_states = np.array(self.next_states_buffer)
        dones = np.array(self.dones_buffer)

        # Get current Q values for all states in batch
        current_q_values = self.model.predict(states, verbose=0, batch_size=self.buffer_size)

        # Get next Q values from target model for all next states
        next_q_values = self.target_model.predict(next_states, verbose=0, batch_size=self.buffer_size)

        # Create target Q values by copying current values and updating only the values for taken actions
        target_q_values = current_q_values.copy()

        # Update in batch using numpy operations
        for i in range(self.buffer_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train model on entire batch at once
        self.model.fit(states, target_q_values, epochs=1, verbose=0, batch_size=min(64, self.buffer_size))

        # Reset buffers
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []
        self.buffer_size = 0

        # Update counter
        self.train_iterations += 1

        # Decay epsilon after each batch update
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Occasionally update target model
        if self.train_iterations % self.target_update_frequency == 0:
            self.update_target_model()
            # Clear prediction cache after target update
            self.prediction_cache.clear()

    def get_state_hash(self, state):
        """Create a hashable representation of a state for caching"""
        return hash(state.tobytes())

    def predict_q_values(self, state):
        """Get Q-values with caching to avoid redundant predictions"""
        state_hash = self.get_state_hash(state)

        if state_hash in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[state_hash]

        self.cache_misses += 1
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

        # Add to cache, with size management
        if len(self.prediction_cache) >= self.max_cache_size:
            # Remove a random key to prevent cache from growing too large
            self.prediction_cache.pop(random.choice(list(self.prediction_cache.keys())))

        self.prediction_cache[state_hash] = q_values
        return q_values

    def act(self, state):
        """Choose action based on epsilon-greedy policy, with caching to speed up predictions"""
        if self.training and np.random.rand() <= self.epsilon:
            # Random action but avoid known invalid actions
            valid_actions = [a for a in range(self.action_size) if a not in self.invalid_actions]
            if not valid_actions:
                self.invalid_actions = set()
                valid_actions = list(range(self.action_size))
            return random.choice(valid_actions)

        # Get Q-values with caching
        act_values = self.predict_q_values(state)

        # Mask out invalid actions
        masked_values = act_values.copy()
        for invalid_action in self.invalid_actions:
            if 0 <= invalid_action < len(masked_values):
                masked_values[invalid_action] = float('-inf')

        # If all actions invalid, reset
        if all(v == float('-inf') for v in masked_values):
            self.invalid_actions = set()
            return np.argmax(act_values)

        return np.argmax(masked_values)

    def replay(self, batch_size=None):
        """Train on random batch from memory - vectorized version"""
        if batch_size is None:
            batch_size = self.replay_batch_size

        if len(self.memory) < batch_size:
            return

        self.step_counter += 1
        if self.step_counter % self.update_frequency != 0:
            return

        # Sample batch
        minibatch = random.sample(self.memory, batch_size)

        # Prepare arrays for vectorized operations
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.zeros(batch_size)
        dones = np.zeros(batch_size, dtype=np.bool_)

        # Extract data
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        # Compute Q values in batches
        target_q_values = self.model.predict(states, batch_size=batch_size, verbose=0)
        next_q_values = self.target_model.predict(next_states, batch_size=batch_size, verbose=0)

        # Vectorized update of target Q values
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_next_q

        # Update only the relevant action values
        for i in range(batch_size):
            target_q_values[i, actions[i]] = targets[i]

        # Train in one batch
        self.model.fit(states, target_q_values, epochs=1, verbose=0, batch_size=batch_size)

    def prompt_action(self, known_hands, hand, game_state, discard, turn_count):
        """Choose an action based on current state, with optimized state processing"""
        self.current_state = (known_hands, hand, game_state, discard, turn_count)

        # Reset invalid actions for new state
        self.invalid_actions = set()

        # Process state faster with numpy operations
        state = self._preprocess_action_state(known_hands, hand, game_state, discard, turn_count)
        self.last_state = state

        action = self.act(state)
        self.last_action = action

        return action

    def prompt_callback(self, known_hands, action):
        """Handle callback actions with smarter heuristics"""
        player_hands = known_hands[self.player_id]
        opp_hands = known_hands[(self.player_id + 1) % 2]
        player_size = len(player_hands)
        opp_size = len(opp_hands)

        if action == 0:
            return 0
        elif action == 1:  # Peek own
            # Strategic peek: prioritize unknown cards based on position
            # Center card often more valuable in Cambio
            priority_order = [1, 0, 2]  # Check center first, then left, then right
            for idx in priority_order:
                if idx < player_size and player_hands[idx] is None:
                    return idx
            return 0

        elif action == 2:  # Peek other
            # Strategic peek: prioritize unknown opponent cards
            priority_order = [1, 0, 2]  # Check center first, then left, then right
            for idx in priority_order:
                if idx < opp_size and opp_hands[idx] is None:
                    return idx
            return 0

        elif action == 3 or action == 4:  # Swap actions
            # Find my highest value card to swap
            my_highest_idx = 0
            my_highest_val = -float('inf')

            for i in range(player_size):
                card = player_hands[i]
                if card is not None:
                    val = card[1]
                    if val > my_highest_val:
                        my_highest_val = val
                        my_highest_idx = i
                elif my_highest_val == -float('inf'):
                    # If no known cards, prioritize center position
                    if i == 1:
                        my_highest_idx = i

            # Find opponent's lowest value known card or unknown card
            opp_lowest_idx = 0
            opp_lowest_val = float('inf')

            for i in range(opp_size):
                card = opp_hands[i]
                if card is not None:
                    val = card[1]
                    if val < opp_lowest_val:
                        opp_lowest_val = val
                        opp_lowest_idx = i
                elif opp_lowest_val == float('inf'):
                    # If no known cards, prioritize center position
                    if i == 1:
                        opp_lowest_idx = i

            return my_highest_idx, opp_lowest_idx

        return 0

    def pass_state(self, state, hand, reward, done):
        """Process the result of last action and learn, with optimized state update"""
        if not self.training or self.last_state is None:
            return

        # Track total rewards for monitoring
        self.total_rewards += reward

        # Process next state
        known_state, hand, game_state, discard, turn_count = self.current_state
        next_state = self._preprocess_action_state(known_state, hand, game_state, discard, turn_count)

        # Track invalid actions to avoid them in the future
        if reward < -5:
            self.invalid_actions.add(self.last_action)

        # Store experience and potentially trigger batch update
        self.remember(self.last_state, self.last_action, reward, next_state, done)

        # Less frequent replay
        if len(self.memory) > self.replay_batch_size * 2:
            self.replay()

        if done:
            # Force a batch update at the end of episode
            self._batch_update()
            # Reset tracking
            self.invalid_actions = set()
            self.last_state = None
            self.last_action = None
        else:
            self.last_state = next_state

    def save_model(self, path):
        """Save model weights with metadata"""
        self.model.save_weights(path)

        # Could also save training progress metrics in a separate file
        # np.savez(f"{path}_stats.npz",
        #          epsilon=self.epsilon,
        #          train_iterations=self.train_iterations,
        #          total_rewards=self.total_rewards)

    def load_model(self, path):
        """Load model weights with optional metadata"""
        try:
            self.model.load_weights(path)
            self.update_target_model()
            self.prediction_cache.clear()  # Clear cache after loading

            # Could also load training progress metrics
            # try:
            #     stats = np.load(f"{path}_stats.npz")
            #     self.epsilon = stats['epsilon']
            #     self.train_iterations = stats['train_iterations']
            #     self.total_rewards = stats['total_rewards']
            # except:
            #     pass

        except Exception as e:
            print(f"Could not load model: {e}")

    def _preprocess_action_state(self, known_hands, hand, game_state, discard, turn_count):
        """Convert game state to a normalized feature vector - optimized version"""
        # Preallocate numpy array for better performance
        state = np.ones(self.state_size) * -1  # Default to -1 (unknown)

        # My hand - first 3 indices
        for i in range(min(3, len(known_hands[self.player_id]))):
            card = known_hands[self.player_id][i]
            if card is not None:
                state[i] = card[1] / 13.0

        # Opponent's hand - next 3 indices
        opp_id = (self.player_id + 1) % 2
        for i in range(min(3, len(known_hands[opp_id]))):
            card = known_hands[opp_id][i]
            if card is not None:
                state[i + 3] = card[1] / 13.0

        # Current hand card value
        state[6] = hand[1] / 13.0

        # Game state (called or last turn)
        state[7] = 1.0 if game_state == CambioState.CALLED or game_state == CambioState.LAST_TURN else 0.0

        # Calculate relative score more efficiently
        my_score = sum(card[1] for card in known_hands[self.player_id] if card is not None)
        opp_score = sum(card[1] for card in known_hands[opp_id] if card is not None)

        # Count unknown cards
        unknown_me = sum(1 for card in known_hands[self.player_id] if card is None)
        unknown_opp = sum(1 for card in known_hands[opp_id] if card is None)

        # Assume average value (7) for unknown cards
        my_score += unknown_me * 7.0
        opp_score += unknown_opp * 7.0

        # Normalize relative score
        state[8] = (opp_score - my_score) / 21.0

        # Turn count (normalized)
        state[9] = min(turn_count / 30.0, 1.0)

        # Top discard card
        if discard:
            state[10] = discard[1] / 13.0

        return state