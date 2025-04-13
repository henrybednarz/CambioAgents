import numpy as np
import tensorflow as tf
import random
from collections import deque
import os
import datetime
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class DQNAgent:
    """
    Accelerated Deep Q-Network (DQN) agent optimized for faster training.
    Includes performance optimizations and efficient training mechanisms.
    """

    def __init__(
            self,
            state_size,
            action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9975,  # Slightly slower decay
            batch_size=128,  # Larger batch size
            memory_size=20000,  # Moderate memory size
            update_target_freq=500,  # More frequent updates
            n_step=3,  # n-step learning
            use_soft_update=True,  # Soft target updates
            tau=0.005,  # Soft update parameter
    ):
        # Set TensorFlow to mixed precision for faster computation
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        # Enable memory growth to avoid CUDA OOM errors
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass

        # Class parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.n_step = n_step
        self.use_soft_update = use_soft_update
        self.tau = tau

        # Create replay memory
        self.memory = deque(maxlen=memory_size)
        self.n_step_buffer = deque(maxlen=n_step)

        # Create main model with optimized architecture
        self.model = self._build_model()
        self.target_model = self._build_model()

        # Initialize target model weights
        self.update_target_model()

        # Training step counter
        self.train_step_counter = 0

        # Logger for tracking metrics
        self.log_dir = "logs/fast_dqn_agent/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        # Set up checkpointing
        self.checkpoint_dir = "checkpoints/dqn"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Pre-allocated arrays for batch processing
        self.states_batch = np.zeros((batch_size, state_size), dtype=np.float32)
        self.next_states_batch = np.zeros((batch_size, state_size), dtype=np.float32)
        self.rewards_batch = np.zeros(batch_size, dtype=np.float32)
        self.actions_batch = np.zeros(batch_size, dtype=np.int32)
        self.dones_batch = np.zeros(batch_size, dtype=np.bool_)

    def _build_model(self):
        """Build an efficient neural network model for DQN."""
        # Functional API for more flexibility and potentially better performance
        inputs = Input(shape=(self.state_size,))

        # First layer (no activation needed before batch norm)
        x = Dense(128)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Second layer
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Output layer
        outputs = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss='huber',  # More stable than MSE
            optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-4)
        )

        return model

    def update_target_model(self):
        """Update target model weights with current model weights."""
        if self.use_soft_update:
            # Soft update - gradually blend weights for stability
            target_weights = self.target_model.get_weights()
            model_weights = self.model.get_weights()

            for i in range(len(target_weights)):
                target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]

            self.target_model.set_weights(target_weights)
        else:
            # Hard update - directly copy weights
            self.target_model.set_weights(self.model.get_weights())

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Get n-step reward, next state, and done for n-step learning."""
        reward, next_state, done = n_step_buffer[-1][-3:]

        for i in range(len(n_step_buffer) - 2, -1, -1):
            r, s, d = n_step_buffer[i][-3:]
            reward = r + gamma * reward * (1 - d)
            next_state, done = (s, d) if d else (next_state, done)

        return reward, next_state, done

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory with n-step learning."""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we don't have enough transitions for n-step learning, return
        if len(self.n_step_buffer) < self.n_step:
            return

        # Get the n-step reward, next state, and done
        state, action = self.n_step_buffer[0][:2]
        reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma)

        # Store in replay memory
        self.memory.append((state, action, reward, next_state, done))

        # If episode is done, empty the n-step buffer and add remaining transitions
        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                state, action = self.n_step_buffer[0][:2]
                reward, next_state, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
                self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action based on epsilon-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose random action
            return random.randrange(self.action_size)

        # Convert state to float32 tensor for efficiency
        state_tensor = np.expand_dims(state, axis=0).astype(np.float32)

        # Get Q-values (use no training mode for faster inference)
        with tf.device('/GPU:0'):  # Force GPU usage if available
            act_values = self.model(state_tensor, training=False).numpy()

        return np.argmax(act_values[0])

    def replay(self):
        """Train the model efficiently with vectorized operations."""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples for training

        # Sample minibatch from memory using numpy operations for speed
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)

        # Extract batches using pre-allocated arrays for efficiency
        for i, idx in enumerate(indices):
            state, action, reward, next_state, done = self.memory[idx]
            self.states_batch[i] = state
            self.next_states_batch[i] = next_state
            self.rewards_batch[i] = reward
            self.actions_batch[i] = action
            self.dones_batch[i] = done

        # Predict Q-values for current states (all at once)
        with tf.device('/GPU:0'):  # Force GPU usage if available
            targets = self.model.predict(self.states_batch, batch_size=self.batch_size, verbose=0)
            target_next = self.target_model.predict(self.next_states_batch, batch_size=self.batch_size, verbose=0)

        # Vectorized update for Q-values (much faster than a loop)
        targets_actions = np.argmax(targets, axis=1)

        # Vectorized calculation of next Q-values with done masking
        max_q_values = target_next[np.arange(self.batch_size), targets_actions]
        targets[np.arange(self.batch_size), self.actions_batch] = self.rewards_batch + \
                                                                  self.gamma * max_q_values * (1 - self.dones_batch)

        # Train the model with GPU acceleration
        with tf.device('/GPU:0'):
            history = self.model.fit(
                self.states_batch,
                targets,
                batch_size=self.batch_size,
                epochs=1,
                verbose=0,
            )

        loss = history.history['loss'][0]

        # Update target model periodically or with soft updates
        self.train_step_counter += 1
        if self.use_soft_update:
            # Soft update every training step
            self.update_target_model()
        elif self.train_step_counter % self.update_target_freq == 0:
            # Hard update periodically
            self.update_target_model()
            print(f"Target model updated. Step: {self.train_step_counter}")

        # More aggressive epsilon decay early in training
        if self.train_step_counter < 10000:
            decay_rate = 0.9995
        else:
            decay_rate = self.epsilon_decay

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= decay_rate

        # Log metrics less frequently to reduce overhead
        if self.train_step_counter % 100 == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=self.train_step_counter)
                tf.summary.scalar('epsilon', self.epsilon, step=self.train_step_counter)

        # Save checkpoints periodically
        if self.train_step_counter % 1000 == 0:
            self.save(f"{self.checkpoint_dir}/model_{self.train_step_counter}.weights.h5")

        return loss

    # Multiple replay iterations for faster learning
    def train_batch(self, iterations=4):
        """Train on multiple batches consecutively for faster learning."""
        total_loss = 0
        for _ in range(iterations):
            loss = self.replay()
            if loss > 0:
                total_loss += loss
        return total_loss / iterations if total_loss > 0 else 0

    def load(self, name):
        """Load model weights from file."""
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        """Save model weights to file."""
        self.model.save_weights(name)

