import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from collections import deque


class DQNAgent:
    def __init__(self,
                 state_size=28,
                 action_size=21,
                 learning_rate=0.00005,
                 gamma=0.85,
                 epsilon=0.5,
                 epsilon_min=0.05,
                 epsilon_decay=0.9999,
                 batch_size=256,
                 update_frequency=25):

        # Enabling GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_steps = 0
        self.update_frequency = update_frequency
        self.use_double_dqn = True

        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_epsilon = 1e-6

        self.memory = deque(maxlen=50000)
        self.priorities = deque(maxlen=50000)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='huber', optimizer=optimizer)

        return model

    def act(self, state, valid_actions=None, training=False):
        if valid_actions is None:
            valid_actions = np.ones(self.action_size, dtype=bool)

        if training and np.random.rand() <= self.epsilon:
            valid_indices = np.where(valid_actions)[0]
            return np.random.choice(valid_indices)

        state_np = np.array(state, dtype=np.float32)
        state_tensor = np.expand_dims(state_np, axis=0)
        q_values = self.model(state_tensor, training=False)[0]

        valid_q_values = np.where(valid_actions, q_values, -np.inf)
        chosen_action = tf.argmax(valid_q_values).numpy()

        return chosen_action

    def remember(self, state, action, reward, next_state, done, valid_actions):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, 1.0 if done else 0.0, valid_actions))
        self.priorities.append(max_priority)

    def replay_sample(self):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.priority_alpha
        probabilities /= np.sum(probabilities)

        batch_indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        batch_sample = [self.memory[i] for i in batch_indices]

        importance = 1.0 / (len(self.memory) * probabilities[batch_indices])
        importance = importance ** self.priority_beta
        importance /= np.max(importance)

        states, actions, rewards, next_states, dones, valid_actions_batch = zip(*batch_sample)

        return (
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=np.int32),
            tf.convert_to_tensor(rewards, dtype=np.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=np.float32),
            importance,
            batch_indices,
            valid_actions_batch
        )

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        if len(self.memory) <= self.batch_size:
            return
        states, actions, rewards, next_states, dones, importance, batch_indices, valid_actions = self.replay_sample()

        q_next = self.target_model(next_states)
        if self.use_double_dqn:
            q_next_online = self.model(next_states)

            valid_actions_mask = np.zeros((self.batch_size, self.action_size), dtype=np.float32)

            for i, valid_acts in enumerate(valid_actions):
                if valid_acts is not None:
                    valid_actions_mask[i] = valid_acts
                else:
                    valid_actions_mask[i] = np.ones(self.action_size)

            q_next_online_masked = tf.where(
                tf.convert_to_tensor(valid_actions_mask, dtype=tf.bool),
                q_next_online,
                tf.ones_like(q_next_online) * -float('inf')
            )

            best_actions = tf.argmax(q_next_online_masked, axis=1)

            indices = tf.stack([tf.range(tf.shape(best_actions)[0], dtype=tf.int64), best_actions], axis=1)
            q_next_max = tf.gather_nd(q_next, indices)
        else:
            q_next_max = tf.reduce_max(q_next, axis=1)

        q_targets = rewards + self.gamma * (1.0 - dones) * q_next_max

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            actions_mask = tf.one_hot(actions, depth=self.action_size)
            q_values_selected = tf.reduce_sum(q_values * actions_mask, axis=1)

            td_errors = tf.abs(q_targets - q_values_selected)

            importance_tensor = tf.convert_to_tensor(importance, dtype=tf.float32)
            weighted_loss = tf.reduce_mean(importance_tensor * tf.square(td_errors))

        for i, idx in enumerate(batch_indices):
            self.priorities[idx] = td_errors[i].numpy() + self.priority_epsilon

        grads = tape.gradient(weighted_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_steps += 1

        if self.train_steps % self.update_frequency == 0:
            self.update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.priority_beta = min(1.0, self.priority_beta + 0.0001)

        return weighted_loss.numpy()

    def load(self, file_name):
        self.model.load_weights(file_name)
        self.target_model.load_weights(file_name)

    def save(self, save_path):
        
        self.model.save_weights(save_path)