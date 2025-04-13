import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from gym.agent import Agent


class DQNAgent(Agent):
    def __init__(self, num):
        super().__init__(num)
        self.state_size = 11
        self.action_space = 5
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=self.state_size))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def _state_input(self, known_hands, hand, game_state, discard, turn_count):
        state = np.zeros(11)





    def prompt_action(self, known_hands, hand, game_state, discard_pile, turn_count):
        return 0

    def prompt_callback(self, state, action):
        if action == 4 or action == 5:
            return 0, 0
        return 0

    def pass_state(self, state, hand, reward, done):
        pass

    def clear_memory(self):
        pass