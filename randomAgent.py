
import random
from agent import Agent

ACTION_SPACE = {"swap1": 0, "swap2": 1, "swap3": 2, "play": 3, "cambio": 4}


class RandomAgent(Agent):
    def __init__(self, num):
        super().__init__(num)
        pass

    def prompt_action(self, state, hand):
        if random.random() < 0.1:
            return ACTION_SPACE["cambio"]
        else:
            return random.randint(0, 3)

    def prompt_callback(self, state, action):
        if action == 3 or action == 4:
            return random.randint(0, 2), random.randint(0, 2)
        return random.randint(0, 2)
    
    def choose_pass(self, state):
        return 0

    def pass_state(self, state, hand, reward, done):
        pass