import random
from agents.agent import Agent
from util import CambioState


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def prompt_action(self, known_hands, hand, game_state, discard_pile, turn_count, valid_actions):
        if random.random() < 0.1 and game_state is CambioState.NOT_CALLED:
            return 4
        valid_actions[4] = False
        valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        return random.choice(valid_indices)

    def prompt_callback(self, state, action, valid_actions):
        if valid_actions is not None:
            valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
            return random.choice(valid_indices)
        return 0
