import random
from agent import Agent
from util import CambioState

ACTION_SPACE = {"swap1": 0, "swap2": 1, "swap3": 2, "play": 3, "cambio": 4}


class RandomAgent(Agent):
    def __init__(self, num):
        super().__init__(num)
        self.player_id = num

    def prompt_action(self, known_hands, hand, game_state, discard_pile, turn_count, valid_actions=None):
        if valid_actions is not None:
            if random.random() < 0.1 and game_state is CambioState.NOT_CALLED:
                return 4
            valid_actions[4] = False
            valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
            return random.choice(valid_indices)

    def prompt_callback(self, state, action, valid_actions=None):
        if valid_actions is not None:
            valid_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
            return random.choice(valid_indices)

        player_size = len(state[self.player_id])
        opp_size = len(state[(self.player_id + 1) % 2])
        if action == 1:
            if player_size == 0:
                return 0
            return random.randrange(0, player_size)
        elif action == 2:
            if opp_size == 0:
                return 0
            return random.randrange(0, opp_size)
        if action == 3 or action == 4:
            if opp_size == 0 or player_size == 0:
                return 0, 0
            return random.randrange(0, player_size), random.randrange(0,  opp_size)
        return 0

    def pass_state(self, state, hand, reward, done):
        pass

    def clear_memory(self):
        pass