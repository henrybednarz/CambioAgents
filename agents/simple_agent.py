import random
from util import CambioState


class SimpleHeuristicAgent:
    def __init__(self):
        pass
    def prompt_action(self, state, hand, game_state, top_discard, turn_count, valid_actions=None):
        known_cards = state[1]

        # If we know all our cards, try to swap highest
        if all(card is not None for card in known_cards):
            values = [card[1] for card in known_cards]
            max_idx = values.index(max(values))
            if hand[1] < values[max_idx]:
                return max_idx
        else:
            # Replace unknown cards (default to first unknown)
            for idx, card in enumerate(known_cards):
                if card is None:
                    return idx

        # If hand is low, call cambio (simple trigger rule)
        if hand[1] < 7 and game_state is CambioState.NOT_CALLED:
            return 4
        return 3


    def prompt_callback(self, state, open_action, valid_actions):
        if open_action == 1:
            return random.choice(valid_actions)
        elif open_action == 2:
            return random.choice(valid_actions)
        elif open_action in [3, 4]:
            return random.choice(valid_actions)
        return 20

