class Agent:
    def __init__(self):
        pass

    def prompt_action(self, known_hands, hand, game_state, discard_pile, turn_count, valid_actions):
        return 0

    def prompt_callback(self, state, action, valid_actions):
        if action == 4 or action == 5:
            return 0, 0
        return 0