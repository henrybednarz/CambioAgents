class Agent:
    def __init__(self, player: int):
        self.player_id = player

    def prompt_action(self, known_hands, hand, game_state, discard_pile, turn_count):
        return 0

    def prompt_callback(self, state, action):
        if action == 4 or action == 5:
            return 0, 0
        return 0

    def pass_state(self, state, hand, reward, done):
        pass