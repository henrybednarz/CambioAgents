from agent import Agent
import numpy as np
from util import CambioState


class QAgent(Agent):
    def __init__(self, num, policy_path=None):
        super().__init__(num)
        self.STATE_SIZE = (100, 100)
        self.training = False
        self.iaction_table = {}
        self.cbaction_table = {}

    def prompt_action(self, known_hands, hand, game_state, discard, turn_count):
        state = self._preprocess_action_state(known_hands, hand, game_state, discard, turn_count)
        # run state through nn
        return 0

    def prompt_callback(self, state, action):
        if action == 4 or action == 5:
            return 0, 0
        return 0

    def _preprocess_action_state(self, known_hands, hand, game_state, discard, turn_count):
        """Convert game state to a normalized feature vector"""
        state = []

        # Add cards to state, -1 if unknown
        for card in known_hands[self.player_id]:
            state.append(card[1]/13.0 if card else -1)
        for card in known_hands[(self.player_id + 1) % 2]:
            state.append(card[1]/13.0 if card else -1)

        # Current hand
        state.append(hand[1] / 13.0)  # Normalized card value

        # Game state features
        state.append(1.0 if game_state == CambioState.CALLED or game_state == CambioState.LAST_TURN else 0.0)

        # Estimated relative score
        my_score = sum(card[1] for card in known_hands[self.player_id] if card is not None)
        opp_score = sum(card[1] for card in known_hands[(self.player_id + 1) % 2] if card is not None)
        # For unknown cards, assume middle value
        unknown_count_me = sum(1 for card in known_hands[self.player_id] if card is None)
        unknown_count_opp = sum(1 for card in known_hands[(self.player_id + 1) % 2] if card is None)
        my_score += unknown_count_me * 7.0  # Assume middle value of 7 for unknown cards
        opp_score += unknown_count_opp * 7.0

        relative_score = (opp_score - my_score) / 21.0  # Normalize by max possible hand value
        state.append(relative_score)

        # Turn count (normalized)
        state.append(min(turn_count / 30.0, 1.0))  # Assuming max 30 turns

        # Top discard card
        if discard:
            state.append(discard[1] / 13.0)
        else:
            state.append(0.0)

        return np.array(state)

    def _preprocess_callback_state(self, known_hands, open_action, game_state, discard, turn_count):
        pass