import numpy as np
import copy
INITIAL_ACTION_SPACE = [0, 1, 2, 3, 4]


class ExpectiAgent:
    def __init__(self, depth=2):
        self.player_id = 0
        self.depth = depth

    # Extracts the numeric value from a card
    def extract_val(self, card):
        try:
            if isinstance(card, tuple) and isinstance(card[1], int):
                return int(card[1])
            elif isinstance(card, int):
                return card
        except:
            pass
        return 7

    # Main decision function: returns the best action to take this turn
    def prompt_action(self, known_hands, hand, game_state, discard, turn_count, valid_actions):
        discard_val = self.extract_val(discard) if discard else -1
        hand_val = self.extract_val(hand)
        known_player_cards = [self.extract_val(card) for card in known_hands[self.player_id]]

        obs = {
            "known_player_cards": known_player_cards,
            "hand": [hand_val],
            "discard": [discard_val],
            "game_state": game_state.value
        }

        best_score = float("-inf")
        best_action = 3

        for action in INITIAL_ACTION_SPACE:
            if not valid_actions[action]:
                continue
            if action == 3 and discard_val != -1 and hand_val > discard_val:
                continue
            if action == 4 and self.evaluate_state(obs) < -12:
                continue

            simulated_obs = self.simulate_action(copy.deepcopy(obs), action)
            score = self.expectiminimax(simulated_obs, self.depth - 1, is_max=False)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # Choose a valid callback action when required (e.g. peek or swap)
    def prompt_callback(self, state, action, valid_actions):
        for i in range(5, 20):
            if valid_actions[i]:
                return i
        return 20

    # Recursive expectiminimax algorithm:
    # simulates future moves and backpropagates scores
    def expectiminimax(self, obs, depth, is_max):
        if depth == 0 or obs["game_state"] == 2:
            return self.evaluate_state(obs)

        best_score = float("-inf") if is_max else float("inf")
        for action in INITIAL_ACTION_SPACE:
            new_obs = self.simulate_action(copy.deepcopy(obs), action)
            score = self.evaluate_state(new_obs) if depth == 1 else self.expectiminimax(new_obs, depth - 1, not is_max)
            if is_max:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)
        return best_score

    # Simulates a new game state after applying the given action
    def simulate_action(self, obs, action):
        if action in [0, 1, 2]:
            card = obs["known_player_cards"][action]
            swap_val = self.extract_val(card)
            new_hand_val = self.extract_val(obs["hand"][0])
            obs["known_player_cards"][action] = new_hand_val
            obs["hand"][0] = swap_val
            obs["discard"][0] = new_hand_val
            obs["hand"][0] = 7
        elif action == 3:
            val = self.extract_val(obs["hand"][0])
            obs["discard"][0] = val
            obs["hand"][0] = 7
        elif action == 4:
            obs["game_state"] = 1
        return obs

    # Function that assigns a score to the current state
    def evaluate_state(self, obs):
        total = 0
        known_count = 0
        hand_val = self.extract_val(obs["hand"][0])
        discard_val = self.extract_val(obs["discard"][0])

        # Score known cards, weighting middle card more
        for i, val in enumerate(obs["known_player_cards"]):
            val = self.extract_val(val)
            if val != -1:
                total += val * (1.0 if i == 1 else 0.9)
                known_count += 1
            else:
                total += 7

        # Add hand value
        total += hand_val

        # Penalize playing a worse card than what's in the discard
        if discard_val != -1 and hand_val > discard_val:
            total += 3

        # Penalize bad or uncertain cambio calls
        if obs["game_state"] == 1:
            if total > 12:
                total += 15
            if known_count < 2:
                total += 10

        return -total
