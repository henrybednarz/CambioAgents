from enum import Enum
import itertools


class CambioState(Enum):
    NOT_CALLED = 0
    CALLED = 1
    LAST_TURN = 2


class CardType(Enum):
    NORMAL = 0
    PEEK_OWN = 1
    PEEK_OTHER = 2
    BLIND_SWAP = 3
    PEEK_AFTER_SWAP = 4
    PEEK_BEFORE_SWAP = 5

def build_deck():
    deck = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (1, 7),
        (1, 8),
        (2, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (4, 13)
    ]
    deck = deck * 2
    return deck

def hash_known_state(state):
    """Returns and int hash of a known state"""
    state = itertools.chain(*state)
    return hash(tuple(state))


def est_hand_value(hand) -> float:
    value = 0.0
    for card in hand:
        value += card[1] if card is not None else 7
    return value


def known_ratio(hand):
    known_cnt = 0
    for card in hand:
        known_cnt += 1 if card is not None else 0
    return known_cnt / len(hand)


def hand_value(hand):
    total = 0
    for card in hand:
        total += card[1] if card is not None else 8
    return total


def est_state_value(known_state):
    player_hand = known_state[0]
    opponent_hand = known_state[1]
    player_value = est_hand_value(player_hand)
    player_known_ratio = known_ratio(player_hand)
    opponent_known_ratio = known_ratio(opponent_hand)
    value_component = max(0.0, 1 - (player_value / 39.0))
    weighted_sum = (0.4 * value_component +
                    0.5 * player_known_ratio +
                    0.1 * opponent_known_ratio)
    scaled_value = -2 + (3 * weighted_sum)
    return max(-2.0, min(1.0, scaled_value))


action_translation = {
    0: "played to card 1",
    1: "played to card 2",
    2: "played to card 3",
    3: "played to center",
    4: "called cambio",
    5: "peeked own card 1",
    6: "peeked own card 2",
    7: "peeked own card 3",
    8: "peeked opp card 1",
    9: "peeked opp card 2",
    10: "peeked opp card 3",
    11: "swap own 1 for opp 1",
    12: "swap own 1 for opp 2",
    13: "swap own 1 for opp 3",
    14: "swap own 2 for opp 1",
    15: "swap own 2 for opp 2",
    16: "swap own 2 for opp 3",
    17: "swap own 3 for opp 1",
    18: "swap own 3 for opp 2",
    19: "swap own 3 for opp 3",
    20: "passed open action"
}


def action_translator(action):
    if type(action) is tuple:
        return str(action[0] + 1) + " for " + str(action[1] + 1)
    return action_translation[action]
