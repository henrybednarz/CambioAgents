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


def value_state(known_states, player_hands, p):
    value = sum(card[1] for card in player_hands[p])
    self_known = sum(1 if card else 0 for card in known_states[p][p])
    opp_known = sum(1 if card else 0 for card in known_states[p][(p + 1) % 2])
    return (value, self_known, opp_known)


def known_difference(self_known, opp_known):
    self_value = sum(card[1] if card else 15 for card in self_known)
    opp_value = sum(card[1] if card else 15 for card in opp_known)
    return self_value - opp_value


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
