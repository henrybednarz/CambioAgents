from enum import Enum

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
    ]
    deck = deck * 2
    deck.append((5, -1))
    deck.append((5, 13))
    return deck