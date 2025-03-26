import gym
from gym import spaces
import numpy as np
from util import build_deck
import random

class Card():
    def __init__(self, type, value):
        self.value = value
        self.type = type
        self.visible = False
        self.locked = False

    def peek(self):
        self.visible = True

    def lock(self):
        self.locked = True

    def can_swap(self):
        return not self.locked

    def clone(self):
        return Card(self.type, self.value)

class Deck:
    def __init__(self, mode='full'):
        self.cards = build_deck()
        self.discard_pile = []
        self.mode = mode

        self.reset

class CambioEnv(gym.Env):
    def __init__(self, mode='full'):
        super(CambioEnv, self).__init__()
        self.HAND_SIZE = 3
        self.PLAYERS = 2
        self.player_hands = [[] for _ in range(self.PLAYERS)]
        self.known_hands = [[[] for _ in range(self.PLAYERS)] for _ in range(self.PLAYERS)]
        self.mode = mode

        self.reset()

    def reset(self):
        self.deck = build_deck()
        self.discard_pile = []
        self.player_turn = 0
        self.open_action = None
        if self.mode != 'full':
            self.deck  = self.deck * 2
        
        random.shuffle(self.deck)

        for p in range(self.PLAYERS):
            for _ in range(self.HAND_SIZE):
                self.player_hands[p].append(self.draw_card())
                self.known_hands[p].append(self.player_hands[p][1])

        self.hand = self.draw_card()
    
    # High Order Turn Taking Actions
    def step(self, action):
        """Begins next player's turn"""
        # Prompt for action_initial(action)
        # Prompt for action_callback()
        return self.state, self.hand, self.reward, self.done

    def action_initial(self, action):
        """Player calls with initial action, 0:play, 1-3:swap w/ own index , 4:cambio"""
        if action == 0:
            self.play_card(self.hand)
        elif action == 4:
            self.cambio()
        else:
            self.swap_card(action-1)

        return self.state, self.open_action, 

    def action_callback(self, target, *target2):
        """Request a callback action from player to determine targetting inputs vary based on open_action:
        action=peek_own 0-2 is card indexes. action=peek_other 0-2 is card index for other player
        action=swap target 0-2 other player index, target2 0-2 own card index"""
        if self.open_action == 0: # Do nothing
            return None

        elif self.open_action == 1: # Peek Own
            self.peek_card(target)

        elif self.open_action == 2: # Peek Other
            self.peek_card(target)

        elif self.open_action == 3: # Swap Blind
            self.swap_card(target, target2)

        elif self.open_action == 4: # Swap and Look
            self.swap_card(target, target2)
            self.peek_card(target)

        return self.state, self.open_action

    # Game maintencance Actions
    def next_turn(self):
        self.player_turn = (self.player_turn + 1) % self.PLAYERS
        self.hand = self.draw_card()

    def draw_card(self):
        if len(self.deck) == 0:
            self.deck = self.discard_pile
            self.discard_pile = []
            random.shuffle(self.deck)
        return self.deck.pop()

    def flip_cards():
        pass

    # Player Actions
    def play_card(self):
        """Plays a card, sets the current open_action for a callback"""
        self.discard_pile.append(self.hand)
        self.hand = None
        action = card[0]

        self.open_action = action
        return 
        

    def switch_card(self, targetIndex):
        """Switches out card with hand, then plays your hand"""
        player = self.player_turn
        opponent = self.player_turn + 1 % self.PLAYERS
        # Swap cards
        self.player_hands[player][targetIndex], self.hand = self.hand, self.player_hands[player][targetIndex]
        # Update known states
        self.known_hands[player][player][targetIndex] = self.hand
        self.known_hands[opponent][player][targetIndex] = None
        # Play the card
        self.play_card(self.hand)
        return


    def peek_card(self, targetIndex):
        """Lets a player look at a card"""
        player = self.player_turn
        # Update known state
        known_hands[player][player][targetIndex] = self.player_hands[player][targetIndex]
        self.open_action = None
    
    def swap_card(self, targetIndex1, targetIndex2):
        """Performs a card swap with another player"""
        player = self.player_turn
        opponent = self.player_turn + 1 % self.PLAYERS
        # Swap Cards
        self.player_hands[player][targetIndex1], self.player_hands[opponent][targetIndex2] = self.player_hands[player][targetIndex2], self.player_hands[opponent][targetIndex1]
        # update known states
        self.known_hands[player][player][targetIndex1] = self.known_hands[player][opponent][targetIndex2]
        self.known_hands[opponent][opponent][targetIndex2] = self.known_hands[opponent][player][targetIndex1]
        self.open_action = None

    def cambio(self):
        pass


t = CambioEnv("half")