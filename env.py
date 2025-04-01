import numpy as np
from util import build_deck, CambioState
import random
import itertools
from randomAgent import RandomAgent
import time
from qagentclade import CambioQAgent
from qlearnAgent import QAgent


class CambioEnv:
    def __init__(self, Agent1, Agent2, mode='full'):
        super(CambioEnv, self).__init__()
        self.hand = None
        self.turn_count = 0
        self.deck = None
        self.game_state = None
        self.open_action = None
        self.player_turn = None
        self.discard_pile = None
        self.known_hands = None
        self.player_hands = None
        self.HAND_SIZE = 3
        self.PLAYERS = 2
        self.agents = [Agent1, Agent2]
        self.mode = mode

        self.reset()

    def __str__(self):
        return f"""
        ------ Game State ------
        Current Turn: {self.player_turn}
        Player 1 hand: {self.player_hands[0]}
        Player 2 hand: {self.player_hands[1]}
        Open Action: {self.open_action}
        Hand: {self.hand}
        P1 Known: 
            Self: {self.known_hands[0][0]}
            Opp:  {self.known_hands[0][1]}
        P2 Known: 
            Self: {self.known_hands[1][1]}
            Opp:  {self.known_hands[1][0]}
        Discard: {self.discard_pile[-1] if self.discard_pile else None}
        ------------------------"""

    def reset(self):
        """Resets the game env to initial (random) conditions"""
        self.deck = build_deck()
        self.player_hands = [[] for _ in range(self.PLAYERS)]
        self.known_hands = [[[None, None, None] for _ in range(self.PLAYERS)] for _ in range(self.PLAYERS)]
        self.discard_pile = []
        self.player_turn = 0
        self.turn_count = 0
        self.open_action = None
        self.game_state = CambioState.NOT_CALLED

        if self.mode != 'full':
            self.deck  = self.deck * 2
        
        random.shuffle(self.deck)

        for p in range(self.PLAYERS):
            for _ in range(self.HAND_SIZE):
                self.player_hands[p].append(self.draw_card())
            self.known_hands[p][p][1] = self.player_hands[p][1]

        self.hand = self.draw_card()
    
    # High Order Turn Taking Actions
    def action_initial(self, action):
        """Player calls with initial action, 0-2:swap w/ own index, 3: play card, 4:cambio"""
        if action == 3:
            self.play_card()
        elif action == 4:
            self.cambio()
        else:
            self.switch_card(action)

        return self.known_hands[self.player_turn], self.open_action, 

    def action_callback(self, target):
        """Request a callback action from player to determine targetting inputs vary based on open_action:
        action=peek_own 0-2 is card indexes. action=peek_other 0-2 is card index for other player
        action=swap target 0-2 other player index, target2 0-2 own card index"""
        # Handle Bad Input -- 

        if self.open_action == 0: # Do nothing
            return None

        elif self.open_action == 1: # Peek Own
            self.peek_card(target)

        elif self.open_action == 2: # Peek Other
            self.peek_card(target)

        elif self.open_action == 3: # Swap Blind
            target1, target2 = target
            self.swap_card(target1, target2)

        elif self.open_action == 4: # Swap and Look
            target1, target2 = target
            self.swap_card(target1, target2)
            self.peek_card(target2)

        return self.known_hands[self.player_turn], self.open_action


    # Game maintencence Actions
    def next_turn(self):
        self.turn_count += 1
        """Begins next turn, prompts agents to make moves"""
        if self.game_state == CambioState.CALLED:
            self.game_state = CambioState.LAST_TURN
        elif self.game_state == CambioState.LAST_TURN:
            final_vals = self.tally_hands()
            winner = np.argmax(final_vals)
            return False

        self.player_turn = (self.player_turn + 1) % self.PLAYERS
        self.hand = self.draw_card()
        agent = self.agents[self.player_turn]

        action = agent.prompt_action(self.known_hands[self.player_turn], self.hand, self.game_state, self.discard_pile[-1] if self.discard_pile else None, self.turn_count)
        self.action_initial(action)

        callback_action = agent.prompt_callback(self.known_hands[self.player_turn], self.open_action)
        self.action_callback(callback_action)

        self.flip_cards()

        agent.pass_state(self.known_hands[self.player_turn], self.hand, 0, False)
        return True

    def draw_card(self):
        """Returns the top card of the deck or shuffles the discard"""
        player = self.player_turn
        opponent = self.player_turn + 1 % self.PLAYERS
        if len(self.deck) == 0:
            self.deck = self.discard_pile
            self.discard_pile = []
            random.shuffle(self.deck)
        return self.deck.pop()

    def flip_cards(self):
        """Removes cards that match the top discard and replaces them"""
        if not self.discard_pile:  # Check if discard pile is empty
            return

        top_discard = self.discard_pile[-1][1]  # Value of the top card

        # Check each player's hand for matches
        for p in range(self.PLAYERS):
            for card_idx, card in enumerate(self.player_hands[p]):
                # Check if card value matches top discard
                if card[1] == top_discard:
                    # Count how many players know about this card
                    knowing_players = 0
                    for player in range(self.PLAYERS):
                        if self.known_hands[player][p][card_idx] is not None:
                            knowing_players += 1

                    # Case 1: Only one player knows the card
                    if knowing_players == 1:
                        knowing_player = None
                        for player in range(self.PLAYERS):
                            if self.known_hands[player][p][card_idx] is not None:
                                knowing_player = player
                                break

                        # Remove and replace the card
                        self.remove_card(p, card_idx, knowing_player)

                    # Case 2: Both players know the card
                    elif knowing_players > 1:
                        # Randomly choose which player gets to remove the card
                        removing_player = random.randint(0, self.PLAYERS - 1)

                        # Remove and replace the card
                        self.remove_card(p, card_idx, removing_player)

    def remove_card(self, card_owner, card_idx, removing_player):
        """Helper method to remove a card and replace it with a new one"""
        # If player is removing opponent's card, give them their highest known valued card
        if card_owner != removing_player:
            opponent = card_owner

            highest_val = -1
            highest_idx = -1
            unknown_idx = -1

            for idx in range(self.HAND_SIZE):
                card_info = self.known_hands[removing_player][opponent][idx]

                if card_info is None and unknown_idx == -1:
                    unknown_idx = idx

                elif card_info is not None and card_info[1] > highest_val:
                    highest_val = card_info[1]
                    highest_idx = idx

            replace_idx = highest_idx if highest_idx != -1 else unknown_idx

            if replace_idx != -1:
                new_card = self.draw_card()
                self.player_hands[opponent][replace_idx] = new_card
                for p in range(self.PLAYERS):
                    self.known_hands[p][opponent].pop(replace_idx)

        self.player_hands[card_owner].pop(card_idx)

        for p in range(self.PLAYERS):
            self.known_hands[p][card_owner].pop(card_idx)

    def tally_hands(self):
        """Returns total values of each player's hand"""
        hand_values = [0 for _ in range(self.PLAYERS)]
        for p in range(self.PLAYERS):
            for card in self.player_hands[p]:
                hand_values[p] += card[1]
        
        return hand_values

    # Player Actions
    def play_card(self):
        """Plays a card, sets the current open_action for a callback"""
        self.open_action = self.hand[0]
        self.discard_pile.append(self.hand)
        self.hand = None
        return 

    def switch_card(self, target_index):
        """Switches out card with hand, then plays your hand"""
        player = self.player_turn
        opponent = (self.player_turn + 1) % self.PLAYERS

        if len(self.player_hands[player]) <= target_index:
            raise ValueError("Invalid Target Index")

        # Swap cards
        self.player_hands[player][target_index], self.hand = self.hand, self.player_hands[player][target_index]
        # Update known states
        self.known_hands[player][player][target_index] = self.player_hands[player][target_index]
        self.known_hands[opponent][player][target_index] = None
        # Play the card
        self.play_card()
        return

    def peek_card(self, target_index):
        """Lets a player look at a card"""
        player = self.player_turn
        if len(self.player_hands[player]) <= target_index:
            raise ValueError("Invalid Target Index")
        # Update known state
        self.known_hands[player][player][target_index] = self.player_hands[player][target_index]
        self.open_action = None

    def swap_card(self, target_index1, target_index2):
        """Performs a card swap with another player"""
        player = self.player_turn
        opponent = (self.player_turn + 1) % self.PLAYERS

        if len(self.player_hands[player]) <= target_index1 or len(self.player_hands[opponent]) <= target_index2:
            raise ValueError("Invalid Target Index")

        # Swap Cards
        self.player_hands[player][target_index1], self.player_hands[opponent][target_index2] = self.player_hands[opponent][target_index2], self.player_hands[player][target_index1]
        # update known states
        self.known_hands[player][player][target_index1], self.known_hands[player][opponent][target_index2] = self.known_hands[player][opponent][target_index2], self.known_hands[player][player][target_index1]
        self.known_hands[opponent][player][target_index1], self.known_hands[opponent][opponent][target_index2] = self.known_hands[opponent][opponent][target_index2], self.known_hands[opponent][player][target_index1]
        self.open_action = None

    def cambio(self):
        if self.game_state != CambioState.NOT_CALLED:
            return False
        self.game_state = CambioState.CALLED
        return True


start = time.time()
t = CambioEnv(QAgent(0), QAgent(1), "half")

t.reset()
print(t)
flag = True
while flag:
    flag = t.next_turn()
    print(t)

end = time.time()
