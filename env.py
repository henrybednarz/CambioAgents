import numpy as np
from gym.util import build_deck, CambioState, calculate_state_value
import random


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
            self.deck = self.deck * 2
        
        random.shuffle(self.deck)

        self.agents.append(self.agents.pop(0))
        for a in self.agents:
            a.clear_memory()
            a.change_num()

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
            if len(self.player_hands[self.player_turn]) == 0:
                return self.known_hands[self.player_turn], self.open_action

            self.peek_card(target, self.player_turn)

        elif self.open_action == 2: # Peek Other
            if len(self.player_hands[(self.player_turn + 1) % 2]) == 0:
                return self.known_hands[self.player_turn], self.open_action

            self.peek_card(target, (self.player_turn + 1) % 2)

        elif self.open_action == 3: # Swap Blind
            if len(self.player_hands[self.player_turn]) == 0 or len(self.player_hands[(self.player_turn + 1) % 2]) == 0:
                return self.known_hands[self.player_turn], self.open_action
            target1, target2 = target
            self.swap_card(target1, target2)

        elif self.open_action == 4: # Swap and Look
            if len(self.player_hands[self.player_turn]) == 0 or len(self.player_hands[(self.player_turn + 1) % 2]) == 0:
                return self.known_hands[self.player_turn], self.open_action
            target1, target2 = target
            self.swap_card(target1, target2)
            self.peek_card(target2, self.player_turn)

        return self.known_hands[self.player_turn], self.open_action

    # Game Order Actions
    def next_turn(self):
        self.turn_count += 1
        """Begins next turn, prompts agents to make moves"""
        if self.game_state == CambioState.CALLED:
            self.game_state = CambioState.LAST_TURN
        elif self.game_state == CambioState.LAST_TURN:
            final_vals = self.tally_hands()
            winner = np.argmax(final_vals)
            for i, agent in enumerate(self.agents):
                agent.pass_state(self.known_hands[i], self.hand, 10000 if i == winner else 0, True)
            return False

        self.player_turn = (self.player_turn + 1) % self.PLAYERS
        agent = self.agents[self.player_turn]
        self.hand = self.draw_card()

        previous_value = calculate_state_value(self.known_hands[self.player_turn], self.player_turn)

        # Get action and check validity
        valid_action = False
        invalid_attempts = 0

        while not valid_action:
            action = agent.prompt_action(self.known_hands[self.player_turn], self.hand,
                                         self.game_state,
                                         self.discard_pile[-1] if self.discard_pile else None,
                                         self.turn_count)

            # Validate action
            valid_action = self.validate_action(action)

            if not valid_action:
                # Penalize agent for invalid move
                agent.pass_state(self.known_hands[self.player_turn], self.hand, -10, False)
                invalid_attempts += 1

                # Prevent infinite loops if agent keeps making invalid moves
                if invalid_attempts > 10:
                    # Force a valid action (play card)
                    action = 3
                    valid_action = True

        # Execute the valid action
        self.action_initial(action)

        # Handle callback action, also with validation
        valid_callback = False
        invalid_callback_attempts = 0

        while not valid_callback and self.open_action is not None and self.open_action > 0:
            callback_action = agent.prompt_callback(self.known_hands[self.player_turn], self.open_action)
            valid_callback = self.validate_callback(callback_action)

            if not valid_callback:
                # Penalize agent for invalid callback
                agent.pass_state(self.known_hands[self.player_turn], self.hand, -5, False)
                invalid_callback_attempts += 1

                # Prevent infinite loops
                if invalid_callback_attempts > 10:
                    # Force a valid callback (typically index 0)
                    if self.open_action == 1 or self.open_action == 2:
                        callback_action = 0
                    elif self.open_action == 3 or self.open_action == 4:
                        callback_action = (0, 0)
                    valid_callback = True

        if self.open_action is not None and self.open_action > 0:
            self.action_callback(callback_action)

        self.flip_cards()

        reward = calculate_state_value(self.known_hands[self.player_turn], player_id=self.player_turn) - previous_value

        # Bonus reward for completing a valid move
        reward += 1

        agent.pass_state(self.known_hands[self.player_turn], self.hand, reward, False)
        return True

    def validate_action(self, action):
        """Validates if the action is valid in the current game state"""
        player = self.player_turn

        # Action 3 (play card) and 4 (cambio) are always valid
        if action == 3 or action == 4:
            return True

        # For actions 0-2 (swap with own card), check if index is valid
        if 0 <= action < len(self.player_hands[player]):
            return True

        return False

    def validate_callback(self, callback_action):
        """Validates if the callback action is valid"""
        player = self.player_turn
        opponent = (player + 1) % self.PLAYERS

        if self.open_action == 0:
            # No callback needed
            return True

        elif self.open_action == 1:  # Peek own
            # Check if target index is valid
            return 0 <= callback_action < len(self.player_hands[player])

        elif self.open_action == 2:  # Peek other
            # Check if target index is valid
            return 0 <= callback_action < len(self.player_hands[opponent])

        elif self.open_action == 3 or self.open_action == 4:  # Swap actions
            target1, target2 = callback_action
            # Check if both indices are valid
            return (0 <= target1 < len(self.player_hands[player]) and
                    0 <= target2 < len(self.player_hands[opponent]))

        return False

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

            for idx in range(len(self.player_hands[card_owner]) - 1):
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
                    self.known_hands[p][opponent][card_idx] = None

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
            return False # Indicate action failed

        # Swap cards
        self.player_hands[player][target_index], self.hand = self.hand, self.player_hands[player][target_index]
        # Update known states
        self.known_hands[player][player][target_index] = self.player_hands[player][target_index]
        self.known_hands[opponent][player][target_index] = None

        # Play the card
        self.play_card()
        return True

    def peek_card(self, target_index, player):
        """Lets a player look at a card"""
        if len(self.player_hands[player]) <= target_index:
            return False
        # Update known state
        self.known_hands[player][player][target_index] = self.player_hands[player][target_index]
        self.open_action = None
        return True

    def swap_card(self, target_index1, target_index2):
        """Performs a card swap with another player"""
        player = self.player_turn
        opponent = (self.player_turn + 1) % self.PLAYERS

        if len(self.player_hands[player]) <= target_index1 or len(self.player_hands[opponent]) <= target_index2:
            return False

        # Swap Cards
        self.player_hands[player][target_index1], self.player_hands[opponent][target_index2] = self.player_hands[opponent][target_index2], self.player_hands[player][target_index1]
        # update known states
        self.known_hands[player][player][target_index1], self.known_hands[player][opponent][target_index2] = self.known_hands[player][opponent][target_index2], self.known_hands[player][player][target_index1]
        self.known_hands[opponent][player][target_index1], self.known_hands[opponent][opponent][target_index2] = self.known_hands[opponent][opponent][target_index2], self.known_hands[opponent][player][target_index1]
        self.open_action = None
        return True

    def cambio(self):
        """Call cambio to end the game"""
        if self.game_state != CambioState.NOT_CALLED:
            return False
        self.game_state = CambioState.CALLED
        return True

    def calculate_reward(self, player, previous_state):
        pass


# start = time.time()
# #t = CambioEnv(RandomAgent(0), DQNAgent(1, policy_path="DQNweights.weights.h5"))
# t = CambioEnv(RandomAgent(0), RandomAgent(1))
# won = 0
# total = 0
# for i in range(100000):
#     t.reset()
#     flag = True
#     turn_cnt = 0
#     while flag:
#         turn_cnt += 1
#         flag = t.next_turn()
#     if 1 == np.argmax(t.tally_hands()):
#         won += 1
#     total += 1
#
# # t.agents[1].save_model("DQNweights.weights.h5")
# end = time.time()
# print(end-start)
# print(f"Qagent won {won/total * 100}%")