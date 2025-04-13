import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from MCTS import MCTSAgent

# Assuming these are imported from the original code
from util import build_deck, CambioState, value_state, action_translator, known_difference


class CambioGymEnv(gym.Env):
    """
    Cambio card game environment following the OpenAI Gym interface.
    This environment allows an agent to play against an opponent that moves automatically.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, opponent_agent, mode='full', render_mode='human'):
        super(CambioGymEnv, self).__init__()

        # Game constants
        self.HAND_SIZE = 3
        self.PLAYERS = 2
        self.mode = mode
        self.render_mode = render_mode

        # Store opponent agent
        self.opponent = opponent_agent

        # Initialize empty game state
        self.deck = None
        self.player_hands = None
        self.known_hands = None
        self.discard_pile = None
        self.turn_count = 0
        self.player_turn = 0  # 0 for RL agent, 1 for opponent
        self.hand = None
        self.open_action = None
        self.game_state = None
        self.game_over = False
        self.reward = 0
        # Define action and observation spaces

        # Action space:
        # 0-2: swap with own card at index
        # 3: play card
        # 4: cambio
        # For callbacks:
        # 5-7: peek own card (0-2)
        # 8-10: peek opponent card (0-2)
        # 11-19: swap cards (player_idx * 3 + opponent_idx)
        self.action_space = spaces.Discrete(21)

        # Observation space contains:
        # - Known player cards (3 cards, each can be 0-13 or None=-1)
        # - Known opponent cards (3 cards, each can be 0-13 or None=-1)
        # - Current hand (1 card, 0-13)
        # - Top of discard pile (0-13 or None=-1)
        # - Game state (0: not called, 1: called, 2: last turn)
        # - Open action (0-4 or None=-1)
        # - Turn count
        self.observation_space = spaces.Dict({
            'known_player_cards': spaces.Box(low=-1, high=13, shape=(3,), dtype=np.int8),
            'known_opponent_cards': spaces.Box(low=-1, high=13, shape=(3,), dtype=np.int8),
            'hand': spaces.Box(low=-1, high=13, shape=(1,), dtype=np.int8),
            'discard': spaces.Box(low=-1, high=13, shape=(1,), dtype=np.int8),
            'game_state': spaces.Discrete(3),
            'open_action': spaces.Box(low=-1, high=4, shape=(1,), dtype=np.int8),
            'turn_count': spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
            'hand_type': spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32),
            'knowledge_flags': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            'estimated_total': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'deck_size': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'cards_remaining': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'card_advantage': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'last_played_card': spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32),
            'match_potential': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        # Reset game components
        self.deck = build_deck()
        self.player_hands = [[] for _ in range(self.PLAYERS)]
        self.known_hands = [[[None, None, None] for _ in range(self.PLAYERS)] for _ in range(self.PLAYERS)]
        self.discard_pile = []
        self.player_turn = 0  # Always start with the RL agent
        self.turn_count = 0
        self.open_action = None
        self.game_state = CambioState.NOT_CALLED
        self.game_over = False

        if self.mode != 'full':
            self.deck = self.deck * 2

        random.shuffle(self.deck)

        # Deal initial cards
        for p in range(self.PLAYERS):
            for _ in range(self.HAND_SIZE):
                self.player_hands[p].append(self.draw_card())
            # Each player knows their middle card
            self.known_hands[p][p][1] = self.player_hands[p][1]

        self.hand = self.draw_card()

        # Return initial observation
        return self._get_observation(), {'valid_actions': self._get_valid_actions()}

    def _get_valid_actions(self):
        """Return a boolean mask of valid actions based on the current game state."""
        valid_actions = np.zeros(self.action_space.n, dtype=bool)
        # Initial actions (0-4) are always valid when it's the agent's turn and there's no pending callback
        if self.player_turn == 0 and self.open_action is None or self.open_action == 0:
            if self.hand is not None:
                for i in range(min(len(self.player_hands[0]), 3)):
                    valid_actions[i] = True

            # Action 3: play card (if we have a card in hand)
            if self.hand is not None:
                valid_actions[3] = True

            # Action 4: cambio (if game state allows it)
            if self.game_state == CambioState.NOT_CALLED and self.hand is not None:
                valid_actions[4] = True

        # Callback actions are only valid when there's a pending callback
        if self.open_action is not None:
            if self.open_action == 1:  # Peek own card
                # Actions 5-7: peek own card at index
                for i in range(min(len(self.player_hands[self.player_turn]), 3)):
                    valid_actions[i + 5] = True

            elif self.open_action == 2:  # Peek opponent card
                # Actions 8-10: peek opponent card at index
                opponent = (self.player_turn + 1) % 2
                for i in range(min(len(self.player_hands[opponent]), 3)):
                    valid_actions[i + 8] = True

            elif self.open_action in [3, 4]:  # Swap cards
                valid_actions[20] = True
                # Actions 11-19: swap cards (player_idx * 3 + opponent_idx)
                player = self.player_turn
                opponent = (self.player_turn + 1) % 2

                for p_idx in range(min(len(self.player_hands[player]), 3)):
                    for o_idx in range(min(len(self.player_hands[opponent]), 3)):
                        action_idx = 11 + (p_idx * 3) + o_idx
                        valid_actions[action_idx] = True
        return valid_actions

    def step(self, action: int):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {'final_values': self.tally_hands()}

        # Set player turn to agent's turn (player 0)
        self.player_turn = 0
        prev_value, prev_self_known, prev_opp_known = value_state(self.known_hands, self.player_hands, self.player_turn)

        # Process the player's action
        if action <= 4:  # Initial actions
            # Execute the initial action
            self._action_initial(action)
        elif action <= 7:  # Peek own card
            if self.open_action == 1:  # Only valid if open_action is peek_own
                target_index = action - 5
                self._action_callback(target_index)
        elif action <= 10:  # Peek opponent card
            if self.open_action == 2:  # Only valid if open_action is peek_other
                target_index = action - 8
                self._action_callback(target_index)
        elif action <= 19:  # Swap cards
            if self.open_action == 3 or self.open_action == 4:  # Only valid for swap actions
                # Convert action to player and opponent indices
                action_idx = action - 11
                player_idx = action_idx // 3
                opponent_idx = action_idx % 3
                self._action_callback((player_idx, opponent_idx))
        elif action == 20:
            if self.open_action == 3 or self.open_action == 4:
                self.open_action = None
        else:
            print("invalid action taken")
            return self._get_observation(), -5, False, False, {'invalid_action': True,
                                                               'valid_actions': self._get_valid_actions()}

        self._flip_cards2()

        # Calculate immediate reward for agent based on state change
        cur_value, cur_self_known, cur_opp_known = value_state(self.known_hands, self.player_hands, self.player_turn)
        reward = self.calculate_reward2(cur_value - prev_value,
                                        cur_self_known - prev_self_known,
                                        cur_opp_known - prev_opp_known,
                                        action)

        # Check for game end after player's action
        player_ended_game = False
        if self.game_state == CambioState.CALLED:
            self.game_state = CambioState.LAST_TURN
        elif self.game_state == CambioState.LAST_TURN:
            self.game_over = True
            player_ended_game = True

        # Create info dictionary with the current game state
        info = {
            'valid_actions': self._get_valid_actions(),
            'player_action_complete': True,
            'opponent_to_play': not self.game_over and self.open_action is None,
        }

        if self.game_over:
            info['final_values'] = self.tally_hands()

        # If the game is over after player's action, return immediately
        if player_ended_game:
            return self._get_observation(), reward, True, False, info

        # Draw a new card for the player
        self.hand = self.draw_card()

        # Return the state after player's action but BEFORE opponent plays
        # Note: Game is not over yet, but the player's action is complete
        return self._get_observation(), reward, False, False, info

    def process_opponent_turn(self):
        """
        New method to handle the opponent's turn separately.
        This should be called by the environment wrapper or main game loop
        after each player step, but before the next step is called.
        """
        if self.game_over:
            return {'opponent_played': False, 'game_over': True}

        # Set player turn to opponent
        self.player_turn = 1

        if self.hand is not None:
            self.discard_pile.append(self.hand)
        self.hand = self.draw_card()

        # Have the opponent take their action
        agent = self.opponent
        if type(agent) is MCTSAgent:
            initial_action = agent.prompt_action(self)
        else:
            initial_action = agent.prompt_action(
                self.known_hands[self.player_turn],
                self.hand,
                self.game_state,
                self.discard_pile[-1] if self.discard_pile else None,
                self.turn_count,
            )

        self._action_initial(initial_action)

        # Handle callback action if needed
        if self.open_action is not None and self.open_action > 0:
            if type(agent) is MCTSAgent:
                callback_action = agent.prompt_action(self)
            else:
                callback_action = agent.prompt_callback(self.known_hands[self.player_turn], self.open_action)
            self._action_callback(callback_action)

        # Process card flipping after opponent's action
        try:
            self._flip_cards2()
        except:
            self.render()

        # Check if game is over after opponent's turn
        if self.game_state == CambioState.CALLED:
            self.game_state = CambioState.LAST_TURN
        elif self.game_state == CambioState.LAST_TURN:
            self.game_over = True

        self.turn_count += 1
        self.player_turn = 0  # Set back to player's turn

        if self.hand is not None:
            self.discard_pile.append(self.hand)
        self.hand = self.draw_card()

        return {
            'opponent_played': True,
            'game_over': self.game_over,
            'valid_actions': self._get_valid_actions()
        }

    def calculate_reward(self, known_states, prev_diff):
        return min((known_difference(known_states[0][0], known_states[0][1]) - prev_diff) / 20, 0.5)
    def calculate_reward2(self, value_change, self_known_change, opp_known_change, action):
        return 0

    def _opponent_turn(self):
        """Handle the opponent's turn automatically."""
        if self.game_over:
            return True

        # Set player turn to opponent
        self.player_turn = 1

        # Draw a card for the opponent
        if self.hand is not None:
            self.discard_pile.append(self.hand)
        self.hand = self.draw_card()

        agent = self.opponent
        if type(agent) is MCTSAgent:
            initial_action = agent.prompt_action(self)
        else:
            initial_action = agent.prompt_action(
                self.known_hands[self.player_turn],
                self.hand,
                self.game_state,
                self.discard_pile[-1] if self.discard_pile else None,
                self.turn_count,
            )

        # print("Opponent Turn")
        # self.render()
        # print("Action Selected:", action_translator(initial_action))

        # Execute the action
        self._action_initial(initial_action)

        # self.render()
        # Handle callback action if needed
        if self.open_action is not None and self.open_action > 0:
            # print("Callback needed")
            if type(agent) is MCTSAgent:
                callback_action = agent.prompt_action(self)
            else:
                callback_action = agent.prompt_callback(self.known_hands[self.player_turn], self.open_action)
            # print("Callback action selected:", action_translator(callback_action))
            self._action_callback(callback_action)
            # self.render()
        # print("checking for flips")
        # Process card flipping

        self._flip_cards2()

        # Check if game is over after opponent's turn
        if self.game_state == CambioState.CALLED:
            self.game_state = CambioState.LAST_TURN
        elif self.game_state == CambioState.LAST_TURN:
            self.game_over = True

        self.turn_count += 1
        self.player_turn = 0
        # print("opponent turn over")
        if self.game_over:
            return True

    def _action_initial(self, action):
        """Player takes an initial action."""
        if action == 3:
            self._play_card()
        elif action == 4:
            self.discard_pile.append(self.hand)
            self.hand = None
            self._cambio()
        else:
            self._switch_card(action)

    def _action_callback(self, target):
        """Process callback action based on the open_action."""
        if self.open_action == 0:
            self.open_action = None
            return

        elif self.open_action == 1:  # Peek own
            if len(self.player_hands[self.player_turn]) > 0:
                self._peek_card(target, self.player_turn)

        elif self.open_action == 2:  # Peek other
            opponent = (self.player_turn + 1) % 2
            if len(self.player_hands[opponent]) > 0:
                self._peek_card(target, opponent)

        elif self.open_action == 3:  # Swap blind
            if (len(self.player_hands[self.player_turn]) > 0 and
                    len(self.player_hands[(self.player_turn + 1) % 2]) > 0):
                target1, target2 = target
                self._swap_card(target1, target2)

        elif self.open_action == 4:  # Swap and look
            if (len(self.player_hands[self.player_turn]) > 0 and
                    len(self.player_hands[(self.player_turn + 1) % 2]) > 0):
                target1, target2 = target
                self._swap_card(target1, target2)
                self._peek_card(target2, self.player_turn)

    def _get_observation(self):
        """Enhanced observation with additional features for better state representation"""
        # Existing features
        known_player_cards = np.full(3, 0, dtype=np.float32)
        known_opponent_cards = np.full(3, 0, dtype=np.float32)

        for i in range(min(len(self.known_hands[0][0]), 3)):
            if self.known_hands[0][0][i] is not None:
                known_player_cards[i] = self.known_hands[0][0][i][1] / 13.0

        for i in range(min(len(self.known_hands[0][1]), 3)):
            if self.known_hands[0][1][i] is not None:
                known_opponent_cards[i] = self.known_hands[0][1][i][1] / 13.0

        hand_value = np.array([self.hand[1] / 13.0 if self.hand else -0.1], dtype=np.float32)
        hand_type = np.zeros(14, dtype=np.float32)  # One-hot encoding of card type
        if self.hand:
            hand_type[self.hand[0]] = 1.0  # Set the corresponding card type to 1

        # Discard pile information
        discard_value = np.array([self.discard_pile[-1][1] / 13.0 if self.discard_pile else 0], dtype=np.float32)

        knowledge_flags = np.zeros(3, dtype=np.float32)
        for i in range(min(len(self.known_hands[0][0]), 3)):
            knowledge_flags[i] = 1.0 if self.known_hands[0][0][i] is not None else 0.0

        known_value_sum = sum(self.known_hands[0][0][i][1] if self.known_hands[0][0][i] is not None else 0
                              for i in range(min(len(self.known_hands[0][0]), 3)))
        unknown_count = 3 - sum(knowledge_flags)
        expected_unknown_value = unknown_count * 5.0
        estimated_total = (known_value_sum + expected_unknown_value) / 30.0

        deck_size = np.array([len(self.deck) / 52.0], dtype=np.float32)

        cards_remaining = np.array([
            len(self.player_hands[0]) / 3.0,  # Normalize by starting hand size
            len(self.player_hands[1]) / 3.0
        ], dtype=np.float32)

        card_advantage = np.array([len(self.player_hands[1]) - len(self.player_hands[0])], dtype=np.float32) / 3.0

        game_state_one_hot = np.zeros(3, dtype=np.float32)
        if self.game_state == CambioState.NOT_CALLED:
            game_state_one_hot[0] = 1.0
        elif self.game_state == CambioState.CALLED:
            game_state_one_hot[1] = 1.0
        elif self.game_state == CambioState.LAST_TURN:
            game_state_one_hot[2] = 1.0

        open_action_one_hot = np.zeros(6, dtype=np.float32)
        if self.open_action is None:
            open_action_one_hot[5] = 1.0
        else:
            open_action_one_hot[self.open_action] = 1.0

        turn_count = np.array([min(1.0, self.turn_count / 50.0)], dtype=np.float32)

        last_played_card = np.zeros(14, dtype=np.float32)
        if len(self.discard_pile) >= 2:
            last_card_type = self.discard_pile[-2][0]
            last_played_card[last_card_type] = 1.0

        # Match potential - are there any known cards that match the top discard?
        match_potential = np.array([0.0], dtype=np.float32)
        if self.discard_pile:
            top_discard_value = self.discard_pile[-1][1]
            for i in range(min(len(self.known_hands[0][0]), 3)):
                if self.known_hands[0][0][i] is not None and self.known_hands[0][0][i][1] == top_discard_value:
                    match_potential[0] = 1.0
                    break

        # Concatenate all components into a single array
        observation = np.concatenate([
            known_player_cards,  # 3 values
            known_opponent_cards,  # 3 values
            hand_value,  # 1 value
            hand_type,  # 14 values (one-hot card type)
            discard_value,  # 1 value
            knowledge_flags,  # 3 values
            np.array([estimated_total]),  # 1 value
            deck_size,  # 1 value
            cards_remaining,  # 2 values
            card_advantage,  # 1 value
            game_state_one_hot,  # 3 values
            open_action_one_hot,  # 6 values
            turn_count,  # 1 value
            last_played_card,  # 14 values
            match_potential  # 1 value
        ])
        return observation

    def render(self):
        """Render the current state of the game."""
        if self.render_mode == 'human':
            print(f"------ Game State ------")
            print(f"Current Turn: {self.player_turn}")
            print(f"Player 0 hand: {self.player_hands[0]}")
            print(f"Player 1 hand: {self.player_hands[1]}")
            print(f"Open Action: {self.open_action}")
            print(f"Hand: {self.hand}")
            print(f"P0 Known: ")
            print(f"  Self: {self.known_hands[0][0]}")
            print(f"  Opp:  {self.known_hands[0][1]}")
            print(f"P1 Known: ")
            print(f"  Self: {self.known_hands[1][1]}")
            print(f"  Opp:  {self.known_hands[1][0]}")
            print(f"Discard: {self.discard_pile[-1] if self.discard_pile else None}")
            print(f"------------------------")

    def close(self):
        """Clean up resources."""
        pass

    # Helper methods from the original code
    def draw_card(self):
        """Returns the top card of the deck or shuffles the discard."""
        if len(self.deck) == 2:
            self.game_state = CambioState.CALLED
        elif len(self.deck) == 1:
            self.game_state = CambioState.LAST_TURN
            self.game_over = True
            return None
        return self.deck.pop()

    def _play_card(self):
        """Plays a card, sets the current open_action for a callback."""
        self.open_action = self.hand[0]
        self.discard_pile.append(self.hand)
        self.hand = None
        return

    def _switch_card(self, target_index):
        """Switches out card with hand, then plays your hand."""
        player = self.player_turn
        opponent = (self.player_turn + 1) % self.PLAYERS

        if len(self.player_hands[player]) <= target_index:
            return False  # Indicate action failed

        # Swap cards
        self.player_hands[player][target_index], self.hand = self.hand, self.player_hands[player][target_index]
        # Update known states
        self.known_hands[player][player][target_index] = self.player_hands[player][target_index]
        self.known_hands[opponent][player][target_index] = None

        # Play the card
        self._play_card()
        return True

    def _peek_card(self, target_index, player):
        """Lets a player look at a card."""
        if len(self.player_hands[player]) <= target_index:
            return False

        # Update known state
        self.known_hands[self.player_turn][player][target_index] = self.player_hands[player][target_index]
        self.open_action = None
        return True

    def _swap_card(self, target_index1, target_index2):
        """Performs a card swap with another player."""
        player = self.player_turn
        opponent = (self.player_turn + 1) % self.PLAYERS

        if len(self.player_hands[player]) <= target_index1 or len(self.player_hands[opponent]) <= target_index2:
            return False

        # Swap Cards
        (self.player_hands[player][target_index1],
         self.player_hands[opponent][target_index2]) = (
            self.player_hands[opponent][target_index2],
            self.player_hands[player][target_index1]
        )

        # Update known states
        (self.known_hands[player][player][target_index1],
         self.known_hands[player][opponent][target_index2]) = (
            self.known_hands[player][opponent][target_index2],
            self.known_hands[player][player][target_index1]
        )

        (self.known_hands[opponent][player][target_index1],
         self.known_hands[opponent][opponent][target_index2]) = (
            self.known_hands[opponent][opponent][target_index2],
            self.known_hands[opponent][player][target_index1]
        )

        self.open_action = None
        return True

    def _cambio(self):
        """Call cambio to end the game."""
        if self.game_state != CambioState.NOT_CALLED:
            return False
        self.game_state = CambioState.CALLED
        return True

    def _flip_cards2(self):
        if not self.discard_pile:
            return

        top_discard = self.discard_pile[-1]

        for p in range(self.PLAYERS):
            for card_idx in range(len(self.player_hands[p]) - 1, -1, -1):
                card = self.player_hands[p][card_idx][1]
                if card != top_discard[1]:
                    continue
                knows = [False] * self.PLAYERS
                for player in range(self.PLAYERS):
                    if self.known_hands[player][p][card_idx] == top_discard:
                        knows[player] = True

                know_count = knows.count(True)
                if know_count == 0:
                    continue
                elif know_count == 1:
                    knowing_player = knows.index(True)
                    if knowing_player == p:
                        self._flip_from_self(knowing_player, card_idx)
                    else:
                        self._flip_from_opp(knowing_player, card_idx)
                else:
                    faster_player = random.randint(0, 1)
                    if faster_player == p:
                        self._flip_from_self(faster_player, card_idx)
                    else:
                        self._flip_from_opp(faster_player, card_idx)

    def _flip_from_self(self, player, card_idx):
        self.player_hands[player].pop(card_idx)
        for p in range(self.PLAYERS):
            self.known_hands[p][player].pop(card_idx)

    def _flip_from_opp(self, player, card_idx):
        if len(self.player_hands[player]) == 0:
            self._flip_from_self((player + 1) % 2, card_idx)
            return

        worst_card_idx = np.argmin([card[1] if card is not None else 10 for card in self.known_hands[player][player]])
        self.player_hands[(player + 1) % 2][card_idx] = self.player_hands[player][worst_card_idx]
        self.known_hands[player][(player + 1) % 2][card_idx] = self.known_hands[player][player][worst_card_idx]
        self.known_hands[(player + 1) % 2][(player + 1) % 2][card_idx] = self.known_hands[(player + 1) % 2][player][
            worst_card_idx]

        self._flip_from_self(player, worst_card_idx)

    def tally_hands(self):
        """Returns total values of each player's hand."""
        hand_values = [0 for _ in range(self.PLAYERS)]
        for p in range(self.PLAYERS):
            for card in self.player_hands[p]:
                if card is None:
                    continue
                hand_values[p] += card[1]

        return hand_values
