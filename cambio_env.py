import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from util import build_deck, CambioState, est_hand_value, known_ratio, hand_value
from agents.dqn_agent import DQNAgent
from agents.mcts_agent import MCTSAgent

"""
ACTION LIST
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
20: pass open action
"""


class CambioEnv(gym.Env):
    def __init__(self, opponent_agent):
        super().__init__()

        self.HAND_SIZE = 3
        self.PLAYERS = 2
        self.opponent = opponent_agent

        self.deck = None
        self.player_hands = None
        self.known_hands = None
        self.discard_pile = None
        self.turn_count = 0
        self.player_turn = 0
        self.hand = None
        self.open_action = None
        self.game_state = None
        self.game_over = False
        self.reward = 0

        self.action_space = spaces.Discrete(21)

        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        self.deck = build_deck() * 2
        self.player_hands = [[] for _ in range(self.PLAYERS)]
        self.known_hands = [[[None, None, None] for _ in range(self.PLAYERS)] for _ in range(self.PLAYERS)]
        self.discard_pile = []
        self.player_turn = 0  # Always start with the RL agent
        self.turn_count = 0
        self.open_action = None
        self.game_state = CambioState.NOT_CALLED
        self.game_over = False

        random.shuffle(self.deck)
        for p in range(self.PLAYERS):
            for _ in range(self.HAND_SIZE):
                self.player_hands[p].append(self._draw_card())
            self.known_hands[p][p][1] = self.player_hands[p][1]

        self.hand = self._draw_card()
        info = self._get_info()
        return self._get_observation(), info

    def step(self, action):
        """Main stepping function for environment interaction"""
        self.turn_count += 1
        self.player_turn = 0
        info = {}

        if self.game_over:
            reward = self._calculate_reward_handknown(0, 0)
            return self._get_observation(), reward, self.game_over, False, self._get_info()
        prev_hand_value = est_hand_value(self.known_hands[0][0])

        self._handle_action(action)
        self._flip_cards()
        self._update_game_state()

        if self.open_action is None:
            if self.hand is not None:
                print("action tried", action, " hand", self.hand)
                raise Exception("STEPERROR: Trying to draw card after turn but card is still in hand!")
            self.hand = self._draw_card()

        # cur_value = est_state_value(self.known_hands[1])
        reward = self._calculate_reward(prev_hand_value)

        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def step_opponent(self):
        """Process the opponent's turn using prompt_action and prompt_callback."""
        self.player_turn = 1
        self.turn_count += 1
        opponent, player = self.player_turn, (self.player_turn + 1) % 2

        agent = self.opponent
        if type(agent) is DQNAgent:
            action = agent.act(self._get_observation(), self._get_valid_actions(), training=False)
        elif type(agent) is MCTSAgent:
            action = agent.select_action(self._get_info(), self)
        else:
            action = agent.prompt_action(
                self.known_hands[opponent],
                self.hand,
                self.game_state,
                self.discard_pile[-1] if self.discard_pile else None,
                self.turn_count,
                valid_actions=self._get_valid_actions()
            )
        self._handle_action(action)

        if self.open_action is not None:
            if type(agent) is DQNAgent:
                callback_action = agent.act(self._get_observation(), self._get_valid_actions(), training=False)
            else:
                callback_action = agent.prompt_callback(self.known_hands[opponent],
                                                        self.open_action,
                                                        valid_actions=self._get_valid_actions())
            self._handle_action(callback_action)

        self._flip_cards()
        self._update_game_state()

        if not self.game_over:
            self.hand = self._draw_card()

        return self._get_observation(), self._get_info()

    def _get_valid_actions(self):
        """Returns list of currently valid actions"""
        valid_actions = np.zeros(self.action_space.n, dtype=bool)

        if self.open_action is None:
            for i in range(len(self.player_hands[0])):  # Replace card slots
                valid_actions[i] = True
            valid_actions[3] = True  # Play to center always valid
            if self.game_state is CambioState.NOT_CALLED:
                valid_actions[4] = True  # Call cambio

        elif self.open_action is not None:
            valid_actions[20] = True
            if self.open_action == 1:  # Peek own card
                for i in range(len(self.player_hands[self.player_turn])):
                    valid_actions[i + 5] = True
            elif self.open_action == 2:  # Peek opponent card
                opponent = (self.player_turn + 1) % 2
                for i in range(len(self.player_hands[opponent])):
                    valid_actions[i + 8] = True
            elif self.open_action in [3, 4]:  # Swap cards
                player = self.player_turn
                opponent = (self.player_turn + 1) % 2

                for p_idx in range(min(len(self.player_hands[player]), 3)):
                    for o_idx in range(min(len(self.player_hands[opponent]), 3)):
                        action_idx = 11 + (p_idx * 3) + o_idx
                        valid_actions[action_idx] = True

        return valid_actions

    def _calculate_reward(self, prev_hand_value):
        player_hand_value = est_hand_value(self.known_hands[0][0])

        hand_value_reward = (player_hand_value - prev_hand_value) / 13.0 * 10.0

        if self.game_over:
            agent_score, opp_score = self._tally_hands()
            if agent_score < opp_score:  # Agent wins
                return 20.0
            else:
                return -5.0

        return hand_value_reward

    def _draw_card(self):
        """Returns top card of the deck, ends the game if deck is empty"""
        if len(self.deck) == 2 and self.game_state is CambioState.NOT_CALLED:
            self.game_state = CambioState.CALLED
        elif len(self.deck) == 0:
            raise Exception("DRAWCARD ERROR: Deck is empty!")
        return self.deck.pop()

    def _handle_action(self, action):
        if 0 <= action <= 4 and self.open_action is None:  # Initial actions
            self._action_initial(action)
        elif self.open_action is not None:
            if 5 <= action <= 7 and self.open_action == 1:  # Peek own card
                target_index = action - 5
                self._peek_own(target_index)
            elif 7 <= action <= 10 and self.open_action == 2:  # Peek opponent card
                target_index = action - 8
                self._peek_other(target_index)
            elif 11 <= action <= 19 and self.open_action in [3, 4]:  # Swap cards
                action_idx = action - 11
                player_idx = action_idx // 3
                opponent_idx = action_idx % 3
                self._swap_cards(player_idx, opponent_idx)
            self.open_action = None
        else:
            raise Exception(f"HANDLEACTION ERROR: Invalid Action attempted ({action})")

    def _action_initial(self, action):
        if action > 4:
            raise Exception(f"INITALACTION ERROR: Invalid Action attempted ({action})")
        if action == 3:
            self._play_to_center()
        elif action == 4:
            self._cambio()
        else:
            self._play_to_slot(action)

    def _play_to_center(self):
        if self.hand is None:
            raise Exception("PLAYTOCENTER ERROR: no card in hand!")
        card_action = self.hand[0]
        if card_action != 0:
            self.open_action = card_action
        self.discard_pile.append(self.hand)
        self.hand = None

    def _play_to_slot(self, card_idx):
        player, opponent = self.player_turn, (self.player_turn + 1) % 2
        if len(self.player_hands[player]) <= card_idx:
            raise Exception("PLAYTOSLOT ERROR: Chosen card index out of bounds!")
        self.player_hands[player][card_idx], self.hand = self.hand, self.player_hands[player][card_idx]
        self.known_hands[player][player][card_idx] = self.player_hands[player][card_idx]
        self.known_hands[opponent][player][card_idx] = None
        self._play_to_center()

    def _cambio(self):
        if self.game_state is not CambioState.NOT_CALLED:
            raise Exception("CAMBIOERROR: already called!")
        self.discard_pile.append(self.hand)
        self.hand = None
        self.open_action = None
        self.game_state = CambioState.CALLED

    def _peek_own(self, card_idx):
        player, opponent = self.player_turn, (self.player_turn + 1) % 2
        if len(self.player_hands[player]) <= card_idx:
            raise Exception("PEEKOWN ERROR: Chosen peek index out of bounds!")
        self.known_hands[player][player][card_idx] = self.player_hands[player][card_idx]

    def _peek_other(self, card_idx):
        player, opponent = self.player_turn, (self.player_turn + 1) % 2
        if len(self.player_hands[opponent]) <= card_idx:
            raise Exception("PEEKOTHER ERROR: Chosen peek index out of bounds!")
        self.known_hands[player][opponent][card_idx] = self.player_hands[opponent][card_idx]

    def _swap_cards(self, player_idx, opponent_idx):
        player, opponent = self.player_turn, (self.player_turn + 1) % 2
        if len(self.player_hands[opponent]) <= player_idx:
            raise Exception("SWAPERROR: Chosen player index out of bounds!")
        if len(self.player_hands[opponent]) <= player_idx:
            raise Exception("SWAPERROR: Chosen opponent index out of bounds!")
        # Swap Card Locations
        self.player_hands[player][player_idx], self.player_hands[opponent][opponent_idx] = (
            self.player_hands[opponent][opponent_idx], self.player_hands[player][player_idx])
        # Update Known States
        self.known_hands[player][player][player_idx], self.known_hands[player][opponent][opponent_idx] = (
            self.known_hands[player][opponent][opponent_idx], self.known_hands[player][player][player_idx])
        self.known_hands[opponent][player][player_idx], self.known_hands[opponent][opponent][opponent_idx] = (
            self.known_hands[opponent][opponent][opponent_idx], self.known_hands[opponent][player][player_idx])

    def _flip_cards(self):
        player, opponent = self.player_turn, (self.player_turn + 1) % 2
        if not self.discard_pile:
            return
        discard_num = self.discard_pile[-1]
        for p in range(self.PLAYERS):
            for i, card in enumerate(self.player_hands[p]):
                card_num = card[1]
                if card_num == discard_num:
                    player_knows = self.known_hands[player][p][i] == card
                    opponent_knows = self.known_hands[opponent][p][i] == card
                    if player_knows and opponent_knows:
                        faster_player = random.randint(0, 1)  # Which player flipped the card
                        if faster_player == p:
                            self._flip_from_self(faster_player, i)
                        else:
                            self._flip_from_opp(faster_player, i)
                    elif player_knows:
                        if player == p:
                            self._flip_from_self(player, i)
                        else:
                            self._flip_from_opp(player, i)
                    elif opponent_knows:
                        if opponent == p:
                            self._flip_from_self(opponent, i)
                        else:
                            self._flip_from_opp(opponent, i)

    def _flip_from_self(self, player, card_idx):
        self.player_hands[player].pop(card_idx)
        for p in range(self.PLAYERS):
            self.known_hands[p][player].pop(card_idx)

    def _flip_from_opp(self, player, card_idx):
        opponent = (player + 1) % 2
        if len(self.player_hands[player]) == 0:  # Pass on flip opportunity bc no cards to give
            self._flip_from_self(opponent, card_idx)
            return
        worst_card_idx = np.argmin([card[1] if card is not None else 11 for card in self.known_hands[player][player]])
        self.player_hands[opponent][card_idx] = self.player_hands[player][worst_card_idx]
        self.known_hands[player][opponent][card_idx] = self.known_hands[player][player][worst_card_idx]
        self.known_hands[opponent][opponent][card_idx] = self.known_hands[opponent][player][worst_card_idx]
        self._flip_from_self(player, worst_card_idx)

    def _update_game_state(self):
        if self.game_state is CambioState.CALLED:
            self.game_state = CambioState.LAST_TURN
        elif self.game_state is CambioState.LAST_TURN:
            self.game_over = True

    def _tally_hands(self):
        p1_score, p2_score = 0, 0
        for p1_card, p2_card in zip(self.player_hands[0], self.player_hands[1]):
            p1_score += p1_card[1]
            p2_score += p2_card[1]
        return p1_score, p2_score

    def render(self):
        state = "not called"
        if self.game_state is CambioState.CALLED:
            state = "called"
        elif self.game_state is CambioState.LAST_TURN:
            state = "last turn"
        out = (
            f'\n------ Game State ------\n'
            f'Turn Num: {self.turn_count}\n'
            f'Card in Hand: {self.hand}\n'
            f'Top Discard: {self.discard_pile[-1] if self.discard_pile else "empty"}\n'
            f'Current Action: {"initial" if self.open_action is None else f"callback ({self.open_action})"}\n'
            f'Agent Hand:  {self.player_hands[0]}\n'
            f'Oppnt Hand: {self.player_hands[1]}\n'
            f'Agent Known:\n'
            f'  Agent Hand:  {self.known_hands[0][0]}\n'
            f'  Oppnt Hand: {self.known_hands[0][1]}\n'
            f'Oppnt Known:\n'
            f'  Agent Hand:  {self.known_hands[1][0]}\n'
            f'  Oppnt Hand: {self.known_hands[1][1]}\n'
            f'Game State: {state}'
        )
        print(out)

    def _get_observation(self):
        turn_count = np.array([self.turn_count / 50.0], dtype=np.int32)
        known_player_cards = np.full(3, -1, dtype=np.float32)
        for i, card in enumerate(self.known_hands[0][0]):
            known_player_cards[i] = card[1] / 13.0 if card is not None else -1
        known_opponent_cards = np.full(3, -1, dtype=np.float32)
        for i, card in enumerate(self.known_hands[0][1]):
            known_opponent_cards[i] = card[1] / 13.0 if card is not None else -1
        hand = np.array([self.hand[1] / 13.0 if self.hand else -1])
        hand_action = np.zeros(6, dtype=np.float32)
        hand_action[self.hand[0] if self.hand else 5] = 1.0
        discard = np.array([self.discard_pile[-1][1] if self.discard_pile else 13])
        game_state = np.zeros(3, dtype=np.float32)
        if self.game_state == CambioState.NOT_CALLED:
            game_state[0] = 1.0
        elif self.game_state == CambioState.CALLED:
            game_state[1] = 1.0
        elif self.game_state == CambioState.LAST_TURN:
            game_state[2] = 1.0
        open_action = np.zeros(5, dtype=np.float32)
        open_action[self.open_action if self.open_action else 4] = 1.0
        estimated_player_total = np.array([est_hand_value(self.known_hands[0][0]) / 39.0], dtype=np.float32)
        estimated_opponent_total = np.array([est_hand_value(self.known_hands[0][1]) / 39.0], dtype=np.float32)
        self_known_ratio = np.array([known_ratio(self.known_hands[0][0])], dtype=np.float32)
        opponent_known_ratio = np.array([known_ratio(self.known_hands[0][1])], dtype=np.float32)
        cards_in_deck = np.array([len(self.deck) / 52.0])

        obs = np.concatenate([
            turn_count,
            known_player_cards,
            known_opponent_cards,
            hand,
            hand_action,
            discard,
            game_state,
            open_action,
            estimated_player_total,
            estimated_opponent_total,
            self_known_ratio,
            opponent_known_ratio,
            cards_in_deck,
        ])
        return obs

    def _get_info(self):
        info = {'callback': True if self.open_action is not None else False,
                'valid_actions': self._get_valid_actions(),
                'final_tally': self._tally_hands() if self.game_over else None}
        return info

    def get_env_state(self):
        """Return a representation of the current env state."""
        return {
            'deck': self.deck.copy(),
            'player_hands': [[card[:] for card in hand] for hand in self.player_hands],
            'known_hands': [[[card[:] if card else None for card in player]
                             for player in players] for players in self.known_hands],
            'discard_pile': self.discard_pile.copy(),
            'turn_count': self.turn_count,
            'player_turn': self.player_turn,
            'hand': self.hand[:] if self.hand else None,
            'open_action': self.open_action,
            'game_state': self.game_state,
            'game_over': self.game_over
        }

    def load_env_state(self, state):
        """Load a saved environment state."""
        self.deck = state['deck'].copy()
        self.player_hands = [[card[:] for card in hand] for hand in state['player_hands']]
        self.known_hands = [[[card[:] if card else None for card in player]
                             for player in players] for players in state['known_hands']]
        self.discard_pile = state['discard_pile'].copy()
        self.turn_count = state['turn_count']
        self.player_turn = state['player_turn']
        self.hand = state['hand'][:] if state['hand'] else None
        self.open_action = state['open_action']
        self.game_state = state['game_state']
        self.game_over = state['game_over']

