import copy
import random
import numpy as np
from agents.random_agent import RandomAgent
from util import CambioState, est_hand_value


class MCTSNode:
    def __init__(self, state=None, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []


class MCTSAgent:
    def __init__(self, simulations=1000, exploration_weight=1.0):
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def select_action(self, info, env):
        """
        Run MCTS algorithm and select the best action
        """
        # create root node with current game state
        root = MCTSNode(state=env.get_env_state())
        root.untried_actions = np.where(info['valid_actions'])[0].tolist()

        for _ in range(self.simulations):
            # Clone the environment for simulation
            sim_env = CambioEnv(opponent_agent=RandomAgent(1))
            sim_env.load_env_state(env.get_env_state())

            node = self._tree_policy(root, sim_env)
            reward = self._rollout(node, sim_env)
            self._backpropagate(node, reward)

        best_child = self._best_child(root, exploration=0.0)
        return best_child.action if best_child.action is not None else random.choice(root.untried_actions)

    def _tree_policy(self, node, env):
        """Selection and expansion phases: navigate tree and expand if needed"""
        while not env.game_over:
            if node.untried_actions:
                return self._expand(node, env)
            else:
                if not node.children:
                    return node

                node = self._best_child(node)

                if node.action is not None:
                    valid_actions = env._get_valid_actions()
                    if not valid_actions[node.action]:
                        return node
                    obs, reward, terminated, truncated, info = env.step(node.action)

                    if not info['callback'] and not terminated:
                        obs, info = env.step_opponent()

        return node

    def _expand(self, node, env):
        """Expand tree by trying an untried action"""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()

        pre_state = env.get_env_state()

        valid_actions = env._get_valid_actions()
        if not valid_actions[action]:
            return node
        obs, reward, terminated, truncated, info = env.step(action)

        if not info['callback'] and not terminated:
            obs, info = env.step_opponent()

        child = MCTSNode(
            state=env.get_env_state(),
            parent=node,
            action=action
        )

        if not terminated:
            child.untried_actions = np.where(info['valid_actions'])[0].tolist()

        node.children[action] = child

        return child

    def _best_child(self, node, exploration=1.0):
        """
        Select best child using UCB formula
        UCB = average_value + exploration * sqrt(2 * ln(parent_visits) / child_visits)
        """
        if not node.children:
            return node

        best_score = float('-inf')
        best_children = []

        for action, child in node.children.items():
            if child.visits == 0:
                continue

            exploit = child.value / child.visits
            explore = exploration * np.sqrt(2 * np.log(node.visits) / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        if not best_children and node.children:
            return random.choice(list(node.children.values()))

        return random.choice(best_children) if best_children else node

    def _rollout(self, node, env):
        """
        Simulate a random game from the current state until the end
        and return the reward
        """
        if env.game_over:
            agent_score, opp_score = env._tally_hands()
            return 1.0 if agent_score < opp_score else -1.0

        # simulate until game is over
        while not env.game_over:
            valid_actions = env._get_valid_actions()

            action = self._rollout_policy(valid_actions, env)

            obs, reward, terminated, truncated, info = env.step(action)

            if not info['callback'] and not terminated:
                obs, info = env.step_opponent()

        # evaluate final state
        agent_score, opp_score = info['final_tally']
        return 1.0 if agent_score < opp_score else 0.0 if agent_score == opp_score else -1.0

    def _rollout_policy(self, valid_actions, env):
        """
        rollout policy with heuristics for Cambio game
        """
        legal_actions = np.where(valid_actions)[0].tolist()

        if not legal_actions:
            return 20  # Pass

        # get current state info
        player_turn = env.player_turn
        known_hands = env.known_hands
        player_hands = env.player_hands
        open_action = env.open_action

        # call Cambio if estimated hand value is good
        if 4 in legal_actions and env.game_state is CambioState.NOT_CALLED:
            hand_value = est_hand_value(env.known_hands[player_turn][player_turn])
            if hand_value < 10 or env.turn_count > 10:
                return 4

        if env.open_action == 1:
            unknown_card_indices = [i + 5 for i in range(len(player_hands[player_turn]))
                                    if i + 5 in legal_actions and known_hands[player_turn][player_turn][i] is None]
            if unknown_card_indices:
                return random.choice(unknown_card_indices)

        if env.open_action == 2:
            opponent = (player_turn + 1) % 2
            unknown_opp_card_indices = [i + 8 for i in range(len(player_hands[opponent]))
                                        if i + 8 in legal_actions and known_hands[player_turn][opponent][i] is None]
            if unknown_opp_card_indices:
                return random.choice(unknown_opp_card_indices)

        if env.open_action in [3, 4]:
            player = player_turn
            opponent = (player_turn + 1) % 2

            our_high_card_idx = -1
            our_high_value = -1
            for i, card in enumerate(known_hands[player][player]):
                if card is not None and card[1] > our_high_value:
                    our_high_value = card[1]
                    our_high_card_idx = i

            opp_low_card_idx = -1
            opp_low_value = 14
            for i, card in enumerate(known_hands[player][opponent]):
                if card is not None and card[1] < opp_low_value:
                    opp_low_value = card[1]
                    opp_low_card_idx = i

            if our_high_card_idx >= 0 and opp_low_card_idx >= 0 and our_high_value > opp_low_value:
                swap_action = 11 + (our_high_card_idx * 3) + opp_low_card_idx
                if swap_action in legal_actions:
                    return swap_action

        if env.open_action is None:
            hand = env.hand
            if hand and hand[0] > 0:
                return 3

            for i in range(3):
                if i in legal_actions and known_hands[player_turn][player_turn][i] is None:
                    return i

        return random.choice(legal_actions)

    def _backpropagate(self, node, reward):
        """Update node values up the tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent