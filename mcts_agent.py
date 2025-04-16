import copy
import random
import numpy as np
from simple_agent import SimpleHeuristicAgent
from util import CambioState

"""
info['valid_actions']
info['callback']
"""


class MCTSAgent:
    def __init__(self, simulations=100):
        self.simulations = simulations

    def select_action(self, next_state, reward, done, truncated, info, env):
        """
        Run MCTS-style simulations for each initial action (0–4) and select the best.
        """
        valid_actions = info['valid_actions']
        callback = info['callback']
        candidate_actions = np.where(valid_actions)
        win_cnts = {}
        for a in valid_actions:
            win_cnts[a] = 0

        for action in candidate_actions:
            for _ in range(self.simulations):
                sim_env = copy.deepcopy(env)
                sim_env.opponent = SimpleHeuristicAgent(1)

                # Take the initial action
                obs, reward, terminated, truncated, info = sim_env.step(action)

                # Simulate the rest of the game using rollouts
                while not terminated:
                    legal_actions = np.where(info['valid_actions'])
                    next_action = random.choice(legal_actions)
                    obs, reward, terminated, truncated, info = sim_env.step(next_action)

                    if not info['callback']:
                        obs, info = env.step_opponent()

                # Check final outcome
                agent_score, opp_score = info['final_tally']
                if agent_score > opp_score:
                    win_cnts[action] += 1

        # Choose the action with the highest win rate
        best_action = max(win_cnts, key=win_cnts.get)
        return best_action
