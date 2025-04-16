import copy
import random
import numpy as np
from agents.random_agent import RandomAgent


class MCTSAgent:
    def __init__(self, simulations=1000):
        self.simulations = simulations

    def select_action(self, info, env):
        """
        Run MCTS-style simulations for each initial action (0–4) and select the best.
        """
        valid_actions = info['valid_actions']
        possible_actions = np.where(valid_actions)[0]
        win_counts = {}

        for action in possible_actions:
            win_counts[action] = 0
            for _ in range(self.simulations):
                sim_env = copy.deepcopy(env)
                sim_env.opponent = RandomAgent(1)

                obs, reward, terminated, truncated, info = sim_env.step(action)

                while not terminated:
                    legal_actions = np.where(info['valid_actions'])[0].tolist()
                    next_action = random.choice(legal_actions)
                    obs, reward, terminated, truncated, info = sim_env.step(next_action)

                    opponent_play_next = not info['callback']
                    if opponent_play_next and not terminated:
                        obs, info = sim_env.step_opponent()

                agent_score, opp_score = info['final_tally']
                if agent_score < opp_score:
                    win_counts[action] += 1

        best_action = max(win_counts, key=win_counts.get)
        return best_action


