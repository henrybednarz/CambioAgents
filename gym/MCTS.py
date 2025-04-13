import copy
import random
import numpy as np
from randomAgent import RandomAgent


class MCTSAgent:
    def __init__(self, simulations=100):
        self.simulations = simulations

    def select_action(self, env, valid_actions=None):
        """
        Run MCTS-style simulations for each initial action (0–4) and select the best.
        """
        candidate_actions = [0, 1, 2, 3, 4]
        action_win_counts = {a: 0 for a in candidate_actions}
        action_total_counts = {a: 0 for a in candidate_actions}

        for action in candidate_actions:
            for _ in range(self.simulations):
                sim_env = copy.deepcopy(env)
                sim_env.opponent = RandomAgent(1)
                sim_env.opponent.change_num()
                sim_env.opponent.clear_memory()

                # Take the initial action
                obs, reward, terminated, truncated, info = sim_env.step(action)
                valid_actions = info['valid_actions']
                # If open_action was set, try a callback
                if sim_env.open_action is not None and not terminated:
                    callback_options = self.get_valid_callback_actions(sim_env)
                    if callback_options:
                        callback_action = random.choice(callback_options)
                        obs, reward, terminated, truncated, info = sim_env.step(callback_action)

                # Simulate the rest of the game using rollouts
                while not terminated and not truncated:
                    legal_actions = self.get_valid_actions(sim_env)
                    smart_action = self.smart_rollout_policy(sim_env, legal_actions)
                    obs, reward, terminated, truncated, info = sim_env.step(smart_action)

                # Check final outcome
                final_vals = info.get("final_values", None)
                if final_vals:
                    if final_vals[0] < final_vals[1]:
                        action_win_counts[action] += 1
                action_total_counts[action] += 1

        # Compute win rates
        action_win_rates = {
            a: action_win_counts[a] / action_total_counts[a] if action_total_counts[a] > 0 else 0
            for a in candidate_actions
        }

        if valid_actions is not None:
            for key in action_win_rates.keys():
                if not valid_actions[key]:
                    action_win_rates[key] = 0

        # Choose the action with the highest win rate
        best_action = max(action_win_rates, key=action_win_rates.get)

        return best_action

    def get_valid_actions(self, env):
        if env.open_action is None:
            return [0, 1, 2, 3, 4]
        elif env.open_action == 1:
            return [5, 6, 7]
        elif env.open_action == 2:
            return [8, 9, 10]
        elif env.open_action in [3, 4]:
            return list(range(11, 20))
        else:
            return [0]

    def get_valid_callback_actions(self, env):
        return self.get_valid_actions(env)

    def smart_rollout_policy(self, env, valid_actions):
        if 4 in valid_actions:
            return 4
        if 3 in valid_actions:
            return 3
        if any(a in valid_actions for a in [0, 1, 2]):
            return random.choice([a for a in [0, 1, 2] if a in valid_actions])
        return random.choice(valid_actions)

    def prompt_action(self, env):
        return self.select_action(env)

    def prompt_callback(self, env):
        return self.select_action(env)