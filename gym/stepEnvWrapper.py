import gymnasium as gym
from GymEnv import CambioGymEnv
from gymnasium import spaces
import numpy as np


class CambioEnvWrapper(gym.Wrapper):
    """
    A wrapper for the CambioGymEnv that handles the modified step function flow.
    This wrapper processes the opponent's turn automatically after the player's turn.
    """

    def __init__(self, env: CambioGymEnv):
        """Initialize the wrapper with the CambioGymEnv."""
        super().__init__(env)
        self.env = env
        # The observation and action spaces remain the same as the base environment

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation."""
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        """
        Modified step function that:
        1. Processes the player's action
        2. Gets the result and reward for the player
        3. Only processes opponent's turn if no callback is needed from the player
        4. Returns the appropriate observation based on game state
        """
        # Process the player's action first
        observation, reward, terminated, truncated, info = self.env.step(action)

        # If the game ended after the player's action, return immediately
        if terminated:
            return observation, reward, terminated, truncated, info

        # Check if there's a pending callback that requires player input
        # open_action will be set to a value other than None if a callback is needed
        if self.env.open_action is not None:
            # Return without processing opponent's turn, as we need player input for the callback
            info['callback_required'] = True
            info['open_action'] = self.env.open_action
            return observation, reward, terminated, truncated, info

        # No callback needed, so process the opponent's turn
        opponent_info = self.env.process_opponent_turn()

        # Check if the game ended after the opponent's turn
        terminated = opponent_info.get('game_over', False)

        # Get the observation after opponent's turn
        new_observation = self.env._get_observation()

        # Update the info dictionary with opponent info
        info.update(opponent_info)

        # If the game ended after opponent's turn, add final values to info
        if terminated:
            info['final_values'] = self.env.tally_hands()

        return new_observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()

