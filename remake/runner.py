import time
import gymnasium as gym
from gymnasium.core import ObsType, WrapperObsType
from cambio_env import CambioEnv
import numpy as np
from random_agent import RandomAgent
from tqdm import tqdm
from agent import Agent
from DQN_agent import DQNAgent
import os


def run_games(save_path, num_games=1000, training=False):
    agent = DQNAgent()
    opponent = RandomAgent(1)
    env = CambioEnv(opponent)

    losses = []

    for episode in tqdm(range(num_games), desc="Training" if training else "Testing", unit="games"):
        total_reward = 0
        episode_loss = []
        episode_rewards = []
        state, info = env.reset()
        steps = 0
        done = False

        while not done and steps < 50:
            steps += 1
            valid_actions = info['valid_actions']
            action = agent.act(state, valid_actions=valid_actions, training=training)
            next_state, reward, done, truncated, info = env.step(action)
            opponent_plays_next = not info['callback']
            total_reward += reward

            if opponent_plays_next and not done:
                agent.remember(state, action, reward, next_state, done)
                state, info = env.step_opponent()
            else:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            if training:
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)

        if steps >= 50:
            print("stalled")

        episode_rewards.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"\nEpisode {episode + 1}/{num_games}")
            print(f"Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")

    if training:
        agent.save("trained_weights/" + save_path)


def eval_agent(path, num_games=1000):
    agent = DQNAgent()
    agent.load("trained_weights/" + path)
    opponent = RandomAgent(1)
    env = CambioEnv(opponent)

    wins = 0
    ties = 0

    for episode in tqdm(range(num_games), desc="Testing", unit="games"):
        state, info = env.reset()
        steps = 0
        done = False

        while not done and steps < 50:
            steps += 1
            valid_actions = info['valid_actions']
            action = agent.act(state, valid_actions=valid_actions, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            if not done and not info['callback']:
                state, info = env.step_opponent()

        agent_score, opp_score = info['final_tally']
        if agent_score > opp_score:
            wins += 1
        elif agent_score == opp_score:
            ties += 1

    print(f'----Finished Testing----\n'
          f'Final Record: {wins}-{ties}-{num_games - wins - ties}\n'
          f'Winrate: {round(wins / num_games, 3) * 100:.1f}%\n'
          f'-------------------------')


run_games("weights2.weights.h5", num_games=2500, training=True)
eval_agent("weights2.weights.h5")
