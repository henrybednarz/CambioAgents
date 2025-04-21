import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


from cambio_env import CambioEnv
from agents.agent import Agent
from agents.simple_agent import SimpleHeuristicAgent
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.dqn_agent import DQNAgent


def evaluate_agent(agent, opponent, agent_save=None, opponent_save=None, num_games=1000):
    if agent_save is not None:
        agent.load(agent_save)
    if opponent_save is not None:
        opponent.load(opponent_save)
    os.makedirs("training_weights", exist_ok=True)
    os.makedirs("evaluation_figures", exist_ok=True)

    env = CambioEnv(opponent)

    wins = 0
    ties = 0
    hand_sizes = []
    turn_counts = []
    for game in tqdm(range(num_games), desc="agent evaluation", unit="games"):
        total_reward = 0
        turn_count = 0
        done = False

        state, info = env.reset()
        while not done:
            valid_actions = info['valid_actions']

            action = agent.act(state, valid_actions=valid_actions, training=False)

            next_state, reward, done, truncated, info = env.step(action)
            opponent_plays_next = not info['callback']

            if opponent_plays_next:
                turn_count += 1

            total_reward += reward

            if opponent_plays_next and not done:
                next_state, info = env.step_opponent()
                turn_count += 1

            state = next_state

        agent_score, opponent_score = info['final_tally']
        if agent_score < opponent_score:
            wins += 1
        elif agent_score == opponent_score:
            ties += 1

        hand_sizes.append(agent_score)
        turn_counts.append(turn_count)

    print(f"\n---- EVALUATION OVER {num_games} GAMES ----")
    print(f"AGENT WINRATE: {round(wins / num_games, 2) * 100}%\nRecord (W-T-L): {wins}-{ties}-{num_games - wins - ties}")
    print(f"AVERAGE TURNS/GAME: {round(sum(turn_counts) / num_games, 2)}")
    print(f"AVERAGE HAND SIZE: {round(sum(hand_sizes) / num_games, 2)} ")


agent = DQNAgent()
opponent = SimpleHeuristicAgent()
evaluate_agent(agent, opponent, agent_save="saved_weights/dqn_randomagent.weights.h5")