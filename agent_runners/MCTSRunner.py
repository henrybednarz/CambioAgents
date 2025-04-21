from env import CambioEnv
from mcts_agent import MCTSAgent
from randomAgent import RandomAgent
import numpy as np
import matplotlib.pyplot as plt
import time

SIM_COUNTS = list(range(10, 1000, 50))
GAMES_PER_SIM = 10000

win_rates = []
avg_mcts_scores = []
avg_random_scores = []
avg_times = []

# pre-generate 10000 consistent initial states
SAVED_INIT_STATES = []
for _ in range(GAMES_PER_SIM):
    env = CambioEnv(opponent_agent=RandomAgent(1))
    obs, info = env.reset()
    SAVED_INIT_STATES.append(env.get_env_state())

# run experiments using the same saved states
for sim_count in SIM_COUNTS:
    print(f"\nRunning with {sim_count} simulations per action")
    mcts_wins = 0
    mcts_scores = []
    random_scores = []
    simulation_times = []

    for game_num in range(GAMES_PER_SIM):
        mcts_agent = MCTSAgent(simulations=sim_count, exploration_weight=1.0)

        env = CambioEnv(opponent_agent=RandomAgent(1))
        env.load_env_state(SAVED_INIT_STATES[game_num])

        obs = env._get_observation()
        info = {'valid_actions': env._get_valid_actions(), 'callback': False}

        done = False

        # play the game through
        while not done:
            action = mcts_agent.select_action(info, env)

            obs, reward, done, _, info = env.step(action)

            if not info['callback'] and not done:
                obs, info = env.step_opponent()

        # get final scores
        final_vals = info['final_tally']
        m_score, r_score = final_vals

        mcts_scores.append(m_score)
        random_scores.append(r_score)

        # determine the winner
        if m_score < r_score:
            mcts_wins += 1
            outcome = "MCTS Wins"
        elif m_score > r_score:
            outcome = "Random Wins"
        else:
            outcome = "Tie"

        print(f"Game {game_num + 1}/{GAMES_PER_SIM} — MCTS: {m_score} | Random: {r_score} → {outcome}")

    # calculate statistics
    win_rate = mcts_wins / GAMES_PER_SIM
    win_rates.append(win_rate)
    avg_mcts_scores.append(np.mean(mcts_scores))
    avg_random_scores.append(np.mean(random_scores))

    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg MCTS Hand Value:   {np.mean(mcts_scores):.2f}")
    print(f"Avg Random Hand Value: {np.mean(random_scores):.2f}")

# Win Rate vs Simulations Graph
plt.figure(figsize=(10, 6))
plt.plot(SIM_COUNTS, win_rates, marker='o', linestyle='-', color='blue', label="MCTS Win Rate")
plt.axhline(y=0.5, color='gray', linestyle='--', label='50% Threshold')
plt.xlabel("Simulations Per Action")
plt.ylabel("Win Rate")
plt.title("Win Rate vs MCTS Simulations per Action")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mcts_win_rate_vs_simulations.png")

# Average Hand Value Graph
plt.figure(figsize=(10, 6))
plt.plot(SIM_COUNTS, avg_mcts_scores, marker='o', label='MCTS Avg Hand Value', color='blue')
plt.xlabel("Simulations Per Action")
plt.ylabel("Average Final Hand Value")
plt.title("Final Hand Value vs MCTS Simulations per Action")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mcts_hand_value_vs_sim.png")