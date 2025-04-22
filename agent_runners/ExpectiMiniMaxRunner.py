import util
from agents.expectiminimax_agent import ExpectiAgent
from agents.random_agent import RandomAgent
from cambio_env import CambioEnv
import matplotlib.pyplot as plt
import traceback
import collections
import numpy as np


def safe_est_hand_value(hand):
    value = 0.0
    for card in hand:
        try:
            if isinstance(card, tuple) and isinstance(card[1], int):
                value += card[1]
            else:
                value += 8
        except:
            value += 8
    return value


util.est_hand_value = safe_est_hand_value


def run_experiment(n=1000):
    simulations_per_action = list(range(1, n + 1))

    expecti_wins = 0
    total_games = 0
    skipped_games = 0
    win_rates = []
    game_lengths = []
    agent_scores = []
    opponent_scores = []
    win_results = []
    all_actions = []

    for i in range(n):
        print(f"\\nStarting Game {i + 1}/{n}")
        agent = ExpectiAgent()
        opponent = RandomAgent()
        env = CambioEnv(opponent)

        try:
            obs, info = env.reset()
            done = False
            turn_counter = 0
            game_actions = []
            cambio_called = False
            cambio_turn = -1

            while not done:
                action = agent.prompt_action(
                    known_hands=env.known_hands[0],
                    hand=env.hand,
                    game_state=env.game_state,
                    discard=env.discard_pile[-1] if env.discard_pile else None,
                    turn_count=env.turn_count,
                    valid_actions=info['valid_actions']
                )
                game_actions.append(action)
                obs, reward, done, _, info = env.step(action)
                turn_counter += 1

                if action == 4 and not cambio_called:
                    cambio_called = True
                    cambio_turn = turn_counter

                if info.get('callback', False):
                    callback_action = agent.prompt_callback(
                        env.known_hands[0],
                        env.open_action,
                        valid_actions=info['valid_actions']
                    )
                    game_actions.append(callback_action)
                    obs, reward, done, _, info = env.step(callback_action)

                if done:
                    break

                obs, info = env.step_opponent()
                turn_counter += 1

            final_vals = info.get('final_tally', None)
            if final_vals:
                print(f"Final hands - ExpectiAgent: {final_vals[0]}, Opponent: {final_vals[1]}")
                total_games += 1
                agent_scores.append(final_vals[0])
                opponent_scores.append(final_vals[1])
                game_lengths.append(turn_counter)
                if cambio_called:
                    cambio_turns.append(cambio_turn)
                if final_vals[0] < final_vals[1]:
                    expecti_wins += 1
                    win_results.append(1)
                    print("ExpectiAgent won!")
                else:
                    win_results.append(0)
                    print("ExpectiAgent lost.")
                all_actions.extend(game_actions)
            else:
                skipped_games += 1
                print("No final tally found. Skipping.")

        except Exception as e:
            print(f"[Game {i + 1}] Unexpected error: {e}")
            traceback.print_exc()
            skipped_games += 1

        if total_games > 0:
            win_rate = expecti_wins / total_games
            win_rates.append(win_rate)
        else:
            win_rates.append(0)

    print("\\nExperiment Summary")
    print(f"ExpectiAgent wins: {expecti_wins} / {total_games}")
    print(f"Skipped games (errors or incomplete): {skipped_games}")

    def moving_average(data, window_size=5):
        """Apply a simple moving average with the given window size."""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_win_rates = moving_average(win_rates, window_size=5)

    plt.figure(figsize=(10, 5))
    plt.plot(simulations_per_action[:len(smoothed_win_rates)], smoothed_win_rates, marker='o',
             label="ExpectiAgent Win Rate (Smoothed)", color="blue")
    plt.axhline(y=0.5, linestyle='--', color='gray', label="50% Threshold")
    plt.title("Win Rate vs Simulations Per Action (Smoothed)")
    plt.xlabel("Simulations Per Action")
    plt.ylabel("Win Rate")
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("win_rate_plot_smoothed.png")
    plt.show()

    def moving_average(values, window_size):
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

    def bin_average(values, bin_size=50):
        """Average values over non-overlapping bins of size bin_size."""
        n_bins = len(values) // bin_size
        binned_means = [np.mean(values[i * bin_size:(i + 1) * bin_size]) for i in range(n_bins)]
        return binned_means

    bin_size = 50

    smoothed_hand_values = bin_average(agent_scores, bin_size)
    smoothed_opponent_scores = bin_average(opponent_scores, bin_size)

    simulations_binned = np.arange(bin_size, bin_size * (len(smoothed_hand_values) + 1), bin_size)

    plt.figure(figsize=(12, 6))
    plt.plot(simulations_binned, smoothed_hand_values, marker='o', color='blue',
             label="ExpectiAgent Avg Hand Value (Binned)")
    plt.plot(simulations_binned, smoothed_opponent_scores, marker='o', color="orange",
             label="Random Opponent Avg Hand Value (Binned)")
    plt.title("Average Final Hand Value vs Simulations Per Action (Binned Every 50 Games)")
    plt.xlabel("Simulations Per Action")
    plt.ylabel("Average Final Hand Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_hand_smoothed.png")
    plt.show()


if __name__ == "__main__":
    run_experiment(1000)
