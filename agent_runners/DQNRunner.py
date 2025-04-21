from cambio_env import CambioEnv
import numpy as np
from agents.random_agent import RandomAgent
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent  # Import your improved DQN agent
from agents.simple_agent import SimpleHeuristicAgent


def run_games(save_path, num_games=5000, training=True, save_interval=500):
    agent = DQNAgent(gamma=0.85, learning_rate=0.00001, epsilon=0.3)
    # agent.load("trained_weights/72at10k.weights.h5")
    opponent = RandomAgent()
    env = CambioEnv(opponent)

    losses = []
    rewards = []
    win_rates = []
    epsilon_values = []

    # Create directory for weights if it doesn't exist
    os.makedirs("trained_weights", exist_ok=True)

    # Training checkpoints
    best_win_rate = 0.0
    evaluation_interval = 100  # Evaluate every 500 episodes

    for episode in tqdm(range(num_games), desc="Training" if training else "Testing", unit="games"):
        total_reward = 0
        episode_loss = []
        done = False

        state, info = env.reset()

        while not done:
            valid_actions = info['valid_actions']

            # Get action from agent
            action = agent.act(state, valid_actions=valid_actions, training=training)

            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            opponent_plays_next = not info['callback']
            total_reward += reward

            # Store experience in agent's memory

            agent.remember(state, action, reward, next_state, done, valid_actions)

            # Update state
            if opponent_plays_next and not done:
                # Let opponent play
                next_state, info = env.step_opponent()

            state = next_state

            if training:
                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)

        rewards.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        epsilon_values.append(agent.epsilon)

        if (episode + 1) % evaluation_interval == 0 and training:
            temp_weights_path = "trained_weights/temp_weights.weights.h5"
            agent.save(temp_weights_path)

            win_rate = evaluate_agent(agent, opponent, 100)
            win_rates.append(win_rate)

            avg_reward = np.mean(rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"\nEpisode {episode + 1}/{num_games}")
            print(
                f"Win Rate: {win_rate * 100:.2f}% | Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save("trained_weights/best_" + save_path)
                print(f"New best model saved with win rate: {best_win_rate * 100:.2f}%")

            if (episode + 1) % save_interval == 0:
                agent.save(f"trained_weights/checkpoint_{episode+1}_{save_path}")

            plot_learning_curves(rewards, losses, win_rates, epsilon_values, evaluation_interval)

    if training:
        agent.save("trained_weights/" + save_path)
        print(f"Final model saved to trained_weights/{save_path}")

        # Final evaluation
        final_win_rate = evaluate_agent(agent, opponent, 500)
        print(f"Final evaluation: Win rate = {final_win_rate * 100:.2f}%")

    return agent


def evaluate_agent(agent, opponent, num_games=100):
    """Evaluate agent without training"""
    env = CambioEnv(opponent)
    wins = 0
    ties = 0
    avg_hand_size = 0
    avg_turns = 0

    # Store original epsilon to restore it later
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    for _ in range(num_games):
        state, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 50:
            steps += 1
            valid_actions = info['valid_actions']
            action = agent.act(state, valid_actions=valid_actions, training=False)
            next_state, reward, done, _, info = env.step(action)

            state = next_state
            if not done and not info['callback']:
                state, info = env.step_opponent()

        agent_score, opp_score = info['final_tally']
        if agent_score < opp_score:  # Lower score is better in Cambio
            wins += 1
        elif agent_score == opp_score:
            ties += 1
        avg_hand_size += agent_score / num_games
        avg_turns += env.turn_count / num_games
    # Restore original epsilon
    agent.epsilon = original_epsilon
    print("average hand value", avg_hand_size, "average game turns", avg_turns)

    return wins / num_games


def plot_learning_curves(rewards, losses, win_rates, epsilon_values, eval_interval):
    """Plot learning curves to track progress"""
    plt.figure(figsize=(15, 10))

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(np.convolve(rewards, np.ones(100) / 100, mode='valid'))
    plt.title('Moving Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(2, 2, 2)
    if losses:
        plt.plot(np.convolve(losses, np.ones(100) / 100, mode='valid'))
        plt.title('Moving Average Loss (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    eval_episodes = [i * eval_interval for i in range(len(win_rates))]
    plt.plot(eval_episodes, win_rates, 'ro-')
    plt.title('Win Rate (every 500 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)

    # Plot epsilon
    plt.subplot(2, 2, 4)
    plt.plot(epsilon_values)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig('dqn_opt_params.png')
    plt.close()


def main():
    # Training phase
    print("Starting training...")
    # agent = run_games("cambio_dqn_weights.weights.h5", num_games=5000, training=True)
    agent = run_games("dqn_opt2_params.weights.h5", num_games=2000, training=True)

    # Full evaluation phase
    print("\nFinal evaluation...")
    opponent = RandomAgent()
    final_win_rate = evaluate_agent(agent, opponent, 1000)
    print(f"Final evaluation over 1000 games: Win rate = {final_win_rate * 100:.2f}%")


def eval_file(file_name):
    agent = DQNAgent()
    agent.load("trained_weights/" + file_name + ".weights.h5")
    opponent = RandomAgent()
    win_rate = evaluate_agent(agent, opponent, 1000)
    print(f"Final evaluation over 1000 games: Win rate = {win_rate * 100:.2f}%")


if __name__ == "__main__":
    main()
    # agent = DQNAgent()
    # agent.load("../trained_weights/checkpoint_4000_dqn_opt_params.weights.h5")
    # opponent = RandomAgent
    # evaluate_agent(agent, opponent, num_games=1000)
