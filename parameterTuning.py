import numpy as np
import matplotlib.pyplot as plt
from cambio_env import CambioEnv
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from tqdm import tqdm
import os
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
import time


def hyperparameter_tuning(num_games=2000, eval_interval=100):
    """
    Test different hyperparameter combinations and track performance metrics.

    Args:
        num_games: Number of training games for each configuration
        eval_interval: Interval for evaluation and metric logging
    """
    # Define hyperparameter ranges to test
    gamma_values = [0.90, 0.95]
    epsilon_values = [0.3]
    learning_rates = [0.00001, 0.00005, 0.0001]

    # Create directory for results
    os.makedirs("hyperparameter_tuning", exist_ok=True)
    os.makedirs("hyperparameter_tuning/plots", exist_ok=True)
    os.makedirs("hyperparameter_tuning/weights", exist_ok=True)

    # Create list of all hyperparameter combinations
    param_combinations = list(itertools.product(gamma_values, epsilon_values, learning_rates))

    # For parallel processing
    results = []

    # Train and evaluate each combination
    for i, (gamma, epsilon, lr) in enumerate(param_combinations):
        print(f"Training configuration {i + 1}/{len(param_combinations)}")
        print(f"Gamma: {gamma}, Epsilon: {epsilon}, Learning Rate: {lr}")

        config_name = f"g{gamma}_e{epsilon}_lr{lr}"
        result = train_and_evaluate(
            gamma=gamma,
            epsilon=epsilon,
            learning_rate=lr,
            config_name=config_name,
            num_games=num_games,
            eval_interval=eval_interval
        )
        results.append(result)

    # Compile and save overall results
    compile_results(results, param_combinations)


def train_and_evaluate(gamma, epsilon, learning_rate, config_name, num_games, eval_interval):
    """
    Train and evaluate a DQN agent with specific hyperparameters.

    Args:
        gamma: Discount factor for future rewards
        epsilon: Initial exploration rate
        learning_rate: Learning rate for optimizer
        config_name: Name for saving files
        num_games: Number of training games
        eval_interval: Interval for evaluation

    Returns:
        Dictionary with training metrics and config info
    """
    # Initialize agent with specified hyperparameters
    agent = DQNAgent(
        gamma=gamma,
        epsilon=epsilon,
        learning_rate=learning_rate
    )

    opponent = RandomAgent()
    env = CambioEnv(opponent)

    # Tracking metrics
    losses = []
    rewards = []
    win_rates = []
    eval_episodes = []

    # Best model tracking
    best_win_rate = 0.0

    for episode in tqdm(range(num_games), desc=f"Training {config_name}", unit="games"):
        total_reward = 0
        episode_losses = []
        done = False

        state, info = env.reset()

        # Play one game
        while not done:
            valid_actions = info['valid_actions']

            # Get action
            action = agent.act(state, valid_actions=valid_actions, training=True)

            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            opponent_plays_next = not info['callback']
            total_reward += reward

            # Remember experience
            agent.remember(state, action, reward, next_state, done, valid_actions)

            # Update state
            if opponent_plays_next and not done:
                next_state, info = env.step_opponent()

            state = next_state

            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

        # Record metrics
        rewards.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))

        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_agent(agent, opponent, 100)
            win_rates.append(win_rate)
            eval_episodes.append(episode + 1)

            # Log progress
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)

            print(f"\nEpisode {episode + 1}/{num_games}")
            print(f"Win Rate: {win_rate * 100:.2f}% | Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")

            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(f"hyperparameter_tuning/weights/best_{config_name}.weights.h5")

    # Final evaluation
    final_win_rate = evaluate_agent(agent, opponent, 200)
    print(f"Final win rate for {config_name}: {final_win_rate * 100:.2f}%")

    # Plot learning curves
    plot_learning_curves(rewards, losses, win_rates, eval_episodes, config_name)

    # Return results
    return {
        'config_name': config_name,
        'gamma': gamma,
        'epsilon': epsilon,
        'learning_rate': learning_rate,
        'rewards': rewards,
        'losses': losses,
        'win_rates': win_rates,
        'eval_episodes': eval_episodes,
        'final_win_rate': final_win_rate
    }


def evaluate_agent(agent, opponent, num_games=100):
    """Evaluate agent without training"""
    env = CambioEnv(opponent)
    wins = 0

    # Store original epsilon to restore it later
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation

    for _ in range(num_games):
        state, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 50:  # Limit to prevent infinite games
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

    # Restore original epsilon
    agent.epsilon = original_epsilon

    return wins / num_games


def plot_learning_curves(rewards, losses, win_rates, eval_episodes, config_name):
    """Plot separate learning curves for each metric"""

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(losses, np.ones(min(100, len(losses))) / min(100, len(losses)), mode='valid'))
    plt.title(f'Moving Average Loss - {config_name}')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'hyperparameter_tuning/plots/{config_name}_loss.png')
    plt.close()

    # Plot average reward
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(rewards, np.ones(min(100, len(rewards))) / min(100, len(rewards)), mode='valid'))
    plt.title(f'Moving Average Reward - {config_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(f'hyperparameter_tuning/plots/{config_name}_reward.png')
    plt.close()

    # Plot win rate
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, win_rates, 'ro-')
    plt.title(f'Win Rate - {config_name}')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(f'hyperparameter_tuning/plots/{config_name}_winrate.png')
    plt.close()


def compile_results(results, param_combinations):
    """Compile and visualize overall results"""

    # Extract final win rates
    final_win_rates = [result['final_win_rate'] for result in results]

    # Create DataFrame
    df = pd.DataFrame({
        'Gamma': [p[0] for p in param_combinations],
        'Epsilon': [p[1] for p in param_combinations],
        'Learning Rate': [p[2] for p in param_combinations],
        'Config Name': [result['config_name'] for result in results],
        'Final Win Rate': final_win_rates
    })

    # Save to CSV
    df.to_csv('hyperparameter_tuning/results_summary.csv', index=False)

    # Plot heatmap for best parameters
    plt.figure(figsize=(15, 10))

    # For each gamma value, plot epsilon vs learning rate
    for i, gamma in enumerate(df['Gamma'].unique()):
        plt.subplot(1, len(df['Gamma'].unique()), i + 1)

        # Filter for this gamma
        df_gamma = df[df['Gamma'] == gamma]

        # Reshape data for heatmap
        pivot = df_gamma.pivot(index='Epsilon', columns='Learning Rate', values='Final Win Rate')

        # Plot heatmap
        plt.imshow(pivot, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Win Rate')
        plt.title(f'Gamma = {gamma}')
        plt.ylabel('Epsilon')
        plt.xlabel('Learning Rate')

        # Add text annotations
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                plt.text(x, y, f'{pivot.iloc[y, x]:.2f}',
                         ha='center', va='center',
                         color='white' if pivot.iloc[y, x] < 0.6 else 'black')

    plt.tight_layout()
    plt.savefig('hyperparameter_tuning/parameter_heatmap.png')
    plt.close()

    # Plot comparison of all configurations
    plt.figure(figsize=(12, 8))

    # Sort by win rate
    df = df.sort_values('Final Win Rate', ascending=False)

    plt.barh(df['Config Name'], df['Final Win Rate'])
    plt.xlabel('Final Win Rate')
    plt.ylabel('Configuration')
    plt.title('Performance Comparison of Different Hyperparameter Configurations')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning/configuration_comparison.png')
    plt.close()

    # Return best configuration
    best_idx = np.argmax(final_win_rates)
    best_config = param_combinations[best_idx]
    best_win_rate = final_win_rates[best_idx]

    print("\n========== RESULTS ==========")
    print(f"Best configuration: Gamma={best_config[0]}, Epsilon={best_config[1]}, Learning Rate={best_config[2]}")
    print(f"Best win rate: {best_win_rate * 100:.2f}%")
    print("=============================\n")

    # Also print top 3 configurations
    print("Top 3 configurations:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"{i + 1}. {row['Config Name']}: {row['Final Win Rate'] * 100:.2f}%")


def main():
    """Run hyperparameter tuning"""
    start_time = time.time()
    hyperparameter_tuning(num_games=1000, eval_interval=100)
    duration = time.time() - start_time
    print(f"Total time: {duration / 3600:.2f} hours")


if __name__ == "__main__":
    main()