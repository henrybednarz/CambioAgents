import numpy as np
import time
from gym.randomAgent import RandomAgent
from gym.agents.DQN2 import OptimizedDQNAgent  # Import the optimized agent
from gym.agents.fastDQN import FastDQNAgent
import multiprocessing
import os
import tensorflow as tf

# Make TensorFlow quieter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def run_game_session(session_id, num_games, save_path=None, load_path=None):
    """Run a session of games with the optimized agent"""
    print(f"Starting session {session_id} with {num_games} games")

    # Initialize agents
    random_agent = RandomAgent(0)
    dqn_agent = OptimizedDQNAgent(1, state_size=11, action_size=5)

    # Load pretrained model if provided
    if load_path:
        dqn_agent.load_model(load_path)

    # Import here to avoid circular imports
    from env import CambioEnv

    # Initialize environment
    env = CambioEnv(random_agent, dqn_agent)

    # Track metrics
    wins = 0
    total_reward = 0
    start_time = time.time()

    for i in range(num_games):
        env.reset()
        game_active = True
        turn_count = 0

        # Run the game
        while game_active:
            turn_count += 1
            game_active = env.next_turn()

            # Break if game runs too long
            if turn_count > 50:
                break

        # Check who won
        hand_values = env.tally_hands()
        if 1 == np.argmax(hand_values):
            wins += 1

        # Progress update
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Session {session_id} - Completed {i + 1}/{num_games} games. Win rate: {wins / (i + 1) * 100:.2f}% "
                  f"({elapsed:.2f}s, {(i + 1) / elapsed:.2f} games/s)")

    # Save model if path provided
    if save_path:
        dqn_agent.save_model(f"weights/{save_path}_session_{session_id}.weights.h5")

    end_time = time.time()
    elapsed = end_time - start_time
    win_rate = wins / num_games * 100

    print(f"Session {session_id} complete - {num_games} games in {elapsed:.2f}s "
          f"({num_games / elapsed:.2f} games/s). Win rate: {win_rate:.2f}%")

    return {
        'session_id': session_id,
        'games': num_games,
        'wins': wins,
        'win_rate': win_rate,
        'elapsed_time': elapsed,
        'games_per_second': num_games / elapsed
    }


def parallel_training(total_games=100000, sessions=4, save_prefix="dqn_model"):
    """Run multiple training sessions in parallel"""
    print(f"Starting parallel training with {sessions} sessions for a total of {total_games} games")

    games_per_session = total_games // sessions
    processes = []

    # Start processes
    for i in range(sessions):
        # Decide whether to load the model from previous session
        load_path = None
        if i > 0:
            load_path = f"weights/{save_prefix}_session_{i - 1}.weights.h5"

        p = multiprocessing.Process(
            target=run_game_session,
            args=(i, games_per_session, save_prefix, load_path)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"All {sessions} training sessions completed.")


def sequential_training_with_evaluation(games_per_epoch=10000, epochs=10, eval_games=1000):
    """Train the agent sequentially with evaluation periods"""
    print(f"Starting sequential training for {epochs} epochs with {games_per_epoch} games per epoch")

    # Track metrics across epochs
    epoch_metrics = []
    best_win_rate = 0
    best_model_path = None

    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")

        # Load model from previous epoch if available
        load_path = None if epoch == 0 else f"weights/dqn_model_epoch_{epoch}.weights.h5"
        save_path = f"dqn_model_epoch_{epoch + 1}"

        # Training phase
        print("Training phase:")
        metrics = run_game_session(
            session_id=f"epoch_{epoch + 1}_train",
            num_games=games_per_epoch,
            save_path=save_path,
            load_path=load_path
        )

        # Evaluation phase with exploration disabled
        print("\nEvaluation phase:")
        # Create agents for evaluation
        random_agent = RandomAgent(0)
        fast_dqn = FastDQNAgent(1)
        dqn_agent = OptimizedDQNAgent(1, state_size=11, action_size=5)
        dqn_agent.load_model(f"weights/{save_path}_session_epoch_{epoch + 1}_train.weights.h5")
        dqn_agent2 = OptimizedDQNAgent(0, state_size=11, action_size=5)
        dqn_agent2.load_model(f"weights/{save_path}_session_epoch_{epoch + 1}_train.weights.h5")

        # Set to evaluation mode
        dqn_agent.epsilon = 0.05  # Small epsilon for some exploration
        dqn_agent.training = False

        # Import here to avoid circular imports
        from env import CambioEnv

        # Initialize environment
        env = CambioEnv(random_agent, fast_dqn)

        # Evaluate
        wins = 0
        start_time = time.time()

        for i in range(eval_games):
            env.reset()
            game_active = True
            while game_active:
                game_active = env.next_turn()

            # Check who won
            hand_values = env.tally_hands()
            if 1 == np.argmax(hand_values):
                wins += 1

        eval_win_rate = wins / eval_games * 100
        eval_time = time.time() - start_time

        print(f"Epoch {epoch + 1} evaluation: {eval_win_rate:.2f}% win rate over {eval_games} games "
              f"({eval_time:.2f}s, {eval_games / eval_time:.2f} games/s)")

        # Save best model
        if eval_win_rate > best_win_rate:
            best_win_rate = eval_win_rate
            best_model_path = f"weights/{save_path}_best.weights.h5"
            dqn_agent.save_model(best_model_path)
            print(f"New best model saved with {best_win_rate:.2f}% win rate")

        # Store metrics
        epoch_metrics.append({
            'epoch': epoch + 1,
            'training_wins': metrics['win_rate'],
            'eval_win_rate': eval_win_rate,
            'training_speed': metrics['games_per_second'],
            'eval_speed': eval_games / eval_time
        })

    print("\n--- Training Complete ---")
    print(f"Best model achieved {best_win_rate:.2f}% win rate and was saved to {best_model_path}")

    # Print metrics summary
    print("\nTraining Metrics Summary:")
    print("Epoch | Train Win% | Eval Win% | Train Speed | Eval Speed")
    print("-" * 60)
    for m in epoch_metrics:
        print(f"{m['epoch']:5d} | {m['training_wins']:9.2f}% | {m['eval_win_rate']:8.2f}% | "
              f"{m['training_speed']:10.2f} | {m['eval_speed']:9.2f} games/s")


if __name__ == "__main__":
    # Choose training method based on system capabilities
    multiprocessing_available = multiprocessing.cpu_count() >= 4

    if multiprocessing_available:
        print(f"System has {multiprocessing.cpu_count()} cores. Using parallel training.")
        parallel_training(total_games=1000, sessions=2)
    else:
        print("Using sequential training with evaluation.")
        sequential_training_with_evaluation(games_per_epoch=2000, epochs=10, eval_games=1000)