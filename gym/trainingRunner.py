import time
import gymnasium as gym
from stepEnvWrapper import CambioEnvWrapper
from randomAgent import RandomAgent
from remake.DQN_agent import DQNAgent
from GymEnv import CambioGymEnv
import numpy as np
from tqdm import tqdm


class CombinedObservationWrapper(gym.ObservationWrapper):
    """Wrapper to flatten the dictionary observation space into a single vector."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Calculate total size of the flattened observation
        total_size = 3 + 3 + 1 + 1 + 1 + 1 + 1  # All observation components
        # Define the new observation space
        self.observation_space = gym.spaces.Box(
            low=-1, high=1000, shape=(total_size,), dtype=np.float32
        )

    def observation(self, obs):
        return obs

    def tally_hands(self):
        return self.env.tally_hands()


def train_with_callback_handling(num_games=1000):
    """Train the DQN agent with the modified game flow that handles callbacks."""
    agent = DQNAgent()
    opponent = RandomAgent(1)

    # Create the base environment
    base_env = CambioGymEnv(opponent)

    # Wrap it with our custom wrapper that handles opponent turns when appropriate
    env = CambioEnvWrapper(base_env)

    # Variables to track metrics
    losses = []
    episode_rewards = []

    for episode in tqdm(range(num_games), desc="Training", unit="game"):
        state, info = env.reset()
        total_reward = 0
        episode_loss = []
        step_count = 0
        terminated = False

        while not terminated and step_count < 100:  # Step limit to prevent infinite loops
            step_count += 1

            # Get valid actions from info
            valid_actions = info.get('valid_actions', np.ones(env.action_space.n, dtype=bool))

            # Agent selects action
            print(info)
            action = agent.act(state, valid_actions=valid_actions, training=True)

            # Execute action and get new state
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, terminated)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            # Train the agent on a batch of experiences
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)

        # Record metrics for this episode
        episode_rewards.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"\nEpisode {episode + 1}/{num_games}")
            print(f"Avg Reward: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")

    # Save the trained model
    agent.save("trained_agent.h5")

    print("\nTraining Complete!")
    print(f"Final Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")

    return {
        'losses': losses,
        'rewards': episode_rewards
    }

def train_dqn(num_games=1000, update_frequency=10):
    try:
        agent = DQNAgent()
        opponent = RandomAgent(1)
        env = CambioGymEnv(opponent)
        env = CambioEnvWrapper(env)
        env = CombinedObservationWrapper(env)

        # Variables to track loss
        losses = []
        episode_rewards = []

        for episode in tqdm(range(num_games), desc="training", unit="game"):
            step = 0
            state, info = env.reset()
            terminated = False
            total_reward = 0
            episode_loss = []

            while not terminated and step < 100:  # Add step limit to prevent infinite loops
                step += 1
                valid_actions = info.get('valid_actions', np.ones(env.action_space.n, dtype=bool))
                action = agent.act(state, valid_actions=valid_actions, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.remember(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward

                loss = agent.train()
                if loss is not None:
                    episode_loss.append(loss)

            # Record metrics
            episode_rewards.append(total_reward)
            if episode_loss:
                losses.append(np.mean(episode_loss))

            # Print progress every 100 episodes
            if (episode + 1) % 500 == 0:
                avg_loss = np.mean(losses[-100:]) if losses else 0
                avg_reward = np.mean(episode_rewards[-100:])
                avg_epsilon = agent.epsilon
                print(f"\nEpisode {episode + 1}/{num_games}")
                print(f"Avg Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.2f} | Epsilon: {avg_epsilon:.4f}")

        # Final statistics
        print("\nTraining Complete!")
        print(f"Final Avg Loss: {np.mean(losses[-100:]):.4f}")
        print(f"Final Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"Final Epsilon: {agent.epsilon:.4f}")

        agent.save("finalagent/dqn_weights3.weights.h5")

        # Return metrics for plotting if needed
        return {
            'losses': losses,
            'rewards': episode_rewards
        }
    except:
        agent.save("finalagent/uhoh.weights.h5")


def test_dqn(file_name, num_games=1000):
    agent = DQNAgent()
    agent.load(file_name)
    opponent = RandomAgent(1)
    env = CambioGymEnv(opponent)
    env = CombinedObservationWrapper(env)
    wins = 0
    ties = 0
    losses = 0
    stall_cnt = 0

    # Track more metrics
    game_lengths = []
    win_margins = []
    loss_margins = []

    for game in tqdm(range(num_games), desc="testing", unit="game"):
        step = 0
        state, info = env.reset()
        terminated = False
        stall_flag = False

        while not terminated and step < 50:
            step += 1
            valid_actions = info.get('valid_actions', np.ones(env.action_space.n, dtype=bool))
            action = agent.act(state, valid_actions=valid_actions, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

        if step >= 50:
            stall_cnt += 1
        game_lengths.append(step)
        try:
            agent_score, opp_score = info['final_values']
        except:
            agent_score, opp_score = env.tally_hands()
        margin = abs(agent_score - opp_score)

        if agent_score < opp_score:
            wins += 1
            win_margins.append(margin)
        elif agent_score == opp_score:
            ties += 1
        else:
            losses += 1
            loss_margins.append(margin)

    # Calculate completed games
    completed_games = num_games - stall_cnt
    win_rate = (wins / completed_games * 100) if completed_games > 0 else 0

    print("\nTesting Complete!")
    print(f"Completed Games: {completed_games}/{num_games} ({completed_games / num_games * 100:.1f}%)")
    print(f"Agent W-T-L Record: {wins}-{ties}-{losses}")
    print(f"Win Rate: {win_rate:.1f}%")

    if game_lengths:
        print(f"Average Game Length: {np.mean(game_lengths):.1f} steps")
    if win_margins:
        print(f"Average Win Margin: {np.mean(win_margins):.1f} points")
    if loss_margins:
        print(f"Average Loss Margin: {np.mean(loss_margins):.1f} points")

    return {
        'wins': wins,
        'ties': ties,
        'losses': losses,
        'stalled': stall_cnt,
        'game_lengths': game_lengths,
        'win_margins': win_margins if win_margins else []
    }

def slow_play(file_name):
    agent = DQNAgent()
    agent.load(file_name)
    opponent = RandomAgent(1)
    env = CambioGymEnv(opponent)
    env = CombinedObservationWrapper(env)
    step = 0
    state, info = env.reset()
    terminated = False
    stall_flag = False

    while not terminated and step < 50:
        time.sleep(0.2)
        # env.render()
        step += 1
        valid_actions = info.get('valid_actions', np.ones(env.action_space.n, dtype=bool))
        action = agent.act(state, valid_actions=valid_actions, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
    try:
        agent_score, opp_score = info['final_values']
    except:
        agent_score, opp_score = env.tally_hands()
    margin = abs(agent_score - opp_score)

    print(f"Game Complete: {agent_score} to {opp_score}")


def main():
    ans = input("Train 't', Evaluate 'e', or slow play 's' agent?: ")
    if ans == 't':
        train_dqn(25000, 2)

        print("Testing agent...")
        test_dqn("finalagent/dqn_weights3.weights.h5", 1000)
    elif ans == 'e':
        test_dqn("finalagent/uhoh.weights.h5", 1000)
    elif ans == 's':
        slow_play("finalagent/dqn_weights3.weights.h5")
    elif ans == 'g':
        train_with_callback_handling()
    else:
        print("thats not a response vro")

main()