import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns


class CambioAnalyzer:
    """Tool for analyzing Cambio gameplay and agent learning"""

    def __init__(self):
        self.action_labels = [
            "played to card 1",
            "played to card 2",
            "played to card 3",
            "played to center",
            "called cambio",
            "peeked own card 1",
            "peeked own card 2",
            "peeked own card 3",
            "peeked opp card 1",
            "peeked opp card 2",
            "peeked opp card 3",
            "swap own 1 for opp 1",
            "swap own 1 for opp 2",
            "swap own 1 for opp 3",
            "swap own 2 for opp 1",
            "swap own 2 for opp 2",
            "swap own 2 for opp 3",
            "swap own 3 for opp 1",
            "swap own 3 for opp 2",
            "swap own 3 for opp 3",
            "pass open action"
        ]
        self.action_stats = defaultdict(int)
        self.actions_by_turn = defaultdict(lambda: defaultdict(int))
        self.win_actions = defaultdict(int)
        self.loss_actions = defaultdict(int)
        self.state_values = []
        self.game_outcomes = []

    def log_action(self, action, turn, win=None):
        """Log an action taken by the agent"""
        self.action_stats[action] += 1
        self.actions_by_turn[turn][action] += 1

        if win is not None:
            if win:
                self.win_actions[action] += 1
            else:
                self.loss_actions[action] += 1

    def log_state_value(self, state_value, is_terminal=False):
        """Log state value estimates"""
        self.state_values.append((state_value, is_terminal))

    def log_game_outcome(self, agent_score, opponent_score):
        """Log game outcome"""
        win = agent_score < opponent_score  # Lower is better in Cambio
        self.game_outcomes.append((agent_score, opponent_score, win))

    def analyze_actions(self):
        """Analyze which actions are most frequently taken"""
        total_actions = sum(self.action_stats.values())
        print("Action distribution:")
        for action, count in sorted(self.action_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                action_name = self.action_labels[action]
                percentage = (count / total_actions) * 100
                print(f"{action_name}: {count} ({percentage:.2f}%)")

    def analyze_win_rate_by_action(self):
        """Analyze win rate by action"""
        print("\nWin rate by action:")
        for action in range(len(self.action_labels)):
            wins = self.win_actions[action]
            losses = self.loss_actions[action]
            total = wins + losses
            if total > 0:
                win_rate = (wins / total) * 100
                action_name = self.action_labels[action]
                print(f"{action_name}: {win_rate:.2f}% ({wins}/{total})")

    def visualize_action_distribution(self):
        """Visualize action distribution"""
        action_counts = [self.action_stats[i] for i in range(len(self.action_labels))]

        plt.figure(figsize=(15, 8))
        plt.bar(self.action_labels, action_counts)
        plt.xticks(rotation=90)
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('action_distribution.png')
        plt.close()

    def visualize_action_by_turn(self):
        """Visualize actions taken at different turns"""
        # Create a matrix of turns x actions
        max_turn = max(self.actions_by_turn.keys())
        action_matrix = np.zeros((max_turn + 1, len(self.action_labels)))

        for turn in range(max_turn + 1):
            for action in range(len(self.action_labels)):
                action_matrix[turn, action] = self.actions_by_turn[turn][action]

        # Create a heatmap
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(action_matrix[:20, :], annot=False, fmt='.1f',
                         xticklabels=self.action_labels,
                         yticklabels=range(1, 21))
        plt.title('Actions by Turn (First 20 Turns)')
        plt.xlabel('Action')
        plt.ylabel('Turn')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('actions_by_turn.png')
        plt.close()

    def visualize_win_rate_by_action(self):
        """Visualize win rate by action"""
        win_rates = []
        for action in range(len(self.action_labels)):
            wins = self.win_actions[action]
            losses = self.loss_actions[action]
            total = wins + losses
            win_rate = (wins / total) * 100 if total > 0 else 0
            win_rates.append(win_rate)

        plt.figure(figsize=(15, 8))
        plt.bar(self.action_labels, win_rates)
        plt.axhline(y=50, color='r', linestyle='--', label='50% Win Rate')
        plt.xticks(rotation=90)
        plt.title('Win Rate by Action')
        plt.xlabel('Action')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('win_rate_by_action.png')
        plt.close()

    def visualize_state_value_progression(self):
        """Visualize how state values progress over time"""
        values = [v[0] for v in self.state_values]
        is_terminal = [v[1] for v in self.state_values]

        plt.figure(figsize=(15, 8))
        plt.plot(values, label='State Value')
        terminal_indices = [i for i, term in enumerate(is_terminal) if term]
        terminal_values = [values[i] for i in terminal_indices]
        plt.scatter(terminal_indices, terminal_values, color='red', label='Terminal States')
        plt.title('State Value Progression')
        plt.xlabel('State Index')
        plt.ylabel('Estimated State Value')
        plt.legend()
        plt.savefig('state_value_progression.png')
        plt.close()

    def run_full_analysis(self):
        """Run all analysis and visualization functions"""
        self.analyze_actions()
        self.analyze_win_rate_by_action()
        self.visualize_action_distribution()
        self.visualize_action_by_turn()
        self.visualize_win_rate_by_action()
        self.visualize_state_value_progression()

        # Calculate overall statistics
        wins = sum(1 for _, _, win in self.game_outcomes if win)
        total_games = len(self.game_outcomes)
        win_rate = (wins / total_games) * 100 if total_games > 0 else 0

        print(f"\nOverall win rate: {win_rate:.2f}% ({wins}/{total_games})")

        # Return win rate for use in training
        return win_rate


"""
# Initialize the analyzer
analyzer = CambioAnalyzer()

# During training/evaluation:
for episode in range(num_episodes):
    # Run episode
    for step in range(num_steps):
        # Get action
        action = agent.act(state)

        # Log action with turn number
        analyzer.log_action(action, step)

        # Log state value
        analyzer.log_state_value(state_value)

        # Take step in environment
        # ...

    # Log game outcome
    analyzer.log_game_outcome(agent_score, opponent_score)

    # Also update win/loss for actions
    for action in episode_actions:
        analyzer.log_action(action, 0, win=(agent_score < opponent_score))

# After training/evaluation
analyzer.run_full_analysis()
"""