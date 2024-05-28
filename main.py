import argparse
import os
import shutil 
import numpy as np
from utils import plot_value_function, plot_policy, plot_convergence, combine_plots

class MazeEnvironment:
    def __init__(self):
        self.maze = np.array([
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 3],
            [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ])
        self.start_pos = (0, 0)  # Start position of the agent
        self.current_pos = self.start_pos
        self.state_penalty = -1
        self.trap_penalty = -100
        self.goal_reward = 100
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        row, col = self.current_pos
        action_prob = np.random.rand()
        
        if action_prob <= 0.75:
            next_pos = (row + self.actions[action][0], col + self.actions[action][1])
        elif action_prob <= 0.80:
            next_pos = (row - self.actions[action][0], col - self.actions[action][1])
        else:
            next_pos = (row + self.actions[(action + 2) % 4][0], col + self.actions[(action + 2) % 4][1])
        
        if (0 <= next_pos[0] < self.maze.shape[0]) and (0 <= next_pos[1] < self.maze.shape[1]) and self.maze[next_pos] != 1:
            self.current_pos = next_pos
        else:
            next_pos = self.current_pos

        reward = self.state_penalty
        if self.maze[next_pos] == 2:
            reward = self.trap_penalty
        elif self.maze[next_pos] == 3:
            reward = self.goal_reward
        
        done = self.maze[next_pos] in (2, 3)

        return next_pos, reward, done

class MazeTD0(MazeEnvironment):
    def __init__(self, maze, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.utility = np.zeros(self.maze.shape)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(self.actions.keys()))
        else:
            action_values = []
            for action in self.actions:
                next_pos = (state[0] + self.actions[action][0], state[1] + self.actions[action][1])
                if (0 <= next_pos[0] < self.maze.shape[0]) and (0 <= next_pos[1] < self.maze.shape[1]) and self.maze[next_pos] != 1:
                    action_values.append(self.utility[next_pos])
                else:
                    action_values.append(float('-inf'))
            return np.argmax(action_values)

    def update_utility_value(self, current_state, reward, new_state):
        current_value = self.utility[current_state]
        new_value = reward + self.gamma * self.utility[new_state]
        self.utility[current_state] += self.alpha * (new_value - current_value)

    def run_episodes(self):
        value_history = []
        for episode in range(self.episodes+1):
            state = self.reset()
            done = False
            steps = 0
            while not done and steps < 1000:
                action = self.choose_action(state)
                new_state, reward, done = self.step(action)
                self.update_utility_value(state, reward, new_state)
                state = new_state
                steps += 1
                if done:
                    # Perform one last update for the terminal state
                    action = self.choose_action(state)
                    new_state, reward, done = self.step(action)
                    self.update_utility_value(state, reward, new_state)
                    state = new_state
                    steps += 1

            value_history.append((episode, self.utility.copy()))
            if episode % 1000 == 0:
                print(f"Episode {episode} completed")
        return value_history

class MazeQLearning(MazeEnvironment):
    def __init__(self, maze, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.random.rand(*self.maze.shape, len(self.actions)) * 0.1

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(self.actions.keys()))
        else:
            state_actions = self.q_table[state[0], state[1], :]
            return np.argmax(state_actions)

    def update_q_table(self, action, current_state, reward, new_state):
        current_q = self.q_table[current_state[0], current_state[1], action]
        max_future_q = np.max(self.q_table[new_state[0], new_state[1], :])
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[current_state[0], current_state[1], action] = new_q

    def run_episodes(self):
        value_history = []
        for episode in range(self.episodes+1):
            state = self.reset()
            done = False
            steps = 0
            while not done and steps < 1000:
                action = self.choose_action(state)
                new_state, reward, done = self.step(action)
                self.update_q_table(action, state, reward, new_state)
                state = new_state
                steps += 1
                if done:
                    # Perform one last update for the terminal state
                    action = self.choose_action(state)
                    new_state, reward, done = self.step(action)
                    self.update_q_table(action, state, reward, new_state)
                    state = new_state
                    steps += 1
                    break
            utility_values = np.max(self.q_table, axis=2)
            value_history.append((episode, utility_values.copy()))
            if episode % 1000 == 0:
                print(f"Episode {episode} completed")
        return value_history

# Running experiments for TD(0) and Q-Learning
def run_experiments(args):
    maze = MazeEnvironment()
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    episodes = args.episodes

    # TD(0) Learning Experiments
    folder_name = f"TD_Learning_alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}"
    experiment_folder = os.path.join("results", folder_name)
    plot_folder = os.path.join(experiment_folder, "plots")

    # Remove the experiment folder if it exists
    if os.path.exists(experiment_folder):
        shutil.rmtree(experiment_folder)
    os.makedirs(plot_folder, exist_ok=True)

    print(f"TD(0) Learning with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
    maze_td0 = MazeTD0(maze.maze, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    td0_value_history = maze_td0.run_episodes()
    for episode, utility_values in td0_value_history:
        if episode in [1, 50, 100, 1000, 5000, 10000]:
            plot_value_function(utility_values, maze.maze, episode, plot_folder)
            plot_policy(utility_values, maze.maze, episode, plot_folder)
    plot_convergence(td0_value_history, plot_folder)
    combine_plots(plot_folder)

    # Q-Learning Experiments
    folder_name = f"Q_Learning_alpha_{alpha}_gamma_{gamma}_epsilon_{epsilon}"
    experiment_folder = os.path.join("results", folder_name)
    plot_folder = os.path.join(experiment_folder, "plots")

    # Remove the experiment folder if it exists
    if os.path.exists(experiment_folder):
        shutil.rmtree(experiment_folder)
    os.makedirs(plot_folder, exist_ok=True)

    print(f"Q-Learning with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
    maze_q_learning = MazeQLearning(maze.maze, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    q_learning_value_history = maze_q_learning.run_episodes()
    for episode, utility_values in q_learning_value_history:
        if episode in [1, 50, 100, 1000, 5000, 10000]:
            plot_value_function(utility_values, maze.maze, episode, plot_folder)
            plot_policy(utility_values, maze.maze, episode, plot_folder)
    plot_convergence(q_learning_value_history, plot_folder)
    combine_plots(plot_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TD(0) and Q-Learning experiments on a maze.")
    parser.add_argument('--alpha', type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument('--gamma', type=float, default=0.95, help="Discount factor (default: 0.95)")
    parser.add_argument('--epsilon', type=float, default=0.2, help="Exploration rate (default: 0.2)")
    parser.add_argument('--episodes', type=int, default=10000, help="Number of episodes (default: 10000)")
    args = parser.parse_args()

    run_experiments(args)
