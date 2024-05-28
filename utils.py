import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

def plot_value_function(value_function, maze, episode, folder_path):
    mask = np.zeros_like(value_function, dtype=bool)
    mask[maze == 1] = True  # Mask obstacles
    mask[maze == 2] = True  # Mask the trap
    mask[maze == 3] = True  # Mask the goal

    trap_position = tuple(np.array(np.where(maze == 2)).transpose(1, 0))
    goal_position = np.where(maze == 3)
    obs_position = tuple(np.array(np.where(maze == 1)).transpose(1, 0))

    plt.figure(figsize=(10, 10))
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    ax = sns.heatmap(value_function, mask=mask, annot=True, fmt=".1f", cmap=cmap,
                     cbar=False, linewidths=1, linecolor='black')
    ax.add_patch(plt.Rectangle(goal_position[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkgreen'))
    for t in trap_position:
        ax.add_patch(plt.Rectangle(t[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkred'))
    for o in obs_position:
        ax.add_patch(plt.Rectangle(o[::-1], 1, 1, fill=True, edgecolor='black', facecolor='gray'))
    ax.set_title(f"Value Function at Episode {episode}")

    # Save the figure
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, f"value_function_episode_{episode}.png"))
    plt.close()

def plot_policy(value_function, maze, episode, folder_path):
    policy_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    policy_grid = np.full(maze.shape, '', dtype='<U2')
    actions = ['up', 'down', 'left', 'right']

    trap_position = tuple(np.array(np.where(maze == 2)).transpose(1, 0))
    goal_position = np.where(maze == 3)
    obs_position = tuple(np.array(np.where(maze == 1)).transpose(1, 0))

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 1 or (i, j) == goal_position:
                continue  # Skip obstacles and the goal
            best_action = None
            best_value = float('-inf')
            for action in actions:
                next_i, next_j = i, j
                if action == 'up':
                    next_i -= 1
                elif action == 'down':
                    next_i += 1
                elif action == 'left':
                    next_j -= 1
                elif action == 'right':
                    next_j += 1
                if 0 <= next_i < maze.shape[0] and 0 <= next_j < maze.shape[1]:
                    if value_function[next_i][next_j] > best_value:
                        best_value = value_function[next_i][next_j]
                        best_action = action
            if best_action:
                policy_grid[i][j] = policy_arrows[best_action]

    mask = np.zeros_like(value_function, dtype=bool)
    mask[maze == 1] = True  # Mask obstacles
    mask[maze == 2] = True  # Mask the trap
    mask[maze == 3] = True  # Mask the goal

    plt.figure(figsize=(10, 10))
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    ax = sns.heatmap(value_function, mask=mask, annot=policy_grid, fmt="", cmap=cmap,
                     cbar=False, linewidths=1, linecolor='black')
    ax.add_patch(plt.Rectangle(goal_position[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkgreen'))
    for t in trap_position:
        ax.add_patch(plt.Rectangle(t[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkred'))
    for o in obs_position:
        ax.add_patch(plt.Rectangle(o[::-1], 1, 1, fill=True, edgecolor='black', facecolor='gray'))
    ax.set_title(f"Policy at Episode {episode}")

    # Save the figure
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, f"policy_episode_{episode}.png"))
    plt.close()

def plot_convergence(value_history, folder_path):
    episodes = [vh[0] for vh in value_history]
    diffs = [np.sum(np.abs(value_history[i+1][1] - value_history[i][1])) for i in range(len(value_history)-1)]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes[1:], diffs, label='Sum of Absolute Differences')
    
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Absolute Differences')
    plt.title('Convergence Plot')
    plt.grid(True)

    # Save the figure
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "convergence.png"))
    plt.close()

def combine_plots(folder_path):
    episodes = [1, 50, 100, 1000, 5000, 10000]
    plot_types = ["policy", "value_function"]

    for plot_type in plot_types:
        images = []
        output_path = os.path.join(folder_path, f"combined_{plot_type}_plots.png")

        for episode in episodes:
            plot_path = os.path.join(folder_path, f"{plot_type}_episode_{episode}.png")
            
            if os.path.exists(plot_path):
                image = Image.open(plot_path)
                images.append(image)
            else:
                print(f"Missing {plot_type} plot for episode {episode}")
        
        # Create a new image with a suitable size
        if images:
            width, height = images[0].size
            combined_image = Image.new('RGB', (width * 3, height * 2))  # 3 columns, 2 rows
            
            for i, image in enumerate(images):
                row = i // 3
                col = i % 3
                combined_image.paste(image, (col * width, row * height))
            
            combined_image.save(output_path)
        else:
            print(f"No {plot_type} images to combine.")


