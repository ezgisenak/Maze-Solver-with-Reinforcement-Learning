# Maze Solver with Reinforcement Learning

This repository contains the implementation of a maze solver using reinforcement learning algorithms as part of the homework assignment for METU EE 449: Computational Intelligence course.

## Overview

In this homework, we explore the application of Reinforcement Learning (RL) to solve a maze navigation problem by employing Temporal Difference (TD) Learning and Q Learning. The task involves modeling a maze, implementing and analyzing TD(0) Learning and Q Learning to solve it, and providing various plots such as heatmaps and learning curves.

## Project Structure

- `main.py`: Main script containing the implementation of the maze environment, TD(0) Learning, and Q Learning algorithms.
- `utils.py`: Utility functions used for plotting heatmaps, policies, and convergence plots.
- `EE449_HW3_2024.pdf`: Homework assignment details.
- `EE449_HW3_2024_Report.pdf`: Homework assignment report.

## Getting Started

### Prerequisites

Ensure you have Python installed. The required libraries are specified in `requirements.txt`.

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/maze-solver-rl.git
   cd maze-solver-rl
2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
### Usage
To run the maze solver, execute the following command:
```sh
python main.py
```

### Changing Hyperparameters
You can change the hyperparameters (learning rate, discount factor, initial exploration rate) via the terminal when running the script. Use the following command format:

```sh
python main.py --alpha <learning_rate> --gamma <discount_factor> --epsilon <exploration_rate>
```
For example, to set the learning rate to 0.1, discount factor to 0.95, and exploration rate to 0.2, you would run:

```sh
python main.py --alpha 0.1 --gamma 0.95 --epsilon 0.2
```


## Implementation Details
### Maze Environment
The maze is modeled as a grid with the following components:

Start position: Blue cell
- Traps: Red cells
- Goal: Green cell
- Free space: White cells
- The environment is stochastic with defined transition probabilities.

### Temporal Difference (TD) Learning
The TD(0) algorithm is implemented to update the utility value estimate after every action taken. The agent uses an ε-greedy strategy for action selection.

### Q Learning
Q Learning algorithm is implemented to directly learn an optimal policy. The agent initializes the Q-values (state-action pair values) arbitrarily and updates them using the Q Learning update rule with an ε-greedy strategy.

### Experimental Work
The project includes experiments with various hyperparameters (learning rate, discount factor, initial exploration rate) to analyze their effects on the learning process. Plots are generated to visualize utility value functions, policies, and convergence.

## Results
The results of the experiments, including utility value function heatmaps, policies, and convergence plots, are presented in the report (EE449_HW3_2024_Report.pdf). The report also includes a discussion on the experimental findings, comparing TD(0) Learning and Q Learning.

## Conclusion
The project successfully demonstrates the application of TD(0) Learning and Q Learning to solve a maze navigation problem. The analysis provides insights into the effects of different hyperparameters on the learning process and the performance of the algorithms.
