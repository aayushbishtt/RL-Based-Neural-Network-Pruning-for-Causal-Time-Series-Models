# RL-Based Neural Network Pruning for Causal Time Series Models

This project implements **Reinforcement Learning-based Neural Network Pruning** for time series models trained on the **Causal Chamber** wind tunnel dataset. The goal is to train an RL agent that learns optimal layer-wise pruning strategies while maintaining predictive accuracy, especially under distributional shifts (interventional data).

## Project Overview

### The Problem

Neural network pruning traditionally uses heuristics (e.g., magnitude-based pruning) that don't account for:
1. **Layer-specific sensitivity** - Different layers have different importance
2. **Accuracy-compression trade-off** - Finding the optimal balance requires careful tuning
3. **Causal robustness** - Pruned models should generalize under interventions

### Our Approach

We frame pruning as a **Markov Decision Process (MDP)**:

| Component | Definition |
|-----------|------------|
| **State** | Layer index, parameter count, current sparsity, validation loss |
| **Action** | Pruning ratio (0%, 10%, 20%, ..., 50%) |
| **Reward** | `-α·accuracy_drop + β·compression + γ·causal_robustness` |

The RL agent learns to make layer-by-layer pruning decisions that optimize this reward, resulting in models that are both compact and causally robust.

### Novel Contribution

Unlike standard pruning approaches that focus on image classification (ResNet, VGG), we apply RL-based pruning to:
1. **Time series architectures** (TCN, LSTM)
2. **Causal robustness objective** - Penalizing pruning decisions that hurt performance on interventional data

## Project Structure

```
rl_pruning/
├── configs/
│   └── default.yaml          # Configuration file
├── src/
│   ├── data/                  # Data loading and preprocessing
│   │   ├── causal_chamber.py  # Causal Chamber dataset loader
│   │   └── preprocessing.py   # Data preprocessing utilities
│   ├── models/                # Neural network architectures
│   │   ├── base.py            # Base model class
│   │   ├── tcn.py             # Temporal Convolutional Network
│   │   └── lstm.py            # LSTM model
│   ├── pruning/               # Pruning utilities
│   │   ├── pruner.py          # Main pruning controller
│   │   └── structured.py      # Structured pruning methods
│   ├── environments/          # RL environments
│   │   └── pruning_env.py     # Gymnasium-compatible pruning environment
│   ├── agents/                # RL agents
│   │   ├── base.py            # Base agent class
│   │   ├── q_learning.py      # Tabular Q-Learning
│   │   └── dqn.py             # Deep Q-Network
│   └── utils/                 # Utility functions
│       ├── config.py          # Configuration management
│       ├── metrics.py         # Model metrics (FLOPs, parameters)
│       ├── logging_utils.py   # Logging utilities
│       └── visualization.py   # Plotting functions
├── scripts/
│   ├── train_base_model.py    # Train the forecasting model
│   ├── train_pruning_agent.py # Train the RL agent
│   └── evaluate.py            # Evaluate and compare methods
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

### 1. Clone and Setup

```bash
cd rl_pruning
pip install -r requirements.txt
```

### 2. Install Causal Chamber Dataset (Optional)

```bash
pip install causalchamber
```

If the `causalchamber` package is not available, the code will use synthetic data that mimics the structure of the wind tunnel experiments.

## Quick Start

### Run Demo (Quick Test)

```bash
python main.py demo
```

This runs a quick pipeline with minimal settings to verify everything works.

### Full Pipeline

```bash
# Step 1: Train base time series model
python main.py train_base --epochs 100 --model_type tcn

# Step 2: Train RL pruning agent
python main.py train_agent --episodes 500 --agent_type dqn

# Step 3: Evaluate and compare with baselines
python main.py evaluate --compare_baselines
```

### Or run everything at once:

```bash
python main.py full_pipeline --epochs 100 --episodes 500
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model architecture
base_model:
  type: "tcn"  # or "lstm"
  tcn:
    num_channels: [32, 64, 64, 32]
    kernel_size: 3

# RL agent
agent:
  type: "dqn"  # or "q_learning"
  dqn:
    learning_rate: 0.0001
    epsilon_decay_steps: 10000

# Reward function weights
environment:
  reward:
    alpha: 1.0   # Accuracy preservation
    beta: 0.5    # Compression reward
    gamma: 0.3   # Causal robustness bonus
```

## MDP Formulation

### State Space

For each layer being pruned, the state includes:
- **Layer index** (normalized)
- **Layer parameters** (normalized by total)
- **Layer type** (one-hot: Linear, Conv1d, LSTM)
- **Cumulative sparsity** (fraction of zeros so far)
- **Validation loss ratio** (current / original)

### Action Space

Discrete pruning ratios: `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`

### Reward Function

```
R = -α·(val_loss_after - val_loss_before)/val_loss_before
    + β·pruning_ratio
    + γ·causal_robustness_bonus
```

Where the **causal robustness bonus** rewards maintaining performance on interventional data.

## Agents

### Q-Learning (Tabular)
- Suitable for small state spaces
- States discretized into bins
- Good for initial experiments

### DQN (Deep Q-Network)
- Neural network function approximation
- Experience replay + target network
- Handles continuous state spaces

## Results and Visualization

After training, check the results in:

```
checkpoints/
├── best_model.pt              # Trained base model
├── rl_agent/
│   ├── best_agent.pt          # Trained RL agent
│   ├── training_history.json  # Training metrics
│   └── tensorboard/           # TensorBoard logs
└── results/
    ├── evaluation_results.json
    ├── comparison.png         # Method comparison plot
    └── pareto_frontier.png    # Sparsity vs accuracy
```

### View TensorBoard Logs

```bash
tensorboard --logdir checkpoints/rl_agent/tensorboard
```

## Key Files for Understanding the Code

1. **`src/environments/pruning_env.py`** - The core RL environment
2. **`src/agents/dqn.py`** - DQN implementation
3. **`src/models/tcn.py`** - TCN architecture
4. **`scripts/train_pruning_agent.py`** - Training loop

## Extending the Project

### Add a New Model Architecture

1. Create a new file in `src/models/`
2. Inherit from `BaseTimeSeriesModel`
3. Implement `forward()` and optionally `get_prunable_layers()`
4. Register in `src/models/__init__.py`

### Add a New RL Agent

1. Create a new file in `src/agents/`
2. Inherit from `BaseAgent`
3. Implement `select_action()`, `learn()`, `save()`, `load()`
4. Register in `src/agents/__init__.py`

### Modify the Reward Function

Edit `_compute_reward()` in `src/environments/pruning_env.py`

## References

### Pruning
- He et al. (2018). "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"

### Causal Chamber
- Gamella, Peters, & Bühlmann (2024). "Causal chambers as a real-world physical testbed for AI methodology." Nature Machine Intelligence.

### TCN
- Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"

### DQN
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.

## License

MIT License

## Author

[Your Name]
