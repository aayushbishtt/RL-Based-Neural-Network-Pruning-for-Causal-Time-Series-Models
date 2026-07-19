#!/usr/bin/env python3
"""
Train the RL agent for neural network pruning.

This script trains a DQN or Q-learning agent to learn optimal pruning
strategies for time series models.

Usage:
    python scripts/train_pruning_agent.py --config configs/default.yaml
    python scripts/train_pruning_agent.py --agent_type dqn --episodes 500
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from tqdm import tqdm

from src.data import load_causal_chamber_data, create_data_loaders
from src.models import create_model, load_model
from src.environments import PruningEnvironment, PruningEnvConfig
from src.agents import QLearningAgent, DQNAgent
from src.utils import (
    load_config,
    get_device,
    set_seed,
    setup_logger,
    TensorBoardLogger,
    MetricTracker,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL pruning agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to pre-trained base model",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["q_learning", "dqn"],
        help="Agent type (overrides config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of training episodes (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/rl_agent",
        help="Directory to save agent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_simple_env",
        action="store_true",
        help="Use simplified environment for faster prototyping",
    )
    return parser.parse_args()


def train_agent(
    agent,
    env,
    num_episodes: int,
    eval_frequency: int = 10,
    save_frequency: int = 50,
    output_dir: Path = None,
    logger=None,
    tb_logger=None,
):
    """
    Train the RL agent.

    Args:
        agent: The RL agent to train.
        env: The pruning environment.
        num_episodes: Number of training episodes.
        eval_frequency: How often to log evaluation metrics.
        save_frequency: How often to save checkpoints.
        output_dir: Directory to save checkpoints.
        logger: Python logger.
        tb_logger: TensorBoard logger.

    Returns:
        Training history.
    """
    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "final_sparsities": [],
        "final_compressions": [],
        "losses": [],
    }

    best_reward = float("-inf")
    tracker = MetricTracker()

    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Learn
            loss = agent.learn(state, action, reward, next_state, done)

            # Track
            episode_reward += reward
            episode_length += 1
            state = next_state

            if loss is not None:
                tracker.update("loss", loss)

        # End of episode
        agent.end_episode(episode_reward)

        # Record history
        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(episode_length)
        history["final_sparsities"].append(info.get("final_sparsity", 0))
        history["final_compressions"].append(info.get("final_compression", 1))

        # Logging
        if episode % eval_frequency == 0:
            avg_reward = np.mean(history["episode_rewards"][-eval_frequency:])
            avg_sparsity = np.mean(history["final_sparsities"][-eval_frequency:])
            avg_compression = np.mean(history["final_compressions"][-eval_frequency:])

            if logger:
                logger.info(
                    f"Episode {episode}/{num_episodes} - "
                    f"Avg Reward: {avg_reward:.3f}, "
                    f"Avg Sparsity: {avg_sparsity:.2%}, "
                    f"Avg Compression: {avg_compression:.2f}x, "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

            if tb_logger:
                tb_logger.log_scalar("reward/episode", episode_reward, episode)
                tb_logger.log_scalar("reward/average", avg_reward, episode)
                tb_logger.log_scalar("sparsity/final", info.get("final_sparsity", 0), episode)
                tb_logger.log_scalar("compression/final", info.get("final_compression", 1), episode)
                tb_logger.log_scalar("epsilon", agent.epsilon, episode)

                avg_loss = tracker.get_average("loss")
                if avg_loss > 0:
                    tb_logger.log_scalar("loss/average", avg_loss, episode)
                tracker.reset("loss")

        # Save checkpoints
        if output_dir and episode % save_frequency == 0:
            agent.save(output_dir / f"agent_episode_{episode}.pt")

        # Save best
        if episode_reward > best_reward:
            best_reward = episode_reward
            if output_dir:
                agent.save(output_dir / "best_agent.pt")

    return history


def evaluate_agent(agent, env, num_episodes: int = 10, logger=None):
    """
    Evaluate trained agent.

    Args:
        agent: Trained RL agent.
        env: Pruning environment.
        num_episodes: Number of evaluation episodes.
        logger: Python logger.

    Returns:
        Evaluation metrics.
    """
    rewards = []
    sparsities = []
    compressions = []
    actions_taken = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_actions = []
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            episode_actions.append(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        sparsities.append(info.get("final_sparsity", 0))
        compressions.append(info.get("final_compression", 1))
        actions_taken.append(episode_actions)

    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_sparsity": np.mean(sparsities),
        "mean_compression": np.mean(compressions),
        "action_distribution": np.bincount(
            [a for actions in actions_taken for a in actions],
            minlength=env.action_space.n,
        ),
    }

    if logger:
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
        logger.info(f"  Mean Sparsity: {metrics['mean_sparsity']:.2%}")
        logger.info(f"  Mean Compression: {metrics['mean_compression']:.2f}x")
        logger.info(f"  Action Distribution: {metrics['action_distribution']}")

    return metrics


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    agent_config = config["agent"]
    if args.agent_type:
        agent_config["type"] = args.agent_type
    if args.episodes:
        agent_config["training"]["num_episodes"] = args.episodes

    # Setup
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("hardware", {}).get("device", "auto"))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger("train_rl", log_dir=output_dir)
    tb_logger = TensorBoardLogger(output_dir / "tensorboard", comment="rl_agent")

    logger.info(f"Using device: {device}")
    logger.info(f"Agent type: {agent_config['type']}")

    # Load data
    logger.info("Loading data...")
    data_config = config["data"]

    obs_data, int_data = load_causal_chamber_data(
        dataset_name=data_config.get("dataset_name", "lt_camera_walks_v1")
    )

    # Determine features
    available_columns = obs_data.columns.tolist()
    input_features = [f for f in data_config["input_features"] if f in available_columns]
    if not input_features:
        input_features = ["pressure", "temperature", "airflow", "fan_speed"]
        input_features = [f for f in input_features if f in available_columns]

    target_feature = data_config["target_feature"]
    if target_feature not in available_columns:
        target_feature = "pol_1"

    # Create data loaders
    loaders, _ = create_data_loaders(
        obs_data=obs_data,
        int_data=int_data,
        input_features=input_features,
        target_feature=target_feature,
        sequence_length=data_config.get("sequence_length", 50),
        batch_size=data_config.get("batch_size", 64),
    )

    # Load or create base model
    model_path = Path(args.model_path)
    model_config = config["base_model"]

    model = create_model(
        model_type=model_config["type"],
        input_size=len(input_features),
        output_size=1,
        sequence_length=data_config.get("sequence_length", 50),
        **model_config.get(model_config["type"], {}),
    )

    if model_path.exists():
        logger.info(f"Loading pre-trained model from {model_path}")
        model, _, _ = load_model(model, model_path, device=device)
    else:
        logger.warning(f"Model not found at {model_path}, using untrained model")
        model = model.to(device)

    # Create environment
    if args.use_simple_env:
        from src.environments.pruning_env import SimplePruningEnv
        env = SimplePruningEnv(
            num_layers=len(model.get_prunable_layers()),
            pruning_ratios=config["pruning"]["pruning_ratios"],
        )
        logger.info("Using simplified environment")
    else:
        env_config = PruningEnvConfig(
            pruning_ratios=config["pruning"]["pruning_ratios"],
            alpha=config["environment"]["reward"]["alpha"],
            beta=config["environment"]["reward"]["beta"],
            gamma=config["environment"]["reward"]["gamma"],
            finetune_epochs=config["pruning"].get("finetune_epochs", 5),
        )

        env = PruningEnvironment(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            interventional_loader=loaders.get("interventional"),
            config=env_config,
            device=device,
        )
        logger.info("Using full pruning environment")

    logger.info(f"State dim: {env.observation_space.shape[0]}")
    logger.info(f"Action dim: {env.action_space.n}")

    # Create agent
    agent_type = agent_config["type"]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if agent_type == "q_learning":
        q_config = agent_config["q_learning"]
        agent = QLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=q_config["learning_rate"],
            discount_factor=q_config["discount_factor"],
            epsilon_start=q_config["epsilon_start"],
            epsilon_end=q_config["epsilon_end"],
            epsilon_decay=q_config["epsilon_decay"],
        )
    elif agent_type == "dqn":
        dqn_config = agent_config["dqn"]
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=dqn_config["learning_rate"],
            discount_factor=dqn_config["discount_factor"],
            epsilon_start=dqn_config["epsilon_start"],
            epsilon_end=dqn_config["epsilon_end"],
            epsilon_decay_steps=dqn_config["epsilon_decay_steps"],
            batch_size=dqn_config["batch_size"],
            buffer_size=dqn_config["buffer_size"],
            target_update_freq=dqn_config["target_update_freq"],
            hidden_sizes=dqn_config["hidden_sizes"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    logger.info(f"Created {agent_type} agent")

    # Train agent
    train_config = agent_config["training"]
    num_episodes = train_config["num_episodes"]
    eval_frequency = train_config["eval_frequency"]
    save_frequency = train_config["save_frequency"]

    logger.info(f"Training for {num_episodes} episodes...")

    history = train_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
        eval_frequency=eval_frequency,
        save_frequency=save_frequency,
        output_dir=output_dir,
        logger=logger,
        tb_logger=tb_logger,
    )

    # Final evaluation
    logger.info("Final evaluation...")
    eval_metrics = evaluate_agent(agent, env, num_episodes=10, logger=logger)

    # Save final agent
    agent.save(output_dir / "final_agent.pt")

    # Save training history
    import json
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(
            {
                "episode_rewards": history["episode_rewards"],
                "final_sparsities": history["final_sparsities"],
                "final_compressions": history["final_compressions"],
                "eval_metrics": {
                    "mean_reward": float(eval_metrics["mean_reward"]),
                    "mean_sparsity": float(eval_metrics["mean_sparsity"]),
                    "mean_compression": float(eval_metrics["mean_compression"]),
                },
            },
            f,
            indent=2,
        )

    tb_logger.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
