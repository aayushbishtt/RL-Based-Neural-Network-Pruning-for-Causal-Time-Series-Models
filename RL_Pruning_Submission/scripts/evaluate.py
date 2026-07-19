#!/usr/bin/env python3
"""
Evaluate trained pruning agent and compare with baselines.

This script evaluates:
1. The trained RL pruning agent
2. Random pruning baseline
3. Uniform pruning baseline
4. Magnitude-based pruning baseline

Usage:
    python scripts/evaluate.py --agent_path checkpoints/rl_agent/best_agent.pt
    python scripts/evaluate.py --compare_baselines --output_dir results
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_causal_chamber_data, create_data_loaders
from src.models import create_model, load_model
from src.environments import PruningEnvironment, PruningEnvConfig
from src.agents import QLearningAgent, DQNAgent
from src.agents.base import RandomAgent
from src.pruning import Pruner
from src.utils import load_config, get_device, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pruning strategies")
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
        "--agent_path",
        type=str,
        default="checkpoints/rl_agent/best_agent.pt",
        help="Path to trained RL agent",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="dqn",
        choices=["q_learning", "dqn"],
        help="Type of agent to load",
    )
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Compare with baseline pruning methods",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
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
        help="Use simplified state representation (for compatibility with demo mode)",
    )
    return parser.parse_args()


def evaluate_agent_policy(agent, env, num_episodes: int = 10):
    """Evaluate an agent and collect detailed metrics."""
    results = {
        "rewards": [],
        "sparsities": [],
        "compressions": [],
        "val_losses": [],
        "action_sequences": [],
    }

    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        actions = []
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            actions.append(action)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        results["rewards"].append(episode_reward)
        results["sparsities"].append(step_info.get("final_sparsity", 0))
        results["compressions"].append(step_info.get("final_compression", 1))
        results["val_losses"].append(step_info.get("val_loss", 0))
        results["action_sequences"].append(actions)

    return results


def uniform_pruning_baseline(env, pruning_ratio: float, num_episodes: int = 10):
    """Baseline: Apply uniform pruning ratio to all layers."""
    results = {
        "rewards": [],
        "sparsities": [],
        "compressions": [],
        "val_losses": [],
    }

    # Find the action index for the desired pruning ratio
    ratios = getattr(env, "pruning_ratios", None)
    if ratios is None and hasattr(env, "config"):
        ratios = env.config.pruning_ratios
        
    action_idx = min(
        range(len(ratios)),
        key=lambda i: abs(ratios[i] - pruning_ratio),
    )

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        results["rewards"].append(episode_reward)
        results["sparsities"].append(info.get("final_sparsity", 0))
        results["compressions"].append(info.get("final_compression", 1))
        results["val_losses"].append(info.get("val_loss", 0))

    return results


def plot_comparison(results_dict: dict, output_path: Path):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = list(results_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    # Reward comparison
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        rewards = results_dict[method]["rewards"]
        ax.bar(i, np.mean(rewards), yerr=np.std(rewards), color=colors[i], capsize=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Comparison")

    # Sparsity comparison
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        sparsities = results_dict[method]["sparsities"]
        ax.bar(i, np.mean(sparsities) * 100, yerr=np.std(sparsities) * 100, color=colors[i], capsize=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Sparsity (%)")
    ax.set_title("Achieved Sparsity")

    # Compression ratio comparison
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        compressions = results_dict[method]["compressions"]
        ax.bar(i, np.mean(compressions), yerr=np.std(compressions), color=colors[i], capsize=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression Achieved")

    # Validation loss comparison
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        losses = results_dict[method]["val_losses"]
        ax.bar(i, np.mean(losses), yerr=np.std(losses), color=colors[i], capsize=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Final Validation Loss")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(results_dict: dict, output_path: Path):
    """Plot sparsity vs. accuracy trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    for i, (method, results) in enumerate(results_dict.items()):
        sparsities = results["sparsities"]
        losses = results["val_losses"]

        ax.scatter(
            sparsities,
            losses,
            c=[colors[i]] * len(sparsities),
            label=method,
            alpha=0.7,
            s=100,
        )

        # Add mean point
        ax.scatter(
            np.mean(sparsities),
            np.mean(losses),
            c=[colors[i]],
            marker="*",
            s=300,
            edgecolors="black",
        )

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Sparsity vs. Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    config = load_config(args.config)
    device = get_device(config.get("hardware", {}).get("device", "auto"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("evaluate", log_dir=output_dir)
    logger.info(f"Using device: {device}")

    # Load data
    data_config = config["data"]
    obs_data, int_data = load_causal_chamber_data()

    available_columns = obs_data.columns.tolist()
    input_features = [f for f in data_config["input_features"] if f in available_columns]
    if not input_features:
        input_features = ["pressure", "temperature", "airflow", "fan_speed"]
        input_features = [f for f in input_features if f in available_columns]

    target_feature = data_config["target_feature"]
    if target_feature not in available_columns:
        target_feature = "pol_1"

    loaders, _ = create_data_loaders(
        obs_data=obs_data,
        int_data=int_data,
        input_features=input_features,
        target_feature=target_feature,
        sequence_length=data_config.get("sequence_length", 50),
        batch_size=data_config.get("batch_size", 64),
    )

    # Load base model
    model_config = config["base_model"]
    model = create_model(
        model_type=model_config["type"],
        input_size=len(input_features),
        output_size=1,
        sequence_length=data_config.get("sequence_length", 50),
        **model_config.get(model_config["type"], {}),
    )

    model_path = Path(args.model_path)
    if model_path.exists():
        model, _, _ = load_model(model, model_path, device=device)
        logger.info(f"Loaded model from {model_path}")
    else:
        model = model.to(device)
        logger.warning("Using untrained model")

    # Create environment
    env_config = PruningEnvConfig(
        alpha=config["environment"]["reward"]["alpha"],
        beta=config["environment"]["reward"]["beta"],
        gamma=config["environment"]["reward"]["gamma"],
        finetune_epochs=0,  # No fine-tuning during evaluation
    )

    if args.use_simple_env:
        logger.info("Using simplified environment for evaluation")
        from src.environments.pruning_env import SimplePruningEnv
        env = SimplePruningEnv(
            num_layers=4,
            layer_sizes=[1000, 2000, 2000, 500],
            pruning_ratios=env_config.pruning_ratios,
        )
    else:
        env = PruningEnvironment(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            interventional_loader=loaders.get("interventional"),
            config=env_config,
            device=device,
        )

    results_dict = {}

    # Evaluate RL agent
    agent_path = Path(args.agent_path)
    if agent_path.exists():
        logger.info(f"Loading agent from {agent_path}")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        if args.agent_type == "dqn":
            agent_config = config["agent"]["dqn"]
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=agent_config["hidden_sizes"],
                device=device,
            )
        else:
            agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim)

        agent.load(agent_path)
        logger.info("Evaluating RL agent...")
        results_dict["RL Agent"] = evaluate_agent_policy(agent, env, args.num_episodes)
    else:
        logger.warning(f"Agent not found at {agent_path}")

    # Evaluate baselines
    if args.compare_baselines:
        logger.info("Evaluating random baseline...")
        random_agent = RandomAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        results_dict["Random"] = evaluate_agent_policy(random_agent, env, args.num_episodes)

        logger.info("Evaluating uniform pruning baselines...")
        for ratio in [0.1, 0.2, 0.3]:
            results_dict[f"Uniform {int(ratio*100)}%"] = uniform_pruning_baseline(
                env, ratio, args.num_episodes
            )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    for method, results in results_dict.items():
        logger.info(f"\n{method}:")
        logger.info(f"  Reward: {np.mean(results['rewards']):.3f} ± {np.std(results['rewards']):.3f}")
        logger.info(f"  Sparsity: {np.mean(results['sparsities']):.2%} ± {np.std(results['sparsities']):.2%}")
        logger.info(f"  Compression: {np.mean(results['compressions']):.2f}x ± {np.std(results['compressions']):.2f}x")
        logger.info(f"  Val Loss: {np.mean(results['val_losses']):.4f} ± {np.std(results['val_losses']):.4f}")

    # Save results
    serializable_results = {}
    for method, results in results_dict.items():
        serializable_results[method] = {
            k: [float(v) for v in vals] if isinstance(vals[0], (int, float, np.floating)) else vals
            for k, vals in results.items()
            if k != "action_sequences"
        }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Create plots
    if len(results_dict) > 1:
        logger.info("Creating comparison plots...")
        plot_comparison(results_dict, output_dir / "comparison.png")
        plot_pareto_frontier(results_dict, output_dir / "pareto_frontier.png")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
