#!/usr/bin/env python3
"""
RL-Based Neural Network Pruning for Causal Time Series Models

Main entry point for the project. This script provides a unified interface
to run all components of the pruning pipeline.

Usage:
    python main.py train_base      # Train the base time series model
    python main.py train_agent     # Train the RL pruning agent
    python main.py evaluate        # Evaluate and compare methods
    python main.py full_pipeline   # Run the complete pipeline

Examples:
    # Quick test with synthetic data
    python main.py train_base --epochs 10

    # Full training pipeline
    python main.py full_pipeline --config configs/default.yaml

    # Evaluate trained agent against baselines
    python main.py evaluate --compare_baselines
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_train_base(args):
    """Run base model training."""
    cmd = [
        sys.executable,
        "scripts/train_base_model.py",
        "--config", args.config,
        "--output_dir", args.output_dir,
    ]
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.model_type:
        cmd.extend(["--model_type", args.model_type])

    subprocess.run(cmd, cwd=Path(__file__).parent)


def run_train_agent(args):
    """Run RL agent training."""
    cmd = [
        sys.executable,
        "scripts/train_pruning_agent.py",
        "--config", args.config,
        "--model_path", f"{args.output_dir}/best_model.pt",
        "--output_dir", f"{args.output_dir}/rl_agent",
    ]
    if args.episodes:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.agent_type:
        cmd.extend(["--agent_type", args.agent_type])
    if args.use_simple_env:
        cmd.append("--use_simple_env")

    subprocess.run(cmd, cwd=Path(__file__).parent)


def run_evaluate(args):
    """Run evaluation."""
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--config", args.config,
        "--model_path", f"{args.output_dir}/best_model.pt",
        "--agent_path", f"{args.output_dir}/rl_agent/best_agent.pt",
        "--output_dir", f"{args.output_dir}/results",
    ]
    if args.compare_baselines:
        cmd.append("--compare_baselines")
    if args.agent_type:
        cmd.extend(["--agent_type", args.agent_type])
    if getattr(args, "use_simple_env", False):
        cmd.append("--use_simple_env")

    subprocess.run(cmd, cwd=Path(__file__).parent)


def run_full_pipeline(args):
    """Run the complete pipeline."""
    print("=" * 60)
    print("STEP 1: Training Base Model")
    print("=" * 60)
    run_train_base(args)

    print("\n" + "=" * 60)
    print("STEP 2: Training RL Pruning Agent")
    print("=" * 60)
    run_train_agent(args)

    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)
    args.compare_baselines = True
    run_evaluate(args)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="RL-Based Neural Network Pruning for Causal Time Series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train base model
    train_base_parser = subparsers.add_parser(
        "train_base", help="Train the base time series model"
    )
    train_base_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_base_parser.add_argument(
        "--model_type", choices=["tcn", "lstm"], help="Model architecture"
    )

    # Train agent
    train_agent_parser = subparsers.add_parser(
        "train_agent", help="Train the RL pruning agent"
    )
    train_agent_parser.add_argument("--episodes", type=int, help="Number of episodes")
    train_agent_parser.add_argument(
        "--agent_type", choices=["q_learning", "dqn"], help="Agent type"
    )
    train_agent_parser.add_argument(
        "--use_simple_env", action="store_true", help="Use simplified environment"
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument(
        "--compare_baselines", action="store_true", help="Compare with baselines"
    )
    eval_parser.add_argument(
        "--agent_type", choices=["q_learning", "dqn"], default="dqn"
    )

    # Full pipeline
    pipeline_parser = subparsers.add_parser(
        "full_pipeline", help="Run complete training and evaluation pipeline"
    )
    pipeline_parser.add_argument("--epochs", type=int, default=50, help="Base model epochs")
    pipeline_parser.add_argument("--episodes", type=int, default=200, help="RL episodes")
    pipeline_parser.add_argument(
        "--model_type", choices=["tcn", "lstm"], default="tcn"
    )
    pipeline_parser.add_argument(
        "--agent_type", choices=["q_learning", "dqn"], default="dqn"
    )
    pipeline_parser.add_argument("--use_simple_env", action="store_true")

    # Demo/test command
    demo_parser = subparsers.add_parser("demo", help="Quick demo with minimal settings")

    args = parser.parse_args()

    if args.command == "train_base":
        run_train_base(args)
    elif args.command == "train_agent":
        run_train_agent(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "full_pipeline":
        run_full_pipeline(args)
    elif args.command == "demo":
        # Quick demo settings
        args.epochs = 5
        args.episodes = 20
        args.model_type = "tcn"
        args.agent_type = "dqn"
        args.use_simple_env = True
        args.output_dir = "demo_output"
        run_full_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
