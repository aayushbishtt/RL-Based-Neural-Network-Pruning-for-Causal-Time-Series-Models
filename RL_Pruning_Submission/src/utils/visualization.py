"""
Visualization utilities for analyzing pruning results.

This module provides functions for creating plots and visualizations
to understand the behavior of the RL pruning agent.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """
    Plot training curves from training history.

    Args:
        history: Dictionary containing training metrics over episodes.
        output_path: Path to save the figure.
        title: Title for the plot.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards
    if "episode_rewards" in history:
        ax = axes[0, 0]
        rewards = history["episode_rewards"]
        ax.plot(rewards, alpha=0.3, label="Raw")

        # Moving average
        window = min(50, len(rewards) // 10 + 1)
        if len(rewards) > window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), ma, label=f"MA({window})")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Rewards")
        ax.legend()

    # Sparsity over episodes
    if "final_sparsities" in history:
        ax = axes[0, 1]
        sparsities = [s * 100 for s in history["final_sparsities"]]
        ax.plot(sparsities, alpha=0.5)
        ax.axhline(y=np.mean(sparsities), color="r", linestyle="--", label="Mean")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sparsity (%)")
        ax.set_title("Achieved Sparsity")
        ax.legend()

    # Compression ratio over episodes
    if "final_compressions" in history:
        ax = axes[1, 0]
        compressions = history["final_compressions"]
        ax.plot(compressions, alpha=0.5)
        ax.axhline(y=np.mean(compressions), color="r", linestyle="--", label="Mean")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Compression Ratio")
        ax.set_title("Compression Ratio")
        ax.legend()

    # Loss curve (if available)
    if "losses" in history and history["losses"]:
        ax = axes[1, 1]
        losses = history["losses"]
        ax.plot(losses, alpha=0.3)

        window = min(100, len(losses) // 10 + 1)
        if len(losses) > window:
            ma = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(losses)), ma, label=f"MA({window})")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_layer_pruning_heatmap(
    action_sequences: List[List[int]],
    pruning_ratios: List[float],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Create a heatmap showing pruning decisions across layers and episodes.

    Args:
        action_sequences: List of action sequences (one per episode).
        pruning_ratios: List of pruning ratio options.
        output_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert actions to pruning ratios
    num_episodes = len(action_sequences)
    num_layers = max(len(seq) for seq in action_sequences) if action_sequences else 0

    heatmap_data = np.zeros((num_episodes, num_layers))
    for i, seq in enumerate(action_sequences):
        for j, action in enumerate(seq):
            heatmap_data[i, j] = pruning_ratios[action]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(pruning_ratios),
        xticklabels=[f"L{i}" for i in range(num_layers)],
        yticklabels=False,
        cbar_kws={"label": "Pruning Ratio"},
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Episode")
    ax.set_title("Pruning Decisions Across Episodes")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_action_distribution(
    action_sequences: List[List[int]],
    pruning_ratios: List[float],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the distribution of pruning actions across all episodes.

    Args:
        action_sequences: List of action sequences.
        pruning_ratios: List of pruning ratio options.
        output_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Flatten all actions
    all_actions = [a for seq in action_sequences for a in seq]

    # Count actions
    action_counts = np.bincount(all_actions, minlength=len(pruning_ratios))
    action_frequencies = action_counts / sum(action_counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of action distribution
    ax = axes[0]
    x = range(len(pruning_ratios))
    ax.bar(x, action_frequencies)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in pruning_ratios])
    ax.set_xlabel("Pruning Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Action Distribution")

    # Per-layer action distribution
    ax = axes[1]
    num_layers = max(len(seq) for seq in action_sequences) if action_sequences else 0

    layer_distributions = []
    for layer_idx in range(num_layers):
        layer_actions = [seq[layer_idx] for seq in action_sequences if layer_idx < len(seq)]
        counts = np.bincount(layer_actions, minlength=len(pruning_ratios))
        layer_distributions.append(counts / sum(counts))

    layer_distributions = np.array(layer_distributions)

    im = ax.imshow(layer_distributions.T, aspect="auto", cmap="Blues")
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)])
    ax.set_yticks(range(len(pruning_ratios)))
    ax.set_yticklabels([f"{r:.0%}" for r in pruning_ratios])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pruning Ratio")
    ax.set_title("Per-Layer Action Distribution")
    plt.colorbar(im, ax=ax, label="Frequency")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sparsity_accuracy_tradeoff(
    sparsities: List[float],
    accuracies: List[float],
    labels: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the trade-off between sparsity and accuracy.

    Args:
        sparsities: List of sparsity values.
        accuracies: List of accuracy values (or inverse loss).
        labels: Optional labels for each point.
        output_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(sparsities, accuracies, c=range(len(sparsities)), cmap="viridis", s=100)

    if labels:
        for i, (s, a, label) in enumerate(zip(sparsities, accuracies, labels)):
            ax.annotate(label, (s, a), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Accuracy (1/Loss)")
    ax.set_title("Sparsity vs. Accuracy Trade-off")

    plt.colorbar(scatter, ax=ax, label="Episode/Trial")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_architecture(
    layer_info: List[Dict],
    pruning_ratios: Optional[List[float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Visualize model architecture and pruning ratios.

    Args:
        layer_info: List of dictionaries with layer information.
        pruning_ratios: Optional list of pruning ratios applied to each layer.
        output_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    num_layers = len(layer_info)
    x = range(num_layers)

    # Plot layer sizes
    params = [info.get("params", 0) for info in layer_info]
    colors = plt.cm.Blues(np.linspace(0.3, 1, num_layers))

    bars = ax.bar(x, params, color=colors)

    # Add pruning ratio overlay if available
    if pruning_ratios:
        for i, (bar, ratio) in enumerate(zip(bars, pruning_ratios)):
            if ratio > 0:
                ax.bar(
                    i,
                    params[i] * ratio,
                    color="red",
                    alpha=0.5,
                    bottom=params[i] * (1 - ratio),
                )

    # Add layer names
    layer_names = [info.get("name", f"Layer {i}") for i, info in enumerate(layer_info)]
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Parameters")
    ax.set_title("Model Architecture and Pruning")

    if pruning_ratios:
        ax.legend(["Original", "Pruned"], loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def create_summary_report(
    results: Dict,
    output_dir: Union[str, Path],
) -> None:
    """
    Create a comprehensive summary report with all visualizations.

    Args:
        results: Dictionary containing training/evaluation results.
        output_dir: Directory to save the report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    if "episode_rewards" in results:
        plot_training_curves(results, output_dir / "training_curves.png")

    # Action analysis
    if "action_sequences" in results and "pruning_ratios" in results:
        plot_action_distribution(
            results["action_sequences"],
            results["pruning_ratios"],
            output_dir / "action_distribution.png",
        )

        plot_layer_pruning_heatmap(
            results["action_sequences"],
            results["pruning_ratios"],
            output_dir / "pruning_heatmap.png",
        )

    # Trade-off analysis
    if "sparsities" in results and "val_losses" in results:
        accuracies = [1.0 / (loss + 1e-8) for loss in results["val_losses"]]
        plot_sparsity_accuracy_tradeoff(
            results["sparsities"],
            accuracies,
            output_dir=output_dir / "tradeoff.png",
        )

    print(f"Summary report saved to {output_dir}")
