"""
Reinforcement Learning Environment for Neural Network Pruning.

This module implements a Gymnasium-compatible environment where an RL agent
learns to prune neural network layers to achieve optimal compression
while maintaining accuracy (especially on interventional/causal data).

The environment follows the MDP formulation:
- State: Layer information and current model performance
- Action: Pruning ratio to apply to the current layer
- Reward: Balance between compression and accuracy preservation
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from ..pruning import Pruner
from ..utils.metrics import compute_flops, count_parameters


@dataclass
class PruningEnvConfig:
    """Configuration for the pruning environment."""

    # Available pruning ratios (discrete action space)
    pruning_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )

    # Reward function weights
    alpha: float = 1.0  # Weight for accuracy preservation
    beta: float = 0.5  # Weight for compression
    gamma: float = 0.3  # Weight for interventional robustness

    # Constraints
    min_remaining_ratio: float = 0.1  # Minimum parameters to keep
    max_accuracy_drop: float = 0.3  # Maximum allowed accuracy drop

    # Fine-tuning
    finetune_epochs: int = 5
    finetune_lr: float = 0.001


class PruningEnvironment(gym.Env):
    """
    Gymnasium environment for RL-based neural network pruning.

    The agent sequentially decides how much to prune each layer of the network.
    An episode consists of pruning all layers once.

    State Space:
        - layer_index: Current layer being pruned (normalized)
        - layer_params: Number of parameters in current layer (normalized)
        - layer_type: One-hot encoding of layer type
        - cumulative_sparsity: Overall sparsity so far
        - current_val_loss: Validation loss after previous pruning
        - val_loss_ratio: Ratio of current to original validation loss

    Action Space:
        Discrete actions corresponding to pruning ratios (e.g., 0%, 10%, ..., 50%)

    Reward:
        R = -alpha * accuracy_drop + beta * compression_gain + gamma * causal_robustness
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        interventional_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[PruningEnvConfig] = None,
        device: torch.device = None,
    ):
        """
        Initialize the pruning environment.

        Args:
            model: The neural network to prune.
            train_loader: DataLoader for training/fine-tuning.
            val_loader: DataLoader for validation (observational).
            interventional_loader: DataLoader for interventional data.
            config: Environment configuration.
            device: Device to run computations on.
        """
        super().__init__()

        self.config = config or PruningEnvConfig()
        self.device = device or torch.device("cpu")

        # Store original model state dict (avoid deepcopy issues with weight_norm)
        self.original_model = model
        self.original_state_dict = copy.deepcopy(model.state_dict())
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.interventional_loader = interventional_loader

        # Create pruner
        self.pruner = Pruner(self.model)

        # Get prunable layers
        self.prunable_layers = self.pruner.prunable_layers
        self.num_layers = len(self.prunable_layers)

        # Store original metrics
        self.original_params = count_parameters(self.model)
        self.original_val_loss = self._evaluate(self.val_loader)
        self.original_int_loss = (
            self._evaluate(self.interventional_loader)
            if self.interventional_loader
            else self.original_val_loss
        )

        # Define action space (discrete pruning ratios)
        self.action_space = spaces.Discrete(len(self.config.pruning_ratios))

        # Define observation space
        # State: [layer_idx, layer_params, layer_type (one-hot), sparsity, val_loss_ratio]
        self.state_dim = 4 + 3  # 4 scalar features + 3 for layer type one-hot
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Episode state
        self.current_layer_idx = 0
        self.episode_rewards = []
        self.current_val_loss = self.original_val_loss

    def _get_layer_type_onehot(self, layer: nn.Module) -> np.ndarray:
        """Get one-hot encoding of layer type."""
        onehot = np.zeros(3, dtype=np.float32)
        if isinstance(layer, nn.Linear):
            onehot[0] = 1.0
        elif isinstance(layer, nn.Conv1d):
            onehot[1] = 1.0
        elif isinstance(layer, (nn.LSTM, nn.GRU)):
            onehot[2] = 1.0
        return onehot

    def _get_state(self) -> np.ndarray:
        """
        Construct the state observation.

        Returns:
            State vector as numpy array.
        """
        if self.current_layer_idx >= self.num_layers:
            # Terminal state
            return np.zeros(self.state_dim, dtype=np.float32)

        layer_name, layer = self.prunable_layers[self.current_layer_idx]

        # Normalized layer index
        layer_idx_norm = self.current_layer_idx / max(1, self.num_layers - 1)

        # Normalized layer parameters
        layer_params = sum(p.numel() for p in layer.parameters())
        layer_params_norm = layer_params / self.original_params

        # Current sparsity
        sparsity = self.pruner.get_model_sparsity()

        # Validation loss ratio
        val_loss_ratio = self.current_val_loss / (self.original_val_loss + 1e-8)

        # Layer type one-hot
        layer_type = self._get_layer_type_onehot(layer)

        state = np.array(
            [layer_idx_norm, layer_params_norm, sparsity, val_loss_ratio],
            dtype=np.float32,
        )
        state = np.concatenate([state, layer_type])

        return state

    def _evaluate(self, loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate model on a data loader.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Average loss value.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                n_batches += 1

                # Limit evaluation batches for speed
                if n_batches >= 10:
                    break

        return total_loss / max(1, n_batches)

    def _finetune(self, epochs: int = 5) -> float:
        """
        Fine-tune the model after pruning.

        Args:
            epochs: Number of fine-tuning epochs.

        Returns:
            Final validation loss after fine-tuning.
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.finetune_lr
        )
        criterion = nn.MSELoss()

        for _ in range(epochs):
            for batch in self.train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Limit batches per epoch for speed
                break

        return self._evaluate(self.val_loader)

    def _compute_reward(
        self,
        pruning_ratio: float,
        val_loss_before: float,
        val_loss_after: float,
    ) -> float:
        """
        Compute the reward for a pruning action.

        Reward function:
        R = -alpha * accuracy_drop + beta * compression_gain + gamma * causal_bonus

        Args:
            pruning_ratio: The pruning ratio applied.
            val_loss_before: Validation loss before pruning.
            val_loss_after: Validation loss after pruning.

        Returns:
            Reward value.
        """
        # Accuracy drop (higher is worse)
        accuracy_drop = (val_loss_after - val_loss_before) / (val_loss_before + 1e-8)
        accuracy_penalty = -self.config.alpha * max(0, accuracy_drop)

        # Compression gain (higher pruning = more reward)
        compression_reward = self.config.beta * pruning_ratio

        # Causal robustness bonus
        causal_bonus = 0.0
        if self.interventional_loader is not None:
            int_loss = self._evaluate(self.interventional_loader)
            int_loss_ratio = int_loss / (self.original_int_loss + 1e-8)

            # Bonus for maintaining performance on interventional data
            if int_loss_ratio < 1.5:  # Less than 50% degradation
                causal_bonus = self.config.gamma * (1.0 - int_loss_ratio / 1.5)

        reward = accuracy_penalty + compression_reward + causal_bonus

        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options.

        Returns:
            Tuple of (initial_state, info_dict).
        """
        super().reset(seed=seed)

        # Reset model to original state by reloading state dict
        self.model.load_state_dict(copy.deepcopy(self.original_state_dict))
        self.pruner = Pruner(self.model)

        # Reset episode state
        self.current_layer_idx = 0
        self.episode_rewards = []
        self.current_val_loss = self.original_val_loss

        state = self._get_state()
        info = {
            "original_params": self.original_params,
            "original_val_loss": self.original_val_loss,
            "num_layers": self.num_layers,
        }

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute a pruning action.

        Args:
            action: Index into the pruning_ratios list.

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info).
        """
        # Get pruning ratio from action
        pruning_ratio = self.config.pruning_ratios[action]

        # Record loss before pruning
        val_loss_before = self.current_val_loss

        # Apply pruning
        prune_stats = self.pruner.prune_layer(
            self.current_layer_idx, pruning_ratio
        )

        # Fine-tune after pruning (if ratio > 0)
        if pruning_ratio > 0 and self.config.finetune_epochs > 0:
            self._finetune(epochs=self.config.finetune_epochs)

        # Evaluate after pruning
        self.current_val_loss = self._evaluate(self.val_loader)

        # Compute reward
        reward = self._compute_reward(
            pruning_ratio, val_loss_before, self.current_val_loss
        )
        self.episode_rewards.append(reward)

        # Move to next layer
        self.current_layer_idx += 1

        # Check if episode is done
        terminated = self.current_layer_idx >= self.num_layers
        truncated = False

        # Check for constraint violation
        sparsity = self.pruner.get_model_sparsity()
        if sparsity > (1 - self.config.min_remaining_ratio):
            truncated = True
            reward -= 10.0  # Heavy penalty for over-pruning

        # Get next state
        next_state = self._get_state()

        # Compile info
        info = {
            "layer_idx": self.current_layer_idx - 1,
            "pruning_ratio": pruning_ratio,
            "val_loss": self.current_val_loss,
            "sparsity": sparsity,
            "compression_ratio": self.pruner.get_compression_ratio(),
            **prune_stats,
        }

        if terminated:
            info["episode_reward"] = sum(self.episode_rewards)
            info["final_sparsity"] = sparsity
            info["final_compression"] = self.pruner.get_compression_ratio()

        return next_state, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render the environment state."""
        if mode == "human":
            print(f"Layer {self.current_layer_idx}/{self.num_layers}")
            print(f"Current sparsity: {self.pruner.get_model_sparsity():.2%}")
            print(f"Current val loss: {self.current_val_loss:.4f}")
            print(f"Original val loss: {self.original_val_loss:.4f}")

    def get_pruned_model(self) -> nn.Module:
        """Get the current pruned model."""
        return self.model

    def close(self) -> None:
        """Clean up resources."""
        pass


class SimplePruningEnv(gym.Env):
    """
    Simplified pruning environment for initial experiments.

    This version uses a synthetic reward based on layer statistics
    without actually training/evaluating the model. Useful for
    rapid prototyping of RL algorithms.
    """

    def __init__(
        self,
        num_layers: int = 4,
        layer_sizes: Optional[List[int]] = None,
        pruning_ratios: Optional[List[float]] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layer_sizes = layer_sizes or [1000, 2000, 2000, 500]
        self.pruning_ratios = pruning_ratios or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        # Ensure layer_sizes matches num_layers
        if len(self.layer_sizes) != num_layers:
            self.layer_sizes = [1000] * num_layers

        self.total_params = sum(self.layer_sizes)

        # Action space
        self.action_space = spaces.Discrete(len(self.pruning_ratios))

        # State space: [layer_idx_norm, layer_size_norm, cumulative_sparsity]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_layer = 0
        self.pruned_params = [0] * self.num_layers
        self.cumulative_sparsity = 0.0

        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        if self.current_layer >= self.num_layers:
            return np.zeros(3, dtype=np.float32)

        return np.array([
            self.current_layer / self.num_layers,
            self.layer_sizes[self.current_layer] / max(self.layer_sizes),
            self.cumulative_sparsity,
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        ratio = self.pruning_ratios[action]

        # Simulate pruning
        layer_size = self.layer_sizes[self.current_layer]
        pruned = int(layer_size * ratio)
        self.pruned_params[self.current_layer] = pruned

        # Update sparsity
        total_pruned = sum(self.pruned_params)
        self.cumulative_sparsity = total_pruned / self.total_params

        # Simple reward: encourage pruning but penalize over-pruning
        reward = ratio * 0.5  # Reward for compression
        if self.cumulative_sparsity > 0.7:
            reward -= 1.0  # Penalty for over-pruning

        self.current_layer += 1
        terminated = self.current_layer >= self.num_layers

        info = {
            "pruning_ratio": ratio,
            "sparsity": self.cumulative_sparsity,
        }

        return self._get_state(), reward, terminated, False, info
