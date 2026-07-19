"""
Base agent class for RL-based pruning.

This module defines the abstract base class that all agents must implement,
ensuring a consistent interface for training and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    All agents must implement the core methods for action selection,
    learning, and state management.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Initialize the base agent.

        Args:
            state_dim: Dimension of the state space.
            action_dim: Number of possible actions.
            learning_rate: Learning rate for updates.
            discount_factor: Discount factor (gamma) for future rewards.
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Decay rate for epsilon.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.total_rewards: List[float] = []

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current state observation.
            training: Whether in training mode (enables exploration).

        Returns:
            Selected action index.
        """
        pass

    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Update the agent based on a transition.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode is done.

        Returns:
            Loss value if applicable, None otherwise.
        """
        pass

    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def reset_epsilon(self) -> None:
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start

    def end_episode(self, episode_reward: float) -> None:
        """
        Called at the end of each episode.

        Args:
            episode_reward: Total reward for the episode.
        """
        self.episode_count += 1
        self.total_rewards.append(episode_reward)
        self.decay_epsilon()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
        }

        if self.total_rewards:
            stats["mean_reward"] = np.mean(self.total_rewards[-100:])
            stats["last_reward"] = self.total_rewards[-1]

        return stats

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent to disk.

        Args:
            path: Path to save the agent.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent from disk.

        Args:
            path: Path to load the agent from.
        """
        pass


class RandomAgent(BaseAgent):
    """
    Random agent that selects actions uniformly at random.

    Useful as a baseline for comparison.
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.rng = np.random.default_rng()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        return self.rng.integers(0, self.action_dim)

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        return None  # Random agent doesn't learn

    def save(self, path: str) -> None:
        pass  # Nothing to save

    def load(self, path: str) -> None:
        pass  # Nothing to load
