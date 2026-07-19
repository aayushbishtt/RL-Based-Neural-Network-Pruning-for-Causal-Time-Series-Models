"""
Tabular Q-Learning Agent.

This module implements a simple tabular Q-learning agent for discrete
state-action spaces. It can be used with discretized states or
small finite state spaces.

This is useful for:
- Initial experiments with small networks
- Understanding the pruning dynamics before scaling up
- Baseline comparisons
"""

import pickle
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent.

    Uses a table (dictionary) to store Q-values for state-action pairs.
    States are discretized by binning continuous values.

    Args:
        state_dim: Dimension of the state space.
        action_dim: Number of possible actions.
        learning_rate: Learning rate (alpha).
        discount_factor: Discount factor (gamma).
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Decay rate for epsilon.
        n_bins: Number of bins for state discretization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        n_bins: int = 10,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )

        self.n_bins = n_bins

        # Q-table: maps discretized state to array of Q-values for each action
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_dim)
        )

        # Statistics
        self.state_visits: Dict[Tuple, int] = defaultdict(int)

    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize a continuous state into bins.

        Args:
            state: Continuous state vector.

        Returns:
            Tuple of discretized state indices.
        """
        # Clip state values to [0, 1] range and bin
        clipped = np.clip(state, 0, 1)
        discretized = np.floor(clipped * self.n_bins).astype(int)
        discretized = np.clip(discretized, 0, self.n_bins - 1)

        return tuple(discretized)

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation.
            training: Whether in training mode.

        Returns:
            Selected action index.
        """
        discrete_state = self._discretize_state(state)

        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            q_values = self.q_table[discrete_state]
            return int(np.argmax(q_values))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Update Q-value using the Q-learning update rule.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.

        Returns:
            TD error (absolute value).
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        # Update visit count
        self.state_visits[discrete_state] += 1
        self.training_step += 1

        # Current Q-value
        current_q = self.q_table[discrete_state][action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.discount_factor * max_next_q

        # TD error
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[discrete_state][action] += self.learning_rate * td_error

        return abs(td_error)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in a state.

        Args:
            state: State observation.

        Returns:
            Array of Q-values for each action.
        """
        discrete_state = self._discretize_state(state)
        return self.q_table[discrete_state].copy()

    def get_policy(self) -> Dict[Tuple, int]:
        """
        Get the current greedy policy.

        Returns:
            Dictionary mapping states to best actions.
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = int(np.argmax(q_values))
        return policy

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = super().get_stats()
        stats["n_states_visited"] = len(self.q_table)
        stats["total_state_visits"] = sum(self.state_visits.values())
        return stats

    def save(self, path: str) -> None:
        """Save agent to disk."""
        data = {
            "q_table": dict(self.q_table),
            "state_visits": dict(self.state_visits),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "total_rewards": self.total_rewards,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "n_bins": self.n_bins,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load agent from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(
            lambda: np.zeros(self.action_dim),
            data["q_table"],
        )
        self.state_visits = defaultdict(int, data["state_visits"])
        self.epsilon = data["epsilon"]
        self.training_step = data["training_step"]
        self.episode_count = data["episode_count"]
        self.total_rewards = data["total_rewards"]


class SARSAAgent(QLearningAgent):
    """
    SARSA (State-Action-Reward-State-Action) agent.

    Similar to Q-learning but uses the actual next action instead of
    the max Q-value for updates (on-policy).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action: Optional[int] = None

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Update Q-value using SARSA update rule.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))

        where a' is the actual next action (not the max).
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        self.state_visits[discrete_state] += 1
        self.training_step += 1

        current_q = self.q_table[discrete_state][action]

        if done:
            target_q = reward
        else:
            # Select next action (will be taken)
            next_action = self.select_action(next_state, training=True)
            next_q = self.q_table[discrete_next_state][next_action]
            target_q = reward + self.discount_factor * next_q

        td_error = target_q - current_q
        self.q_table[discrete_state][action] += self.learning_rate * td_error

        return abs(td_error)
