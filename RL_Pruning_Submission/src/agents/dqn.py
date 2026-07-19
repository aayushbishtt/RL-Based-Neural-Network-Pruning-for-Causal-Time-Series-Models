"""
Deep Q-Network (DQN) Agent.

This module implements DQN with experience replay and target network,
suitable for continuous state spaces.

Reference:
    Mnih et al. (2015). Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529-533.
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseAgent


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.

    Args:
        state_dim: Dimension of input state.
        action_dim: Number of output actions.
        hidden_sizes: List of hidden layer sizes.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        # Build network layers
        layers = []
        prev_size = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch_size, state_dim).

        Returns:
            Q-values tensor of shape (batch_size, action_dim).
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and allows random sampling for training.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and target network.

    Args:
        state_dim: Dimension of the state space.
        action_dim: Number of possible actions.
        learning_rate: Learning rate for optimizer.
        discount_factor: Discount factor (gamma).
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay_steps: Steps to decay epsilon to minimum.
        batch_size: Batch size for training.
        buffer_size: Size of replay buffer.
        target_update_freq: Frequency of target network updates.
        hidden_sizes: Hidden layer sizes for Q-network.
        device: Device to run on (cuda/cpu).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        batch_size: int = 32,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        hidden_sizes: List[int] = None,
        device: torch.device = None,
    ):
        # Calculate decay to achieve epsilon_end in epsilon_decay_steps
        epsilon_decay = (epsilon_end / epsilon_start) ** (1 / epsilon_decay_steps)

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or torch.device("cpu")

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_sizes).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Loss history
        self.losses: List[float] = []

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
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Store transition and update network.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.

        Returns:
            Loss value if update performed, None otherwise.
        """
        # Store transition
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.training_step += 1

        # Update epsilon
        self.decay_epsilon()

        # Skip if not enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.discount_factor * next_q * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in a state.

        Args:
            state: State observation.

        Returns:
            Array of Q-values for each action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = super().get_stats()

        if self.losses:
            stats["mean_loss"] = np.mean(self.losses[-100:])
            stats["last_loss"] = self.losses[-1]

        stats["buffer_size"] = len(self.replay_buffer)

        return stats

    def save(self, path: str) -> None:
        """Save agent to disk."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "episode_count": self.episode_count,
                "total_rewards": self.total_rewards,
                "losses": self.losses,
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.total_rewards = checkpoint["total_rewards"]
        self.losses = checkpoint.get("losses", [])


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent.

    Uses the online network to select actions and the target network
    to evaluate them, reducing overestimation bias.

    Reference:
        Van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning.
        AAAI 2016.
    """

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Update using Double DQN algorithm.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.training_step += 1
        self.decay_epsilon()

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.discount_factor * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value
