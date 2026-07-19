"""
Base model classes and factory functions.

This module provides abstract base classes for time series models
and factory functions to create models from configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseTimeSeriesModel(nn.Module, ABC):
    """
    Abstract base class for time series forecasting models.

    All time series models should inherit from this class to ensure
    a consistent interface for training and pruning.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        sequence_length: int = 50,
    ):
        """
        Initialize the base model.

        Args:
            input_size: Number of input features.
            output_size: Number of output features (prediction targets).
            sequence_length: Length of input sequences.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        pass

    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get list of layers that can be pruned.

        Returns:
            List of (name, module) tuples for prunable layers.
        """
        prunable_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)
        prunable_layers = []

        for name, module in self.named_modules():
            if isinstance(module, prunable_types):
                prunable_layers.append((name, module))

        return prunable_layers

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get information about each layer for the RL agent's state.

        Returns:
            List of dictionaries containing layer information.
        """
        layer_info = []

        for name, module in self.get_prunable_layers():
            info = {
                "name": name,
                "type": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters()),
            }

            if isinstance(module, nn.Linear):
                info["in_features"] = module.in_features
                info["out_features"] = module.out_features
            elif isinstance(module, nn.Conv1d):
                info["in_channels"] = module.in_channels
                info["out_channels"] = module.out_channels
                info["kernel_size"] = module.kernel_size[0]
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                info["input_size"] = module.input_size
                info["hidden_size"] = module.hidden_size
                info["num_layers"] = module.num_layers

            layer_info.append(info)

        return layer_info

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_nonzero_parameters(self) -> int:
        """Count non-zero parameters."""
        total = 0
        for param in self.parameters():
            total += torch.count_nonzero(param).item()
        return total


def create_model(
    model_type: str,
    input_size: int,
    output_size: int = 1,
    sequence_length: int = 50,
    **kwargs,
) -> BaseTimeSeriesModel:
    """
    Factory function to create a model from configuration.

    Args:
        model_type: Type of model ("tcn" or "lstm").
        input_size: Number of input features.
        output_size: Number of output features.
        sequence_length: Length of input sequences.
        **kwargs: Additional model-specific arguments.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    from .tcn import TCNModel
    from .lstm import LSTMModel

    model_type = model_type.lower()

    if model_type == "tcn":
        return TCNModel(
            input_size=input_size,
            output_size=output_size,
            sequence_length=sequence_length,
            num_channels=kwargs.get("num_channels", [32, 64, 64, 32]),
            kernel_size=kwargs.get("kernel_size", 3),
            dropout=kwargs.get("dropout", 0.2),
        )

    elif model_type == "lstm":
        return LSTMModel(
            input_size=input_size,
            output_size=output_size,
            sequence_length=sequence_length,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
            bidirectional=kwargs.get("bidirectional", False),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'tcn' or 'lstm'.")


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save.
        path: Path to save the checkpoint.
        optimizer: Optional optimizer state to save.
        epoch: Current epoch number.
        metrics: Optional metrics dictionary.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)


def load_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
) -> Tuple[nn.Module, int, Optional[Dict[str, float]]]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into.
        path: Path to the checkpoint.
        optimizer: Optional optimizer to load state into.
        device: Device to load the model on.

    Returns:
        Tuple of (model, epoch, metrics).
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", None)

    return model, epoch, metrics
