"""
Neural network pruning utilities.

This module provides functions for pruning neural network weights
using various strategies (magnitude-based, structured, etc.).

The pruning is designed to be reversible and compatible with
the RL environment for iterative pruning decisions.
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class Pruner:
    """
    Main pruning controller for neural network models.

    This class manages the pruning process, tracking which layers
    have been pruned and allowing for rollback if needed.

    Args:
        model: The PyTorch model to prune.
        pruning_method: Pruning method ("magnitude", "random", "l1_structured").
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_method: str = "magnitude",
    ):
        self.model = model
        self.pruning_method = pruning_method

        # Store original state for potential rollback
        self.original_state = copy.deepcopy(model.state_dict())

        # Track pruning history
        self.pruning_history: List[Dict] = []

        # Get prunable layers
        self.prunable_layers = self._get_prunable_layers()

    def _get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get list of layers that can be pruned."""
        prunable_types = (nn.Linear, nn.Conv1d, nn.Conv2d)
        layers = []

        for name, module in self.model.named_modules():
            if isinstance(module, prunable_types):
                layers.append((name, module))

        return layers

    def prune_layer(
        self,
        layer_index: int,
        pruning_ratio: float,
    ) -> Dict[str, float]:
        """
        Prune a specific layer by the given ratio.

        Args:
            layer_index: Index of the layer to prune.
            pruning_ratio: Fraction of weights to prune (0 to 1).

        Returns:
            Dictionary with pruning statistics.
        """
        if layer_index >= len(self.prunable_layers):
            raise IndexError(f"Layer index {layer_index} out of range")

        if pruning_ratio <= 0:
            return {"pruned_params": 0, "remaining_params": 0, "sparsity": 0}

        layer_name, layer = self.prunable_layers[layer_index]

        # Count parameters before pruning
        params_before = sum(p.numel() for p in layer.parameters())

        # Apply pruning based on method
        if self.pruning_method == "magnitude":
            prune.l1_unstructured(layer, name="weight", amount=pruning_ratio)
        elif self.pruning_method == "random":
            prune.random_unstructured(layer, name="weight", amount=pruning_ratio)
        elif self.pruning_method == "l1_structured":
            if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                prune.ln_structured(layer, name="weight", amount=pruning_ratio, n=1, dim=0)
            else:
                prune.l1_unstructured(layer, name="weight", amount=pruning_ratio)
        else:
            raise ValueError(f"Unknown pruning method: {self.pruning_method}")

        # Calculate statistics
        nonzero_params = torch.count_nonzero(layer.weight).item()
        total_params = layer.weight.numel()
        sparsity = 1.0 - (nonzero_params / total_params)

        # Record history
        self.pruning_history.append({
            "layer_index": layer_index,
            "layer_name": layer_name,
            "pruning_ratio": pruning_ratio,
            "sparsity": sparsity,
        })

        return {
            "pruned_params": total_params - nonzero_params,
            "remaining_params": nonzero_params,
            "sparsity": sparsity,
        }

    def get_model_sparsity(self) -> float:
        """
        Calculate overall model sparsity.

        Returns:
            Fraction of zero weights in the model.
        """
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0

    def get_layer_sparsities(self) -> List[float]:
        """
        Get sparsity of each prunable layer.

        Returns:
            List of sparsity values for each layer.
        """
        sparsities = []

        for _, layer in self.prunable_layers:
            total = layer.weight.numel()
            zeros = (layer.weight == 0).sum().item()
            sparsities.append(zeros / total if total > 0 else 0)

        return sparsities

    def remove_pruning_reparametrization(self) -> None:
        """
        Make pruning permanent by removing the pruning reparametrization.

        This converts the pruning mask into actual zero weights.
        """
        for _, layer in self.prunable_layers:
            if prune.is_pruned(layer):
                prune.remove(layer, "weight")

    def reset(self) -> None:
        """Reset model to original unpruned state."""
        self.model.load_state_dict(copy.deepcopy(self.original_state))
        self.pruning_history = []

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio achieved by pruning.

        Returns:
            Ratio of original to current non-zero parameters.
        """
        original_params = sum(
            p.numel() for p in self.model.parameters()
        )
        nonzero_params = sum(
            torch.count_nonzero(p).item() for p in self.model.parameters()
        )

        return original_params / nonzero_params if nonzero_params > 0 else float("inf")


def prune_layer(
    layer: nn.Module,
    amount: float,
    method: str = "l1_unstructured",
) -> None:
    """
    Prune a single layer.

    Args:
        layer: The layer to prune.
        amount: Fraction of weights to prune (0 to 1).
        method: Pruning method to use.
    """
    if amount <= 0:
        return

    if method == "l1_unstructured":
        prune.l1_unstructured(layer, name="weight", amount=amount)
    elif method == "random_unstructured":
        prune.random_unstructured(layer, name="weight", amount=amount)
    elif method == "ln_structured":
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
        else:
            prune.l1_unstructured(layer, name="weight", amount=amount)
    else:
        raise ValueError(f"Unknown pruning method: {method}")


def remove_pruning(model: nn.Module) -> None:
    """
    Remove pruning reparametrization from all layers.

    Args:
        model: The model to remove pruning from.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if prune.is_pruned(module):
                prune.remove(module, "weight")


def get_pruning_mask(layer: nn.Module) -> Optional[torch.Tensor]:
    """
    Get the pruning mask for a layer.

    Args:
        layer: The layer to get the mask from.

    Returns:
        Pruning mask tensor or None if layer is not pruned.
    """
    if hasattr(layer, "weight_mask"):
        return layer.weight_mask
    return None


def apply_pruning_mask(
    layer: nn.Module,
    mask: torch.Tensor,
) -> None:
    """
    Apply a custom pruning mask to a layer.

    Args:
        layer: The layer to apply the mask to.
        mask: Binary mask tensor (1 = keep, 0 = prune).
    """
    prune.custom_from_mask(layer, name="weight", mask=mask)


def count_pruned_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count pruned and total parameters in a model.

    Args:
        model: The model to analyze.

    Returns:
        Dictionary with total, pruned, and remaining parameter counts.
    """
    total = 0
    pruned = 0

    for param in model.parameters():
        total += param.numel()
        pruned += (param == 0).sum().item()

    return {
        "total": total,
        "pruned": pruned,
        "remaining": total - pruned,
        "sparsity": pruned / total if total > 0 else 0,
    }
