"""
Structured pruning utilities.

Structured pruning removes entire filters/neurons rather than individual weights,
which can lead to actual speedups on hardware (unlike unstructured pruning).
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class StructuredPruner:
    """
    Structured pruning for convolutional and linear layers.

    This pruner removes entire filters/channels from Conv layers
    and entire neurons from Linear layers, enabling actual
    computational savings.

    Args:
        model: The PyTorch model to prune.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())
        self.pruning_history: List[Dict] = []

    def compute_filter_importance(
        self,
        layer: nn.Module,
        method: str = "l1_norm",
    ) -> torch.Tensor:
        """
        Compute importance scores for each filter/neuron.

        Args:
            layer: The layer to analyze.
            method: Importance scoring method.
                - "l1_norm": Sum of absolute weight values
                - "l2_norm": L2 norm of weights
                - "geometric_median": Distance from geometric median

        Returns:
            Tensor of importance scores for each filter/neuron.
        """
        weight = layer.weight.data

        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            # For conv layers, compute norm across all dimensions except output channels
            if method == "l1_norm":
                importance = weight.abs().sum(dim=tuple(range(1, weight.dim())))
            elif method == "l2_norm":
                importance = weight.pow(2).sum(dim=tuple(range(1, weight.dim()))).sqrt()
            elif method == "geometric_median":
                # Flatten filters and compute distance from median
                flat = weight.view(weight.size(0), -1)
                median = flat.median(dim=0).values
                importance = (flat - median).pow(2).sum(dim=1).sqrt()
            else:
                raise ValueError(f"Unknown method: {method}")

        elif isinstance(layer, nn.Linear):
            # For linear layers, compute norm across input features
            if method == "l1_norm":
                importance = weight.abs().sum(dim=1)
            elif method == "l2_norm":
                importance = weight.pow(2).sum(dim=1).sqrt()
            elif method == "geometric_median":
                median = weight.median(dim=0).values
                importance = (weight - median).pow(2).sum(dim=1).sqrt()
            else:
                raise ValueError(f"Unknown method: {method}")

        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

        return importance

    def prune_layer_structured(
        self,
        layer_name: str,
        pruning_ratio: float,
        importance_method: str = "l1_norm",
    ) -> Dict:
        """
        Prune a layer by removing least important filters/neurons.

        Args:
            layer_name: Name of the layer to prune.
            pruning_ratio: Fraction of filters/neurons to remove.
            importance_method: Method to compute importance scores.

        Returns:
            Dictionary with pruning statistics and indices.
        """
        # Get the layer
        layer = dict(self.model.named_modules())[layer_name]

        if not isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            raise TypeError(f"Cannot prune layer of type {type(layer)}")

        # Compute importance scores
        importance = self.compute_filter_importance(layer, importance_method)

        # Determine number of filters to prune
        n_filters = importance.size(0)
        n_to_prune = int(n_filters * pruning_ratio)

        if n_to_prune == 0:
            return {"pruned_indices": [], "remaining_indices": list(range(n_filters))}

        # Get indices of filters to prune (least important)
        _, indices = torch.sort(importance)
        prune_indices = indices[:n_to_prune].tolist()
        keep_indices = indices[n_to_prune:].tolist()

        # Zero out pruned filters (soft pruning)
        with torch.no_grad():
            for idx in prune_indices:
                layer.weight.data[idx] = 0
                if layer.bias is not None:
                    layer.bias.data[idx] = 0

        # Record history
        self.pruning_history.append({
            "layer_name": layer_name,
            "pruning_ratio": pruning_ratio,
            "pruned_indices": prune_indices,
            "remaining_indices": keep_indices,
            "original_size": n_filters,
            "pruned_size": len(keep_indices),
        })

        return {
            "pruned_indices": prune_indices,
            "remaining_indices": keep_indices,
            "original_size": n_filters,
            "pruned_size": len(keep_indices),
        }

    def get_prunable_layers(self) -> List[Tuple[str, nn.Module, Dict]]:
        """
        Get list of layers that can be structurally pruned with their info.

        Returns:
            List of (name, module, info) tuples.
        """
        layers = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                info = {
                    "type": "Conv1d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size[0],
                    "params": sum(p.numel() for p in module.parameters()),
                }
                layers.append((name, module, info))

            elif isinstance(module, nn.Conv2d):
                info = {
                    "type": "Conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "params": sum(p.numel() for p in module.parameters()),
                }
                layers.append((name, module, info))

            elif isinstance(module, nn.Linear):
                info = {
                    "type": "Linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "params": sum(p.numel() for p in module.parameters()),
                }
                layers.append((name, module, info))

        return layers

    def reset(self) -> None:
        """Reset model to original unpruned state."""
        self.model.load_state_dict(copy.deepcopy(self.original_state))
        self.pruning_history = []


def prune_filters(
    layer: nn.Module,
    indices_to_prune: List[int],
) -> None:
    """
    Zero out specific filters in a layer.

    Args:
        layer: The layer to prune.
        indices_to_prune: Indices of filters to zero out.
    """
    with torch.no_grad():
        for idx in indices_to_prune:
            layer.weight.data[idx] = 0
            if layer.bias is not None:
                layer.bias.data[idx] = 0


def prune_neurons(
    layer: nn.Linear,
    indices_to_prune: List[int],
) -> None:
    """
    Zero out specific neurons in a linear layer.

    Args:
        layer: The linear layer to prune.
        indices_to_prune: Indices of neurons to zero out.
    """
    prune_filters(layer, indices_to_prune)


def get_layer_importance_ranking(
    model: nn.Module,
    method: str = "l1_norm",
) -> List[Tuple[str, float]]:
    """
    Rank all prunable layers by their importance.

    Args:
        model: The model to analyze.
        method: Importance scoring method.

    Returns:
        List of (layer_name, importance_score) sorted by importance.
    """
    pruner = StructuredPruner(model)
    layer_scores = []

    for name, module, _ in pruner.get_prunable_layers():
        importance = pruner.compute_filter_importance(module, method)
        avg_importance = importance.mean().item()
        layer_scores.append((name, avg_importance))

    # Sort by importance (ascending = least important first)
    layer_scores.sort(key=lambda x: x[1])

    return layer_scores
