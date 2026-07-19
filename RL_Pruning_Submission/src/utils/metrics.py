"""
Metrics computation utilities for model analysis.

This module provides functions to compute various metrics related to
model size, computational cost, and pruning effectiveness.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model.
        only_trainable: If True, count only trainable parameters.

    Returns:
        Total number of parameters.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """
    Count the number of non-zero parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of non-zero parameters.
    """
    total = 0
    for param in model.parameters():
        total += torch.count_nonzero(param).item()
    return total


def compute_sparsity(model: nn.Module) -> float:
    """
    Compute the sparsity ratio of a model (fraction of zero weights).

    Args:
        model: PyTorch model.

    Returns:
        Sparsity ratio between 0 and 1.
    """
    total_params = count_parameters(model, only_trainable=False)
    nonzero_params = count_nonzero_parameters(model)

    if total_params == 0:
        return 0.0

    return 1.0 - (nonzero_params / total_params)


def compute_layer_info(model: nn.Module) -> List[Dict]:
    """
    Compute information about each layer in the model.

    Args:
        model: PyTorch model.

    Returns:
        List of dictionaries containing layer information.
    """
    layer_info = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
            info = {
                "name": name,
                "type": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters()),
                "nonzero_params": sum(
                    torch.count_nonzero(p).item() for p in module.parameters()
                ),
            }

            # Add layer-specific information
            if isinstance(module, nn.Linear):
                info["in_features"] = module.in_features
                info["out_features"] = module.out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                info["in_channels"] = module.in_channels
                info["out_channels"] = module.out_channels
                info["kernel_size"] = module.kernel_size
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                info["input_size"] = module.input_size
                info["hidden_size"] = module.hidden_size
                info["num_layers"] = module.num_layers

            layer_info.append(info)

    return layer_info


def compute_flops(
    model: nn.Module, input_shape: Tuple[int, ...], device: torch.device = None
) -> int:
    """
    Estimate the FLOPs (Floating Point Operations) of a model.

    This is a simplified estimation that counts multiply-accumulate operations.

    Args:
        model: PyTorch model.
        input_shape: Shape of the input tensor (excluding batch dimension).
        device: Device to run the computation on.

    Returns:
        Estimated number of FLOPs.
    """
    if device is None:
        device = next(model.parameters()).device

    total_flops = 0
    hooks = []

    def count_conv1d_flops(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        output_length = output.size(2)

        kernel_ops = module.kernel_size[0] * module.in_channels
        total_flops += batch_size * output_length * module.out_channels * kernel_ops

    def count_conv2d_flops(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        output_h, output_w = output.size(2), output.size(3)

        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        total_flops += (
            batch_size * output_h * output_w * module.out_channels * kernel_ops
        )

    def count_linear_flops(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        total_flops += batch_size * module.in_features * module.out_features

    def count_lstm_flops(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        seq_length = input[0].size(1)

        # LSTM has 4 gates, each with input and hidden connections
        gates_ops = 4 * (module.input_size + module.hidden_size) * module.hidden_size
        total_flops += batch_size * seq_length * module.num_layers * gates_ops

    # Register hooks
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(count_conv1d_flops))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv2d_flops))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear_flops))
        elif isinstance(module, nn.LSTM):
            hooks.append(module.register_forward_hook(count_lstm_flops))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape).to(device)
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops


def compute_compression_ratio(
    original_params: int, pruned_params: int
) -> float:
    """
    Compute the compression ratio after pruning.

    Args:
        original_params: Number of parameters in original model.
        pruned_params: Number of non-zero parameters after pruning.

    Returns:
        Compression ratio (original / pruned).
    """
    if pruned_params == 0:
        return float("inf")
    return original_params / pruned_params


def compute_accuracy_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute various accuracy/error metrics for regression tasks.

    Args:
        predictions: Model predictions.
        targets: Ground truth values.

    Returns:
        Dictionary containing MSE, MAE, RMSE, and R² score.
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = mse**0.5

    # R² score (coefficient of determination)
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}
