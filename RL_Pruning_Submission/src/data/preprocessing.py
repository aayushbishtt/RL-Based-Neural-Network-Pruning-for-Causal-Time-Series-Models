"""
Data preprocessing utilities for time series data.

This module provides functions for normalizing, splitting, and transforming
time series data for training neural networks.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def normalize_data(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Normalize data columns.

    Args:
        data: Input DataFrame.
        columns: Columns to normalize. If None, normalize all numeric columns.
        method: Normalization method. Options: "standard" (z-score), "minmax".

    Returns:
        Tuple of (normalized_data, scaler_params).
        scaler_params is a dict mapping column names to (mean, std) or (min, max).
    """
    data = data.copy()

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    scaler_params = {}

    for col in columns:
        if col not in data.columns:
            continue

        if method == "standard":
            mean = data[col].mean()
            std = data[col].std()
            data[col] = (data[col] - mean) / (std + 1e-8)
            scaler_params[col] = (mean, std)

        elif method == "minmax":
            min_val = data[col].min()
            max_val = data[col].max()
            data[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
            scaler_params[col] = (min_val, max_val)

    return data, scaler_params


def denormalize_data(
    data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    scaler_params: Dict[str, Tuple[float, float]],
    columns: Optional[List[str]] = None,
    method: str = "standard",
) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
    """
    Denormalize data using saved scaler parameters.

    Args:
        data: Normalized data.
        scaler_params: Dictionary of scaler parameters from normalize_data.
        columns: Columns to denormalize.
        method: Normalization method that was used.

    Returns:
        Denormalized data in the same format as input.
    """
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        if columns is None:
            columns = list(scaler_params.keys())

        for col in columns:
            if col in scaler_params and col in data.columns:
                if method == "standard":
                    mean, std = scaler_params[col]
                    data[col] = data[col] * std + mean
                elif method == "minmax":
                    min_val, max_val = scaler_params[col]
                    data[col] = data[col] * (max_val - min_val) + min_val
        return data

    elif isinstance(data, (np.ndarray, torch.Tensor)):
        # Assume single column denormalization
        if len(scaler_params) != 1:
            raise ValueError(
                "For array/tensor input, scaler_params must have exactly one column"
            )
        col = list(scaler_params.keys())[0]

        if method == "standard":
            mean, std = scaler_params[col]
            return data * std + mean
        elif method == "minmax":
            min_val, max_val = scaler_params[col]
            return data * (max_val - min_val) + min_val

    return data


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    For time series, we typically don't shuffle to maintain temporal order.

    Args:
        data: Input DataFrame.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        shuffle: Whether to shuffle before splitting.
        random_state: Random state for shuffling.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    n = len(data)

    if shuffle:
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end].reset_index(drop=True)
    val_data = data.iloc[train_end:val_end].reset_index(drop=True)
    test_data = data.iloc[val_end:].reset_index(drop=True)

    return train_data, val_data, test_data


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    prediction_horizon: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data using sliding window.

    Args:
        data: Input array of shape (n_samples, n_features).
        sequence_length: Length of input sequences.
        prediction_horizon: Steps ahead to predict.
        stride: Step size between sequences.

    Returns:
        Tuple of (X, y) arrays.
        X: Shape (n_sequences, sequence_length, n_features)
        y: Shape (n_sequences, n_features) or (n_sequences,) for single target
    """
    n_samples = len(data)
    n_sequences = (n_samples - sequence_length - prediction_horizon + 1) // stride

    if n_sequences <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for sequence_length={sequence_length} "
            f"and prediction_horizon={prediction_horizon}"
        )

    X = np.zeros((n_sequences, sequence_length, data.shape[1] if data.ndim > 1 else 1))
    y = np.zeros((n_sequences,) + data.shape[1:] if data.ndim > 1 else (n_sequences,))

    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        target_idx = end_idx + prediction_horizon - 1

        if data.ndim > 1:
            X[i] = data[start_idx:end_idx]
            y[i] = data[target_idx]
        else:
            X[i, :, 0] = data[start_idx:end_idx]
            y[i] = data[target_idx]

    return X, y


class TimeSeriesDataset(Dataset):
    """
    Generic PyTorch Dataset for time series data.

    This can be used with pre-created sequences or raw data.

    Args:
        X: Input sequences of shape (n_samples, sequence_length, n_features).
        y: Target values of shape (n_samples,) or (n_samples, n_targets).
        transform: Optional transform to apply to inputs.
        target_transform: Optional transform to apply to targets.
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float()

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).float()
        else:
            self.y = y.float()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y


def add_noise(
    data: Union[np.ndarray, torch.Tensor],
    noise_level: float = 0.01,
    noise_type: str = "gaussian",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add noise to data for augmentation.

    Args:
        data: Input data.
        noise_level: Standard deviation of noise (for Gaussian) or range (for uniform).
        noise_type: Type of noise ("gaussian" or "uniform").

    Returns:
        Noisy data.
    """
    is_tensor = isinstance(data, torch.Tensor)

    if is_tensor:
        if noise_type == "gaussian":
            noise = torch.randn_like(data) * noise_level
        else:
            noise = (torch.rand_like(data) * 2 - 1) * noise_level
        return data + noise
    else:
        if noise_type == "gaussian":
            noise = np.random.randn(*data.shape) * noise_level
        else:
            noise = (np.random.rand(*data.shape) * 2 - 1) * noise_level
        return data + noise


def compute_statistics(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute descriptive statistics for each column.

    Args:
        data: Input DataFrame.

    Returns:
        Dictionary mapping column names to their statistics.
    """
    stats = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            "mean": data[col].mean(),
            "std": data[col].std(),
            "min": data[col].min(),
            "max": data[col].max(),
            "median": data[col].median(),
            "q25": data[col].quantile(0.25),
            "q75": data[col].quantile(0.75),
        }
    return stats
