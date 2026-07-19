"""
Causal Chamber dataset loading utilities.

This module provides functions to load and process the Causal Chamber
wind tunnel dataset for time series forecasting tasks.

The Causal Chamber dataset contains both observational and interventional
data from physical experiments, making it ideal for testing causal robustness.

Reference:
    Gamella, J.L., Peters, J., & Bühlmann, P. (2024).
    Causal chambers as a real-world physical testbed for AI methodology.
    Nature Machine Intelligence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CausalChamberDataset(Dataset):
    """
    PyTorch Dataset for Causal Chamber time series data.

    This dataset creates sliding window sequences for time series forecasting.

    Args:
        data: DataFrame containing the sensor data.
        input_features: List of column names to use as inputs.
        target_feature: Column name to predict.
        sequence_length: Number of timesteps to look back.
        prediction_horizon: Number of steps ahead to predict.
        transform: Optional transform to apply to the data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        input_features: List[str],
        target_feature: str,
        sequence_length: int = 50,
        prediction_horizon: int = 1,
        transform: Optional[callable] = None,
    ):
        self.data = data
        self.input_features = input_features
        self.target_feature = target_feature
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform

        # Validate features exist
        missing_inputs = set(input_features) - set(data.columns)
        if missing_inputs:
            raise ValueError(f"Missing input features in data: {missing_inputs}")
        if target_feature not in data.columns:
            raise ValueError(f"Target feature '{target_feature}' not in data")

        # Extract numpy arrays for faster indexing
        self.X_data = data[input_features].values.astype(np.float32)
        self.y_data = data[target_feature].values.astype(np.float32)

        # Compute valid indices
        self.valid_indices = len(data) - sequence_length - prediction_horizon + 1

    def __len__(self) -> int:
        return max(0, self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (input_sequence, target) tensors.
            - input_sequence: Shape (sequence_length, num_features)
            - target: Shape (1,) for single-step prediction
        """
        # Extract sequence
        start_idx = idx
        end_idx = idx + self.sequence_length

        X = self.X_data[start_idx:end_idx]
        y = self.y_data[end_idx + self.prediction_horizon - 1]

        # Convert to tensors
        X = torch.from_numpy(X)
        y = torch.tensor([y], dtype=torch.float32)

        if self.transform is not None:
            X = self.transform(X)

        return X, y


def load_causal_chamber_data(
    dataset_name: str = "lt_camera_walks_v1",
    data_dir: Optional[Union[str, Path]] = None,
    experiment_indices: Optional[List[int]] = None,
    use_synthetic: bool = True,  # Use synthetic data by default for faster testing
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Causal Chamber dataset.

    This function attempts to load data using the causalchamber package.
    If not available, it provides instructions for manual download.

    Args:
        dataset_name: Name of the dataset to load.
        data_dir: Directory containing the data files.
        experiment_indices: Specific experiment indices to load.
        use_synthetic: If True, use synthetic data (faster for testing).

    Returns:
        Tuple of (observational_data, interventional_data) DataFrames.
    """
    # Use synthetic data for faster testing
    if use_synthetic:
        print("Using synthetic Causal Chamber-like data for demonstration.")
        return _load_synthetic_data()

    try:
        from causalchamber.datasets import Dataset

        # Set default data directory if not provided
        if data_dir is None:
            data_dir = Path.home() / ".causalchamber"
            data_dir.mkdir(parents=True, exist_ok=True)

        # Load the dataset
        dataset = Dataset(name=dataset_name, root=str(data_dir))

        # Get available experiments
        experiments = dataset.available_experiments()

        if experiment_indices is None:
            # Load all experiments
            experiment_indices = list(range(min(5, len(experiments))))

        # Load observational data (no interventions)
        obs_dfs = []
        int_dfs = []

        for idx in experiment_indices:
            if idx < len(experiments):
                exp_name = experiments[idx]
                df = dataset.get_experiment(exp_name)

                # Simple heuristic: if experiment name contains "intervention"
                # or specific manipulation indicators, treat as interventional
                if "intervention" in exp_name.lower() or "manip" in exp_name.lower():
                    int_dfs.append(df)
                else:
                    obs_dfs.append(df)

        obs_data = pd.concat(obs_dfs, ignore_index=True) if obs_dfs else pd.DataFrame()
        int_data = pd.concat(int_dfs, ignore_index=True) if int_dfs else pd.DataFrame()

        return obs_data, int_data

    except ImportError:
        print("=" * 60)
        print("CausalChamber package not found.")
        print("Install it with: pip install causalchamber")
        print("Or download data manually from:")
        print("https://github.com/causalchamber/causalchamber")
        print("=" * 60)
        return _load_synthetic_data()

    except Exception as e:
        print("=" * 60)
        print(f"Error loading Causal Chamber data: {e}")
        print("Falling back to synthetic data for demonstration.")
        print("=" * 60)
        return _load_synthetic_data()


def _load_synthetic_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data for testing when real data is unavailable.

    This creates time series data that mimics the structure of the
    Causal Chamber wind tunnel data.

    Returns:
        Tuple of (observational_data, interventional_data) DataFrames.
    """
    np.random.seed(42)
    n_samples = 10000

    # Simulate sensor readings
    t = np.linspace(0, 100, n_samples)

    # Create correlated sensor readings with some causal structure
    # fan_speed -> airflow -> pressure, temperature
    fan_speed = np.sin(t * 0.1) + 0.5 * np.random.randn(n_samples)
    airflow = 0.8 * fan_speed + 0.2 * np.random.randn(n_samples)
    pressure = 0.6 * airflow + 0.3 * np.sin(t * 0.2) + 0.1 * np.random.randn(n_samples)
    temperature = 0.4 * pressure + 0.2 * fan_speed + 0.1 * np.random.randn(n_samples)

    # Light sensors (spurious correlation with temperature)
    red = 0.5 + 0.3 * np.sin(t * 0.05) + 0.1 * np.random.randn(n_samples)
    green = 0.5 + 0.3 * np.cos(t * 0.05) + 0.1 * np.random.randn(n_samples)
    blue = 0.5 + 0.2 * np.sin(t * 0.03) + 0.1 * np.random.randn(n_samples)

    # Target variable
    pol_1 = 0.5 * pressure + 0.3 * temperature + 0.15 * np.random.randn(n_samples)

    obs_data = pd.DataFrame({
        "time": t,
        "fan_speed": fan_speed,
        "airflow": airflow,
        "pressure": pressure,
        "temperature": temperature,
        "red": red,
        "green": green,
        "blue": blue,
        "current": 0.7 * fan_speed + 0.1 * np.random.randn(n_samples),
        "voltage": 5.0 + 0.2 * np.random.randn(n_samples),
        "pol_1": pol_1,
    })

    # Interventional data: change fan speed distribution
    fan_speed_int = 2.0 * np.sin(t * 0.15) + 0.3 * np.random.randn(n_samples)
    airflow_int = 0.8 * fan_speed_int + 0.2 * np.random.randn(n_samples)
    pressure_int = 0.6 * airflow_int + 0.3 * np.sin(t * 0.2) + 0.1 * np.random.randn(n_samples)
    temperature_int = 0.4 * pressure_int + 0.2 * fan_speed_int + 0.1 * np.random.randn(n_samples)
    pol_1_int = 0.5 * pressure_int + 0.3 * temperature_int + 0.15 * np.random.randn(n_samples)

    int_data = pd.DataFrame({
        "time": t,
        "fan_speed": fan_speed_int,
        "airflow": airflow_int,
        "pressure": pressure_int,
        "temperature": temperature_int,
        "red": red,  # Light unchanged (spurious)
        "green": green,
        "blue": blue,
        "current": 0.7 * fan_speed_int + 0.1 * np.random.randn(n_samples),
        "voltage": 5.0 + 0.2 * np.random.randn(n_samples),
        "pol_1": pol_1_int,
    })

    print("Using synthetic data for testing.")
    print(f"Observational samples: {len(obs_data)}")
    print(f"Interventional samples: {len(int_data)}")

    return obs_data, int_data


def create_data_loaders(
    obs_data: pd.DataFrame,
    int_data: pd.DataFrame,
    input_features: List[str],
    target_feature: str,
    sequence_length: int = 50,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        obs_data: Observational data DataFrame.
        int_data: Interventional data DataFrame.
        input_features: List of input feature names.
        target_feature: Target feature name.
        sequence_length: Length of input sequences.
        batch_size: Batch size for data loaders.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        num_workers: Number of workers for data loading.

    Returns:
        Dictionary containing data loaders for each split.
    """
    from .preprocessing import normalize_data, split_data

    # Normalize observational data
    obs_normalized, scaler = normalize_data(
        obs_data, input_features + [target_feature]
    )

    # Split observational data
    train_data, val_data, test_data = split_data(
        obs_normalized, train_ratio, val_ratio
    )

    # Create datasets
    train_dataset = CausalChamberDataset(
        train_data, input_features, target_feature, sequence_length
    )
    val_dataset = CausalChamberDataset(
        val_data, input_features, target_feature, sequence_length
    )
    test_dataset = CausalChamberDataset(
        test_data, input_features, target_feature, sequence_length
    )

    # Handle interventional data if available
    if len(int_data) > 0:
        # Normalize using the same scaler
        int_normalized = obs_data.copy()
        for col in input_features + [target_feature]:
            if col in int_data.columns and col in scaler:
                mean, std = scaler[col]
                int_normalized[col] = (int_data[col] - mean) / (std + 1e-8)

        int_dataset = CausalChamberDataset(
            int_normalized, input_features, target_feature, sequence_length
        )
    else:
        int_dataset = None

    # Create data loaders
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if int_dataset is not None:
        loaders["interventional"] = DataLoader(
            int_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders, scaler
