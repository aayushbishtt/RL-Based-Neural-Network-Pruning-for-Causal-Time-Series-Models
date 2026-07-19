"""
Configuration management utilities.

This module handles loading, saving, and managing configuration files
for the RL pruning project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        save_path: Path where to save the configuration.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        device_config: Device specification. Can be:
            - "auto": Automatically select best available device
            - "cuda": Use NVIDIA GPU
            - "mps": Use Apple Silicon GPU
            - "cpu": Use CPU

    Returns:
        torch.device object for the selected device.
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration with values to override.

    Returns:
        Merged configuration dictionary.
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


class Config:
    """
    Configuration wrapper class for easy attribute access.

    Example:
        config = Config(load_config("configs/default.yaml"))
        print(config.data.batch_size)  # Access nested values
    """

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
