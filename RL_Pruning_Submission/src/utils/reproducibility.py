"""
Reproducibility utilities.

This module provides functions to ensure reproducible experiments
by setting random seeds across all libraries.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def ensure_reproducibility(seed: int, deterministic: bool = True) -> None:
    """
    Ensure reproducibility by setting seeds and configuring PyTorch.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (may be slower).
    """
    set_seed(seed)

    if deterministic:
        # Use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for cublas
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Enable deterministic mode in PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass


def get_rng_state() -> dict:
    """
    Get the current random state of all random number generators.

    Returns:
        Dictionary containing RNG states.
    """
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict) -> None:
    """
    Restore random state from a saved state dictionary.

    Args:
        state: Dictionary containing RNG states.
    """
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
