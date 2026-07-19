"""
Logging utilities for training and evaluation.

This module provides logging functionality including console logging
and TensorBoard integration.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def setup_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Name of the logger.
        log_dir: Directory to save log files. If None, only console logging.
        level: Logging level.
        console: Whether to also log to console.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """
    TensorBoard logging wrapper for easy metric tracking.

    Example:
        logger = TensorBoardLogger("logs/experiment_1")
        logger.log_scalar("loss", 0.5, step=100)
        logger.log_histogram("weights", model.fc.weight, step=100)
        logger.close()
    """

    def __init__(self, log_dir: Union[str, Path], comment: str = ""):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs.
            comment: Optional comment to append to log directory.
        """
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if comment:
            self.log_dir = log_dir / f"{timestamp}_{comment}"
        else:
            self.log_dir = log_dir / timestamp

        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram of values."""
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """Log the model graph."""
        self.writer.add_graph(model, input_tensor)

    def log_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]) -> None:
        """Log hyperparameters and associated metrics."""
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        """Flush the writer."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


class MetricTracker:
    """
    Track and compute running statistics for metrics.

    Example:
        tracker = MetricTracker()
        for batch in batches:
            tracker.update("loss", loss_value)
        avg_loss = tracker.get_average("loss")
        tracker.reset()
    """

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    def update(self, name: str, value: float) -> None:
        """Add a value to a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_average(self, name: str) -> float:
        """Get the average value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_last(self, name: str) -> float:
        """Get the last value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]

    def get_all(self, name: str) -> list:
        """Get all values of a metric."""
        return self.metrics.get(name, [])

    def reset(self, name: Optional[str] = None) -> None:
        """Reset metrics. If name is None, reset all metrics."""
        if name is None:
            self.metrics = {}
        elif name in self.metrics:
            self.metrics[name] = []

    def get_summary(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {name: self.get_average(name) for name in self.metrics}
