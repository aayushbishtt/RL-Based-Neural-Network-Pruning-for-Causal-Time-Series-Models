#!/usr/bin/env python3
"""
Train the base time series forecasting model.

This script trains a TCN or LSTM model on the Causal Chamber dataset
before applying RL-based pruning.

Usage:
    python scripts/train_base_model.py --config configs/default.yaml
    python scripts/train_base_model.py --model_type tcn --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from src.data import load_causal_chamber_data, create_data_loaders
from src.models import create_model, save_model
from src.utils import load_config, get_device, set_seed, setup_logger, TensorBoardLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train base time series model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["tcn", "lstm"],
        help="Model type (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.model_type:
        config["base_model"]["type"] = args.model_type
    if args.epochs:
        config["base_model"]["training"]["epochs"] = args.epochs
    if args.learning_rate:
        config["base_model"]["training"]["learning_rate"] = args.learning_rate

    # Set up
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("hardware", {}).get("device", "auto"))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logger("train_base", log_dir=output_dir)
    tb_logger = TensorBoardLogger(output_dir / "tensorboard", comment="base_model")

    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config['base_model']}")

    # Load data
    logger.info("Loading data...")
    data_config = config["data"]

    obs_data, int_data = load_causal_chamber_data(
        dataset_name=data_config.get("dataset_name", "lt_camera_walks_v1")
    )

    # Determine input features - use available columns
    available_columns = obs_data.columns.tolist()
    input_features = [f for f in data_config["input_features"] if f in available_columns]

    if not input_features:
        # Use default features from synthetic data
        input_features = ["pressure", "temperature", "airflow", "fan_speed"]
        input_features = [f for f in input_features if f in available_columns]

    target_feature = data_config["target_feature"]
    if target_feature not in available_columns:
        target_feature = "pol_1"

    logger.info(f"Using input features: {input_features}")
    logger.info(f"Target feature: {target_feature}")

    # Create data loaders
    loaders, scaler = create_data_loaders(
        obs_data=obs_data,
        int_data=int_data,
        input_features=input_features,
        target_feature=target_feature,
        sequence_length=data_config.get("sequence_length", 50),
        batch_size=data_config.get("batch_size", 64),
        train_ratio=data_config.get("train_ratio", 0.7),
        val_ratio=data_config.get("val_ratio", 0.15),
    )

    logger.info(f"Train batches: {len(loaders['train'])}")
    logger.info(f"Val batches: {len(loaders['val'])}")

    # Create model
    model_config = config["base_model"]
    model_type = model_config["type"]

    model_kwargs = model_config.get(model_type, {})
    model = create_model(
        model_type=model_type,
        input_size=len(input_features),
        output_size=1,
        sequence_length=data_config.get("sequence_length", 50),
        **model_kwargs,
    )
    model = model.to(device)

    logger.info(f"Model: {model_type}")
    logger.info(f"Parameters: {model.count_parameters():,}")

    # Training setup
    train_config = model_config["training"]
    epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_config.get("weight_decay", 0.0001),
    )

    # Learning rate scheduler
    scheduler_type = train_config.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # Early stopping
    patience = train_config.get("early_stopping_patience", 10)
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    logger.info("Starting training...")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss = evaluate(model, loaders["val"], criterion, device)

        if scheduler:
            scheduler.step()

        # Logging
        tb_logger.log_scalar("loss/train", train_loss, epoch)
        tb_logger.log_scalar("loss/val", val_loss, epoch)
        tb_logger.log_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_model(
                model,
                output_dir / "best_model.pt",
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "train_loss": train_loss},
            )
            logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation
    test_loss = evaluate(model, loaders["test"], criterion, device)
    logger.info(f"Test Loss: {test_loss:.6f}")

    # Evaluate on interventional data if available
    if "interventional" in loaders:
        int_loss = evaluate(model, loaders["interventional"], criterion, device)
        logger.info(f"Interventional Loss: {int_loss:.6f}")
        tb_logger.log_scalar("loss/interventional", int_loss, epochs)

    # Save final model
    save_model(
        model,
        output_dir / "final_model.pt",
        optimizer=optimizer,
        epoch=epochs,
        metrics={"val_loss": val_loss, "test_loss": test_loss},
    )

    tb_logger.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
