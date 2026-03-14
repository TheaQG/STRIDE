"""
Lightweight logging utilities for STRIDE training.

This module centralises training-time reporting without turning the trainer into
an output-formatting script. It is intentionally small: console summaries, metric
formatting, and a simple in-memory history container that can be saved to JSON
later if desired.

Responsibilities
----------------
- Format compact console messages for train/valid epochs
- Optionally format step-level progress messages
- Store simple per-epoch metric history
- Save history to disk as JSON
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def print_section(title: str) -> None:
    """
    Log a section header.
    """
    line = "=" * len(title)
    logger.info(f"\n{line}\n{title}\n{line}")


def format_metric(value: float | int | None, precision: int = 6) -> str:
    """
    Format a scalar metric for console output.
    """
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{precision}f}"


def format_epoch_summary(
    stage: str,
    epoch: int,
    metrics: dict[str, float],
    *,
    lr: float | None = None,
    best_val_loss: float | None = None,
    improved: bool | None = None,
) -> str:
    """
    Format a compact epoch-level summary line.
    """
    parts = [
        f"[{stage}]",
        f"epoch={epoch + 1:04d}",
        f"loss={format_metric(metrics.get('loss'))}",
    ]

    if "num_batches" in metrics:
        parts.append(f"num_batches={int(metrics['num_batches'])}")

    if lr is not None:
        parts.append(f"lr={format_metric(lr, precision=8)}")

    if best_val_loss is not None:
        parts.append(f"best_val={format_metric(best_val_loss)}")

    if improved is True:
        parts.append("improved=yes")
    elif improved is False:
        parts.append("improved=no")

    return " ".join(parts)


def log_epoch_summary(message: str) -> None:
    """
    Log a formatted epoch summary line.
    """
    logger.info(message)


def format_step_progress(
    stage: str,
    *,
    epoch: int,
    batch_idx: int,
    num_batches_total: int,
    loss_value: float,
    global_step: int | None = None,
) -> str:
    """
    Format a compact step-level progress line.
    """
    global_step_str = "" if global_step is None else f" step={global_step:06d}"
    return (
        f"[{stage}] epoch={epoch + 1:04d}"
        f"{global_step_str} "
        f"batch={batch_idx + 1:04d}/{num_batches_total:04d} "
        f"loss={loss_value:.6f}"
    )


def log_step_progress(message: str) -> None:
    """
    Log a formatted step progress line.
    """
    logger.info(message)


def summarize_run_setup(
    *,
    run_name: str,
    device: str,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    model_config_path: str | Path,
    dataset_config_path: str | Path,
    generation_config_path: str | Path | None,
    train_batches: int,
    valid_batches: int,
    total_params: int,
    trainable_params: int,
    ema_enabled: bool,
) -> list[str]:
    """
    Return formatted run-summary lines.
    """
    return [
        f"Run name:           {run_name}",
        f"Device:             {device}",
        f"Output dir:         {output_dir}",
        f"Checkpoint dir:     {checkpoint_dir}",
        f"Model config:       {model_config_path}",
        f"Dataset config:     {dataset_config_path}",
        f"Generation config:  {generation_config_path}",
        f"Train batches:      {train_batches}",
        f"Valid batches:      {valid_batches}",
        f"Total params:       {total_params:,}",
        f"Trainable params:   {trainable_params:,}",
        f"EMA enabled:        {ema_enabled}",
    ]


def log_run_setup(lines: list[str]) -> None:
    """
    Log formatted run-setup lines.
    """
    for line in lines:
        logger.info(line)


# -----------------------------------------------------------------------------
# History container
# -----------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """
    Simple in-memory training history.

    Records one dictionary per epoch for training and validation.
    """

    train_epochs: list[dict[str, Any]] = field(default_factory=list)
    valid_epochs: list[dict[str, Any]] = field(default_factory=list)

    def add_train_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.train_epochs.append({"epoch": int(epoch), **dict(metrics)})

    def add_valid_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.valid_epochs.append({"epoch": int(epoch), **dict(metrics)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_epochs": self.train_epochs,
            "valid_epochs": self.valid_epochs,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path