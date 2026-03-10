"""
Checkpoint utilities for STRIDE training.
This module keeps checkpoint persistence separate from the trainer so the trainer
can stay focused on orchestration. It preserves the useful ideas from legacy setup
(saving full training state, resume support, clear file naming) while keeping the 
first STRIDE version small and explicit


Current scope
-------------
- save checkpoints
- load checkpoints
- save "latest" and optional named checkpoints
- resolve checkpoint paths
- persist trainer state components:
    - model state_dict
    - optimizer state_dict
    - scheduler state_dict (optional)
    - EMA state_dict (optional)
    - epoch
    - global step
    - best_val_loss (optional)
    - training/model/dataset/generation config paths (optional)
    - extra metadata dict (optional)

Typical usage
-------------
    save_checkpoint(
        checkpoint_dir=...,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        scheduler=scheduler,  # optional
        ema=ema,  # optional
        best_val_loss=best_val_loss,  # optional
        tag=f"epoch_{epoch:04d}",
    )

    state = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)

Notes
-----
- This module saves plain PyTorch checkpoint dictionaries.
- The trainer should decide when to save. This module only handles how.
- The returned loaded state dictionary includes non-module metadata so the
  trainer can restore counters and bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import yaml

from stride_core.training.ema import EMA


@dataclass(frozen=True)
class CheckpointConfig:
    """
    Minimal checkpoint configuration extracted from the training YAML.
    """

    enabled: bool
    save_every_n_epochs: int
    save_best: bool
    resume_from: str | None

    @classmethod
    def from_training_yaml(cls, training_config_path: str | Path) -> "CheckpointConfig":
        path = Path(training_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Training config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected training config YAML to load into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CheckpointConfig":
        training_cfg = _get_training_section(cfg)
        checkpoint_cfg = training_cfg.get("checkpointing", {})

        if not isinstance(checkpoint_cfg, dict):
            raise ValueError("Expected 'training.checkpointing' to be a dict")

        enabled = bool(checkpoint_cfg.get("enabled", True))
        save_every_n_epochs = int(checkpoint_cfg.get("save_every_n_epochs", 1))
        save_best = bool(checkpoint_cfg.get("save_best", False))
        resume_from_raw = checkpoint_cfg.get("resume_from", None)
        resume_from = None if resume_from_raw is None else str(resume_from_raw)

        if save_every_n_epochs <= 0:
            raise ValueError(
                f"checkpointing.save_every_n_epochs must be > 0, got {save_every_n_epochs}"
            )

        return cls(
            enabled=enabled,
            save_every_n_epochs=save_every_n_epochs,
            save_best=save_best,
            resume_from=resume_from,
        )


@dataclass(frozen=True)
class LoadedCheckpointState:
    """
    Metadata restored from a checkpoint after module states are loaded.
    """

    checkpoint_path: Path
    epoch: int
    global_step: int
    best_val_loss: float | None
    config_paths: dict[str, str | None]
    extra_state: dict[str, Any]


def _get_training_section(cfg: dict[str, Any]) -> dict[str, Any]:
    if "training" not in cfg:
        raise KeyError("Expected top-level key 'training' in training config")

    training_cfg = cfg["training"]
    if not isinstance(training_cfg, dict):
        raise ValueError(
            f"Expected 'training' section to be a dict, got {type(training_cfg)}"
        )

    return training_cfg


def ensure_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """
    Create the checkpoint directory if it does not already exist.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_checkpoint_path(
    checkpoint_dir: str | Path,
    *,
    tag: str | None = None,
    filename: str | None = None,
) -> Path:
    """
    Build a checkpoint file path.

    Rules
    -----
    - if `filename` is provided, use it directly
    - else if `tag` is provided, save as `checkpoint_<tag>.pt`
    - else save as `checkpoint_latest.pt`
    """
    checkpoint_dir = ensure_checkpoint_dir(checkpoint_dir)

    if filename is not None:
        return checkpoint_dir / filename

    if tag is not None:
        return checkpoint_dir / f"checkpoint_{tag}.pt"

    return checkpoint_dir / "checkpoint_latest.pt"


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    ema: EMA | None = None,
    best_val_loss: float | None = None,
    training_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    dataset_config_path: str | Path | None = None,
    generation_config_path: str | Path | None = None,
    extra_state: dict[str, Any] | None = None,
    tag: str | None = None,
    save_latest: bool = True,
) -> dict[str, Path]:
    """
    Save a STRIDE training checkpoint.

    Returns
    -------
    dict[str, Path]
        Dictionary of written checkpoint paths. Always contains the tagged path
        when a tag is provided or `checkpoint_latest.pt` when no tag is provided.
        If `save_latest=True` and `tag` is provided, also contains a `latest`
        entry pointing to the latest checkpoint path.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")

    checkpoint_dir = ensure_checkpoint_dir(checkpoint_dir)

    config_paths = {
        "training_config_path": _maybe_str(training_config_path),
        "model_config_path": _maybe_str(model_config_path),
        "dataset_config_path": _maybe_str(dataset_config_path),
        "generation_config_path": _maybe_str(generation_config_path),
    }

    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "config_paths": config_paths,
        "extra_state": {} if extra_state is None else dict(extra_state),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "ema_state_dict": None if ema is None else ema.state_dict(),
    }

    written_paths: dict[str, Path] = {}

    primary_path = build_checkpoint_path(checkpoint_dir, tag=tag)
    torch.save(checkpoint, primary_path)
    written_paths["primary"] = primary_path

    if save_latest and tag is not None:
        latest_path = build_checkpoint_path(checkpoint_dir, filename="checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        written_paths["latest"] = latest_path

    return written_paths


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    ema: EMA | None = None,
    map_location: str | torch.device | None = "cpu",
    strict_model: bool = True,
) -> LoadedCheckpointState:
    """
    Load a STRIDE training checkpoint and restore module states.

    Parameters
    ----------
    checkpoint_path:
        Path to the saved `.pt` checkpoint file.
    model:
        Model to restore.
    optimizer:
        Optimizer to restore, if desired.
    scheduler:
        Scheduler to restore, if desired.
    ema:
        EMA object to restore, if desired.
    map_location:
        Torch map_location passed to `torch.load`.
    strict_model:
        Passed to `model.load_state_dict`.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint to load into a dict, got {type(checkpoint)}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing required key 'model_state_dict'")

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict_model)

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    if ema is not None:
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None:
            if not isinstance(ema_state, dict):
                raise ValueError("Checkpoint key 'ema_state_dict' must be a dict")
            ema.load_state_dict(ema_state)

    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))

    best_val_loss_raw = checkpoint.get("best_val_loss", None)
    best_val_loss = None if best_val_loss_raw is None else float(best_val_loss_raw)

    config_paths_raw = checkpoint.get("config_paths", {})
    if not isinstance(config_paths_raw, dict):
        raise ValueError("Checkpoint key 'config_paths' must be a dict")

    config_paths = {
        "training_config_path": _maybe_str(config_paths_raw.get("training_config_path")),
        "model_config_path": _maybe_str(config_paths_raw.get("model_config_path")),
        "dataset_config_path": _maybe_str(config_paths_raw.get("dataset_config_path")),
        "generation_config_path": _maybe_str(config_paths_raw.get("generation_config_path")),
    }

    extra_state_raw = checkpoint.get("extra_state", {})
    if not isinstance(extra_state_raw, dict):
        raise ValueError("Checkpoint key 'extra_state' must be a dict")

    return LoadedCheckpointState(
        checkpoint_path=checkpoint_path,
        epoch=epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        config_paths=config_paths,
        extra_state=dict(extra_state_raw),
    )


def resolve_resume_checkpoint(
    resume_from: str | Path | None,
    checkpoint_dir: str | Path | None = None,
) -> Path | None:
    """
    Resolve a resume checkpoint path.

    Behavior
    --------
    - if `resume_from` is None, return None
    - if `resume_from` is a path-like string, resolve and return it
    - if `resume_from == "latest"`, require `checkpoint_dir` and return
      `checkpoint_latest.pt` inside it
    """
    if resume_from is None:
        return None

    resume_from_str = str(resume_from)
    if resume_from_str.lower() == "latest":
        if checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir must be provided when resume_from='latest'"
            )
        candidate = build_checkpoint_path(checkpoint_dir, filename="checkpoint_latest.pt")
    else:
        candidate = Path(resume_from_str)

    if not candidate.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")

    return candidate


def _maybe_str(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value)