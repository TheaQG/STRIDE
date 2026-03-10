"""
Optimizer and scheduler factories for STRIDE training.

This module keeps optimization setup separate from the trainer so the trainer can 
stay focused on orchestration. The design takes the useful ideas from legacy setup 
(central optimizer/scheduler construction, and config-driven selection) but keeps the
first STRIDE version deliberately small and explicit.

Current scope
-------------
- Optimizers:
    - AdamW
    - Adam
    - SGD
- Schedulers
    - None/disabled
    - StepLR
    - ReduceLROnPlateau
    - CosineAnnealingLR

The training YAML is expected to contain:

training:
    optimizer:
        name: "adamw"
        lr: 1.0e-4
        weight_decay: 1.0e-4
        betas: [0.9, 0.999]
        eps: 1.0e-8
        momentum: 0.9
    scheduler:
        enabled: false
        name: null
        params: {}    
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)
import yaml


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Minimal optimizer configuration extracted from the training YAML.
    """

    name: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    momentum: float

    @classmethod
    def from_training_yaml(cls, training_config_path: str | Path) -> "OptimizerConfig":
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
    def from_dict(cls, cfg: dict[str, Any]) -> "OptimizerConfig":
        training_cfg = _get_training_section(cfg)
        optimizer_cfg = training_cfg.get("optimizer", {})

        if not isinstance(optimizer_cfg, dict):
            raise ValueError("Expected 'training.optimizer' to be a dict")

        name = str(optimizer_cfg.get("name", "adamw")).lower()
        lr = float(optimizer_cfg.get("lr", 1.0e-4))
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
        betas_raw = optimizer_cfg.get("betas", (0.9, 0.999))
        eps = float(optimizer_cfg.get("eps", 1.0e-8))
        momentum = float(optimizer_cfg.get("momentum", 0.9))

        if lr <= 0:
            raise ValueError(f"optimizer.lr must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError(
                f"optimizer.weight_decay must be >= 0, got {weight_decay}"
            )
        if eps <= 0:
            raise ValueError(f"optimizer.eps must be positive, got {eps}")
        if momentum < 0:
            raise ValueError(f"optimizer.momentum must be >= 0, got {momentum}")

        betas = _coerce_betas(betas_raw)

        return cls(
            name=name,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            momentum=momentum,
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """
    Minimal scheduler configuration extracted from the training YAML.
    """

    enabled: bool
    name: str | None
    params: dict[str, Any]

    @classmethod
    def from_training_yaml(cls, training_config_path: str | Path) -> "SchedulerConfig":
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
    def from_dict(cls, cfg: dict[str, Any]) -> "SchedulerConfig":
        training_cfg = _get_training_section(cfg)
        scheduler_cfg = training_cfg.get("scheduler", {})

        if not isinstance(scheduler_cfg, dict):
            raise ValueError("Expected 'training.scheduler' to be a dict")

        enabled = bool(scheduler_cfg.get("enabled", False))
        raw_name = scheduler_cfg.get("name", None)
        name = None if raw_name is None else str(raw_name)
        params = scheduler_cfg.get("params", {})

        if not isinstance(params, dict):
            raise ValueError("Expected 'training.scheduler.params' to be a dict")

        if not enabled:
            name = None

        return cls(enabled=enabled, name=name, params=params)


def _get_training_section(cfg: dict[str, Any]) -> dict[str, Any]:
    if "training" not in cfg:
        raise KeyError("Expected top-level key 'training' in training config")

    training_cfg = cfg["training"]
    if not isinstance(training_cfg, dict):
        raise ValueError(
            f"Expected 'training' section to be a dict, got {type(training_cfg)}"
        )

    return training_cfg


def _coerce_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"optimizer.betas must be a list/tuple of length 2, got {value}"
        )

    beta1 = float(value[0])
    beta2 = float(value[1])

    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"optimizer.betas[0] must be in [0, 1), got {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"optimizer.betas[1] must be in [0, 1), got {beta2}")

    return beta1, beta2


def build_optimizer(
    model: torch.nn.Module,
    optimizer_config: OptimizerConfig | dict[str, Any] | str | Path,
) -> Optimizer:
    """
    Build the optimizer for a model.

    Supported inputs for `optimizer_config`
    --------------------------------------
    - OptimizerConfig instance
    - full training config dict
    - path to training YAML
    """
    cfg = _coerce_optimizer_config(optimizer_config)

    if cfg.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
        )

    if cfg.name == "adam":
        return Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
        )

    if cfg.name == "sgd":
        return SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
        )

    raise ValueError(
        f"Optimizer '{cfg.name}' not recognized. Use 'adamw', 'adam', or 'sgd'."
    )


def build_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig | dict[str, Any] | str | Path,
) -> LRScheduler | ReduceLROnPlateau | None:
    """
    Build the learning-rate scheduler.

    Supported inputs for `scheduler_config`
    --------------------------------------
    - SchedulerConfig instance
    - full training config dict
    - path to training YAML
    """
    cfg = _coerce_scheduler_config(scheduler_config)

    if not cfg.enabled or cfg.name is None:
        return None

    name = cfg.name.lower()
    params = cfg.params

    if name in {"step", "steplr"}:
        step_size = int(params.get("step_size", 1))
        gamma = float(params.get("gamma", 0.1))
        if step_size <= 0:
            raise ValueError(f"scheduler.params.step_size must be > 0, got {step_size}")
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name in {"reducelronplateau", "reduce_lr_on_plateau", "plateau"}:
        mode = str(params.get("mode", "min"))
        factor = float(params.get("factor", 0.1))
        patience = int(params.get("patience", 10))
        min_lr = float(params.get("min_lr", 0.0))
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    if name in {"cosineannealing", "cosine", "cosineannealinglr"}:
        t_max = int(params.get("T_max", 10))
        eta_min = float(params.get("eta_min", 0.0))
        if t_max <= 0:
            raise ValueError(f"scheduler.params.T_max must be > 0, got {t_max}")
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    raise ValueError(
        "Scheduler not recognized. Use one of: 'Step', 'ReduceLROnPlateau', or 'CosineAnnealing'."
    )


def _coerce_optimizer_config(
    optimizer_config: OptimizerConfig | dict[str, Any] | str | Path,
) -> OptimizerConfig:
    if isinstance(optimizer_config, OptimizerConfig):
        return optimizer_config

    if isinstance(optimizer_config, (str, Path)):
        return OptimizerConfig.from_training_yaml(optimizer_config)

    if isinstance(optimizer_config, dict):
        return OptimizerConfig.from_dict(optimizer_config)

    raise TypeError(
        f"Unsupported optimizer_config type {type(optimizer_config)}. "
        "Expected OptimizerConfig, dict, str, or Path."
    )


def _coerce_scheduler_config(
    scheduler_config: SchedulerConfig | dict[str, Any] | str | Path,
) -> SchedulerConfig:
    if isinstance(scheduler_config, SchedulerConfig):
        return scheduler_config

    if isinstance(scheduler_config, (str, Path)):
        return SchedulerConfig.from_training_yaml(scheduler_config)

    if isinstance(scheduler_config, dict):
        return SchedulerConfig.from_dict(scheduler_config)

    raise TypeError(
        f"Unsupported scheduler_config type {type(scheduler_config)}. "
        "Expected SchedulerConfig, dict, str, or Path."
    )