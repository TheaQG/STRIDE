"""
Data-loading utilities for STRIDE training.

This module is the training-side bridge between:
- training YAML configuration
- dataset adapters
- PyTorch datasets and dataloaders

Responsibilities
----------------
- parse the training YAML for data-related settings
- resolve the dataset config path
- build the dataset adapter
- build train/validation datasets
- build DataLoaders
- provide a safe custom collate function for STRIDE samples (collate means to combine individual samples into a batch, and is used by the DataLoader)
- provide a lightweight batch inspection helper

Design principles
-----------------
- adapter owns dataset logic
- this module only orchestrates loading
- keep batching predictable and explicit
- avoid PyTorch default_collate for nested metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from data_adapters.danra_era5_small.adapter import DanraEra5SmallAdapter
from stride_core.configs.adapter_config import AdapterConfig


# -----------------------------------------------------------------------------
# Training data configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingDataConfig:
    """
    Minimal training-side data configuration extracted from the training YAML.
    """

    dataset_config_path: Path

    batch_size: int
    num_workers: int

    pin_memory: bool

    shuffle_train: bool
    drop_last_train: bool

    shuffle_val: bool
    drop_last_val: bool

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_training_yaml(cls, training_config_path: str | Path) -> "TrainingDataConfig":
        path = Path(training_config_path)

        if not path.exists():
            raise FileNotFoundError(f"Training config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected training config YAML to load into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg, config_path=path)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        config_path: str | Path | None = None,
    ) -> "TrainingDataConfig":

        if "training" not in cfg:
            raise KeyError("Expected top-level key 'training' in training config")

        training_cfg = cfg["training"]

        if not isinstance(training_cfg, dict):
            raise ValueError(
                f"Expected 'training' section to be a dict, got {type(training_cfg)}"
            )

        configs_cfg = training_cfg.get("configs", {})
        data_cfg = training_cfg.get("data", {})

        if not isinstance(configs_cfg, dict):
            raise ValueError("Expected 'training.configs' to be a dict")

        if not isinstance(data_cfg, dict):
            raise ValueError("Expected 'training.data' to be a dict")

        dataset_config_raw = configs_cfg.get("dataset_config")

        if dataset_config_raw is None:
            raise KeyError("Missing required key 'training.configs.dataset_config'")

        dataset_config_path = cls._resolve_config_path(
            raw_path=dataset_config_raw,
            config_path=config_path,
        )

        batch_size = int(data_cfg.get("batch_size", 1))
        num_workers = int(data_cfg.get("num_workers", 0))

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")

        return cls(
            dataset_config_path=dataset_config_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=bool(data_cfg.get("pin_memory", False)),
            shuffle_train=bool(data_cfg.get("shuffle_train", True)),
            drop_last_train=bool(data_cfg.get("drop_last_train", False)),
            shuffle_val=bool(data_cfg.get("shuffle_val", False)),
            drop_last_val=bool(data_cfg.get("drop_last_val", False)),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config_path(
        raw_path: str | Path,
        config_path: str | Path | None = None,
    ) -> Path:

        path = Path(raw_path)

        if path.is_absolute():
            return path

        if config_path is None:
            return path

        config_path = Path(config_path).resolve()
        repo_root = Path(__file__).resolve().parents[2]

        if "configs" in config_path.parts:
            return repo_root / path

        return config_path.parent / path


# -----------------------------------------------------------------------------
# Built training data container
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltTrainingData:
    """
    Container for datasets and dataloaders used by the trainer.
    """

    adapter_config: AdapterConfig

    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]

    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------


def build_dataset_adapter(adapter_config: AdapterConfig) -> DanraEra5SmallAdapter:
    """
    Build the STRIDE v1 dataset adapter.

    In STRIDE v1 we only support the danra_era5_small adapter.
    Later this should become registry-driven.
    """

    return DanraEra5SmallAdapter(adapter_config)


# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------


def stride_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function for STRIDE samples.

    PyTorch's default_collate fails on nested metadata structures
    and optional None values. This collate keeps the structure
    predictable while stacking tensor fields, including temporal
    conditioning features such as DOY sin/cos.
    """

    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch")

    collated: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # tensor fields
    # ------------------------------------------------------------------

    collated["target"] = torch.stack([sample["target"] for sample in batch], dim=0)

    collated["cond_dynamic"] = torch.stack(
        [sample["cond_dynamic"] for sample in batch],
        dim=0,
    )

    cond_static_values = [sample.get("cond_static") for sample in batch]

    if all(v is None for v in cond_static_values):
        collated["cond_static"] = None

    elif any(v is None for v in cond_static_values):
        raise ValueError(
            "Inconsistent batch: some samples contain cond_static and others do not"
        )

    else:
        collated["cond_static"] = torch.stack([v for v in cond_static_values if v is not None], dim=0)

    # Optional temporal conditioning features
    time_feature_values = [sample.get("time_features") for sample in batch]

    if all(v is None for v in time_feature_values):
        collated["time_features"] = None

    elif any(v is None for v in time_feature_values):
        raise ValueError(
            "Inconsistent batch: some samples contain time_features and others do not"
        )

    else:
        collated["time_features"] = torch.stack(
            [v for v in time_feature_values if v is not None],
            dim=0,
        )

    # ------------------------------------------------------------------
    # coordinate fields
    # ------------------------------------------------------------------

    collated["cond_coord"] = [sample.get("cond_coord") for sample in batch]

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    meta_values = [sample.get("meta", {}) for sample in batch]

    meta_keys: set[str] = set()
    for meta in meta_values:
        if isinstance(meta, dict):
            meta_keys.update(meta.keys())

    collated_meta: dict[str, Any] = {}

    for key in sorted(meta_keys):
        values = [meta.get(key) for meta in meta_values]

        if all(v is None for v in values):
            collated_meta[key] = None

        elif all(isinstance(v, torch.Tensor) for v in values if v is not None):
            if any(v is None for v in values):
                collated_meta[key] = values
            else:
                collated_meta[key] = torch.stack(values, dim=0)

        else:
            collated_meta[key] = values

    collated["meta"] = collated_meta

    return collated


# -----------------------------------------------------------------------------
# Dataloader builder
# -----------------------------------------------------------------------------


def build_dataloader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader[Any]:
    """
    Build a PyTorch DataLoader with STRIDE collate logic.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=stride_collate_fn,
    )



# -----------------------------------------------------------------------------
# Training data orchestration
# -----------------------------------------------------------------------------


def build_training_data(training_config_path: str | Path) -> BuiltTrainingData:
    """
    Build train and validation datasets + DataLoaders
    """

    training_data_cfg = TrainingDataConfig.from_training_yaml(training_config_path)

    adapter_cfg = AdapterConfig.from_yaml(training_data_cfg.dataset_config_path)

    adapter = build_dataset_adapter(adapter_cfg)

    datasets = adapter.build_datasets()

    if "train" not in datasets:
        raise KeyError("Adapter did not return a 'train' dataset")

    if "valid" not in datasets:
        raise KeyError("Adapter did not return a 'valid' dataset")

    train_dataset = datasets["train"]
    val_dataset = datasets["valid"]

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_data_cfg.batch_size,
        shuffle=training_data_cfg.shuffle_train,
        num_workers=training_data_cfg.num_workers,
        pin_memory=training_data_cfg.pin_memory,
        drop_last=training_data_cfg.drop_last_train,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=training_data_cfg.batch_size,
        shuffle=training_data_cfg.shuffle_val,
        num_workers=training_data_cfg.num_workers,
        pin_memory=training_data_cfg.pin_memory,
        drop_last=training_data_cfg.drop_last_val,
    )

    return BuiltTrainingData(
        adapter_config=adapter_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )


# -----------------------------------------------------------------------------
# Batch debugging helper
# -----------------------------------------------------------------------------


def describe_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a compact summary of a STRIDE batch for debugging.
    """

    summary: dict[str, Any] = {}

    for key in ("target", "cond_dynamic", "cond_static", "time_features"):

        value = batch.get(key)

        if value is None:
            summary[key] = None
            continue

        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "mean": float(value.mean().item()),
            }

        else:
            summary[key] = {"type": type(value).__name__}

    cond_coord = batch.get("cond_coord")

    if isinstance(cond_coord, list):
        summary["cond_coord"] = f"list[{len(cond_coord)}]"
    else:
        summary["cond_coord"] = type(cond_coord).__name__

    meta = batch.get("meta")

    if isinstance(meta, dict):
        summary["meta_keys"] = sorted(meta.keys())
        summary["meta_value_types"] = {
            key: type(value).__name__ for key, value in meta.items()
        }
    else:
        summary["meta_keys"] = None
        summary["meta_value_types"] = None

    return summary