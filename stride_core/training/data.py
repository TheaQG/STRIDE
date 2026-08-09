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
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import yaml

from data_adapters.danra_era5_small.adapter import DanraEra5SmallAdapter
from data_adapters.norcp.adapter import NorCPAdapter
from stride_core.configs.adapter_config import AdapterConfig
from stride_core.utils.distributed import is_distributed, rank, world_size


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
    persistent_workers: bool
    prefetch_factor_train: int | None
    prefetch_factor_val: int | None

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

        pin_memory = bool(data_cfg.get("pin_memory", False))
        persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
        prefetch_factor_train = data_cfg.get("prefetch_factor_train", None)
        prefetch_factor_val = data_cfg.get("prefetch_factor_val", None)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")

        return cls(
            dataset_config_path=dataset_config_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor_train=(
                None if prefetch_factor_train is None else int(prefetch_factor_train)
            ),
            prefetch_factor_val=(
                None if prefetch_factor_val is None else int(prefetch_factor_val)
            ),
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

    train_sampler: DistributedSampler | None
    val_sampler: DistributedSampler | None


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------


def build_dataset_adapter(adapter_config: AdapterConfig) -> DanraEra5SmallAdapter | NorCPAdapter:
    """
    Build the dataset adapter for the configured STRIDE dataset.

    Current supported adapters:
    - DanraEra5SmallAdapter
    - NorCPAdapter

    Still an explicit dispatcher rather than a full registry, but currently enough to support multiple datasets without overengineering.
    """

    if (
        adapter_config.scenario_name is not None
        or adapter_config.target_source == "NORCP_HR"
    ):
        return NorCPAdapter(adapter_config)

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
    sampler: Sampler[Any] | None = None,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader[Any]:
    """
    Build a PyTorch DataLoader with STRIDE collate logic.
    """
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": stride_collate_fn,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)



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

    if "val" not in datasets:
        raise KeyError("Adapter did not return a 'val' dataset")

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=training_data_cfg.shuffle_train,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=training_data_cfg.shuffle_val,
        )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_data_cfg.batch_size,
        shuffle=training_data_cfg.shuffle_train,
        sampler=train_sampler,
        num_workers=training_data_cfg.num_workers,
        pin_memory=training_data_cfg.pin_memory,
        drop_last=training_data_cfg.drop_last_train,
        persistent_workers=training_data_cfg.persistent_workers,
        prefetch_factor=training_data_cfg.prefetch_factor_train,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=training_data_cfg.batch_size,
        shuffle=training_data_cfg.shuffle_val,
        sampler=val_sampler,
        num_workers=training_data_cfg.num_workers,
        pin_memory=training_data_cfg.pin_memory,
        drop_last=training_data_cfg.drop_last_val,
        persistent_workers=training_data_cfg.persistent_workers,
        prefetch_factor=training_data_cfg.prefetch_factor_val,
    )

    return BuiltTrainingData(
        adapter_config=adapter_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
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
