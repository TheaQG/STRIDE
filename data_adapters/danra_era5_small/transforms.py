"""
Transform utilities for the small DANRA/ERA5 STRIDE adapter.
Supports:
    - IdentityTransform
    - ZScoreTransform
    - LogZScoreTransform

Also provides:
    - Lightweight stats-loading interface from dict / JSON
    - Builder utilities to construct transforms from config-like names

Notes
-----
This module intentionally stays simple for the first STRIDE setup.
It focuses on invertible transforms needed for:
    - target precipitation: log-zscore
    - ERA5 precipitation: log-zscore
    - ERA5 temperature: zscore
    - statics: identity or zscore later
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EPS = 1e-6
DEFAULT_CLIP_MIN = 0.0
DEFAULT_FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class TransformMetadata:
    """
    Lightweight metadata describing a transform instance.
    """

    name: str
    stats: dict[str, float]
    clip_min: float | None = None
    eps: float = DEFAULT_EPS


class BaseTransform:
    """
    Base interface for STRIDE transforms.
    """

    name: str = "base"

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_metadata(self) -> TransformMetadata:
        raise NotImplementedError


class IdentityTransform(BaseTransform):
    name = "identity"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(name=self.name, stats={})


class ZScoreTransform(BaseTransform):
    name = "zscore"

    def __init__(self, mean: float, std: float, eps: float = DEFAULT_EPS) -> None:
        if std <= 0:
            raise ValueError(f"std must be positive for z-score transform, got {std}")
        self.mean = float(mean)
        self.std = float(std)
        self.eps = float(eps)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)
        return x * (self.std + self.eps) + self.mean

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name=self.name,
            stats={"mean": self.mean, "std": self.std},
            eps=self.eps,
        )


class LogZScoreTransform(BaseTransform):
    name = "log_zscore"

    def __init__(
        self,
        mean_log: float,
        std_log: float,
        clip_min: float = DEFAULT_CLIP_MIN,
        eps: float = DEFAULT_EPS,
    ) -> None:
        if std_log <= 0:
            raise ValueError(
                f"std_log must be positive for log-zscore transform, got {std_log}"
            )
        self.mean_log = float(mean_log)
        self.std_log = float(std_log)
        self.clip_min = float(clip_min)
        self.eps = float(eps)

    def _pre_log(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)
        x = np.clip(x, self.clip_min, None)
        return np.log1p(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_log = self._pre_log(x)
        return (x_log - self.mean_log) / (self.std_log + self.eps)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)
        x_log = x * (self.std_log + self.eps) + self.mean_log
        x_phys = np.expm1(x_log)
        x_phys = np.clip(x_phys, self.clip_min, None)
        return np.asarray(x_phys, dtype=DEFAULT_FLOAT_DTYPE)

    def get_metadata(self) -> TransformMetadata:
        return TransformMetadata(
            name=self.name,
            stats={"mean_log": self.mean_log, "std_log": self.std_log},
            clip_min=self.clip_min,
            eps=self.eps,
        )


def load_stats_from_json(stats_path: str | Path) -> dict[str, Any]:
    """
    Load transform statistics from a JSON file.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if not isinstance(stats, dict):
        raise ValueError(f"Expected stats JSON to contain a dict, got {type(stats)}")

    return stats


def build_transform(
    transform_name: str,
    stats: dict[str, Any] | None = None,
    stats_path: str | Path | None = None,
    clip_min: float = DEFAULT_CLIP_MIN,
    eps: float = DEFAULT_EPS,
) -> BaseTransform:
    """
    Build a transform from a transform name and either stats dict or JSON path.

    Supported names:
        - identity
        - zscore
        - log_zscore
    """
    if stats is not None and stats_path is not None:
        raise ValueError("Provide either stats or stats_path, not both")

    if stats is None and stats_path is not None:
        stats = load_stats_from_json(stats_path)

    if transform_name == "identity":
        return IdentityTransform()

    if stats is None:
        raise ValueError(
            f"Transform '{transform_name}' requires stats, but none were provided"
        )

    if transform_name == "zscore":
        return ZScoreTransform(
            mean=float(stats["mean"]),
            std=float(stats["std"]),
            eps=eps,
        )

    if transform_name == "log_zscore":
        return LogZScoreTransform(
            mean_log=float(stats["mean_log"]),
            std_log=float(stats["std_log"]),
            clip_min=clip_min,
            eps=eps,
        )

    raise ValueError(
        f"Unknown transform_name '{transform_name}'. "
        f"Supported: ['identity', 'zscore', 'log_zscore']"
    )


def compute_zscore_stats(x: np.ndarray) -> dict[str, float]:
    """
    Compute simple z-score stats from an array.

    This is mainly for debugging / inspection scripts.
    Offline global stats should still be the canonical training-time source.
    """
    x = np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
    }


def compute_log_zscore_stats(
    x: np.ndarray,
    clip_min: float = DEFAULT_CLIP_MIN,
) -> dict[str, float]:
    """
    Compute log-zscore stats from an array.

    This is mainly for debugging / inspection scripts.
    Offline global stats should still be the canonical training-time source.
    """
    x = np.asarray(x, dtype=DEFAULT_FLOAT_DTYPE)
    x = np.clip(x, clip_min, None)
    x_log = np.log1p(x)
    return {
        "mean_log": float(np.nanmean(x_log)),
        "std_log": float(np.nanstd(x_log)),
    }
