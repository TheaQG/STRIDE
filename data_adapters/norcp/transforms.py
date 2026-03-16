

"""
Transform utilities for the NorCP STRIDE adapter.

This module defines per-variable preprocessing transforms used after raw NorCP
fields are loaded and converted to canonical physical units.

Current transform families
--------------------------
- identity
- zscore
- log_zscore

Design notes
------------
- precipitation should typically use `log_zscore`
- most other variables use `zscore`
- transform parameters are expected to come from saved statistics JSON files
  later in the pipeline
- inverse transforms are included so evaluation/generation code can map back to
  physical space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from data_adapters.norcp.variable_registry import get_default_transform


DEFAULT_LOG_EPSILON = 1e-6


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TransformMetadata:
    """
    Lightweight description of one configured transform.
    """

    variable: str
    source: str
    transform_name: str
    parameters: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "source": self.source,
            "transform_name": self.transform_name,
            "parameters": dict(self.parameters),
        }


# -------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------


def _to_float_array(x: np.ndarray | Any) -> np.ndarray:
    """
    Convert input to float32 numpy array.
    """
    return np.asarray(x, dtype=np.float32)



def _require_keys(stats: Mapping[str, Any], keys: tuple[str, ...], *, context: str) -> None:
    """
    Ensure required parameter keys are present.
    """
    missing = [key for key in keys if key not in stats]
    if missing:
        raise KeyError(f"Missing required transform stats {missing} for {context}")



def _as_float_dict(stats: Mapping[str, Any] | None) -> dict[str, float]:
    """
    Normalize transform stats to a plain float dictionary.
    """
    if stats is None:
        return {}
    return {str(key): float(value) for key, value in stats.items()}


# -------------------------------------------------------------------
# Transform classes
# -------------------------------------------------------------------

class BaseTransform:
    """
    Base interface for NorCP transforms.
    """

    name: str = "base"

    def __init__(self, *, variable: str, source: str, stats: Mapping[str, Any] | None = None):
        self.variable = variable
        self.source = source
        self.stats = _as_float_dict(stats)

    def forward(self, x: np.ndarray | Any) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, x: np.ndarray | Any) -> np.ndarray:
        raise NotImplementedError

    def metadata(self) -> TransformMetadata:
        return TransformMetadata(
            variable=self.variable,
            source=self.source,
            transform_name=self.name,
            parameters=dict(self.stats),
        )

    def __call__(self, x: np.ndarray | Any) -> np.ndarray:
        return self.forward(x)


class IdentityTransform(BaseTransform):
    """
    No-op transform.
    """

    name = "identity"

    def forward(self, x: np.ndarray | Any) -> np.ndarray:
        return _to_float_array(x)

    def inverse(self, x: np.ndarray | Any) -> np.ndarray:
        return _to_float_array(x)


class ZScoreTransform(BaseTransform):
    """
    Standard affine normalization: (x - mean) / std.
    """

    name = "zscore"

    def __init__(self, *, variable: str, source: str, stats: Mapping[str, Any] | None = None):
        super().__init__(variable=variable, source=source, stats=stats)
        _require_keys(self.stats, ("mean", "std"), context=f"zscore:{variable}:{source}")
        self.mean = float(self.stats["mean"])
        self.std = float(self.stats["std"])
        if self.std <= 0.0:
            raise ValueError(
                f"ZScoreTransform requires std > 0 for variable='{variable}', source='{source}', got {self.std}"
            )

    def forward(self, x: np.ndarray | Any) -> np.ndarray:
        x = _to_float_array(x)
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray | Any) -> np.ndarray:
        x = _to_float_array(x)
        return x * self.std + self.mean


class LogZScoreTransform(BaseTransform):
    """
    Log-plus-zscore transform, primarily intended for precipitation.

    Forward:
        z = (log(x + epsilon) - log_mean) / log_std

    Inverse:
        x = exp(z * log_std + log_mean) - epsilon
    """

    name = "log_zscore"

    def __init__(self, *, variable: str, source: str, stats: Mapping[str, Any] | None = None):
        super().__init__(variable=variable, source=source, stats=stats)
        _require_keys(
            self.stats,
            ("log_mean", "log_std"),
            context=f"log_zscore:{variable}:{source}",
        )
        self.log_mean = float(self.stats["log_mean"])
        self.log_std = float(self.stats["log_std"])
        self.epsilon = float(self.stats.get("epsilon", DEFAULT_LOG_EPSILON))
        if self.log_std <= 0.0:
            raise ValueError(
                f"LogZScoreTransform requires log_std > 0 for variable='{variable}', source='{source}', got {self.log_std}"
            )
        if self.epsilon <= 0.0:
            raise ValueError(
                f"LogZScoreTransform requires epsilon > 0 for variable='{variable}', source='{source}', got {self.epsilon}"
            )

    def forward(self, x: np.ndarray | Any) -> np.ndarray:
        x = _to_float_array(x)
        if np.any(x < 0.0):
            raise ValueError(
                f"LogZScoreTransform received negative values for variable='{self.variable}', source='{self.source}'"
            )
        logged = np.log(x + self.epsilon)
        return (logged - self.log_mean) / self.log_std

    def inverse(self, x: np.ndarray | Any) -> np.ndarray:
        x = _to_float_array(x)
        out = np.exp(x * self.log_std + self.log_mean) - self.epsilon
        return np.maximum(out, 0.0)


# -------------------------------------------------------------------
# Builder helpers
# -------------------------------------------------------------------


def build_transform(
    *,
    variable: str,
    source: str,
    transform_name: str | None = None,
    stats: Mapping[str, Any] | None = None,
) -> BaseTransform:
    """
    Build one transform instance.

    If `transform_name` is omitted, the variable registry default is used.
    """
    name = transform_name or get_default_transform(variable)

    if name == "identity":
        return IdentityTransform(variable=variable, source=source, stats=stats)
    if name == "zscore":
        return ZScoreTransform(variable=variable, source=source, stats=stats)
    if name == "log_zscore":
        return LogZScoreTransform(variable=variable, source=source, stats=stats)

    raise ValueError(
        f"Unsupported transform '{name}' for variable='{variable}', source='{source}'"
    )



def build_transform_from_stats_block(
    *,
    variable: str,
    source: str,
    stats_block: Mapping[str, Any],
) -> BaseTransform:
    """
    Build a transform from a saved statistics JSON block.

    Expected shape:
        {
            "transform_name": "zscore" | "log_zscore" | "identity",
            "transform_stats": {...}
        }

    This mirrors the structure planned for NorCP saved statistics files.
    """
    if "transform_name" not in stats_block:
        raise KeyError("Expected 'transform_name' in stats block")

    transform_name = str(stats_block["transform_name"])
    transform_stats = stats_block.get("transform_stats", {})
    if not isinstance(transform_stats, Mapping):
        raise TypeError("Expected 'transform_stats' to be a mapping")

    return build_transform(
        variable=variable,
        source=source,
        transform_name=transform_name,
        stats=transform_stats,
    )


# -------------------------------------------------------------------
# Batch helpers
# -------------------------------------------------------------------


def apply_transform_to_stack(
    stack: np.ndarray,
    transforms: list[BaseTransform],
) -> np.ndarray:
    """
    Apply per-channel transforms to a [C, H, W] stack.
    """
    stack = _to_float_array(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack with shape [C,H,W], got {stack.shape}")
    if stack.shape[0] != len(transforms):
        raise ValueError(
            f"Number of transforms ({len(transforms)}) must match stack channels ({stack.shape[0]})"
        )

    transformed_channels = [
        transforms[channel_idx].forward(stack[channel_idx])
        for channel_idx in range(stack.shape[0])
    ]
    return np.stack(transformed_channels, axis=0).astype(np.float32, copy=False)



def inverse_transform_stack(
    stack: np.ndarray,
    transforms: list[BaseTransform],
) -> np.ndarray:
    """
    Apply inverse per-channel transforms to a [C, H, W] stack.
    """
    stack = _to_float_array(stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected stack with shape [C,H,W], got {stack.shape}")
    if stack.shape[0] != len(transforms):
        raise ValueError(
            f"Number of transforms ({len(transforms)}) must match stack channels ({stack.shape[0]})"
        )

    inverted_channels = [
        transforms[channel_idx].inverse(stack[channel_idx])
        for channel_idx in range(stack.shape[0])
    ]
    return np.stack(inverted_channels, axis=0).astype(np.float32, copy=False)



def build_transform_list(
    *,
    variables: list[str],
    source: str,
    transform_name_by_variable: Mapping[str, str] | None = None,
    stats_by_variable: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[BaseTransform]:
    """
    Build an ordered transform list matching a variable order.
    """
    transform_name_by_variable = dict(transform_name_by_variable or {})
    stats_by_variable = dict(stats_by_variable or {})

    return [
        build_transform(
            variable=variable,
            source=source,
            transform_name=transform_name_by_variable.get(variable),
            stats=stats_by_variable.get(variable),
        )
        for variable in variables
    ]