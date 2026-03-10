"""
Top-level model builder for STRIDE.

This module turns a validated `ModelSpec` into an instantiated model by routing
through the model registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from stride_core.configs.model_config import ModelSpec
from stride_core.models.model_registry import get_model_builder


SpecLike = ModelSpec | str | Path | dict[str, Any]


def _coerce_model_spec(spec_like: SpecLike) -> ModelSpec:
    """
    Normalize supported model-spec inputs into a `ModelSpec`.

    Supported inputs
    ----------------
    - ModelSpec instance
    - YAML path as `str` or `Path`
    - config dictionary with top-level `model:` section
    """
    if isinstance(spec_like, ModelSpec):
        return spec_like

    if isinstance(spec_like, (str, Path)):
        return ModelSpec.from_yaml(spec_like)

    if isinstance(spec_like, dict):
        return ModelSpec.from_dict(spec_like)

    raise TypeError(
        f"Unsupported spec_like type {type(spec_like)}. "
        "Expected ModelSpec, str, Path, or dict."
    )


def build_model(spec_like: SpecLike) -> nn.Module:
    """
    Build a STRIDE model from a model spec, YAML path, or config dict.

    Examples
    --------
    >>> model = build_model(ModelSpec(...))
    >>> model = build_model("configs/models/edm_small.yaml")
    >>> model = build_model({"model": {...}})
    """
    spec = _coerce_model_spec(spec_like)
    builder = get_model_builder(spec.name)
    return builder(spec)
