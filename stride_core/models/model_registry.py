"""
Model registry for STRIDE.

The registry maps a framework-level model name (from `ModelSpec.name`) to a
builder function. Builders are kept lightweight and lazy so imports only happen
when a model is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable

import torch.nn as nn

from stride_core.configs.model_config import ModelSpec


ModelBuilder = Callable[[ModelSpec], nn.Module]


def _build_edm_unet(spec: ModelSpec) -> nn.Module:
    """
    Build the default EDM-preconditioned UNet.

    The implementation is imported lazily so the registry itself stays cheap to
    import and so model code can be developed independently.
    """
    candidate_imports: list[tuple[str, str]] = [
        ("stride_core.models.edm_precond_unet", "EDMPrecondUNet"),
        ("stride_core.models.edm_unet", "EDMPrecondUNet"),
        ("stride_core.models.edm_unet", "EDMUNet"),
    ]

    last_error: Exception | None = None
    for module_name, class_name in candidate_imports:
        try:
            module = import_module(module_name)
            model_cls = getattr(module, class_name)
            model = model_cls(spec)
            if not isinstance(model, nn.Module):
                raise TypeError(
                    f"Registered model class {class_name} from {module_name} did not return an nn.Module"
                )
            return model
        except (ImportError, AttributeError, TypeError) as exc:
            last_error = exc
            continue

    raise ImportError(
        "Could not build model 'edm_unet'. Expected one of the following "
        "implementations to exist: "
        "stride_core.models.edm_precond_unet.EDMPrecondUNet, "
        "stride_core.models.edm_unet.EDMPrecondUNet, or "
        "stride_core.models.edm_unet.EDMUNet."
    ) from last_error


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "edm_unet": _build_edm_unet,
}


def get_model_builder(model_name: str) -> ModelBuilder:
    """
    Return the registered builder for a model name.
    """
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model name '{model_name}'. Available models: {available}"
        ) from exc
