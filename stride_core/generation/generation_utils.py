"""
Utilities for STRIDE generation.

This module keeps reusable generation-side helpers out of the trainer and out
of preview plotting code. Its job is to provide a small, practical bridge from
batched model-space tensors to sampler calls and preview-ready payloads.

Current scope
-------------
- load generation config from YAML
- move nested batch structures to device
- call the EDM sampler with the current STRIDE interface
- convert tensors to numpy safely
- build physical-space-ready preview payloads when transform metadata is present
- save generated preview arrays
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import yaml

from stride_core.generation.edm_sampler import edm_sampler


@dataclass(frozen=True)
class GenerationConfig:
    """
    Minimal generation configuration extracted from the generation YAML.
    """

    num_steps: int
    sigma_min: float
    sigma_max: float
    rho: float
    S_churn: float
    S_min: float
    S_max: float
    S_noise: float
    return_intermediates: bool

    @classmethod
    def from_yaml(cls, generation_config_path: str | Path) -> "GenerationConfig":
        path = Path(generation_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Generation config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected generation config YAML to load into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "GenerationConfig":
        gen_cfg = cfg.get("generation", cfg)
        if not isinstance(gen_cfg, dict):
            raise ValueError("Expected generation config to be a dict")

        num_steps = int(gen_cfg.get("num_steps", 18))
        sigma_min = float(gen_cfg.get("sigma_min", 0.002))
        sigma_max = float(gen_cfg.get("sigma_max", 80.0))
        rho = float(gen_cfg.get("rho", 7.0))
        s_churn = float(gen_cfg.get("S_churn", 0.0))
        s_min = float(gen_cfg.get("S_min", 0.0))
        s_max_raw = gen_cfg.get("S_max", float("inf"))
        s_max = float("inf") if s_max_raw in {"inf", "INF", None} else float(s_max_raw)
        s_noise = float(gen_cfg.get("S_noise", 1.0))
        return_intermediates = bool(gen_cfg.get("return_intermediates", False))

        if num_steps <= 0:
            raise ValueError(f"generation.num_steps must be > 0, got {num_steps}")
        if sigma_min <= 0:
            raise ValueError(f"generation.sigma_min must be > 0, got {sigma_min}")
        if sigma_max <= 0:
            raise ValueError(f"generation.sigma_max must be > 0, got {sigma_max}")
        if rho <= 0:
            raise ValueError(f"generation.rho must be > 0, got {rho}")

        return cls(
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            S_churn=s_churn,
            S_min=s_min,
            S_max=s_max,
            S_noise=s_noise,
            return_intermediates=return_intermediates,
        )

    def to_sampler_kwargs(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "S_churn": self.S_churn,
            "S_min": self.S_min,
            "S_max": self.S_max,
            "S_noise": self.S_noise,
            "return_intermediates": self.return_intermediates,
        }


def load_generation_config(
    generation_config: GenerationConfig | dict[str, Any] | str | Path,
) -> GenerationConfig:
    if isinstance(generation_config, GenerationConfig):
        return generation_config
    if isinstance(generation_config, (str, Path)):
        return GenerationConfig.from_yaml(generation_config)
    if isinstance(generation_config, dict):
        return GenerationConfig.from_dict(generation_config)
    raise TypeError(
        f"Unsupported generation_config type {type(generation_config)}. "
        "Expected GenerationConfig, dict, str, or Path."
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def _move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _move(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_move(v) for v in value]
        return value

    return {key: _move(value) for key, value in batch.items()}


def tensor_to_numpy(value: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def get_batch_meta_value(batch: dict[str, Any], key: str) -> Any:
    """
    Retrieve a metadata value from a batched STRIDE sample.

    The training collate function currently keeps most metadata entries as lists
    of per-sample values. For preview plotting we often only use fixed preview
    cases, so returning the single shared value or the first value is usually
    the most practical behavior.
    """
    meta = batch.get("meta", {})
    if not isinstance(meta, dict):
        return None

    value = meta.get(key)
    if isinstance(value, list):
        if len(value) == 0:
            return None
        # If all entries match, return the shared value. Otherwise return the
        # first entry; preview batches are typically fixed and small.
        first = value[0]
        if all(v == first for v in value):
            return first
        return first
    return value



def get_transform_metadata_from_batch(batch: dict[str, Any]) -> dict[str, Any] | None:
    """
    Retrieve transform metadata from the batch, if present.
    """
    value = get_batch_meta_value(batch, "transform_metadata")
    if isinstance(value, dict):
        return value
    return None



class VariableNames(TypedDict):
    target: str | None
    dynamic: list[str]
    static: list[str]


def get_variable_names_from_batch(batch: dict[str, Any]) -> VariableNames:
    """
    Extract target / dynamic / static variable names from batch metadata.
    """
    target_var = get_batch_meta_value(batch, "target_var")
    cond_dynamic_vars = get_batch_meta_value(batch, "cond_dynamic_vars")
    cond_static_vars = get_batch_meta_value(batch, "cond_static_vars")

    dynamic_names = [] if cond_dynamic_vars is None else list(cond_dynamic_vars)
    static_names = [] if cond_static_vars is None else list(cond_static_vars)

    return {
        "target": None if target_var is None else str(target_var),
        "dynamic": dynamic_names,
        "static": static_names,
    }



def _inverse_transform_single_tensor(
    tensor: torch.Tensor,
    transform_cfg: dict[str, Any] | None,
) -> torch.Tensor:
    """
    Invert a single transformed tensor using adapter-provided metadata.

    Supported transforms
    --------------------
    - identity
    - zscore
    - log_zscore  (inverse: exp(z * std_log + mean_log) - clip_min)
    """
    if transform_cfg is None:
        return tensor

    name = str(transform_cfg.get("name", "identity"))
    stats = transform_cfg.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}

    if name == "identity":
        return tensor

    if name == "zscore":
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        return tensor * std + mean

    if name == "log_zscore":
        mean_log = float(stats.get("mean_log", 0.0))
        std_log = float(stats.get("std_log", 1.0))
        clip_min = transform_cfg.get("clip_min", 0.0)
        clip_min_value = 0.0 if clip_min is None else float(clip_min)
        return torch.exp(tensor * std_log + mean_log) - clip_min_value

    raise ValueError(f"Unsupported inverse transform '{name}'")



def inverse_transform_batch_tensors(
    batch: dict[str, Any],
    generated: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    """
    Build a physical-space version of the preview tensors when transform
    metadata is available. If metadata is missing, the original tensors are
    returned unchanged.
    """
    transform_metadata = get_transform_metadata_from_batch(batch)

    target = batch.get("target")
    cond_dynamic = batch.get("cond_dynamic")
    cond_static = batch.get("cond_static")

    if not isinstance(target, torch.Tensor):
        raise TypeError("batch['target'] must be a torch.Tensor")
    if not isinstance(cond_dynamic, torch.Tensor):
        raise TypeError("batch['cond_dynamic'] must be a torch.Tensor")
    if not isinstance(generated, torch.Tensor):
        raise TypeError("generated must be a torch.Tensor")

    if transform_metadata is None:
        return {
            "target": target,
            "generated": generated,
            "cond_dynamic": cond_dynamic,
            "cond_static": cond_static if isinstance(cond_static, torch.Tensor) else None,
        }

    target_cfg = transform_metadata.get("target")
    dynamic_cfg = transform_metadata.get("dynamic", {})
    static_cfg = transform_metadata.get("static", {})

    variable_names = get_variable_names_from_batch(batch)
    dynamic_names = variable_names["dynamic"]
    static_names = variable_names["static"]

    target_phys = _inverse_transform_single_tensor(target, target_cfg)
    generated_phys = _inverse_transform_single_tensor(generated, target_cfg)

    cond_dynamic_phys = cond_dynamic.clone()
    for idx in range(cond_dynamic.shape[1]):
        var_name = dynamic_names[idx] if idx < len(dynamic_names) else f"dynamic_{idx}"
        cfg = dynamic_cfg.get(var_name)
        cond_dynamic_phys[:, idx : idx + 1, :, :] = _inverse_transform_single_tensor(
            cond_dynamic[:, idx : idx + 1, :, :],
            cfg,
        )

    cond_static_phys: torch.Tensor | None = None
    if isinstance(cond_static, torch.Tensor):
        cond_static_phys = cond_static.clone()
        for idx in range(cond_static.shape[1]):
            var_name = static_names[idx] if idx < len(static_names) else f"static_{idx}"
            cfg = static_cfg.get(var_name)
            cond_static_phys[:, idx : idx + 1, :, :] = _inverse_transform_single_tensor(
                cond_static[:, idx : idx + 1, :, :],
                cfg,
            )

    return {
        "target": target_phys,
        "generated": generated_phys,
        "cond_dynamic": cond_dynamic_phys,
        "cond_static": cond_static_phys,
    }



def build_preview_payload(
    batch: dict[str, Any],
    generated: torch.Tensor,
) -> dict[str, Any]:
    """
    Build a preview-ready payload containing both raw tensors and physical-space
    tensors when transform metadata is available.
    """
    variable_names = get_variable_names_from_batch(batch)
    physical = inverse_transform_batch_tensors(batch, generated)

    return {
        "target": batch.get("target"),
        "generated": generated,
        "cond_dynamic": batch.get("cond_dynamic"),
        "cond_static": batch.get("cond_static"),
        "target_physical": physical["target"],
        "generated_physical": physical["generated"],
        "cond_dynamic_physical": physical["cond_dynamic"],
        "cond_static_physical": physical["cond_static"],
        "variable_names": variable_names,
        "date": get_batch_meta_value(batch, "date"),
        "domain_tag": get_batch_meta_value(batch, "domain_tag"),
        "transform_metadata": get_transform_metadata_from_batch(batch),
    }


def generate_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    generation_config: GenerationConfig | dict[str, Any] | str | Path,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a batch of samples using the current STRIDE EDM sampler.

    The live sampler interface is:

        edm_sampler(
            model,
            cond_dynamic,
            cond_static=None,
            *,
            num_steps=...,
            sigma_min=...,
            sigma_max=...,
            rho=...,
            S_churn=...,
            S_min=...,
            S_max=...,
            S_noise=...,
            y=None,
            variable_labels=None,
            return_intermediates=False,
        )

    so this wrapper should stay explicit rather than trying multiple legacy call
    variants.
    """
    gen_cfg = load_generation_config(generation_config)
    batch = move_batch_to_device(batch, device)

    cond_dynamic = batch.get("cond_dynamic")
    cond_static = batch.get("cond_static")
    meta = batch.get("meta", {})
    time_features = batch.get("time_features")

    if not isinstance(cond_dynamic, torch.Tensor):
        raise TypeError("batch['cond_dynamic'] must be a torch.Tensor")
    if cond_dynamic.ndim != 4:
        raise ValueError(
            "Expected batch['cond_dynamic'] to have shape (B, C, H, W), "
            f"got {tuple(cond_dynamic.shape)}"
        )

    if cond_static is not None:
        if not isinstance(cond_static, torch.Tensor):
            raise TypeError(
                "batch['cond_static'] must be a torch.Tensor or None"
            )
        if cond_static.ndim != 4:
            raise ValueError(
                "Expected batch['cond_static'] to have shape (B, C, H, W), "
                f"got {tuple(cond_static.shape)}"
            )

    if time_features is not None:
        if not isinstance(time_features, torch.Tensor):
            raise TypeError(
                "batch['time_features'] must be a torch.Tensor or None"
            )
        if time_features.ndim != 2:
            raise ValueError(
                "Expected batch['time_features'] to have shape (B, 2), "
                f"got {tuple(time_features.shape)}"
            )
        if time_features.shape[0] != cond_dynamic.shape[0]:
            raise ValueError(
                "batch['time_features'] batch dimension mismatch: "
                f"expected {cond_dynamic.shape[0]}, got {time_features.shape[0]}"
            )
        if time_features.shape[1] != 2:
            raise ValueError(
                "batch['time_features'] must have final dimension 2 for "
                "[sin(DOY), cos(DOY)], got "
                f"{tuple(time_features.shape)}"
            )
        time_features = time_features.to(device=device, dtype=cond_dynamic.dtype)

    sampler_kwargs = gen_cfg.to_sampler_kwargs()

    variable_labels: torch.Tensor | None = None
    if isinstance(meta, dict):
        raw_variable_labels = meta.get("variable_labels")

        if isinstance(raw_variable_labels, torch.Tensor):
            variable_labels = raw_variable_labels.to(device)
        elif isinstance(raw_variable_labels, list) and len(raw_variable_labels) > 0:
            if all(isinstance(v, (int, float)) for v in raw_variable_labels):
                variable_labels = torch.tensor(
                    raw_variable_labels,
                    device=device,
                    dtype=cond_dynamic.dtype,
                )

    generated = edm_sampler(
        model,
        cond_dynamic,
        cond_static=cond_static,
        y=time_features,
        variable_labels=variable_labels,
        **sampler_kwargs,
    )

    if isinstance(generated, dict):
        sample = generated.get("sample")
        if not isinstance(sample, torch.Tensor):
            raise TypeError(
                "edm_sampler returned a dict, but generated['sample'] is not a torch.Tensor"
            )
        return sample

    if not isinstance(generated, torch.Tensor):
        raise TypeError(
            f"edm_sampler returned type {type(generated)}, expected torch.Tensor"
        )

    return generated


def save_preview_arrays(
    output_dir: str | Path,
    *,
    epoch: int,
    batch: dict[str, Any],
    generated: torch.Tensor,
    prefix: str = "preview",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{prefix}_epoch_{epoch + 1:04d}.npz"

    preview_payload = build_preview_payload(batch, generated)

    payload: dict[str, np.ndarray | None] = {
        "generated": tensor_to_numpy(generated),
    }

    for key in (
        "target",
        "cond_dynamic",
        "cond_static",
        "target_physical",
        "generated_physical",
        "cond_dynamic_physical",
        "cond_static_physical",
    ):
        value = preview_payload.get(key)
        if isinstance(value, torch.Tensor):
            payload[key] = tensor_to_numpy(value)

    # Filter out None values before saving
    filtered_payload = {k: v for k, v in payload.items() if v is not None}
    np.savez_compressed(save_path, **filtered_payload)
    return save_path
