"""
Training-time generation preview utilities for STRIDE.

This module keeps preview-generation logic out of trainer.py. The trainer should
only decide *when* to preview; this module handles *how* to build a fixed
preview batch, generate samples, save arrays, and save quick-look figures.

This version is preview-oriented and climate-aware:
- plots target / generated in physical units when transform metadata is present
- uses variable names from batch metadata when available
- overlays an LSM coastline-style contour when available
- uses more meaningful default colormaps for common climate variables
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from stride_core.generation.generation_utils import (
    build_preview_payload,
    generate_batch,
    move_batch_to_device,
    save_preview_arrays,
    tensor_to_numpy,
)
from stride_core.plotting.colormaps import get_variable_cmap
from stride_core.plotting.plotting_utils import (
    compute_shared_range,
    find_matching_dynamic_index,
    get_channel_name,
    overlay_lsm_contour,
    plot_field_with_colorbar_and_optional_boxplot,
)
from stride_core.training.data import stride_collate_fn


@dataclass(frozen=True)
class TrainingPreviewConfig:
    enabled: bool
    use_ema_model: bool
    every_n_epochs: int
    on_improved_val: bool
    min_epoch: int
    fixed_val_indices: tuple[int, ...]
    save_arrays: bool
    save_plots: bool
    output_subdir: str

    @classmethod
    def from_training_yaml(cls, training_config_path: str | Path) -> "TrainingPreviewConfig":
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
    def from_dict(cls, cfg: dict[str, Any]) -> "TrainingPreviewConfig":
        training_cfg = cfg.get("training", cfg)
        if not isinstance(training_cfg, dict):
            raise ValueError("Expected training config to be a dict")

        preview_cfg = training_cfg.get("generation_preview", {})
        if not isinstance(preview_cfg, dict):
            raise ValueError("Expected 'training.generation_preview' to be a dict")

        fixed_val_indices_raw = preview_cfg.get("fixed_val_indices", [0])
        if not isinstance(fixed_val_indices_raw, (list, tuple)):
            raise ValueError("generation_preview.fixed_val_indices must be a list")

        every_n_epochs = int(preview_cfg.get("every_n_epochs", 5))
        if every_n_epochs <= 0:
            raise ValueError(
                f"generation_preview.every_n_epochs must be > 0, got {every_n_epochs}"
            )

        return cls(
            enabled=bool(preview_cfg.get("enabled", False)),
            use_ema_model=bool(preview_cfg.get("use_ema_model", True)),
            every_n_epochs=every_n_epochs,
            on_improved_val=bool(preview_cfg.get("on_improved_val", True)),
            min_epoch=int(preview_cfg.get("min_epoch", 0)),
            fixed_val_indices=tuple(int(i) for i in fixed_val_indices_raw),
            save_arrays=bool(preview_cfg.get("save_arrays", True)),
            save_plots=bool(preview_cfg.get("save_plots", True)),
            output_subdir=str(preview_cfg.get("output_subdir", "training_generation_preview")),
        )


# -----------------------------------------------------------------------------
# Preview scheduling / batch creation
# -----------------------------------------------------------------------------


def should_run_training_preview(
    *,
    epoch: int,
    improved_val: bool,
    preview_config: TrainingPreviewConfig,
) -> bool:
    if not preview_config.enabled:
        return False
    if epoch + 1 < preview_config.min_epoch:
        return False

    on_schedule = (epoch + 1) % preview_config.every_n_epochs == 0
    on_improved = preview_config.on_improved_val and improved_val
    return on_schedule or on_improved


@torch.no_grad()
def build_preview_batch(
    val_dataset: Any,
    *,
    fixed_indices: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    if len(fixed_indices) == 0:
        raise ValueError("fixed_indices must contain at least one validation index")

    samples = [val_dataset[i] for i in fixed_indices]
    batch = stride_collate_fn(samples)
    return move_batch_to_device(batch, device)


@torch.no_grad()
def run_training_preview(
    *,
    model: torch.nn.Module,
    val_dataset: Any,
    generation_config_path: str | Path,
    output_dir: str | Path,
    epoch: int,
    device: torch.device,
    preview_config: TrainingPreviewConfig,
) -> dict[str, Any]:
    output_dir = Path(output_dir) / preview_config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = build_preview_batch(
        val_dataset,
        fixed_indices=preview_config.fixed_val_indices,
        device=device,
    )
    generated = generate_batch(
        model,
        batch,
        generation_config=generation_config_path,
        device=device,
    )

    preview_payload = build_preview_payload(batch, generated)

    saved_arrays: Path | None = None
    saved_figure: Path | None = None

    if preview_config.save_arrays:
        saved_arrays = save_preview_arrays(
            output_dir,
            epoch=epoch,
            batch=batch,
            generated=generated,
            prefix="training_preview",
        )

    if preview_config.save_plots:
        saved_figure = save_training_preview_figure(
            output_dir,
            epoch=epoch,
            preview_payload=preview_payload,
        )

    return {
        "batch": batch,
        "generated": generated,
        "preview_payload": preview_payload,
        "arrays_path": None if saved_arrays is None else str(saved_arrays),
        "figure_path": None if saved_figure is None else str(saved_figure),
    }


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def _maybe_get_lsm_overlay(preview_payload: dict[str, Any]) -> np.ndarray | None:
    cond_static_phys = preview_payload.get("cond_static_physical")
    cond_static = preview_payload.get("cond_static")
    variable_names = preview_payload.get("variable_names", {})
    static_names = variable_names.get("static", []) if isinstance(variable_names, dict) else []

    source = cond_static_phys if isinstance(cond_static_phys, torch.Tensor) else cond_static
    if not isinstance(source, torch.Tensor):
        return None
    if source.ndim != 4 or source.shape[1] == 0:
        return None

    lsm_idx = None
    for idx, name in enumerate(static_names):
        if str(name).lower() == "lsm":
            lsm_idx = idx
            break

    if lsm_idx is None:
        lsm_idx = 0

    return tensor_to_numpy(source[:, lsm_idx : lsm_idx + 1, :, :])


def _prefer_physical_preview_tensor(
    preview_payload: dict[str, Any],
    physical_key: str,
    fallback_key: str,
) -> Any:
    value = preview_payload.get(physical_key)
    if value is not None:
        return value
    return preview_payload.get(fallback_key)



def _upsample_preview_condition_to_target(
    array: np.ndarray | None,
    *,
    target_hw: tuple[int, int],
    mode: str = "nearest",
) -> np.ndarray | None:
    """
    Upsample LR conditioning fields for preview visualization only.
    This must never affect the actual model inputs.
    """
    if array is None:
        return None

    arr = np.asarray(array)
    if arr.ndim != 4:
        return arr

    if tuple(arr.shape[-2:]) == tuple(target_hw):
        return arr

    tensor = torch.from_numpy(arr).float()
    upsampled = torch.nn.functional.interpolate(
        tensor,
        size=target_hw,
        mode=mode,
    )
    return upsampled.cpu().numpy()


# -----------------------------------------------------------------------------
# Figure creation
# -----------------------------------------------------------------------------


def save_training_preview_figure(
    output_dir: str | Path,
    *,
    epoch: int,
    preview_payload: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_tensor = _prefer_physical_preview_tensor(
        preview_payload, "target_physical", "target"
    )
    generated_tensor = _prefer_physical_preview_tensor(
        preview_payload, "generated_physical", "generated"
    )
    cond_dynamic_tensor = _prefer_physical_preview_tensor(
        preview_payload, "cond_dynamic_physical", "cond_dynamic"
    )
    cond_static_tensor = _prefer_physical_preview_tensor(
        preview_payload, "cond_static_physical", "cond_static"
    )

    target = tensor_to_numpy(target_tensor)
    generated_np = tensor_to_numpy(generated_tensor)
    cond_dynamic = tensor_to_numpy(cond_dynamic_tensor)
    cond_static = tensor_to_numpy(cond_static_tensor)

    if target is None or generated_np is None:
        raise ValueError("target and generated must not be None")

    target_hw = tuple(target.shape[-2:])
    cond_dynamic = _upsample_preview_condition_to_target(
        cond_dynamic,
        target_hw=cast(tuple[int, int], target_hw),
        mode="nearest",
    )

    variable_names = preview_payload.get("variable_names", {})
    if not isinstance(variable_names, dict):
        variable_names = {}

    target_name = variable_names.get("target")
    dynamic_names = variable_names.get("dynamic", [])
    static_names = variable_names.get("static", [])
    date = preview_payload.get("date")
    domain_tag = preview_payload.get("domain_tag")

    n_cases = int(target.shape[0])
    matching_dynamic_idx = find_matching_dynamic_index(
        None if target_name is None else str(target_name),
        [str(name) for name in dynamic_names],
    )

    dynamic_row_indices = []
    if cond_dynamic is not None:
        dynamic_row_indices = list(range(cond_dynamic.shape[1]))
        if matching_dynamic_idx is not None and matching_dynamic_idx in dynamic_row_indices:
            dynamic_row_indices.remove(matching_dynamic_idx)

    row1_cols = 2 + (1 if matching_dynamic_idx is not None and cond_dynamic is not None else 0)
    row2_cols = max(len(dynamic_row_indices), 1)
    row3_cols = max(0 if cond_static is None else int(cond_static.shape[1]), 1)
    n_cols = max(row1_cols, row2_cols, row3_cols)

    fig, axes = plt.subplots(
        3 * n_cases,
        n_cols,
        figsize=(4.0 * n_cols, 3.0 * (3 * n_cases)),
        squeeze=False,
        constrained_layout=True,
    )

    lsm_overlay_all = _maybe_get_lsm_overlay(preview_payload)

    for case_idx in range(n_cases):
        row_offset = 3 * case_idx
        lsm_overlay_case = None
        if isinstance(lsm_overlay_all, np.ndarray):
            lsm_overlay_case = lsm_overlay_all[case_idx, 0]

        target_img = np.asarray(target[case_idx, 0])
        gen_img = np.asarray(generated_np[case_idx, 0])

        row1_fields = [gen_img, target_img]
        cond_match_img = None
        if matching_dynamic_idx is not None and cond_dynamic is not None:
            cond_match_img = np.asarray(cond_dynamic[case_idx, matching_dynamic_idx])
            row1_fields.append(cond_match_img)
        shared_vmin, shared_vmax = compute_shared_range(row1_fields)

        # Row 1: generated, HR target, matching LR condition
        row1_panels: list[tuple[np.ndarray, str, bool]] = [
            (
                gen_img,
                "generated (physical)"
                if target_name is None
                else f"generated: {target_name} (physical)",
                True,
            ),
            (
                target_img,
                "target (physical)"
                if target_name is None
                else f"target: {target_name} (physical)",
                True,
            ),
        ]
        if cond_match_img is not None:
            row1_panels.append(
                (
                    cond_match_img,
                    "LR cond (physical, upsampled)"
                    if target_name is None
                    else f"LR cond: {target_name} (physical, upsampled)",
                    True,
                )
            )

        for col_idx, (field, title, add_boxplot) in enumerate(row1_panels):
            plot_field_with_colorbar_and_optional_boxplot(
                axes[row_offset, col_idx],
                field,
                title=title,
                cmap=get_variable_cmap(None if target_name is None else str(target_name)),
                vmin=shared_vmin,
                vmax=shared_vmax,
                lsm_overlay=lsm_overlay_case,
                add_boxplot=add_boxplot,
            )
        for col_idx in range(len(row1_panels), n_cols):
            axes[row_offset, col_idx].axis("off")

        # Row 2: remaining dynamic conditions
        if cond_dynamic is not None and len(dynamic_row_indices) > 0:
            for col_idx, dyn_idx in enumerate(dynamic_row_indices):
                name = get_channel_name(dynamic_names, dyn_idx, "cond_dynamic")
                img = np.asarray(cond_dynamic[case_idx, dyn_idx])
                plot_field_with_colorbar_and_optional_boxplot(
                    axes[row_offset + 1, col_idx],
                    img,
                    title=f"cond: {name} (physical, upsampled)",
                    cmap=get_variable_cmap(name),
                    lsm_overlay=lsm_overlay_case,
                    add_boxplot=True,
                )
            for col_idx in range(len(dynamic_row_indices), n_cols):
                axes[row_offset + 1, col_idx].axis("off")
        else:
            for col_idx in range(n_cols):
                axes[row_offset + 1, col_idx].axis("off")

        # Row 3: static fields without boxplots
        if cond_static is not None:
            for col_idx in range(cond_static.shape[1]):
                name = get_channel_name(static_names, col_idx, "cond_static")
                img = np.asarray(cond_static[case_idx, col_idx])
                plot_field_with_colorbar_and_optional_boxplot(
                    axes[row_offset + 2, col_idx],
                    img,
                    title=f"{name} (physical)",
                    cmap=get_variable_cmap(name),
                    lsm_overlay=None if str(name).lower() == "lsm" else lsm_overlay_case,
                    add_boxplot=False,
                )
            for col_idx in range(cond_static.shape[1], n_cols):
                axes[row_offset + 2, col_idx].axis("off")
        else:
            for col_idx in range(n_cols):
                axes[row_offset + 2, col_idx].axis("off")

    subtitle_parts = [f"Training preview – epoch {epoch + 1}"]
    if date is not None:
        subtitle_parts.append(f"date={date}")
    if domain_tag is not None:
        subtitle_parts.append(f"domain={domain_tag}")
    fig.suptitle(" | ".join(subtitle_parts))

    save_path = output_dir / f"training_preview_epoch_{epoch + 1:04d}.png"
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return save_path
