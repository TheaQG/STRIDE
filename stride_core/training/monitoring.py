"""
Monitoring utilities for STRIDE training previews.

This module is intentionally lightweight. It is not the full evaluation suite;
its job is to compute a small set of informative preview-time metrics on fixed
validation cases during training, so model progress can be monitored alongside
loss.

Current scope
-------------
- simple scalar metrics on generated vs target preview samples
- precipitation-focused monitoring helpers
- optional basic history smoothing
- optional plotting of metric history from `history.json`

Non-goals for now
-----------------
- full benchmark evaluation
- large-batch test-set statistics
- spectral diagnostics with domain-specific fitting choices
- external logging backends

The functions here are designed to return plain Python dictionaries so they can
be saved directly into the trainer history.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringConfig:
    """
    Minimal training-monitoring configuration.

    This is kept separate from the trainer so the trainer only needs to ask
    whether monitoring should run and where plots should be saved.
    """

    enabled: bool = False
    compute_every_n_epochs: int = 1
    save_history_plots: bool = True
    output_subdir: str = "monitoring"
    wet_threshold: float = 0.1
    high_quantiles: tuple[float, ...] = (0.99, 0.999)
    smoothing_alpha: float | None = None
    preview_metric_keys: tuple[str, str, str] = ("rmse", "corr", "q99_generated")
    preview_metrics_layout: str = "column"

    @classmethod
    def from_training_dict(cls, cfg: dict[str, Any]) -> "MonitoringConfig":
        training_cfg = cfg.get("training", cfg)
        if not isinstance(training_cfg, dict):
            raise ValueError("Expected training config to be a dict")

        mon_cfg = training_cfg.get("monitoring", {})
        if not isinstance(mon_cfg, dict):
            raise ValueError("Expected 'training.monitoring' to be a dict")

        raw_quantiles = mon_cfg.get("high_quantiles", [0.99, 0.999])
        if not isinstance(raw_quantiles, (list, tuple)):
            raise ValueError("monitoring.high_quantiles must be a list or tuple")

        quantiles = tuple(float(q) for q in raw_quantiles)
        for q in quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(
                    f"All monitoring.high_quantiles must lie in (0, 1), got {q}"
                )

        compute_every_n_epochs = int(mon_cfg.get("compute_every_n_epochs", 1))
        if compute_every_n_epochs <= 0:
            raise ValueError(
                "monitoring.compute_every_n_epochs must be > 0, got "
                f"{compute_every_n_epochs}"
            )

        smoothing_alpha_raw = mon_cfg.get("smoothing_alpha", None)

        preview_metric_keys_raw = mon_cfg.get(
            "preview_metric_keys",
            ["rmse", "corr", "q99_generated"],
        )
        if not isinstance(preview_metric_keys_raw, (list, tuple)):
            raise ValueError("monitoring.preview_metric_keys must be a list or tuple")
        if len(preview_metric_keys_raw) != 3:
            raise ValueError(
                "monitoring.preview_metric_keys must contain exactly 3 metric names"
            )
        preview_metric_keys_tuple = tuple(str(key) for key in preview_metric_keys_raw)
        preview_metric_keys = (preview_metric_keys_tuple[0], preview_metric_keys_tuple[1], preview_metric_keys_tuple[2])

        preview_metrics_layout = str(mon_cfg.get("preview_metrics_layout", "column"))
        if preview_metrics_layout not in {"row", "column"}:
            raise ValueError(
                "monitoring.preview_metrics_layout must be either 'row' or 'column'"
            )

        smoothing_alpha = None
        if smoothing_alpha_raw is not None:
            smoothing_alpha = float(smoothing_alpha_raw)
            if not 0.0 < smoothing_alpha <= 1.0:
                raise ValueError(
                    "monitoring.smoothing_alpha must lie in (0, 1], got "
                    f"{smoothing_alpha}"
                )

        return cls(
            enabled=bool(mon_cfg.get("enabled", False)),
            compute_every_n_epochs=compute_every_n_epochs,
            save_history_plots=bool(mon_cfg.get("save_history_plots", True)),
            output_subdir=str(mon_cfg.get("output_subdir", "monitoring")),
            wet_threshold=float(mon_cfg.get("wet_threshold", 0.1)),
            high_quantiles=quantiles,
            smoothing_alpha=smoothing_alpha,
            preview_metric_keys=preview_metric_keys,
            preview_metrics_layout=preview_metrics_layout,
        )


# -----------------------------------------------------------------------------
# Array helpers
# -----------------------------------------------------------------------------



def _to_numpy_2d_or_3d(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in {2, 3, 4}:
        raise ValueError(f"{name} must have ndim in {{2, 3, 4}}, got {array.ndim}")
    return array



def _collapse_to_spatial_samples(array: np.ndarray) -> np.ndarray:
    """
    Convert `[B, C, H, W]`, `[B, H, W]`, or `[H, W]` into a flat vector.
    """
    return np.asarray(array, dtype=np.float64).reshape(-1)



def _select_single_channel(array: np.ndarray) -> np.ndarray:
    """
    For preview monitoring we usually care about one generated target field.

    Accepted inputs:
    - [H, W]
    - [B, H, W]
    - [B, 1, H, W]
    - [1, H, W]
    """
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(
                "Expected single-channel target/generated arrays for preview monitoring, "
                f"got shape {array.shape}"
            )
        return array[:, 0, :, :]
    raise ValueError(f"Unsupported array shape {array.shape}")


# -----------------------------------------------------------------------------
# Core scalar metrics
# -----------------------------------------------------------------------------



def compute_basic_field_metrics(
    target: np.ndarray,
    generated: np.ndarray,
) -> dict[str, float]:
    """
    Compute general scalar comparison metrics.
    """
    target_flat = _collapse_to_spatial_samples(target)
    generated_flat = _collapse_to_spatial_samples(generated)

    if target_flat.shape != generated_flat.shape:
        raise ValueError(
            f"target and generated must have matching flattened shapes, got "
            f"{target_flat.shape} vs {generated_flat.shape}"
        )

    diff = generated_flat - target_flat
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    target_std = float(np.std(target_flat))
    generated_std = float(np.std(generated_flat))

    corr = np.nan
    if target_flat.size > 1 and target_std > 0.0 and generated_std > 0.0:
        corr = float(np.corrcoef(target_flat, generated_flat)[0, 1])

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": float(np.mean(diff)),
        "target_mean": float(np.mean(target_flat)),
        "generated_mean": float(np.mean(generated_flat)),
        "target_std": target_std,
        "generated_std": generated_std,
        "corr": corr,
    }



def compute_precipitation_metrics(
    target: np.ndarray,
    generated: np.ndarray,
    *,
    wet_threshold: float = 0.1,
    high_quantiles: tuple[float, ...] = (0.99, 0.999),
) -> dict[str, float]:
    """
    Compute a compact precipitation-oriented metric set.

    These are useful early training diagnostics for downscaling:
    - wet fraction
    - intensity means
    - extreme quantiles
    - max value
    """
    target_flat = _collapse_to_spatial_samples(target)
    generated_flat = _collapse_to_spatial_samples(generated)

    target_wet = target_flat > wet_threshold
    generated_wet = generated_flat > wet_threshold

    metrics: dict[str, float] = {
        "wet_fraction_target": float(np.mean(target_wet)),
        "wet_fraction_generated": float(np.mean(generated_wet)),
        "wet_fraction_bias": float(np.mean(generated_wet) - np.mean(target_wet)),
        "max_target": float(np.max(target_flat)),
        "max_generated": float(np.max(generated_flat)),
        "mean_target": float(np.mean(target_flat)),
        "mean_generated": float(np.mean(generated_flat)),
    }

    if np.any(target_wet):
        metrics["wet_mean_target"] = float(np.mean(target_flat[target_wet]))
    else:
        metrics["wet_mean_target"] = 0.0

    if np.any(generated_wet):
        metrics["wet_mean_generated"] = float(np.mean(generated_flat[generated_wet]))
    else:
        metrics["wet_mean_generated"] = 0.0

    for q in high_quantiles:
        label = str(q).replace("0.", "q")
        metrics[f"{label}_target"] = float(np.quantile(target_flat, q))
        metrics[f"{label}_generated"] = float(np.quantile(generated_flat, q))

    return metrics



def compute_preview_metrics(
    *,
    preview_payload: dict[str, Any],
    target_key: str = "target_physical",
    generated_key: str = "generated_physical",
    monitoring_config: MonitoringConfig | None = None,
) -> dict[str, Any]:
    """
    Compute monitoring metrics from a training preview payload.

    By default this uses physical-space tensors if available. If they are not,
    it falls back to raw target / generated arrays.
    """
    cfg = monitoring_config or MonitoringConfig(enabled=True)

    target_value = preview_payload.get(target_key)
    generated_value = preview_payload.get(generated_key)

    if target_value is None:
        target_value = preview_payload.get("target")
    if generated_value is None:
        generated_value = preview_payload.get("generated")

    if target_value is None or generated_value is None:
        raise ValueError("preview_payload must contain target and generated arrays")

    target_array = _select_single_channel(_to_numpy_2d_or_3d(target_value, "target"))
    generated_array = _select_single_channel(
        _to_numpy_2d_or_3d(generated_value, "generated")
    )

    metrics: dict[str, Any] = compute_basic_field_metrics(target_array, generated_array)

    variable_names = preview_payload.get("variable_names", {})
    target_name = None
    if isinstance(variable_names, dict):
        raw_target_name = variable_names.get("target")
        target_name = None if raw_target_name is None else str(raw_target_name).lower()

    if target_name in {"prcp", "precip", "precipitation", "tp"}:
        metrics.update(
            compute_precipitation_metrics(
                target_array,
                generated_array,
                wet_threshold=cfg.wet_threshold,
                high_quantiles=cfg.high_quantiles,
            )
        )

    metrics["n_values"] = int(target_array.size)
    metrics["target_variable"] = target_name
    return metrics


# -----------------------------------------------------------------------------
# Scheduling helpers
# -----------------------------------------------------------------------------



def should_run_monitoring(
    *,
    epoch: int,
    monitoring_config: MonitoringConfig,
) -> bool:
    if not monitoring_config.enabled:
        return False
    return (epoch + 1) % monitoring_config.compute_every_n_epochs == 0


# -----------------------------------------------------------------------------
# Smoothing / plotting helpers
# -----------------------------------------------------------------------------



def smooth_series(values: list[float], alpha: float) -> list[float]:
    """
    Exponential moving average smoothing.
    """
    if len(values) == 0:
        return []
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")

    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed



def load_history_json(history_path: str | Path) -> dict[str, Any]:
    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"History file does not exist: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, dict):
        raise ValueError(
            f"Expected history JSON to decode into a dict, got {type(history)}"
        )
    return history



def _extract_metric_series(
    epoch_records: list[dict[str, Any]],
    metric_key: str,
) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []

    for record in epoch_records:
        if not isinstance(record, dict):
            continue
        if metric_key not in record:
            continue

        value = record[metric_key]
        if value is None:
            continue
        if isinstance(value, str):
            continue

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue

        epochs.append(int(record.get("epoch", len(epochs) + 1)))
        values.append(value_float)

    return epochs, values



def plot_history_metric(
    history: dict[str, Any],
    *,
    metric_key: str,
    split: str,
    output_path: str | Path,
    smoothing_alpha: float | None = None,
) -> Path:
    """
    Plot one scalar metric from training history.
    """
    records = history.get(split, [])
    if not isinstance(records, list):
        raise ValueError(f"History split '{split}' must be a list")

    epochs, values = _extract_metric_series(records, metric_key)
    if len(values) == 0:
        raise ValueError(
            f"Could not find numeric metric '{metric_key}' in history split '{split}'"
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, values, label=metric_key)

    if smoothing_alpha is not None:
        smoothed = smooth_series(values, smoothing_alpha)
        ax.plot(epochs, smoothed, label=f"{metric_key} (EMA)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{split}: {metric_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_metric_on_axis(
    ax: plt.Axes,
    history: dict[str, Any],
    *,
    split: str,
    metric_key: str,
    smoothing_alpha: float | None,
) -> None:
    records = history.get(split, [])
    if not isinstance(records, list):
        raise ValueError(f"History split '{split}' must be a list")

    epochs, values = _extract_metric_series(records, metric_key)
    if len(values) == 0:
        raise ValueError(
            f"Could not find numeric metric '{metric_key}' in history split '{split}'"
        )

    ax.plot(epochs, values, label=metric_key)
    if smoothing_alpha is not None:
        smoothed = smooth_series(values, smoothing_alpha)
        ax.plot(epochs, smoothed, label=f"{metric_key} (EMA)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{split}: {metric_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)



def plot_loss_history_figure(
    history: dict[str, Any],
    *,
    output_path: str | Path,
    smoothing_alpha: float | None = None,
) -> Path:
    """
    Plot train and validation loss side by side with a shared y-range.
    """
    train_epochs, train_values = _extract_metric_series(history.get("train_epochs", []), "loss")
    valid_epochs, valid_values = _extract_metric_series(history.get("valid_epochs", []), "loss")

    if len(train_values) == 0 or len(valid_values) == 0:
        raise ValueError("Could not find train/valid loss in history")

    all_loss_values = list(train_values) + list(valid_values)
    y_min = float(min(all_loss_values))
    y_max = float(max(all_loss_values))
    if y_min == y_max:
        pad = 1.0
    else:
        pad = 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)

    axes[0].plot(train_epochs, train_values, label="loss")
    if smoothing_alpha is not None:
        axes[0].plot(
            train_epochs,
            smooth_series(train_values, smoothing_alpha),
            label="loss (EMA)",
        )
    axes[0].set_title("train_epochs: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(y_min - pad, y_max + pad)

    axes[1].plot(valid_epochs, valid_values, label="loss")
    if smoothing_alpha is not None:
        axes[1].plot(
            valid_epochs,
            smooth_series(valid_values, smoothing_alpha),
            label="loss (EMA)",
        )
    axes[1].set_title("valid_epochs: loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(y_min - pad, y_max + pad)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_preview_metrics_figure(
    history: dict[str, Any],
    *,
    metric_keys: tuple[str, str, str],
    output_path: str | Path,
    smoothing_alpha: float | None = None,
    layout: str = "column",
) -> Path:
    """
    Plot three preview-monitoring metrics in one combined figure.
    """
    if layout == "row":
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
    elif layout == "column":
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 12.5))
    else:
        raise ValueError("layout must be either 'row' or 'column'")

    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax, metric_key in zip(axes_array, metric_keys):
        _plot_metric_on_axis(
            ax,
            history,
            split="valid_epochs",
            metric_key=metric_key,
            smoothing_alpha=smoothing_alpha,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_default_history_metrics(
    history_path: str | Path,
    *,
    output_dir: str | Path,
    smoothing_alpha: float | None = None,
    preview_metric_keys: tuple[str, str, str] = ("rmse", "corr", "q99_generated"),
    preview_metrics_layout: str = "column",
) -> list[Path]:
    """
    Plot the default compact monitoring figure set:
    - one combined loss figure
    - one combined preview-metrics figure
    """
    history = load_history_json(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    try:
        saved_paths.append(
            plot_loss_history_figure(
                history,
                output_path=output_dir / "loss_history.png",
                smoothing_alpha=smoothing_alpha,
            )
        )
    except ValueError:
        pass

    try:
        saved_paths.append(
            plot_preview_metrics_figure(
                history,
                metric_keys=preview_metric_keys,
                output_path=output_dir / "preview_metrics.png",
                smoothing_alpha=smoothing_alpha,
                layout=preview_metrics_layout,
            )
        )
    except ValueError:
        pass

    return saved_paths