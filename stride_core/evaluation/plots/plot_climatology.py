"""
Climatology plotting utilities for STRIDE evaluation.

Current scope
-------------
This module provides lightweight family-level plotting for climatological /
distributional verification outputs already computed by ``Evaluator`` and
stored inside the serialized ``metrics`` dictionary.

Implemented figures
-------------------
- histogram / pooled distribution comparison
- QQ plot
- extremes comparison
- annual precipitation sum summary

Design notes
------------
- plotting is a second pass over saved metric outputs
- this module does not recompute metrics
- plots are written defensively: missing or malformed metrics are skipped
- aggregated case-level plotting is preferred over one figure per case
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path



def _extract_case_entries(metrics: dict[str, Any]) -> dict[str, Any]:
    climatology = metrics.get("climatology", {})
    if not isinstance(climatology, dict):
        return {}
    case_level = climatology.get("case_level", {})
    if not isinstance(case_level, dict):
        return {}
    return case_level



def _safe_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return None
    return None



def _safe_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        out = float(value)
        if np.isnan(out):
            return None
        return out
    return None



def _collect_metric_payloads(metrics: dict[str, Any], metric_name: str) -> list[dict[str, Any]]:
    case_entries = _extract_case_entries(metrics)
    payloads: list[dict[str, Any]] = []
    for case_id, case_entry in case_entries.items():
        if not isinstance(case_entry, dict):
            continue
        payload = case_entry.get(metric_name)
        if isinstance(payload, dict):
            enriched = dict(payload)
            enriched.setdefault("case_id", case_id)
            payloads.append(enriched)

    if len(payloads) == 0:
        climatology = metrics.get("climatology", {})
        if isinstance(climatology, dict):
            aggregate = climatology.get("aggregate", {})
            if isinstance(aggregate, dict):
                payload = aggregate.get(metric_name)
                if isinstance(payload, dict):
                    enriched = dict(payload)
                    enriched.setdefault("case_id", "aggregate")
                    payloads.append(enriched)

    return payloads



def _flatten_numeric(obj: Any) -> list[float]:
    values: list[float] = []

    def _collect(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                _collect(v)
        elif isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x)
            if arr.dtype.kind in {"i", "u", "f"}:
                values.extend([float(v) for v in arr.ravel() if np.isfinite(float(v))])
            else:
                for v in x:
                    _collect(v)
        else:
            scalar = _safe_scalar(x)
            if scalar is not None:
                values.append(scalar)

    _collect(obj)
    return values


# -----------------------------------------------------------------------------
# Histogram / pooled distribution comparison
# -----------------------------------------------------------------------------


def _extract_histogram_pair(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    bin_edges = _safe_array(payload.get("bin_edges"))
    forecast_hist = _safe_array(payload.get("forecast_hist"))
    target_hist = _safe_array(payload.get("target_hist"))

    if bin_edges is not None and forecast_hist is not None and target_hist is not None:
        bin_edges = np.ravel(bin_edges).astype(float)
        forecast_hist = np.ravel(forecast_hist).astype(float)
        target_hist = np.ravel(target_hist).astype(float)
        n = min(forecast_hist.size, target_hist.size)
        if bin_edges.size == n:
            step = float(bin_edges[1] - bin_edges[0]) if bin_edges.size > 1 else 1.0
            bin_edges = np.concatenate([bin_edges, [bin_edges[-1] + step]])
        if bin_edges.size != n + 1 or n == 0:
            return None
        return bin_edges[: n + 1], forecast_hist[:n], target_hist[:n]

    bin_edges = _safe_array(payload.get("bin_edges"))
    forecast_density = _safe_array(payload.get("forecast_density"))
    target_density = _safe_array(payload.get("target_density"))
    if bin_edges is not None and forecast_density is not None and target_density is not None:
        bin_edges = np.ravel(bin_edges).astype(float)
        forecast_density = np.ravel(forecast_density).astype(float)
        target_density = np.ravel(target_density).astype(float)
        n = min(forecast_density.size, target_density.size)
        if bin_edges.size == n:
            step = float(bin_edges[1] - bin_edges[0]) if bin_edges.size > 1 else 1.0
            bin_edges = np.concatenate([bin_edges, [bin_edges[-1] + step]])
        if bin_edges.size != n + 1 or n == 0:
            return None
        return bin_edges[: n + 1], forecast_density[:n], target_density[:n]

    forecast = None
    target = None

    for key in (
        "forecast_values",
        "forecast_distribution",
        "forecast_samples",
        "generated_values",
        "generated_distribution",
    ):
        forecast = _safe_array(payload.get(key))
        if forecast is not None and forecast.size > 0:
            break

    for key in (
        "target_values",
        "target_distribution",
        "target_samples",
        "reference_values",
        "reference_distribution",
    ):
        target = _safe_array(payload.get(key))
        if target is not None and target.size > 0:
            break

    if forecast is None:
        forecast_vals = _flatten_numeric(payload.get("forecast", []))
        if len(forecast_vals) > 0:
            forecast = np.asarray(forecast_vals, dtype=float)
    if target is None:
        target_vals = _flatten_numeric(payload.get("target", []))
        if len(target_vals) > 0:
            target = np.asarray(target_vals, dtype=float)

    if forecast is None or target is None:
        return None

    forecast = np.ravel(forecast)
    target = np.ravel(target)
    forecast = forecast[np.isfinite(forecast)]
    target = target[np.isfinite(target)]

    if forecast.size == 0 or target.size == 0:
        return None

    all_values = np.concatenate([forecast, target])
    bin_edges = np.histogram_bin_edges(all_values, bins=40)
    forecast_hist, _ = np.histogram(forecast, bins=bin_edges, density=True)
    target_hist, _ = np.histogram(target, bins=bin_edges, density=True)
    return bin_edges, forecast_hist, target_hist



def _plot_histogram_comparison(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "histogram_comparison")

    edge_arrays: list[np.ndarray] = []
    forecast_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []
    distance_values: list[float] = []

    for payload in payloads:
        extracted = _extract_histogram_pair(payload)
        if extracted is not None:
            bin_edges, forecast_hist, target_hist = extracted
            edge_arrays.append(bin_edges)
            forecast_arrays.append(forecast_hist)
            target_arrays.append(target_hist)

        for key in ("wasserstein", "distance", "score", "value", "l1_distance", "l2_distance"):
            scalar = _safe_scalar(payload.get(key))
            if scalar is not None:
                distance_values.append(scalar)
                break

    if len(forecast_arrays) == 0 or len(target_arrays) == 0:
        return None

    min_bins = min(arr.size for arr in forecast_arrays + target_arrays)
    forecast_stack = np.stack([arr[:min_bins] for arr in forecast_arrays], axis=0)
    target_stack = np.stack([arr[:min_bins] for arr in target_arrays], axis=0)
    edge_stack = np.stack([arr[: min_bins + 1] for arr in edge_arrays], axis=0)

    forecast_mean = np.nanmean(forecast_stack, axis=0)
    target_mean = np.nanmean(target_stack, axis=0)
    mean_edges = np.nanmean(edge_stack, axis=0)
    centers = 0.5 * (mean_edges[:-1] + mean_edges[1:])

    figure_path = output_dir / "histogram_comparison.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.step(centers, target_mean, where="mid", label="Target")
    ax.step(centers, forecast_mean, where="mid", label="Forecast")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density / frequency")
    title = "Distribution comparison"
    if len(distance_values) > 0:
        title += f"\nDistance ≈ {float(np.mean(distance_values)):.3g}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# QQ plot
# -----------------------------------------------------------------------------


def _extract_qq_pair(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    forecast_q = None
    target_q = None

    for key in (
        "forecast_quantiles",
        "forecast_q",
        "generated_quantiles",
        "x",
    ):
        forecast_q = _safe_array(payload.get(key))
        if forecast_q is not None and forecast_q.size > 0:
            break

    for key in (
        "target_quantiles",
        "target_q",
        "reference_quantiles",
        "y",
    ):
        target_q = _safe_array(payload.get(key))
        if target_q is not None and target_q.size > 0:
            break

    if forecast_q is None or target_q is None:
        return None

    forecast_q = np.ravel(forecast_q)
    target_q = np.ravel(target_q)
    n = min(forecast_q.size, target_q.size)
    if n == 0:
        return None

    forecast_q = forecast_q[:n]
    target_q = target_q[:n]
    mask = np.isfinite(forecast_q) & np.isfinite(target_q)
    if not np.any(mask):
        return None

    return forecast_q[mask], target_q[mask]



def _plot_qq(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "qq_plot")
    curves: list[tuple[np.ndarray, np.ndarray]] = []

    for payload in payloads:
        pair = _extract_qq_pair(payload)
        if pair is not None:
            curves.append(pair)

    if len(curves) == 0:
        return None

    n = min(len(fq) for fq, tq in curves)
    forecast_stack = np.stack([fq[:n] for fq, tq in curves], axis=0)
    target_stack = np.stack([tq[:n] for fq, tq in curves], axis=0)

    forecast_mean = np.nanmean(forecast_stack, axis=0)
    target_mean = np.nanmean(target_stack, axis=0)

    figure_path = output_dir / "qq_plot.png"

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(target_mean, forecast_mean, s=18)

    lower = min(float(np.nanmin(target_mean)), float(np.nanmin(forecast_mean)))
    upper = max(float(np.nanmax(target_mean)), float(np.nanmax(forecast_mean)))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Target quantiles")
    ax.set_ylabel("Forecast quantiles")
    ax.set_title("QQ plot")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Extremes
# -----------------------------------------------------------------------------


def _extract_extremes(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    quantile_levels = _safe_array(payload.get("quantile_levels"))
    forecast_quantiles = _safe_array(payload.get("forecast_quantiles"))
    target_quantiles = _safe_array(payload.get("target_quantiles"))

    if (
        quantile_levels is not None
        and forecast_quantiles is not None
        and target_quantiles is not None
    ):
        q_arr = np.ravel(quantile_levels).astype(float)
        f_arr = np.ravel(forecast_quantiles).astype(float)
        t_arr = np.ravel(target_quantiles).astype(float)
        n = min(q_arr.size, f_arr.size, t_arr.size)
        if n == 0:
            return None
        mask = np.isfinite(q_arr[:n]) & np.isfinite(f_arr[:n]) & np.isfinite(t_arr[:n])
        if not np.any(mask):
            return None
        return q_arr[:n][mask], f_arr[:n][mask], t_arr[:n][mask]

    quantiles: list[float] = []
    forecast_vals: list[float] = []
    target_vals: list[float] = []

    for key, value in payload.items():
        try:
            q = float(key)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue

        forecast = _safe_scalar(value.get("forecast"))
        target = _safe_scalar(value.get("target"))
        if forecast is None or target is None:
            continue

        quantiles.append(q)
        forecast_vals.append(forecast)
        target_vals.append(target)

    if len(quantiles) == 0:
        return None

    order = np.argsort(np.asarray(quantiles, dtype=float))
    q_arr = np.asarray(quantiles, dtype=float)[order]
    f_arr = np.asarray(forecast_vals, dtype=float)[order]
    t_arr = np.asarray(target_vals, dtype=float)[order]
    return q_arr, f_arr, t_arr



def _plot_extremes(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "extremes")
    curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for payload in payloads:
        result = _extract_extremes(payload)
        if result is not None:
            curves.append(result)

    if len(curves) == 0:
        return None

    n = min(len(q) for q, f, t in curves)
    q_stack = np.stack([q[:n] for q, f, t in curves], axis=0)
    f_stack = np.stack([f[:n] for q, f, t in curves], axis=0)
    t_stack = np.stack([t[:n] for q, f, t in curves], axis=0)

    quantiles = np.nanmean(q_stack, axis=0)
    forecast_mean = np.nanmean(f_stack, axis=0)
    target_mean = np.nanmean(t_stack, axis=0)

    x = np.arange(n)
    width = 0.36
    figure_path = output_dir / "extremes.png"

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, target_mean, width=width, label="Target")
    ax.bar(x + width / 2, forecast_mean, width=width, label="Forecast")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q:.4g}" for q in quantiles], rotation=45, ha="right")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Value")
    ax.set_title("Extreme quantile comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Annual precipitation sum
# -----------------------------------------------------------------------------


def _plot_annual_precipitation_sum(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "annual_precipitation_sum")

    forecast_means: list[float] = []
    target_means: list[float] = []
    forecast_stds: list[float] = []
    target_stds: list[float] = []
    case_ids: list[str] = []

    for payload in payloads:
        forecast_mean = _safe_scalar(payload.get("forecast_mean"))
        if forecast_mean is None:
            forecast_mean = _safe_scalar(payload.get("mean_forecast_sum"))
        target_mean = _safe_scalar(payload.get("target_mean"))
        if target_mean is None:
            target_mean = _safe_scalar(payload.get("mean_target_sum"))

        forecast_std = _safe_scalar(payload.get("forecast_std"))
        if forecast_std is None:
            forecast_std = _safe_scalar(payload.get("forecast_std_sum"))
        target_std = _safe_scalar(payload.get("target_std"))
        if target_std is None:
            target_std = _safe_scalar(payload.get("target_std_sum"))

        if (forecast_std is None or target_std is None) and isinstance(payload.get("per_case"), list):
            per_case = payload.get("per_case", [])
            if len(per_case) > 0 and all(isinstance(item, dict) for item in per_case):
                fvals = []
                tvals = []
                for item in per_case:
                    f_item = _safe_scalar(item.get("forecast_mean_sum"))
                    if f_item is None:
                        f_item = _safe_scalar(item.get("forecast_sum"))
                    t_item = _safe_scalar(item.get("target_sum"))
                    if f_item is not None:
                        fvals.append(f_item)
                    if t_item is not None:
                        tvals.append(t_item)
                if forecast_std is None and len(fvals) > 1:
                    forecast_std = float(np.std(fvals, ddof=0))
                if target_std is None and len(tvals) > 1:
                    target_std = float(np.std(tvals, ddof=0))

        if forecast_mean is None or target_mean is None:
            continue

        forecast_means.append(forecast_mean)
        target_means.append(target_mean)
        forecast_stds.append(np.nan if forecast_std is None else forecast_std)
        target_stds.append(np.nan if target_std is None else target_std)
        case_ids.append(str(payload.get("case_id", f"case_{len(case_ids):03d}")))

    if len(case_ids) == 0:
        return None

    x = np.arange(len(case_ids), dtype=float)
    width = 0.36
    figure_path = output_dir / "annual_precipitation_sum.png"

    fig = plt.figure(figsize=(7, 4.8))
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, target_means, width=width, label="Target mean")
    ax.bar(x + width / 2, forecast_means, width=width, label="Forecast mean")

    if np.any(np.isfinite(target_stds)):
        ax.errorbar(
            x - width / 2,
            target_means,
            yerr=np.asarray(target_stds, dtype=float),
            fmt="none",
            capsize=3,
        )
    if np.any(np.isfinite(forecast_stds)):
        ax.errorbar(
            x + width / 2,
            forecast_means,
            yerr=np.asarray(forecast_stds, dtype=float),
            fmt="none",
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha="right")
    ax.set_ylabel("Annual precipitation sum")
    ax.set_title("Annual precipitation sum mean ± std")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Public family entrypoint
# -----------------------------------------------------------------------------


def plot_climatology(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Generate climatology evaluation plots.
    """
    out_dir = _ensure_output_dir(output_dir)
    saved: list[Path] = []

    for builder in (
        _plot_histogram_comparison,
        _plot_qq,
        _plot_extremes,
        _plot_annual_precipitation_sum,
    ):
        result = builder(metrics, out_dir)
        if result is not None:
            saved.append(result)

    return saved



def run_climatology_plots(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Backward-compatible alias for the evaluator plot-dispatch scaffold.
    """
    return plot_climatology(metrics=metrics, output_dir=output_dir, cfg=cfg)