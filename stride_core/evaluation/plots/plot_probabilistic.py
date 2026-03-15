"""
Probabilistic plotting utilities for STRIDE evaluation.

Current scope
-------------
This module provides lightweight family-level plotting for probabilistic
verification outputs already computed by ``Evaluator`` and stored inside the
serialized ``metrics`` dictionary.

Implemented figures
-------------------
- PIT histogram
- rank histogram
- reliability diagrams
- spread-skill scatter / curve

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
    probabilistic = metrics.get("probabilistic", {})
    if not isinstance(probabilistic, dict):
        return {}
    case_level = probabilistic.get("case_level", {})
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
    return payloads


# -----------------------------------------------------------------------------
# PIT histogram
# -----------------------------------------------------------------------------


def _extract_pit_histogram_counts(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    counts = None
    bin_edges = None

    for key in ("normalized_counts", "counts", "histogram"):
        counts = _safe_array(payload.get(key))
        if counts is not None and counts.size > 0:
            break

    for key in ("bin_edges", "bins"):
        bin_edges = _safe_array(payload.get(key))
        if bin_edges is not None and bin_edges.size > 0:
            break

    if counts is None:
        return None

    counts = np.ravel(counts).astype(float)
    counts = counts[np.isfinite(counts)]
    if counts.size == 0:
        return None

    if bin_edges is None:
        bin_edges = np.linspace(0.0, 1.0, counts.size + 1)
    else:
        bin_edges = np.ravel(bin_edges).astype(float)
        if bin_edges.size == counts.size:
            step = float(bin_edges[1] - bin_edges[0]) if bin_edges.size > 1 else 1.0 / counts.size
            bin_edges = np.concatenate([bin_edges, [bin_edges[-1] + step]])
        elif bin_edges.size != counts.size + 1:
            return None

    return counts, bin_edges



def _plot_pit_histogram(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "pit_histogram")
    count_arrays: list[np.ndarray] = []
    edge_arrays: list[np.ndarray] = []

    for payload in payloads:
        extracted = _extract_pit_histogram_counts(payload)
        if extracted is not None:
            counts, bin_edges = extracted
            count_arrays.append(counts)
            edge_arrays.append(bin_edges)

    if len(count_arrays) == 0:
        return None

    min_bins = min(arr.size for arr in count_arrays)
    counts_stack = np.stack([arr[:min_bins] for arr in count_arrays], axis=0)
    mean_counts = np.nanmean(counts_stack, axis=0)

    edge_stack = np.stack([arr[: min_bins + 1] for arr in edge_arrays], axis=0)
    mean_edges = np.nanmean(edge_stack, axis=0)
    if mean_edges.size != min_bins + 1:
        mean_edges = np.linspace(0.0, 1.0, min_bins + 1)

    widths = np.diff(mean_edges)
    centers = 0.5 * (mean_edges[:-1] + mean_edges[1:])

    total = float(np.sum(mean_counts))
    if total > 0.0:
        expected = total / mean_counts.size
    else:
        expected = 0.0

    figure_path = output_dir / "pit_histogram.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(centers, mean_counts, width=widths, align="center")
    ax.axhline(expected, linestyle="--")
    ax.set_xlim(float(mean_edges[0]), float(mean_edges[-1]))
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Mean count")
    ax.set_title("PIT histogram")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Rank histogram
# -----------------------------------------------------------------------------


def _extract_rank_counts(payload: dict[str, Any]) -> np.ndarray | None:
    for key in ("rank_counts", "counts", "histogram"):
        arr = _safe_array(payload.get(key))
        if arr is not None and arr.size > 0:
            return arr.ravel()
    return None



def _plot_rank_histogram(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "rank_histogram")
    counts_list: list[np.ndarray] = []

    for payload in payloads:
        counts = _extract_rank_counts(payload)
        if counts is not None and counts.size > 0:
            counts_list.append(counts)

    if len(counts_list) == 0:
        return None

    min_len = min(arr.size for arr in counts_list)
    stacked = np.stack([arr[:min_len] for arr in counts_list], axis=0)
    mean_counts = np.nanmean(stacked, axis=0)

    figure_path = output_dir / "rank_histogram.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    x = np.arange(min_len)
    ax.bar(x, mean_counts)
    ax.set_xlabel("Rank bin")
    ax.set_ylabel("Mean count")
    ax.set_title("Rank histogram")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Reliability diagram
# -----------------------------------------------------------------------------


def _extract_reliability_curves(payload: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    curves: list[tuple[np.ndarray, np.ndarray]] = []

    direct_x = None
    direct_y = None
    for key in ("forecast_probabilities", "forecast_probability", "probabilities", "x", "bin_centers"):
        direct_x = _safe_array(payload.get(key))
        if direct_x is not None and direct_x.size > 0:
            break
    for key in ("observed_frequencies", "observed_frequency", "frequencies", "event_frequencies", "y"):
        direct_y = _safe_array(payload.get(key))
        if direct_y is not None and direct_y.size > 0:
            break
    if direct_x is not None and direct_y is not None:
        x = np.ravel(direct_x)
        y = np.ravel(direct_y)
        n = min(x.size, y.size)
        if n > 0:
            mask = np.isfinite(x[:n]) & np.isfinite(y[:n])
            if np.any(mask):
                curves.append((x[:n][mask], y[:n][mask]))

    nested_curves = payload.get("curves")
    if isinstance(nested_curves, dict):
        for curve_payload in nested_curves.values():
            if not isinstance(curve_payload, dict):
                continue
            x_arr = None
            y_arr = None
            for key in ("forecast_probabilities", "bin_centers", "probabilities", "x"):
                x_arr = _safe_array(curve_payload.get(key))
                if x_arr is not None and x_arr.size > 0:
                    break
            for key in ("observed_frequencies", "event_frequencies", "frequencies", "y"):
                y_arr = _safe_array(curve_payload.get(key))
                if y_arr is not None and y_arr.size > 0:
                    break
            if x_arr is None or y_arr is None:
                continue
            x = np.ravel(x_arr)
            y = np.ravel(y_arr)
            n = min(x.size, y.size)
            if n == 0:
                continue
            mask = np.isfinite(x[:n]) & np.isfinite(y[:n])
            if np.any(mask):
                curves.append((x[:n][mask], y[:n][mask]))

    return curves



def _plot_reliability_diagram(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "reliability_diagram")
    curves: list[tuple[np.ndarray, np.ndarray]] = []

    for payload in payloads:
        curves.extend(_extract_reliability_curves(payload))

    if len(curves) == 0:
        return None

    n = min(len(x) for x, y in curves)
    x_stack = np.stack([x[:n] for x, y in curves], axis=0)
    y_stack = np.stack([y[:n] for x, y in curves], axis=0)
    x_mean = np.nanmean(x_stack, axis=0)
    y_mean = np.nanmean(y_stack, axis=0)

    figure_path = output_dir / "reliability_diagram.png"

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(x_mean, y_mean, marker="o")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Spread-skill
# -----------------------------------------------------------------------------


def _extract_spread_skill_xy(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    spread_keys = (
        "spread_values",
        "spread",
        "bin_spread",
        "forecast_spread",
        "x",
    )
    skill_keys = (
        "skill_values",
        "skill",
        "rmse",
        "bin_skill",
        "y",
    )

    spread = None
    skill = None
    for key in spread_keys:
        spread = _safe_array(payload.get(key))
        if spread is not None and spread.size > 0:
            break
    for key in skill_keys:
        skill = _safe_array(payload.get(key))
        if skill is not None and skill.size > 0:
            break

    if spread is None or skill is None:
        return None

    spread = spread.ravel()
    skill = skill.ravel()
    n = min(spread.size, skill.size)
    if n == 0:
        return None

    mask = np.isfinite(spread[:n]) & np.isfinite(skill[:n])
    if not np.any(mask):
        return None
    return spread[:n][mask], skill[:n][mask]



def _plot_spread_skill(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "spread_skill")
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    scalar_points_x: list[float] = []
    scalar_points_y: list[float] = []

    for payload in payloads:
        curve = _extract_spread_skill_xy(payload)
        if curve is not None:
            curves.append(curve)
            continue

        spread_candidates = (
            payload.get("mean_spread"),
            payload.get("spread_mean"),
            payload.get("forecast_spread_mean"),
        )
        skill_candidates = (
            payload.get("mean_skill"),
            payload.get("skill_mean"),
            payload.get("rmse_mean"),
        )

        spread_val = next((_safe_scalar(v) for v in spread_candidates if _safe_scalar(v) is not None), None)
        skill_val = next((_safe_scalar(v) for v in skill_candidates if _safe_scalar(v) is not None), None)
        if spread_val is not None and skill_val is not None:
            scalar_points_x.append(spread_val)
            scalar_points_y.append(skill_val)

    if len(curves) == 0 and len(scalar_points_x) == 0:
        return None

    figure_path = output_dir / "spread_skill.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)

    if len(curves) > 0:
        n = min(len(x) for x, y in curves)
        x_stack = np.stack([x[:n] for x, y in curves], axis=0)
        y_stack = np.stack([y[:n] for x, y in curves], axis=0)
        x_mean = np.nanmean(x_stack, axis=0)
        y_mean = np.nanmean(y_stack, axis=0)
        ax.plot(x_mean, y_mean, marker="o")
    else:
        ax.scatter(scalar_points_x, scalar_points_y)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ref_min = min(xlim[0], ylim[0])
    ref_max = max(xlim[1], ylim[1])
    ax.plot([ref_min, ref_max], [ref_min, ref_max], linestyle="--")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Skill")
    ax.set_title("Spread-skill")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Public family entrypoint
# -----------------------------------------------------------------------------


def plot_probabilistic(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Generate probabilistic evaluation plots.
    """
    out_dir = _ensure_output_dir(output_dir)
    saved: list[Path] = []

    for builder in (
        _plot_pit_histogram,
        _plot_rank_histogram,
        _plot_reliability_diagram,
        _plot_spread_skill,
    ):
        result = builder(metrics, out_dir)
        if result is not None:
            saved.append(result)

    return saved



def run_probabilistic_plots(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Backward-compatible alias for the evaluator plot-dispatch scaffold.
    """
    return plot_probabilistic(metrics=metrics, output_dir=output_dir, cfg=cfg)