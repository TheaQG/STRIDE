"""
Temporal plotting utilities for STRIDE evaluation.

Current scope
-------------
This module provides lightweight family-level plotting for temporal verification
outputs already computed by ``Evaluator`` and stored inside the serialized
``metrics`` dictionary.

Implemented figures
-------------------
- lag autocorrelation curves
- wet spell length distributions
- dry spell length distributions

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
    temporal = metrics.get("temporal", {})
    if not isinstance(temporal, dict):
        return {}
    case_level = temporal.get("case_level", {})
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
        temporal = metrics.get("temporal", {})
        if isinstance(temporal, dict):
            aggregate = temporal.get("aggregate", {})
            if isinstance(aggregate, dict):
                payload = aggregate.get(metric_name)
                if isinstance(payload, dict):
                    enriched = dict(payload)
                    enriched.setdefault("case_id", "aggregate")
                    payloads.append(enriched)

    return payloads


# -----------------------------------------------------------------------------
# Lag autocorrelation
# -----------------------------------------------------------------------------


def _extract_lag_autocorrelation(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    def _from_payload_dict(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        lags = None
        autocorr = None

        for key in ("lags", "lags_days", "autocorrelation_lags", "x"):
            lags = _safe_array(source.get(key))
            if lags is not None and lags.size > 0:
                break

        for key in (
            "autocorrelation",
            "lag_autocorrelation",
            "autocorr",
            "values",
            "y",
        ):
            autocorr = _safe_array(source.get(key))
            if autocorr is not None and autocorr.size > 0:
                break

        if lags is None or autocorr is None:
            scalar = _safe_scalar(source.get("value"))
            if scalar is not None:
                return np.asarray([1.0], dtype=float), np.asarray([scalar], dtype=float)
            return None

        lags = np.ravel(lags)
        autocorr = np.ravel(autocorr)
        n = min(lags.size, autocorr.size)
        if n == 0:
            return None

        lags = lags[:n]
        autocorr = autocorr[:n]
        mask = np.isfinite(lags) & np.isfinite(autocorr)
        if not np.any(mask):
            return None

        return lags[mask], autocorr[mask]

    direct = _from_payload_dict(payload)
    if direct is not None:
        return direct

    per_case = payload.get("per_case")
    if isinstance(per_case, list):
        lag_values: list[np.ndarray] = []
        autocorr_values: list[np.ndarray] = []
        for item in per_case:
            if not isinstance(item, dict):
                continue
            extracted = _from_payload_dict(item)
            if extracted is None:
                continue
            lag_values.append(extracted[0])
            autocorr_values.append(extracted[1])
        if len(lag_values) > 0:
            n = min(len(arr) for arr in lag_values + autocorr_values)
            lag_stack = np.stack([arr[:n] for arr in lag_values], axis=0)
            autocorr_stack = np.stack([arr[:n] for arr in autocorr_values], axis=0)
            return np.nanmean(lag_stack, axis=0), np.nanmean(autocorr_stack, axis=0)

    return None



def _plot_lag_autocorrelation(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "lag_autocorrelation")
    curves: list[tuple[np.ndarray, np.ndarray]] = []

    for payload in payloads:
        result = _extract_lag_autocorrelation(payload)
        if result is not None:
            curves.append(result)

    if len(curves) == 0:
        return None

    n = min(len(lags) for lags, values in curves)
    lag_stack = np.stack([lags[:n] for lags, values in curves], axis=0)
    value_stack = np.stack([values[:n] for lags, values in curves], axis=0)

    lag_mean = np.nanmean(lag_stack, axis=0)
    value_mean = np.nanmean(value_stack, axis=0)

    figure_path = output_dir / "lag_autocorrelation.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(lag_mean, value_mean, marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Lag autocorrelation")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Spell-length distributions
# -----------------------------------------------------------------------------


def _extract_spell_lengths(payload: dict[str, Any]) -> np.ndarray | None:
    def _from_payload_dict(source: dict[str, Any]) -> np.ndarray | None:
        for key in (
            "spell_lengths",
            "lengths",
            "durations",
            "values",
            "samples",
        ):
            arr = _safe_array(source.get(key))
            if arr is not None and arr.size > 0:
                arr = np.ravel(arr)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    return arr

        counts = None
        bins = None
        for key in ("counts", "histogram", "spell_counts"):
            counts = _safe_array(source.get(key))
            if counts is not None and counts.size > 0:
                break
        for key in ("bins", "length_bins", "spell_bins"):
            bins = _safe_array(source.get(key))
            if bins is not None and bins.size > 0:
                break

        if counts is None or bins is None:
            return None

        counts = np.ravel(counts)
        bins = np.ravel(bins)
        n = min(counts.size, bins.size)
        if n == 0:
            return None

        counts = counts[:n]
        bins = bins[:n]
        values: list[float] = []
        for b, c in zip(bins, counts):
            c_int = int(round(float(c)))
            if c_int > 0 and np.isfinite(b):
                values.extend([float(b)] * c_int)

        if len(values) == 0:
            return None
        return np.asarray(values, dtype=float)

    direct = _from_payload_dict(payload)
    if direct is not None:
        return direct

    per_case = payload.get("per_case")
    if isinstance(per_case, list):
        all_values: list[np.ndarray] = []
        for item in per_case:
            if not isinstance(item, dict):
                continue
            extracted = _from_payload_dict(item)
            if extracted is not None and extracted.size > 0:
                all_values.append(extracted)
        if len(all_values) > 0:
            return np.concatenate(all_values)

    return None



def _plot_spell_distribution(
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    metric_name: str,
    filename: str,
    title: str,
) -> Path | None:
    payloads = _collect_metric_payloads(metrics, metric_name)
    all_lengths: list[np.ndarray] = []

    for payload in payloads:
        values = _extract_spell_lengths(payload)
        if values is not None and values.size > 0:
            all_lengths.append(values)

    if len(all_lengths) == 0:
        return None

    lengths = np.concatenate(all_lengths)
    lengths = lengths[np.isfinite(lengths)]
    if lengths.size == 0:
        return None

    max_length = int(max(1, np.nanmax(lengths)))
    bins = np.arange(0.5, max_length + 1.5, 1.0)
    figure_path = output_dir / filename

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.hist(lengths, bins=bins)
    ax.set_xlabel("Spell length")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Public family entrypoint
# -----------------------------------------------------------------------------


def plot_temporal(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Generate temporal evaluation plots.
    """
    out_dir = _ensure_output_dir(output_dir)
    saved: list[Path] = []

    builders = (
        lambda m, o: _plot_lag_autocorrelation(m, o),
        lambda m, o: _plot_spell_distribution(
            m,
            o,
            metric_name="wet_spell_length",
            filename="wet_spell_length.png",
            title="Wet spell length distribution",
        ),
        lambda m, o: _plot_spell_distribution(
            m,
            o,
            metric_name="dry_spell_length",
            filename="dry_spell_length.png",
            title="Dry spell length distribution",
        ),
    )

    for builder in builders:
        result = builder(metrics, out_dir)
        if result is not None:
            saved.append(result)

    return saved



def run_temporal_plots(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Backward-compatible alias for the evaluator plot-dispatch scaffold.
    """
    return plot_temporal(metrics=metrics, output_dir=output_dir, cfg=cfg)