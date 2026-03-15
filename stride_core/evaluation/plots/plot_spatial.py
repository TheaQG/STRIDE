"""
Spatial plotting utilities for STRIDE evaluation.

Current scope
-------------
This module provides lightweight family-level plotting for spatial verification
outputs already computed by ``Evaluator`` and stored inside the serialized
``metrics`` dictionary.

Implemented figures
-------------------
- PSD curves
- ISS heatmap / matrix summary
- SAL component distributions

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
    spatial = metrics.get("spatial", {})
    if not isinstance(spatial, dict):
        return {}
    case_level = spatial.get("case_level", {})
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
        spatial = metrics.get("spatial", {})
        if isinstance(spatial, dict):
            aggregate = spatial.get("aggregate", {})
            if isinstance(aggregate, dict):
                payload = aggregate.get(metric_name)
                if isinstance(payload, dict):
                    enriched = dict(payload)
                    enriched.setdefault("case_id", "aggregate")
                    payloads.append(enriched)

    return payloads



def _find_first_array(payload: dict[str, Any], candidate_keys: tuple[str, ...]) -> np.ndarray | None:
    for key in candidate_keys:
        arr = _safe_array(payload.get(key))
        if arr is not None and arr.size > 0:
            return arr
    return None


# -----------------------------------------------------------------------------
# PSD
# -----------------------------------------------------------------------------


def _extract_psd_curve(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None:
    def _from_payload_dict(source: dict[str, Any]):
        scales = _find_first_array(
            source,
            (
                "scales_km",
                "wavelengths_km",
                "spatial_scales_km",
                "mean_wavelengths_km",
                "k_km",
                "k",
            ),
        )
        target = _find_first_array(
            source,
            (
                "target_psd",
                "psd_target",
                "target_spectrum",
                "reference_psd",
                "reference_spectrum",
                "mean_target_psd",
            ),
        )
        forecast = _find_first_array(
            source,
            (
                "forecast_psd",
                "psd_forecast",
                "forecast_spectrum",
                "generated_psd",
                "generated_spectrum",
                "mean_forecast_psd",
            ),
        )

        member_psd = None
        for key in (
            "forecast_member_psd",
            "forecast_psd_members",
            "psd_members",
            "member_psd",
            "ensemble_member_psd",
            "ensemble_psd_members",
            "generated_member_psd",
            "mean_forecast_member_psd",
        ):
            arr = source.get(key)
            if arr is None:
                continue
            try:
                member_psd = np.asarray(arr, dtype=float)
            except Exception:
                member_psd = None
            if member_psd is not None and member_psd.size > 0:
                break

        if scales is None or target is None:
            return None

        scales = np.ravel(scales).astype(float)
        target = np.ravel(target).astype(float)

        if member_psd is not None:
            if member_psd.ndim == 1:
                member_psd = member_psd[None, :]
            if member_psd.ndim >= 2:
                member_psd = np.reshape(member_psd, (member_psd.shape[0], -1))
                n = min(scales.size, target.size, member_psd.shape[1])
                if n == 0:
                    return None

                scales_n = scales[:n]
                target_n = target[:n]
                member_psd_n = member_psd[:, :n]

                valid_cols = (
                    np.isfinite(scales_n)
                    & np.isfinite(target_n)
                    & (scales_n > 0)
                    & (target_n > 0)
                )
                valid_cols &= np.all(np.isfinite(member_psd_n), axis=0)
                valid_cols &= np.all(member_psd_n > 0, axis=0)
                if not np.any(valid_cols):
                    return None

                scales_n = scales_n[valid_cols]
                target_n = target_n[valid_cols]
                member_psd_n = member_psd_n[:, valid_cols]
                forecast_mean = np.nanmean(member_psd_n, axis=0)
                forecast_lower = np.nanpercentile(member_psd_n, 10.0, axis=0)
                forecast_upper = np.nanpercentile(member_psd_n, 90.0, axis=0)
                return scales_n, forecast_mean, target_n, forecast_lower, forecast_upper

        if forecast is None:
            return None

        forecast = np.ravel(forecast).astype(float)
        n = min(scales.size, forecast.size, target.size)
        if n == 0:
            return None

        scales_n = scales[:n]
        forecast_n = forecast[:n]
        target_n = target[:n]
        mask = np.isfinite(scales_n) & np.isfinite(forecast_n) & np.isfinite(target_n)
        mask &= scales_n > 0
        mask &= forecast_n > 0
        mask &= target_n > 0
        if not np.any(mask):
            return None

        return scales_n[mask], forecast_n[mask], target_n[mask], None, None

    direct = _from_payload_dict(payload)
    if direct is not None:
        return direct

    per_case = payload.get("per_case")
    if isinstance(per_case, list):
        for item in per_case:
            if isinstance(item, dict):
                extracted = _from_payload_dict(item)
                if extracted is not None:
                    return extracted

    return None



def _plot_psd(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "psd")
    curves: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ] = []

    for payload in payloads:
        curve = _extract_psd_curve(payload)
        if curve is not None:
            curves.append(curve)

    if len(curves) == 0:
        return None

    n = min(len(scales) for scales, forecast, target, lower, upper in curves)
    scale_stack = np.stack(
        [scales[:n] for scales, forecast, target, lower, upper in curves],
        axis=0,
    )
    forecast_stack = np.stack(
        [forecast[:n] for scales, forecast, target, lower, upper in curves],
        axis=0,
    )
    target_stack = np.stack(
        [target[:n] for scales, forecast, target, lower, upper in curves],
        axis=0,
    )

    lower_curves = [lower[:n] for scales, forecast, target, lower, upper in curves if lower is not None]
    upper_curves = [upper[:n] for scales, forecast, target, lower, upper in curves if upper is not None]

    scales = np.nanmean(scale_stack, axis=0)
    forecast_mean = np.nanmean(forecast_stack, axis=0)
    target_mean = np.nanmean(target_stack, axis=0)

    forecast_lower = None
    forecast_upper = None
    if len(lower_curves) == len(curves) and len(upper_curves) == len(curves):
        forecast_lower = np.nanmean(np.stack(lower_curves, axis=0), axis=0)
        forecast_upper = np.nanmean(np.stack(upper_curves, axis=0), axis=0)

    figure_path = output_dir / "psd.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.loglog(scales, target_mean, label="Target")
    ax.loglog(scales, forecast_mean, label="Forecast mean")

    if forecast_lower is not None and forecast_upper is not None:
        ax.fill_between(
            scales,
            forecast_lower,
            forecast_upper,
            alpha=0.25,
            label="Forecast 10–90%",
        )

    ax.set_xlabel("Scale [km]")
    ax.set_ylabel("PSD")
    ax.set_title("Power spectral density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# ISS
# -----------------------------------------------------------------------------


def _extract_iss_matrix(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    def _from_payload_dict(source: dict[str, Any]):
        thresholds = _find_first_array(
            source,
            (
                "thresholds_mm",
                "thresholds",
                "precip_thresholds_mm",
            ),
        )
        scales = _find_first_array(
            source,
            (
                "scales_km",
                "spatial_scales_km",
                "iss_scales_km",
            ),
        )

        matrix = _find_first_array(
            source,
            (
                "iss_matrix",
                "score_matrix",
                "scores",
                "iss",
            ),
        )

        if matrix is None and isinstance(source.get("scores"), dict):
            score_dict = source["scores"]
            try:
                threshold_keys = sorted(score_dict.keys(), key=float)
                row_arrays = []
                parsed_thresholds = []
                for key in threshold_keys:
                    row = _safe_array(score_dict[key])
                    if row is None:
                        return None
                    row_arrays.append(np.ravel(row))
                    parsed_thresholds.append(float(key))
                if len(row_arrays) == 0:
                    return None
                min_len = min(row.size for row in row_arrays)
                matrix = np.stack([row[:min_len] for row in row_arrays], axis=0)
                thresholds = np.asarray(parsed_thresholds, dtype=float)
                if scales is None:
                    scales = np.arange(1, min_len + 1, dtype=float)
            except Exception:
                return None

        if matrix is None:
            return None

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2:
            return None

        n_thresholds, n_scales = matrix.shape

        if thresholds is None:
            thresholds = np.arange(1, n_thresholds + 1, dtype=float)
        else:
            thresholds = np.ravel(thresholds)[:n_thresholds]

        if scales is None:
            scales = np.arange(1, n_scales + 1, dtype=float)
        else:
            scales = np.ravel(scales)[:n_scales]

        if thresholds.size == 0 or scales.size == 0:
            return None

        return thresholds, scales, matrix[: thresholds.size, : scales.size]

    direct = _from_payload_dict(payload)
    if direct is not None:
        return direct

    per_case = payload.get("per_case")
    if isinstance(per_case, list):
        for item in per_case:
            if isinstance(item, dict):
                extracted = _from_payload_dict(item)
                if extracted is not None:
                    return extracted

    return None



def _plot_iss(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "iss")
    matrices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for payload in payloads:
        result = _extract_iss_matrix(payload)
        if result is not None:
            matrices.append(result)

    if len(matrices) == 0:
        return None

    min_nt = min(thresholds.size for thresholds, scales, matrix in matrices)
    min_ns = min(scales.size for thresholds, scales, matrix in matrices)

    threshold_stack = np.stack([thresholds[:min_nt] for thresholds, scales, matrix in matrices], axis=0)
    scale_stack = np.stack([scales[:min_ns] for thresholds, scales, matrix in matrices], axis=0)
    matrix_stack = np.stack([matrix[:min_nt, :min_ns] for thresholds, scales, matrix in matrices], axis=0)

    thresholds = np.nanmean(threshold_stack, axis=0)
    scales = np.nanmean(scale_stack, axis=0)
    matrix_mean = np.nanmean(matrix_stack, axis=0)

    figure_path = output_dir / "iss.png"

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix_mean, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(scales.size))
    ax.set_xticklabels([f"{v:.0f}" for v in scales], rotation=45, ha="right")
    ax.set_yticks(np.arange(thresholds.size))
    ax.set_yticklabels([f"{v:.1f}" for v in thresholds])
    ax.set_xlabel("Scale [km]")
    ax.set_ylabel("Threshold [mm]")
    ax.set_title("ISS")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# SAL
# -----------------------------------------------------------------------------


def _extract_sal_components(payload: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    def _from_payload_dict(source: dict[str, Any]):
        s_val = _safe_scalar(source.get("S"))
        if s_val is None:
            s_val = _safe_scalar(source.get("structure"))
        a_val = _safe_scalar(source.get("A"))
        if a_val is None:
            a_val = _safe_scalar(source.get("amplitude"))
        l_val = _safe_scalar(source.get("L"))
        if l_val is None:
            l_val = _safe_scalar(source.get("location"))
        return s_val, a_val, l_val

    direct = _from_payload_dict(payload)
    if any(v is not None for v in direct):
        return direct

    per_case = payload.get("per_case")
    if isinstance(per_case, list):
        for item in per_case:
            if isinstance(item, dict):
                extracted = _from_payload_dict(item)
                if any(v is not None for v in extracted):
                    return extracted

    return None, None, None



def _plot_sal(metrics: dict[str, Any], output_dir: Path) -> Path | None:
    payloads = _collect_metric_payloads(metrics, "sal")
    s_vals: list[float] = []
    a_vals: list[float] = []
    l_vals: list[float] = []

    for payload in payloads:
        s_val, a_val, l_val = _extract_sal_components(payload)
        if s_val is not None:
            s_vals.append(s_val)
        if a_val is not None:
            a_vals.append(a_val)
        if l_val is not None:
            l_vals.append(l_val)

    if len(s_vals) == 0 and len(a_vals) == 0 and len(l_vals) == 0:
        return None

    figure_path = output_dir / "sal.png"

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)

    data = []
    labels = []
    if len(s_vals) > 0:
        data.append(s_vals)
        labels.append("S")
    if len(a_vals) > 0:
        data.append(a_vals)
        labels.append("A")
    if len(l_vals) > 0:
        data.append(l_vals)
        labels.append("L")

    ax.boxplot(data, labels=labels)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("Component value")
    ax.set_title("SAL component distributions")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_path


# -----------------------------------------------------------------------------
# Public family entrypoint
# -----------------------------------------------------------------------------


def plot_spatial(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Generate spatial evaluation plots.

    Parameters
    ----------
    metrics:
        Full evaluator metrics dictionary.
    output_dir:
        Directory where spatial figures should be saved.
    cfg:
        Optional evaluation config. Currently unused, but kept in the signature
        so the evaluator can call all family plotters consistently.

    Returns
    -------
    list[Path]
        Paths to the figures that were written.
    """
    out_dir = _ensure_output_dir(output_dir)
    saved: list[Path] = []

    for builder in (
        _plot_psd,
        _plot_iss,
        _plot_sal,
    ):
        result = builder(metrics, out_dir)
        if result is not None:
            saved.append(result)

    return saved



def run_spatial_plots(
    metrics: dict[str, Any],
    output_dir: str | Path,
    cfg: Any | None = None,
) -> list[Path]:
    """
    Backward-compatible alias for the evaluator plot-dispatch scaffold.
    """
    return plot_spatial(metrics=metrics, output_dir=output_dir, cfg=cfg)