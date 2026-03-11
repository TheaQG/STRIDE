"""
Pillar 4 - Temporal evaluation.

This module evaluates whether generated fields reproduce temporal persistence
and spell characteristics that matter for hydrological and climate-impact
applications.

Implemented diagnostics
-----------------------
- Lag autocorrelation
- Wet spell length distribution
- Dry spell length distribution

Design notes
------------
- Temporal metrics are inherently sequence-based. Inputs are expected to span a
  time dimension T.
- Forecast input can be deterministic or ensemble. For ensemble forecasts, the
  module preserves member-wise temporal diagnostics where appropriate and also
  provides ensemble-mean summaries.
- The primary use case is evaluating a sequence of daily fields over a fixed
  spatial domain.

Accepted forecast input shapes
------------------------------
- deterministic sequence: [T, H, W]
- ensemble sequence:      [T, M, H, W]

Accepted target input shapes
----------------------------
- [T, H, W]
- [T, 1, H, W]

Optional mask input shapes
--------------------------
- [H, W]
- [1, H, W]
- [T, H, W]
- [T, 1, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Typed outputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LagAutocorrelationResult:
    lags: list[int]
    forecast_member_mean_autocorr: list[float]
    forecast_member_std_autocorr: list[float]
    forecast_ensemble_mean_autocorr: list[float]
    target_autocorr: list[float]
    num_timesteps: int
    num_members: int


@dataclass(frozen=True)
class SpellLengthDistributionResult:
    threshold_mm: float
    max_length: int
    forecast_member_mean_hist: list[float]
    forecast_member_std_hist: list[float]
    forecast_all_lengths: list[int]
    target_hist: list[float]
    target_lengths: list[int]
    num_members: int
    num_timesteps: int


# -----------------------------------------------------------------------------
# Shape normalization helpers
# -----------------------------------------------------------------------------


ArrayLike = np.ndarray



def _as_numpy(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr



def _ensure_tmhw_forecast(forecast: ArrayLike) -> np.ndarray:
    arr = _as_numpy(forecast, name="forecast")
    if arr.ndim == 3:
        # [T,H,W] -> [T,1,H,W]
        return arr[:, np.newaxis, ...].astype(np.float64, copy=False)
    if arr.ndim == 4:
        # [T,M,H,W]
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"forecast must have shape [T,H,W] or [T,M,H,W], got {tuple(arr.shape)}"
    )



def _ensure_thw_target(target: ArrayLike, *, num_timesteps: int) -> np.ndarray:
    arr = _as_numpy(target, name="target")
    if arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"4D target must have channel dimension 1, got {tuple(arr.shape)}"
            )
        arr = arr[:, 0, ...]
    else:
        raise ValueError(
            f"target must have shape [T,H,W] or [T,1,H,W], got {tuple(arr.shape)}"
        )

    if arr.shape[0] != num_timesteps:
        raise ValueError(
            f"target time length mismatch: expected {num_timesteps}, got {arr.shape[0]}"
        )
    return arr.astype(np.float64, copy=False)



def _ensure_thw_mask(
    mask: ArrayLike | None,
    *,
    num_timesteps: int,
    height: int,
    width: int,
) -> np.ndarray:
    if mask is None:
        return np.ones((num_timesteps, height, width), dtype=bool)

    arr = _as_numpy(mask, name="mask")
    if arr.ndim == 2:
        arr = np.repeat(arr[np.newaxis, ...], num_timesteps, axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] == 1 and num_timesteps > 1:
            arr = np.repeat(arr, num_timesteps, axis=0)
    elif arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"4D mask must have channel dimension 1, got {tuple(arr.shape)}"
            )
        arr = arr[:, 0, ...]
        if arr.shape[0] == 1 and num_timesteps > 1:
            arr = np.repeat(arr, num_timesteps, axis=0)
    else:
        raise ValueError(
            f"mask must have shape [H,W], [T,H,W], [1,H,W], or [T,1,H,W], got {tuple(arr.shape)}"
        )

    arr = np.asarray(arr > 0.5, dtype=bool)
    if arr.shape != (num_timesteps, height, width):
        raise ValueError(
            "mask shape mismatch after normalization: "
            f"expected {(num_timesteps, height, width)}, got {tuple(arr.shape)}"
        )
    return arr



def _normalize_inputs(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast_tmhw = _ensure_tmhw_forecast(forecast)
    num_timesteps, _, height, width = forecast_tmhw.shape
    target_thw = _ensure_thw_target(target, num_timesteps=num_timesteps)
    if target_thw.shape[1:] != (height, width):
        raise ValueError(
            "forecast/target spatial mismatch: "
            f"forecast={(height, width)}, target={tuple(target_thw.shape[1:])}"
        )
    mask_thw = _ensure_thw_mask(
        mask,
        num_timesteps=num_timesteps,
        height=height,
        width=width,
    )
    return forecast_tmhw, target_thw, mask_thw


# -----------------------------------------------------------------------------
# Series helpers
# -----------------------------------------------------------------------------



def _spatial_mean_series(field_thw: np.ndarray, mask_thw: np.ndarray) -> np.ndarray:
    series = np.empty((field_thw.shape[0],), dtype=np.float64)
    for t in range(field_thw.shape[0]):
        valid = mask_thw[t]
        if int(np.sum(valid)) == 0:
            series[t] = np.nan
        else:
            series[t] = float(np.mean(field_thw[t][valid]))
    return series



def _safe_autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    if lag >= series.size:
        return float("nan")

    x = np.asarray(series[:-lag], dtype=np.float64)
    y = np.asarray(series[lag:], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 2:
        return float("nan")

    x = x[valid]
    y = y[valid]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])



def _binary_series_from_sequence(
    field_thw: np.ndarray,
    mask_thw: np.ndarray,
    *,
    threshold_mm: float,
    mode: str,
) -> np.ndarray:
    if mode not in {"wet", "dry"}:
        raise ValueError(f"mode must be 'wet' or 'dry', got {mode!r}")

    series = np.empty((field_thw.shape[0],), dtype=bool)
    for t in range(field_thw.shape[0]):
        valid = mask_thw[t]
        if int(np.sum(valid)) == 0:
            series[t] = False
            continue
        mean_val = float(np.mean(field_thw[t][valid]))
        if mode == "wet":
            series[t] = mean_val >= threshold_mm
        else:
            series[t] = mean_val < threshold_mm
    return series



def _run_lengths(binary_series: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for flag in binary_series:
        if bool(flag):
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths



def _length_histogram(lengths: list[int], *, max_length: int) -> np.ndarray:
    hist = np.zeros((max_length,), dtype=np.float64)
    if len(lengths) == 0:
        return hist
    clipped = np.clip(np.asarray(lengths, dtype=np.int64), 1, max_length)
    counts = np.bincount(clipped, minlength=max_length + 1)[1 : max_length + 1]
    total = int(np.sum(counts))
    if total > 0:
        hist = counts.astype(np.float64) / float(total)
    return hist


# -----------------------------------------------------------------------------
# Lag autocorrelation
# -----------------------------------------------------------------------------



def compute_lag_autocorrelation(
    forecast: ArrayLike,
    target: ArrayLike,
    lags: list[int],
    mask: ArrayLike | None = None,
) -> LagAutocorrelationResult:
    if len(lags) == 0:
        raise ValueError("lags must not be empty")

    forecast_tmhw, target_thw, mask_thw = _normalize_inputs(forecast, target, mask)
    num_timesteps, num_members, _, _ = forecast_tmhw.shape

    forecast_member_series = np.stack(
        [
            _spatial_mean_series(forecast_tmhw[:, m, ...], mask_thw)
            for m in range(num_members)
        ],
        axis=0,
    )
    forecast_ensemble_mean_series = np.nanmean(forecast_member_series, axis=0)
    target_series = _spatial_mean_series(target_thw, mask_thw)

    member_autocorr = np.empty((num_members, len(lags)), dtype=np.float64)
    for m in range(num_members):
        for i, lag in enumerate(lags):
            member_autocorr[m, i] = _safe_autocorrelation(forecast_member_series[m], int(lag))

    ensemble_mean_autocorr = np.array(
        [_safe_autocorrelation(forecast_ensemble_mean_series, int(lag)) for lag in lags],
        dtype=np.float64,
    )
    target_autocorr = np.array(
        [_safe_autocorrelation(target_series, int(lag)) for lag in lags],
        dtype=np.float64,
    )

    return LagAutocorrelationResult(
        lags=[int(x) for x in lags],
        forecast_member_mean_autocorr=np.nanmean(member_autocorr, axis=0).astype(float).tolist(),
        forecast_member_std_autocorr=np.nanstd(member_autocorr, axis=0).astype(float).tolist(),
        forecast_ensemble_mean_autocorr=ensemble_mean_autocorr.astype(float).tolist(),
        target_autocorr=target_autocorr.astype(float).tolist(),
        num_timesteps=num_timesteps,
        num_members=num_members,
    )


# -----------------------------------------------------------------------------
# Spell distributions
# -----------------------------------------------------------------------------



def _compute_spell_distribution(
    forecast: ArrayLike,
    target: ArrayLike,
    *,
    threshold_mm: float,
    mode: str,
    max_length: int | None,
    mask: ArrayLike | None = None,
) -> SpellLengthDistributionResult:
    forecast_tmhw, target_thw, mask_thw = _normalize_inputs(forecast, target, mask)
    num_timesteps, num_members, _, _ = forecast_tmhw.shape

    if max_length is None:
        max_len = num_timesteps
    else:
        max_len = max(int(max_length), 1)

    member_hists = []
    all_forecast_lengths: list[int] = []
    for m in range(num_members):
        binary = _binary_series_from_sequence(
            forecast_tmhw[:, m, ...],
            mask_thw,
            threshold_mm=threshold_mm,
            mode=mode,
        )
        lengths = _run_lengths(binary)
        all_forecast_lengths.extend(lengths)
        member_hists.append(_length_histogram(lengths, max_length=max_len))

    if len(member_hists) == 0:
        member_hist_stack = np.zeros((0, max_len), dtype=np.float64)
        member_hist_mean = np.full((max_len,), np.nan, dtype=np.float64)
        member_hist_std = np.full((max_len,), np.nan, dtype=np.float64)
    else:
        member_hist_stack = np.stack(member_hists, axis=0)
        member_hist_mean = np.nanmean(member_hist_stack, axis=0)
        member_hist_std = np.nanstd(member_hist_stack, axis=0)

    target_binary = _binary_series_from_sequence(
        target_thw,
        mask_thw,
        threshold_mm=threshold_mm,
        mode=mode,
    )
    target_lengths = _run_lengths(target_binary)
    target_hist = _length_histogram(target_lengths, max_length=max_len)

    return SpellLengthDistributionResult(
        threshold_mm=float(threshold_mm),
        max_length=max_len,
        forecast_member_mean_hist=member_hist_mean.astype(float).tolist(),
        forecast_member_std_hist=member_hist_std.astype(float).tolist(),
        forecast_all_lengths=[int(x) for x in all_forecast_lengths],
        target_hist=target_hist.astype(float).tolist(),
        target_lengths=[int(x) for x in target_lengths],
        num_members=num_members,
        num_timesteps=num_timesteps,
    )



def compute_wet_spell_lengths(
    forecast: ArrayLike,
    target: ArrayLike,
    *,
    threshold_mm: float = 1.0,
    max_length: int | None = None,
    mask: ArrayLike | None = None,
) -> SpellLengthDistributionResult:
    return _compute_spell_distribution(
        forecast,
        target,
        threshold_mm=threshold_mm,
        mode="wet",
        max_length=max_length,
        mask=mask,
    )



def compute_dry_spell_lengths(
    forecast: ArrayLike,
    target: ArrayLike,
    *,
    threshold_mm: float = 1.0,
    max_length: int | None = None,
    mask: ArrayLike | None = None,
) -> SpellLengthDistributionResult:
    return _compute_spell_distribution(
        forecast,
        target,
        threshold_mm=threshold_mm,
        mode="dry",
        max_length=max_length,
        mask=mask,
    )