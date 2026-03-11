"""
Pillar 3 - Climatology / distribution evaluation.

This module evaluates whether generated fields reproduce the bulk and tail
statistics that matter for climate and impact-relevant applications.

Implemented diagnostics
-----------------------
- Pixel value distribution (pooled over all valid pixels)
- Histogram comparison
- Q-Q plot data
- Extremes at configurable quantiles
- Wet-day frequency
- Annual precipitation sum (per case and aggregate)
- Seasonal accumulations (DJF, MAM, JJA, SON)

Ensemble-first design
---------------------
Whenever possible, this module uses the full ensemble rather than only a single
summary product such as PMM or ensemble mean.

Accepted forecast input shapes
------------------------------
- deterministic field: [H, W] or [B, H, W]
- ensemble field:      [M, H, W] or [B, M, H, W]

Accepted target input shapes
----------------------------
- [H, W]
- [B, H, W]
- [1, H, W]
- [B, 1, H, W]

Optional mask input shapes
--------------------------
- [H, W]
- [B, H, W]
- [1, H, W]
- [B, 1, H, W]

Notes
-----
- Forecast and target values are assumed to already be in physical units.
- For ensemble inputs, pooled forecast distributions and quantiles are computed
  across all members and all valid pixels, which is usually what you want for
  climatological verification of a generative model.
- Annual and seasonal accumulation summaries are also reported member-wise when
  an ensemble is supplied, so spread information is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Typed outputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelValueDistributionResult:
    bin_edges: list[float]
    forecast_density: list[float]
    target_density: list[float]
    num_forecast_values: int
    num_target_values: int


@dataclass(frozen=True)
class HistogramComparisonResult:
    bin_edges: list[float]
    forecast_hist: list[float]
    target_hist: list[float]
    l1_distance: float
    l2_distance: float


@dataclass(frozen=True)
class QQResult:
    quantile_levels: list[float]
    forecast_quantiles: list[float]
    target_quantiles: list[float]


@dataclass(frozen=True)
class ExtremesResult:
    quantile_levels: list[float]
    forecast_quantiles: list[float]
    target_quantiles: list[float]
    quantile_bias: list[float]


@dataclass(frozen=True)
class WetDayFrequencyResult:
    threshold_mm: float
    forecast_frequency: float
    target_frequency: float
    frequency_bias: float


@dataclass(frozen=True)
class AnnualAccumulationCaseResult:
    forecast_member_sums: list[float]
    forecast_mean_sum: float
    forecast_std_sum: float
    target_sum: float
    sum_bias: float


@dataclass(frozen=True)
class AnnualAccumulationResult:
    per_case: list[AnnualAccumulationCaseResult]
    mean_forecast_sum: float
    mean_target_sum: float
    mean_sum_bias: float
    num_cases: int


@dataclass(frozen=True)
class SeasonalAccumulationSeasonResult:
    season: str
    case_indices: list[int]
    forecast_member_sums: list[float]
    forecast_mean_sum: float
    forecast_std_sum: float
    target_sums: list[float]
    target_mean_sum: float
    mean_sum_bias: float


@dataclass(frozen=True)
class SeasonalAccumulationResult:
    seasons: dict[str, SeasonalAccumulationSeasonResult]
    num_cases: int


# -----------------------------------------------------------------------------
# Shape normalization helpers
# -----------------------------------------------------------------------------


ArrayLike = np.ndarray
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")



def _as_numpy(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr



def _ensure_bmhw_forecast(forecast: ArrayLike) -> np.ndarray:
    arr = _as_numpy(forecast, name="forecast")
    if arr.ndim == 2:
        # [H, W] -> [1, 1, H, W]
        return arr[np.newaxis, np.newaxis, ...].astype(np.float64, copy=False)
    if arr.ndim == 3:
        # Ambiguous: interpret as ensemble [M, H, W] for climatology. This is
        # the desired default when evaluating full ensembles case-by-case.
        return arr[np.newaxis, ...].astype(np.float64, copy=False)
    if arr.ndim == 4:
        # [B, M, H, W]
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"forecast must have shape [H,W], [M,H,W], [B,H,W]-like already expanded, or [B,M,H,W], got {tuple(arr.shape)}"
    )



def _ensure_bhw_target(target: ArrayLike, *, batch_size: int) -> np.ndarray:
    arr = _as_numpy(target, name="target")
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"4D target must have channel dimension 1, got {tuple(arr.shape)}"
            )
        arr = arr[:, 0, ...]
    else:
        raise ValueError(
            f"target must have shape [H,W], [B,H,W], [1,H,W], or [B,1,H,W], got {tuple(arr.shape)}"
        )

    if arr.ndim != 3:
        raise ValueError(f"normalized target must be [B,H,W], got {tuple(arr.shape)}")

    if arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)

    if arr.shape[0] != batch_size:
        raise ValueError(
            f"target batch size mismatch: expected {batch_size}, got {arr.shape[0]}"
        )
    return arr.astype(np.float64, copy=False)



def _ensure_bhw_mask(
    mask: ArrayLike | None,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> np.ndarray:
    if mask is None:
        return np.ones((batch_size, height, width), dtype=bool)

    arr = _as_numpy(mask, name="mask")
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"4D mask must have channel dimension 1, got {tuple(arr.shape)}"
            )
        arr = arr[:, 0, ...]
    else:
        raise ValueError(
            f"mask must have shape [H,W], [B,H,W], [1,H,W], or [B,1,H,W], got {tuple(arr.shape)}"
        )

    if arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)

    arr = np.asarray(arr > 0.5, dtype=bool)
    if arr.shape != (batch_size, height, width):
        raise ValueError(
            "mask shape mismatch after normalization: "
            f"expected {(batch_size, height, width)}, got {tuple(arr.shape)}"
        )
    return arr



def _normalize_inputs(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast_bmhw = _ensure_bmhw_forecast(forecast)
    batch_size, _, height, width = forecast_bmhw.shape
    target_bhw = _ensure_bhw_target(target, batch_size=batch_size)
    if target_bhw.shape[1:] != (height, width):
        raise ValueError(
            "forecast/target spatial mismatch: "
            f"forecast={(height, width)}, target={tuple(target_bhw.shape[1:])}"
        )
    mask_bhw = _ensure_bhw_mask(
        mask,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    return forecast_bmhw, target_bhw, mask_bhw



def _flatten_valid_values(
    forecast_bmhw: np.ndarray,
    target_bhw: np.ndarray,
    mask_bhw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forecast_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []

    for b in range(forecast_bmhw.shape[0]):
        valid = mask_bhw[b].reshape(-1)
        if int(valid.sum()) == 0:
            continue
        forecast_flat = forecast_bmhw[b].reshape(forecast_bmhw.shape[1], -1)[:, valid]
        target_flat = target_bhw[b].reshape(-1)[valid]
        forecast_values.append(forecast_flat.reshape(-1))
        target_values.append(target_flat)

    if len(forecast_values) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return (
        np.concatenate(forecast_values, axis=0).astype(np.float64, copy=False),
        np.concatenate(target_values, axis=0).astype(np.float64, copy=False),
    )



def _resolve_bin_edges(
    forecast_values: np.ndarray,
    target_values: np.ndarray,
    *,
    num_bins: int,
    value_range: tuple[float, float] | None,
) -> np.ndarray:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
    else:
        pooled = np.concatenate([forecast_values, target_values], axis=0)
        if pooled.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.nanmin(pooled))
            hi = float(np.nanmax(pooled))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                hi = lo + 1.0
    return np.linspace(lo, hi, num_bins + 1, dtype=np.float64)


# -----------------------------------------------------------------------------
# Distribution diagnostics
# -----------------------------------------------------------------------------



def compute_pixel_value_distribution(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    num_bins: int = 50,
    value_range: tuple[float, float] | None = None,
) -> PixelValueDistributionResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    forecast_values, target_values = _flatten_valid_values(
        forecast_bmhw,
        target_bhw,
        mask_bhw,
    )

    bin_edges = _resolve_bin_edges(
        forecast_values,
        target_values,
        num_bins=num_bins,
        value_range=value_range,
    )

    forecast_hist, _ = np.histogram(forecast_values, bins=bin_edges, density=True)
    target_hist, _ = np.histogram(target_values, bins=bin_edges, density=True)

    return PixelValueDistributionResult(
        bin_edges=bin_edges.astype(float).tolist(),
        forecast_density=forecast_hist.astype(float).tolist(),
        target_density=target_hist.astype(float).tolist(),
        num_forecast_values=int(forecast_values.size),
        num_target_values=int(target_values.size),
    )



def compute_histogram_comparison(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    num_bins: int = 50,
    value_range: tuple[float, float] | None = None,
) -> HistogramComparisonResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    forecast_values, target_values = _flatten_valid_values(
        forecast_bmhw,
        target_bhw,
        mask_bhw,
    )

    bin_edges = _resolve_bin_edges(
        forecast_values,
        target_values,
        num_bins=num_bins,
        value_range=value_range,
    )

    forecast_hist, _ = np.histogram(forecast_values, bins=bin_edges, density=True)
    target_hist, _ = np.histogram(target_values, bins=bin_edges, density=True)

    l1_distance = float(np.sum(np.abs(forecast_hist - target_hist)))
    l2_distance = float(np.sqrt(np.sum((forecast_hist - target_hist) ** 2)))

    return HistogramComparisonResult(
        bin_edges=bin_edges.astype(float).tolist(),
        forecast_hist=forecast_hist.astype(float).tolist(),
        target_hist=target_hist.astype(float).tolist(),
        l1_distance=l1_distance,
        l2_distance=l2_distance,
    )



def compute_qq_data(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    quantile_levels: np.ndarray | list[float] | None = None,
) -> QQResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    forecast_values, target_values = _flatten_valid_values(
        forecast_bmhw,
        target_bhw,
        mask_bhw,
    )

    if quantile_levels is None:
        quantiles = np.linspace(0.001, 0.999, 100, dtype=np.float64)
    else:
        quantiles = np.asarray(quantile_levels, dtype=np.float64)

    forecast_q = np.quantile(forecast_values, quantiles) if forecast_values.size > 0 else np.full_like(quantiles, np.nan)
    target_q = np.quantile(target_values, quantiles) if target_values.size > 0 else np.full_like(quantiles, np.nan)

    return QQResult(
        quantile_levels=quantiles.astype(float).tolist(),
        forecast_quantiles=forecast_q.astype(float).tolist(),
        target_quantiles=target_q.astype(float).tolist(),
    )



def compute_extremes(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    quantile_levels: list[float],
) -> ExtremesResult:
    if len(quantile_levels) == 0:
        raise ValueError("quantile_levels must not be empty")

    qs = np.asarray(quantile_levels, dtype=np.float64)
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    forecast_values, target_values = _flatten_valid_values(
        forecast_bmhw,
        target_bhw,
        mask_bhw,
    )

    forecast_q = np.quantile(forecast_values, qs) if forecast_values.size > 0 else np.full_like(qs, np.nan)
    target_q = np.quantile(target_values, qs) if target_values.size > 0 else np.full_like(qs, np.nan)
    bias = forecast_q - target_q

    return ExtremesResult(
        quantile_levels=qs.astype(float).tolist(),
        forecast_quantiles=forecast_q.astype(float).tolist(),
        target_quantiles=target_q.astype(float).tolist(),
        quantile_bias=bias.astype(float).tolist(),
    )



def compute_wet_day_frequency(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    threshold_mm: float = 1.0,
) -> WetDayFrequencyResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    forecast_values, target_values = _flatten_valid_values(
        forecast_bmhw,
        target_bhw,
        mask_bhw,
    )

    forecast_frequency = float(np.mean(forecast_values >= threshold_mm)) if forecast_values.size > 0 else float("nan")
    target_frequency = float(np.mean(target_values >= threshold_mm)) if target_values.size > 0 else float("nan")

    return WetDayFrequencyResult(
        threshold_mm=float(threshold_mm),
        forecast_frequency=forecast_frequency,
        target_frequency=target_frequency,
        frequency_bias=float(forecast_frequency - target_frequency),
    )


# -----------------------------------------------------------------------------
# Accumulation diagnostics
# -----------------------------------------------------------------------------



def _case_forecast_member_sums(
    forecast_mhw: np.ndarray,
    mask_hw: np.ndarray,
) -> np.ndarray:
    valid = mask_hw.astype(np.float64)
    weighted = forecast_mhw * valid[None, ...]
    return np.sum(weighted, axis=(1, 2)).astype(np.float64)



def _case_target_sum(
    target_hw: np.ndarray,
    mask_hw: np.ndarray,
) -> float:
    return float(np.sum(target_hw * mask_hw.astype(np.float64)))



def compute_annual_precipitation_sum(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> AnnualAccumulationResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    per_case: list[AnnualAccumulationCaseResult] = []

    case_forecast_means: list[float] = []
    case_target_sums: list[float] = []

    for b in range(forecast_bmhw.shape[0]):
        member_sums = _case_forecast_member_sums(forecast_bmhw[b], mask_bhw[b])
        target_sum = _case_target_sum(target_bhw[b], mask_bhw[b])
        forecast_mean_sum = float(np.mean(member_sums))
        forecast_std_sum = float(np.std(member_sums, ddof=0))
        sum_bias = float(forecast_mean_sum - target_sum)

        per_case.append(
            AnnualAccumulationCaseResult(
                forecast_member_sums=member_sums.astype(float).tolist(),
                forecast_mean_sum=forecast_mean_sum,
                forecast_std_sum=forecast_std_sum,
                target_sum=target_sum,
                sum_bias=sum_bias,
            )
        )
        case_forecast_means.append(forecast_mean_sum)
        case_target_sums.append(target_sum)

    if len(per_case) == 0:
        return AnnualAccumulationResult(
            per_case=[],
            mean_forecast_sum=float("nan"),
            mean_target_sum=float("nan"),
            mean_sum_bias=float("nan"),
            num_cases=0,
        )

    mean_forecast_sum = float(np.mean(case_forecast_means))
    mean_target_sum = float(np.mean(case_target_sums))

    return AnnualAccumulationResult(
        per_case=per_case,
        mean_forecast_sum=mean_forecast_sum,
        mean_target_sum=mean_target_sum,
        mean_sum_bias=float(mean_forecast_sum - mean_target_sum),
        num_cases=len(per_case),
    )



def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"



def compute_seasonal_accumulations(
    forecast: ArrayLike,
    target: ArrayLike,
    dates: list[str] | tuple[str, ...],
    mask: ArrayLike | None = None,
) -> SeasonalAccumulationResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)

    if len(dates) != forecast_bmhw.shape[0]:
        raise ValueError(
            f"dates length must match batch size, got len(dates)={len(dates)} and batch={forecast_bmhw.shape[0]}"
        )

    grouped_indices: dict[str, list[int]] = {season: [] for season in SEASON_ORDER}
    for idx, date_str in enumerate(dates):
        month = int(str(date_str)[4:6])
        grouped_indices[_month_to_season(month)].append(idx)

    seasons: dict[str, SeasonalAccumulationSeasonResult] = {}

    for season in SEASON_ORDER:
        indices = grouped_indices[season]
        if len(indices) == 0:
            seasons[season] = SeasonalAccumulationSeasonResult(
                season=season,
                case_indices=[],
                forecast_member_sums=[],
                forecast_mean_sum=float("nan"),
                forecast_std_sum=float("nan"),
                target_sums=[],
                target_mean_sum=float("nan"),
                mean_sum_bias=float("nan"),
            )
            continue

        forecast_member_sums_per_case: list[np.ndarray] = []
        target_sums: list[float] = []
        for idx in indices:
            forecast_member_sums_per_case.append(
                _case_forecast_member_sums(forecast_bmhw[idx], mask_bhw[idx])
            )
            target_sums.append(_case_target_sum(target_bhw[idx], mask_bhw[idx]))

        stacked_members = np.concatenate(forecast_member_sums_per_case, axis=0)
        forecast_mean_sum = float(np.mean(stacked_members))
        forecast_std_sum = float(np.std(stacked_members, ddof=0))
        target_mean_sum = float(np.mean(target_sums))

        seasons[season] = SeasonalAccumulationSeasonResult(
            season=season,
            case_indices=[int(i) for i in indices],
            forecast_member_sums=stacked_members.astype(float).tolist(),
            forecast_mean_sum=forecast_mean_sum,
            forecast_std_sum=forecast_std_sum,
            target_sums=[float(x) for x in target_sums],
            target_mean_sum=target_mean_sum,
            mean_sum_bias=float(forecast_mean_sum - target_mean_sum),
        )

    return SeasonalAccumulationResult(
        seasons=seasons,
        num_cases=forecast_bmhw.shape[0],
    )