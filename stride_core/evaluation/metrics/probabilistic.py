

"""
Pillar 1 - Probabilistic ensemble evaluation.

This module implements ensemble-first probabilistic diagnostics for STRIDE.
All functions operate on univariate forecast ensembles and corresponding
reference targets, with optional spatial masks.

Supported diagnostics
---------------------
- CRPS (Continuous Ranked Probability Score)
- CRPS decomposition using the standard ensemble representation
  (forecast-vs-observation term and ensemble self-dispersion term)
- Spread-skill relationship
- PIT histogram
- Rank histogram
- Reliability diagrams for exceedance events

Expected tensor/array shapes
----------------------------
Forecast ensembles:
- [M, H, W]
- [B, M, H, W]

Targets:
- [H, W]
- [B, H, W]
- [1, H, W]
- [B, 1, H, W]

Masks:
- [H, W]
- [B, H, W]
- [1, H, W]
- [B, 1, H, W]

Notes
-----
- This module is intentionally NumPy-first, since evaluation is run on saved
  generation outputs, not on live PyTorch tensors.
- All metrics are computed in physical space.
- The implementation is conservative and explicit rather than aggressively
  vectorized in every place; correctness and readability matter more here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Typed outputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CRPSResult:
    mean_crps: float
    per_case_crps: list[float]
    per_case_num_points: list[int]
    num_cases: int
    total_num_points: int


@dataclass(frozen=True)
class CRPSDecompositionResult:
    mean_crps: float
    mean_obs_term: float
    mean_ensemble_term: float
    per_case_crps: list[float]
    per_case_obs_term: list[float]
    per_case_ensemble_term: list[float]
    per_case_num_points: list[int]
    num_cases: int
    total_num_points: int


@dataclass(frozen=True)
class SpreadSkillResult:
    spread_values: list[float]
    skill_values: list[float]
    correlation: float | None
    fitted_slope: float | None
    fitted_intercept: float | None
    num_cases: int


@dataclass(frozen=True)
class HistogramResult:
    counts: list[int]
    bin_edges: list[float]
    normalized_counts: list[float]
    num_samples: int


@dataclass(frozen=True)
class ReliabilityResult:
    thresholds_mm: list[float]
    bin_centers: list[float]
    curves: dict[str, dict[str, list[float] | int]]


# -----------------------------------------------------------------------------
# Shape normalization helpers
# -----------------------------------------------------------------------------


ArrayLike = np.ndarray



def _as_numpy(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr



def _ensure_bmhw_forecast(forecast: ArrayLike) -> np.ndarray:
    arr = _as_numpy(forecast, name="forecast")
    if arr.ndim == 3:
        # [M, H, W] -> [1, M, H, W]
        return arr[np.newaxis, ...]
    if arr.ndim == 4:
        # [B, M, H, W]
        return arr
    raise ValueError(
        f"forecast must have shape [M,H,W] or [B,M,H,W], got {tuple(arr.shape)}"
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
    return arr



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

    if arr.ndim != 3:
        raise ValueError(f"normalized mask must be [B,H,W], got {tuple(arr.shape)}")

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
    forecast_bmhw = _ensure_bmhw_forecast(forecast).astype(np.float64, copy=False)
    batch_size, _, height, width = forecast_bmhw.shape
    target_bhw = _ensure_bhw_target(target, batch_size=batch_size).astype(
        np.float64, copy=False
    )
    mask_bhw = _ensure_bhw_mask(
        mask,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    if target_bhw.shape[1:] != (height, width):
        raise ValueError(
            "forecast/target spatial mismatch: "
            f"forecast={(height, width)}, target={tuple(target_bhw.shape[1:])}"
        )
    return forecast_bmhw, target_bhw, mask_bhw



def _flatten_case(
    forecast_mhw: np.ndarray,
    target_hw: np.ndarray,
    mask_hw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask_hw.reshape(-1)
    if int(valid.sum()) == 0:
        return np.empty((forecast_mhw.shape[0], 0), dtype=np.float64), np.empty(
            (0,), dtype=np.float64
        )

    forecast_flat = forecast_mhw.reshape(forecast_mhw.shape[0], -1)[:, valid]
    target_flat = target_hw.reshape(-1)[valid]
    return forecast_flat, target_flat


# -----------------------------------------------------------------------------
# CRPS
# -----------------------------------------------------------------------------



def _crps_ensemble_1d(forecast_members: np.ndarray, observation: float) -> float:
    """
    Ensemble CRPS for one scalar observation.

    Uses the standard empirical ensemble form:
        CRPS(F, y) = mean(|x_m - y|) - 0.5 * mean(|x_m - x_n|)
    """
    members = np.asarray(forecast_members, dtype=np.float64)
    if members.ndim != 1:
        raise ValueError(f"members must be 1D, got shape {tuple(members.shape)}")
    if members.size == 0:
        raise ValueError("members must not be empty")

    obs_term = np.mean(np.abs(members - observation))
    pairwise = np.abs(members[:, None] - members[None, :])
    ens_term = 0.5 * np.mean(pairwise)
    return float(obs_term - ens_term)



def compute_crps(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> CRPSResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)

    per_case_crps: list[float] = []
    per_case_num_points: list[int] = []
    weighted_sum = 0.0
    total_num_points = 0

    for b in range(forecast_bmhw.shape[0]):
        forecast_flat, target_flat = _flatten_case(
            forecast_bmhw[b],
            target_bhw[b],
            mask_bhw[b],
        )
        num_points = int(target_flat.size)
        per_case_num_points.append(num_points)

        if num_points == 0:
            per_case_crps.append(float("nan"))
            continue

        values = np.empty((num_points,), dtype=np.float64)
        for i in range(num_points):
            values[i] = _crps_ensemble_1d(forecast_flat[:, i], target_flat[i])

        case_crps = float(np.mean(values))
        per_case_crps.append(case_crps)
        weighted_sum += case_crps * num_points
        total_num_points += num_points

    mean_crps = float(weighted_sum / total_num_points) if total_num_points > 0 else float("nan")
    return CRPSResult(
        mean_crps=mean_crps,
        per_case_crps=per_case_crps,
        per_case_num_points=per_case_num_points,
        num_cases=forecast_bmhw.shape[0],
        total_num_points=total_num_points,
    )



def compute_crps_decomposition(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> CRPSDecompositionResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)

    per_case_crps: list[float] = []
    per_case_obs_term: list[float] = []
    per_case_ensemble_term: list[float] = []
    per_case_num_points: list[int] = []

    weighted_crps = 0.0
    weighted_obs = 0.0
    weighted_ens = 0.0
    total_num_points = 0

    for b in range(forecast_bmhw.shape[0]):
        forecast_flat, target_flat = _flatten_case(
            forecast_bmhw[b],
            target_bhw[b],
            mask_bhw[b],
        )
        num_points = int(target_flat.size)
        per_case_num_points.append(num_points)

        if num_points == 0:
            per_case_crps.append(float("nan"))
            per_case_obs_term.append(float("nan"))
            per_case_ensemble_term.append(float("nan"))
            continue

        obs_terms = np.mean(np.abs(forecast_flat - target_flat[None, :]), axis=0)
        # Average over full member-member matrix, then multiply by 0.5.
        pairwise = np.abs(
            forecast_flat[:, None, :] - forecast_flat[None, :, :]
        )
        ens_terms = 0.5 * np.mean(pairwise, axis=(0, 1))
        crps_values = obs_terms - ens_terms

        case_obs = float(np.mean(obs_terms))
        case_ens = float(np.mean(ens_terms))
        case_crps = float(np.mean(crps_values))

        per_case_obs_term.append(case_obs)
        per_case_ensemble_term.append(case_ens)
        per_case_crps.append(case_crps)

        weighted_obs += case_obs * num_points
        weighted_ens += case_ens * num_points
        weighted_crps += case_crps * num_points
        total_num_points += num_points

    mean_obs_term = float(weighted_obs / total_num_points) if total_num_points > 0 else float("nan")
    mean_ensemble_term = float(weighted_ens / total_num_points) if total_num_points > 0 else float("nan")
    mean_crps = float(weighted_crps / total_num_points) if total_num_points > 0 else float("nan")

    return CRPSDecompositionResult(
        mean_crps=mean_crps,
        mean_obs_term=mean_obs_term,
        mean_ensemble_term=mean_ensemble_term,
        per_case_crps=per_case_crps,
        per_case_obs_term=per_case_obs_term,
        per_case_ensemble_term=per_case_ensemble_term,
        per_case_num_points=per_case_num_points,
        num_cases=forecast_bmhw.shape[0],
        total_num_points=total_num_points,
    )


# -----------------------------------------------------------------------------
# Spread-skill
# -----------------------------------------------------------------------------



def compute_spread_skill(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> SpreadSkillResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)

    spread_values: list[float] = []
    skill_values: list[float] = []

    for b in range(forecast_bmhw.shape[0]):
        forecast_flat, target_flat = _flatten_case(
            forecast_bmhw[b],
            target_bhw[b],
            mask_bhw[b],
        )
        if target_flat.size == 0:
            continue

        ensemble_mean = np.mean(forecast_flat, axis=0)
        spread = float(np.mean(np.std(forecast_flat, axis=0, ddof=0)))
        skill = float(np.sqrt(np.mean((ensemble_mean - target_flat) ** 2)))

        spread_values.append(spread)
        skill_values.append(skill)

    if len(spread_values) >= 2:
        x = np.asarray(spread_values, dtype=np.float64)
        y = np.asarray(skill_values, dtype=np.float64)
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
        slope, intercept = np.polyfit(x, y, 1)
        fitted_slope = float(slope)
        fitted_intercept = float(intercept)
    else:
        corr = None
        fitted_slope = None
        fitted_intercept = None

    return SpreadSkillResult(
        spread_values=spread_values,
        skill_values=skill_values,
        correlation=corr,
        fitted_slope=fitted_slope,
        fitted_intercept=fitted_intercept,
        num_cases=len(spread_values),
    )


# -----------------------------------------------------------------------------
# PIT and rank histograms
# -----------------------------------------------------------------------------



def compute_pit_histogram(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    num_bins: int = 10,
) -> HistogramResult:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    pit_values: list[np.ndarray] = []

    rng = np.random.default_rng(0)

    for b in range(forecast_bmhw.shape[0]):
        forecast_flat, target_flat = _flatten_case(
            forecast_bmhw[b],
            target_bhw[b],
            mask_bhw[b],
        )
        if target_flat.size == 0:
            continue

        # Empirical CDF with randomized tie handling.
        less = np.sum(forecast_flat < target_flat[None, :], axis=0)
        equal = np.sum(forecast_flat == target_flat[None, :], axis=0)
        tie_fraction = rng.uniform(size=target_flat.size)
        pits = (less + tie_fraction * equal) / float(forecast_flat.shape[0])
        pit_values.append(pits)

    if len(pit_values) == 0:
        counts = np.zeros((num_bins,), dtype=int)
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        normalized = np.zeros((num_bins,), dtype=np.float64)
        return HistogramResult(
            counts=counts.tolist(),
            bin_edges=bin_edges.tolist(),
            normalized_counts=normalized.tolist(),
            num_samples=0,
        )

    pits_all = np.concatenate(pit_values, axis=0)
    counts, bin_edges = np.histogram(pits_all, bins=num_bins, range=(0.0, 1.0))
    normalized = counts / max(int(counts.sum()), 1)
    return HistogramResult(
        counts=counts.astype(int).tolist(),
        bin_edges=bin_edges.astype(float).tolist(),
        normalized_counts=normalized.astype(float).tolist(),
        num_samples=int(pits_all.size),
    )



def compute_rank_histogram(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
) -> HistogramResult:
    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)

    all_ranks: list[np.ndarray] = []
    rng = np.random.default_rng(0)

    num_members = forecast_bmhw.shape[1]
    num_bins = num_members + 1

    for b in range(forecast_bmhw.shape[0]):
        forecast_flat, target_flat = _flatten_case(
            forecast_bmhw[b],
            target_bhw[b],
            mask_bhw[b],
        )
        if target_flat.size == 0:
            continue

        less = np.sum(forecast_flat < target_flat[None, :], axis=0)
        equal = np.sum(forecast_flat == target_flat[None, :], axis=0)

        ranks = less.astype(np.int64)
        tied = equal > 0
        if np.any(tied):
            # Randomized discrete rank among tied intervals.
            offsets = np.floor(rng.uniform(size=int(np.sum(tied))) * (equal[tied] + 1)).astype(np.int64)
            ranks[tied] = ranks[tied] + offsets
        all_ranks.append(ranks)

    if len(all_ranks) == 0:
        counts = np.zeros((num_bins,), dtype=int)
        bin_edges = np.arange(num_bins + 1, dtype=float) - 0.5
        normalized = np.zeros((num_bins,), dtype=np.float64)
        return HistogramResult(
            counts=counts.tolist(),
            bin_edges=bin_edges.tolist(),
            normalized_counts=normalized.tolist(),
            num_samples=0,
        )

    ranks_all = np.concatenate(all_ranks, axis=0)
    counts = np.bincount(ranks_all, minlength=num_bins)
    bin_edges = np.arange(num_bins + 1, dtype=float) - 0.5
    normalized = counts / max(int(counts.sum()), 1)
    return HistogramResult(
        counts=counts.astype(int).tolist(),
        bin_edges=bin_edges.astype(float).tolist(),
        normalized_counts=normalized.astype(float).tolist(),
        num_samples=int(ranks_all.size),
    )


# -----------------------------------------------------------------------------
# Reliability
# -----------------------------------------------------------------------------



def compute_reliability(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    thresholds_mm: list[float],
    num_probability_bins: int = 10,
) -> ReliabilityResult:
    if num_probability_bins <= 0:
        raise ValueError(
            f"num_probability_bins must be positive, got {num_probability_bins}"
        )
    if len(thresholds_mm) == 0:
        raise ValueError("thresholds_mm must not be empty")

    forecast_bmhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)

    bin_edges = np.linspace(0.0, 1.0, num_probability_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    curves: dict[str, dict[str, list[float] | int]] = {}

    for threshold in thresholds_mm:
        all_probabilities: list[np.ndarray] = []
        all_observations: list[np.ndarray] = []

        for b in range(forecast_bmhw.shape[0]):
            forecast_flat, target_flat = _flatten_case(
                forecast_bmhw[b],
                target_bhw[b],
                mask_bhw[b],
            )
            if target_flat.size == 0:
                continue

            exceedance_probability = np.mean(forecast_flat >= threshold, axis=0)
            observed_exceedance = (target_flat >= threshold).astype(np.float64)

            all_probabilities.append(exceedance_probability)
            all_observations.append(observed_exceedance)

        if len(all_probabilities) == 0:
            observed_frequency = np.full((num_probability_bins,), np.nan, dtype=np.float64)
            counts = np.zeros((num_probability_bins,), dtype=int)
        else:
            probs = np.concatenate(all_probabilities, axis=0)
            obs = np.concatenate(all_observations, axis=0)

            counts = np.zeros((num_probability_bins,), dtype=int)
            observed_frequency = np.full((num_probability_bins,), np.nan, dtype=np.float64)

            bin_ids = np.digitize(probs, bin_edges[1:-1], right=False)
            for bin_idx in range(num_probability_bins):
                members = bin_ids == bin_idx
                counts[bin_idx] = int(np.sum(members))
                if counts[bin_idx] > 0:
                    observed_frequency[bin_idx] = float(np.mean(obs[members]))

        curves[str(float(threshold))] = {
            "observed_frequency": observed_frequency.tolist(),
            "counts": counts.astype(int).tolist(),
            "num_samples": int(np.sum(counts)),
        }

    return ReliabilityResult(
        thresholds_mm=[float(x) for x in thresholds_mm],
        bin_centers=bin_centers.astype(float).tolist(),
        curves=curves,
    )