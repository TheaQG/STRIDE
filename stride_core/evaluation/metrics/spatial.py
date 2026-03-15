"""
Pillar 2 - Spatial structure evaluation.

This module implements spatial diagnostics for STRIDE evaluation. The emphasis is
on precipitation-field structure rather than pointwise probabilistic behavior.

Implemented diagnostics
-----------------------
- Power Spectral Density (PSD), using isotropic radial averaging of the 2D power spectrum
- PSD slope, fitted in log-log space over a configurable wavelength range in km
- Intensity-Scale Skill (ISS), using neighborhood exceedance fractions and an MSE skill score
- SAL (Structure, Amplitude, Location) decomposition

Expected inputs
---------------
Forecast fields:
- [H, W]
- [B, H, W]
- [1, H, W]
- [B, 1, H, W]

Targets:
- same shapes as forecast

Optional masks:
- [H, W]
- [B, H, W]
- [1, H, W]
- [B, 1, H, W]

Notes
-----
- All calculations are NumPy-first and designed for saved evaluation outputs.
- Spatial metrics are deterministic-field diagnostics. They are therefore best
  applied to PMM, ensemble mean, or individual members, depending on the chosen
  evaluation target.
- SAL here uses a clean first implementation aimed at robustness and readability.
  It is appropriate for a first evaluation framework iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Typed outputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PSDCaseResult:
    wavelengths_km: list[float]
    forecast_psd: list[float]
    target_psd: list[float]
    forecast_member_psd: list[list[float]]
    num_members: int
    num_points: int


@dataclass(frozen=True)
class PSDResult:
    per_case: list[PSDCaseResult]
    mean_wavelengths_km: list[float]
    mean_forecast_psd: list[float]
    mean_target_psd: list[float]
    mean_forecast_member_psd: list[list[float]]
    num_members: int
    num_cases: int


@dataclass(frozen=True)
class PSDSlopeResult:
    forecast_slope: float
    target_slope: float
    slope_difference: float
    fit_range_km: list[float]


@dataclass(frozen=True)
class ISSCaseResult:
    thresholds_mm: list[float]
    scales_km: list[int]
    skill_matrix: list[list[float]]


@dataclass(frozen=True)
class ISSResult:
    per_case: list[ISSCaseResult]
    thresholds_mm: list[float]
    scales_km: list[int]
    mean_skill_matrix: list[list[float]]
    num_cases: int


@dataclass(frozen=True)
class SALCaseResult:
    structure: float
    amplitude: float
    location: float


@dataclass(frozen=True)
class SALResult:
    per_case: list[SALCaseResult]
    mean_structure: float
    mean_amplitude: float
    mean_location: float
    num_cases: int


# -----------------------------------------------------------------------------
# Shape normalization helpers
# -----------------------------------------------------------------------------


ArrayLike = np.ndarray



def _as_numpy(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr



def _ensure_bhw(field: ArrayLike, *, name: str) -> np.ndarray:
    arr = _as_numpy(field, name=name)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"4D {name} must have channel dimension 1, got {tuple(arr.shape)}"
            )
        arr = arr[:, 0, ...]
    else:
        raise ValueError(
            f"{name} must have shape [H,W], [B,H,W], [1,H,W], or [B,1,H,W], got {tuple(arr.shape)}"
        )
    if arr.ndim != 3:
        raise ValueError(f"normalized {name} must be [B,H,W], got {tuple(arr.shape)}")
    return arr.astype(np.float64, copy=False)



def _ensure_mask_bhw(
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
    forecast_bhw = _ensure_bhw(forecast, name="forecast")
    target_bhw = _ensure_bhw(target, name="target")

    if forecast_bhw.shape[0] == 1 and target_bhw.shape[0] > 1:
        forecast_bhw = np.repeat(forecast_bhw, target_bhw.shape[0], axis=0)
    if target_bhw.shape[0] == 1 and forecast_bhw.shape[0] > 1:
        target_bhw = np.repeat(target_bhw, forecast_bhw.shape[0], axis=0)

    if forecast_bhw.shape != target_bhw.shape:
        raise ValueError(
            "forecast/target shape mismatch after normalization: "
            f"forecast={tuple(forecast_bhw.shape)}, target={tuple(target_bhw.shape)}"
        )

    batch_size, height, width = forecast_bhw.shape
    mask_bhw = _ensure_mask_bhw(
        mask,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    return forecast_bhw, target_bhw, mask_bhw


# -----------------------------------------------------------------------------
# PSD helpers
# -----------------------------------------------------------------------------



def _apply_mask_fill(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.all(mask):
        return field
    valid_values = field[mask]
    fill_value = float(np.mean(valid_values)) if valid_values.size > 0 else 0.0
    out = field.copy()
    out[~mask] = fill_value
    return out



def _radial_psd(field: np.ndarray, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    if dx_km <= 0:
        raise ValueError(f"dx_km must be positive, got {dx_km}")

    ny, nx = field.shape
    centered = field - np.mean(field)
    fft = np.fft.fft2(centered)
    power = np.abs(fft) ** 2 / float(nx * ny)

    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dx_km)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    kr = np.sqrt(kx_grid**2 + ky_grid**2)

    power = np.fft.fftshift(power)
    kr = np.fft.fftshift(kr)

    positive = kr > 0.0
    kr_pos = kr[positive]
    power_pos = power[positive]
    if kr_pos.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    bin_indices = np.floor(kr_pos / np.min(np.diff(np.unique(np.sort(kr_pos)[: min(10, kr_pos.size)])))) if kr_pos.size > 1 else np.zeros_like(kr_pos)
    # Use a simpler robust scheme: integer bins based on rounded normalized radius.
    delta_k = 1.0 / (max(nx, ny) * dx_km)
    bin_indices = np.floor(kr_pos / delta_k).astype(np.int64)

    unique_bins = np.unique(bin_indices)
    radial_k = np.empty((unique_bins.size,), dtype=np.float64)
    radial_psd = np.empty((unique_bins.size,), dtype=np.float64)

    for i, b in enumerate(unique_bins):
        members = bin_indices == b
        radial_k[i] = float(np.mean(kr_pos[members]))
        radial_psd[i] = float(np.mean(power_pos[members]))

    valid = (radial_k > 0.0) & np.isfinite(radial_psd)
    return radial_k[valid], radial_psd[valid]



def _wavenumber_to_wavelength_km(k: np.ndarray) -> np.ndarray:
    wavelength = np.full_like(k, np.inf, dtype=np.float64)
    positive = k > 0.0
    wavelength[positive] = 1.0 / k[positive]
    return wavelength


# -----------------------------------------------------------------------------
# PSD
# -----------------------------------------------------------------------------



def compute_psd(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    dx_km: float = 1.0,
) -> PSDResult:
    """
    Compute isotropic radial PSD diagnostics with optional ensemble support.

    Supported forecast/target shape combinations
    --------------------------------------------
    Deterministic single case:
        forecast: [H, W]
        target:   [H, W]

    Deterministic batch:
        forecast: [B, H, W] or [B, 1, H, W]
        target:   [B, H, W], [B, 1, H, W], [H, W], or [1, H, W]

    Ensemble single case:
        forecast: [M, H, W]
        target:   [H, W] or [1, H, W]

    Ensemble batch:
        forecast: [B, M, H, W]
        target:   [B, H, W], [B, 1, H, W], [H, W], or [1, H, W]

    Notes
    -----
    - Each case stores PSD for every ensemble member separately.
    - `forecast_psd` is the per-case ensemble-mean PSD.
    - `mean_forecast_psd` is the mean of those per-case ensemble-mean PSDs.
    - `mean_forecast_member_psd` preserves a member axis across cases using the
      minimum common member count.
    """

    def _normalize_target_cases(
        target_in: ArrayLike,
        *,
        batch_size: int,
    ) -> np.ndarray:
        arr = _as_numpy(target_in, name="target")

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
                "target must have shape [H,W], [B,H,W], [1,H,W], or [B,1,H,W], "
                f"got {tuple(arr.shape)}"
            )

        if arr.ndim != 3:
            raise ValueError(f"normalized target must be [B,H,W], got {tuple(arr.shape)}")

        if arr.shape[0] == 1 and batch_size > 1:
            arr = np.repeat(arr, batch_size, axis=0)

        if arr.shape[0] != batch_size:
            raise ValueError(
                "target batch size mismatch after normalization: "
                f"expected {batch_size}, got {arr.shape[0]}"
            )

        return arr.astype(np.float64, copy=False)

    def _normalize_mask_cases(
        mask_in: ArrayLike | None,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        return _ensure_mask_bhw(
            mask_in,
            batch_size=batch_size,
            height=height,
            width=width,
        )

    forecast_arr = _as_numpy(forecast, name="forecast")

    if forecast_arr.ndim == 2:
        # One deterministic case with one member.
        forecast_bmhw = forecast_arr[np.newaxis, np.newaxis, ...]
    elif forecast_arr.ndim == 3:
        target_arr = _as_numpy(target, name="target")

        # Ensemble single case: forecast [M,H,W], target [H,W] or [1,H,W].
        if target_arr.ndim == 2 or (target_arr.ndim == 3 and target_arr.shape[0] == 1):
            forecast_bmhw = forecast_arr[np.newaxis, ...]
        else:
            # Deterministic batch: forecast [B,H,W].
            forecast_bmhw = forecast_arr[:, np.newaxis, ...]
    elif forecast_arr.ndim == 4:
        # Ensemble batch: forecast [B,M,H,W].
        forecast_bmhw = forecast_arr
    elif forecast_arr.ndim == 5:
        # Optional channelized ensemble batch: [B,M,1,H,W].
        if forecast_arr.shape[2] != 1:
            raise ValueError(
                "5D forecast must have singleton channel dimension at axis 2, "
                f"got {tuple(forecast_arr.shape)}"
            )
        forecast_bmhw = forecast_arr[:, :, 0, ...]
    else:
        raise ValueError(
            "forecast must have shape [H,W], [M,H,W], [B,H,W], [B,M,H,W], or "
            f"[B,M,1,H,W], got {tuple(forecast_arr.shape)}"
        )

    if forecast_bmhw.ndim != 4:
        raise ValueError(
            f"normalized forecast must be [B,M,H,W], got {tuple(forecast_bmhw.shape)}"
        )

    forecast_bmhw = forecast_bmhw.astype(np.float64, copy=False)
    batch_size, num_members_in, height, width = forecast_bmhw.shape

    target_bhw = _normalize_target_cases(target, batch_size=batch_size)
    if target_bhw.shape[1:] != (height, width):
        raise ValueError(
            "forecast/target spatial shape mismatch after normalization: "
            f"forecast={(height, width)}, target={tuple(target_bhw.shape[1:])}"
        )

    mask_bhw = _normalize_mask_cases(
        mask,
        batch_size=batch_size,
        height=height,
        width=width,
    )

    per_case: list[PSDCaseResult] = []
    forecast_psd_list: list[np.ndarray] = []
    target_psd_list: list[np.ndarray] = []
    forecast_member_psd_list: list[np.ndarray] = []
    wavelength_list: list[np.ndarray] = []

    for b in range(batch_size):
        target_field = _apply_mask_fill(target_bhw[b], mask_bhw[b])
        k_t, psd_t = _radial_psd(target_field, dx_km=dx_km)
        if k_t.size == 0:
            per_case.append(
                PSDCaseResult(
                    wavelengths_km=[],
                    forecast_psd=[],
                    target_psd=[],
                    forecast_member_psd=[],
                    num_members=int(num_members_in),
                    num_points=0,
                )
            )
            continue

        member_curves: list[np.ndarray] = []
        member_wavelength_ref: np.ndarray | None = None
        target_curve_ref: np.ndarray | None = None

        for m in range(num_members_in):
            forecast_field = _apply_mask_fill(forecast_bmhw[b, m], mask_bhw[b])
            k_f, psd_f = _radial_psd(forecast_field, dx_km=dx_km)
            if k_f.size == 0:
                continue

            n = min(k_f.size, k_t.size)
            if n == 0:
                continue

            wavelength = _wavenumber_to_wavelength_km(k_f[:n])
            forecast_vals = psd_f[:n]
            target_vals = psd_t[:n]

            mask_valid = (
                np.isfinite(wavelength)
                & np.isfinite(forecast_vals)
                & np.isfinite(target_vals)
                & (wavelength > 0.0)
                & (forecast_vals > 0.0)
                & (target_vals > 0.0)
            )
            if not np.any(mask_valid):
                continue

            wavelength = wavelength[mask_valid]
            forecast_vals = forecast_vals[mask_valid]
            target_vals = target_vals[mask_valid]

            if member_wavelength_ref is None:
                member_wavelength_ref = wavelength
                target_curve_ref = target_vals
                member_curves.append(forecast_vals)
            else:
                n_common = min(
                    member_wavelength_ref.size,
                    wavelength.size,
                    target_curve_ref.size if target_curve_ref is not None else wavelength.size,
                )
                if n_common == 0:
                    continue

                member_wavelength_ref = member_wavelength_ref[:n_common]
                if target_curve_ref is not None:
                    target_curve_ref = target_curve_ref[:n_common]
                member_curves = [curve[:n_common] for curve in member_curves]
                member_curves.append(forecast_vals[:n_common])

        if member_wavelength_ref is None or target_curve_ref is None or len(member_curves) == 0:
            per_case.append(
                PSDCaseResult(
                    wavelengths_km=[],
                    forecast_psd=[],
                    target_psd=[],
                    forecast_member_psd=[],
                    num_members=0,
                    num_points=0,
                )
            )
            continue

        member_psd = np.stack(member_curves, axis=0)
        forecast_mean = np.mean(member_psd, axis=0)

        per_case.append(
            PSDCaseResult(
                wavelengths_km=member_wavelength_ref.astype(float).tolist(),
                forecast_psd=forecast_mean.astype(float).tolist(),
                target_psd=target_curve_ref.astype(float).tolist(),
                forecast_member_psd=member_psd.astype(float).tolist(),
                num_members=int(member_psd.shape[0]),
                num_points=int(member_psd.shape[1]),
            )
        )

        wavelength_list.append(member_wavelength_ref)
        forecast_psd_list.append(forecast_mean)
        target_psd_list.append(target_curve_ref)
        forecast_member_psd_list.append(member_psd)

    if len(forecast_psd_list) == 0:
        return PSDResult(
            per_case=per_case,
            mean_wavelengths_km=[],
            mean_forecast_psd=[],
            mean_target_psd=[],
            mean_forecast_member_psd=[],
            num_members=0,
            num_cases=batch_size,
        )

    min_len = min(arr.size for arr in wavelength_list + forecast_psd_list + target_psd_list)
    mean_wavelength = np.mean(
        np.stack([arr[:min_len] for arr in wavelength_list], axis=0),
        axis=0,
    )
    mean_forecast = np.mean(
        np.stack([arr[:min_len] for arr in forecast_psd_list], axis=0),
        axis=0,
    )
    mean_target = np.mean(
        np.stack([arr[:min_len] for arr in target_psd_list], axis=0),
        axis=0,
    )

    member_counts = [arr.shape[0] for arr in forecast_member_psd_list]
    common_num_members = min(member_counts) if len(member_counts) > 0 else 0
    if common_num_members > 0:
        mean_member_psd = np.mean(
            np.stack(
                [arr[:common_num_members, :min_len] for arr in forecast_member_psd_list],
                axis=0,
            ),
            axis=0,
        )
        mean_member_psd_out = mean_member_psd.astype(float).tolist()
    else:
        mean_member_psd_out = []

    return PSDResult(
        per_case=per_case,
        mean_wavelengths_km=mean_wavelength.astype(float).tolist(),
        mean_forecast_psd=mean_forecast.astype(float).tolist(),
        mean_target_psd=mean_target.astype(float).tolist(),
        mean_forecast_member_psd=mean_member_psd_out,
        num_members=int(common_num_members),
        num_cases=batch_size,
    )



def compute_psd_slope(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    dx_km: float = 1.0,
    fit_range_km: tuple[float, float] | list[float] = (5.0, 20.0),
) -> PSDSlopeResult:
    fit_range = tuple(float(x) for x in fit_range_km)
    if len(fit_range) != 2:
        raise ValueError(f"fit_range_km must contain exactly two values, got {fit_range_km}")
    wl_min, wl_max = fit_range
    if wl_min <= 0 or wl_max <= 0 or wl_max <= wl_min:
        raise ValueError(f"Invalid fit_range_km: {fit_range}")

    psd = compute_psd(forecast, target, mask=mask, dx_km=dx_km)
    wavelengths = np.asarray(psd.mean_wavelengths_km, dtype=np.float64)
    forecast_psd = np.asarray(psd.mean_forecast_psd, dtype=np.float64)
    target_psd = np.asarray(psd.mean_target_psd, dtype=np.float64)

    band = (
        np.isfinite(wavelengths)
        & (wavelengths >= wl_min)
        & (wavelengths <= wl_max)
        & (forecast_psd > 0.0)
        & (target_psd > 0.0)
    )
    if int(np.sum(band)) < 2:
        return PSDSlopeResult(
            forecast_slope=float("nan"),
            target_slope=float("nan"),
            slope_difference=float("nan"),
            fit_range_km=[wl_min, wl_max],
        )

    log_w = np.log(wavelengths[band])
    slope_f, _ = np.polyfit(log_w, np.log(forecast_psd[band]), 1)
    slope_t, _ = np.polyfit(log_w, np.log(target_psd[band]), 1)

    return PSDSlopeResult(
        forecast_slope=float(slope_f),
        target_slope=float(slope_t),
        slope_difference=float(slope_f - slope_t),
        fit_range_km=[wl_min, wl_max],
    )


# -----------------------------------------------------------------------------
# ISS helpers
# -----------------------------------------------------------------------------



def _window_size_from_scale(scale_km: int, dx_km: float) -> int:
    if scale_km <= 0:
        raise ValueError(f"scale_km must be positive, got {scale_km}")
    size = int(round(scale_km / dx_km))
    return max(size, 1)



def _integral_image(a: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))



def _box_mean(field: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return field.astype(np.float64, copy=False)

    ny, nx = field.shape
    pad_before = window // 2
    pad_after = window - 1 - pad_before
    padded = np.pad(field, ((pad_before, pad_after), (pad_before, pad_after)), mode="edge")
    ii = _integral_image(padded)

    y0 = np.arange(ny)
    x0 = np.arange(nx)
    y1 = y0 + window
    x1 = x0 + window

    out = np.empty((ny, nx), dtype=np.float64)
    for iy in range(ny):
        yy0 = y0[iy]
        yy1 = y1[iy]
        row_sum = ii[yy1, x1] - ii[yy0, x1] - ii[yy1, x0] + ii[yy0, x0]
        out[iy, :] = row_sum / float(window * window)
    return out


# -----------------------------------------------------------------------------
# ISS
# -----------------------------------------------------------------------------



def compute_iss(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    thresholds_mm: list[float],
    scales_km: list[int],
    dx_km: float = 1.0,
) -> ISSResult:
    if len(thresholds_mm) == 0:
        raise ValueError("thresholds_mm must not be empty")
    if len(scales_km) == 0:
        raise ValueError("scales_km must not be empty")

    forecast_bhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    per_case: list[ISSCaseResult] = []
    all_case_mats: list[np.ndarray] = []

    for b in range(forecast_bhw.shape[0]):
        f = forecast_bhw[b]
        t = target_bhw[b]
        m = mask_bhw[b]

        mat = np.full((len(thresholds_mm), len(scales_km)), np.nan, dtype=np.float64)
        for i_thr, thr in enumerate(thresholds_mm):
            f_bin = (f >= float(thr)).astype(np.float64)
            t_bin = (t >= float(thr)).astype(np.float64)

            for i_scale, scale in enumerate(scales_km):
                window = _window_size_from_scale(int(scale), dx_km)
                f_frac = _box_mean(f_bin, window)
                t_frac = _box_mean(t_bin, window)

                if int(np.sum(m)) == 0:
                    continue

                f_valid = f_frac[m]
                t_valid = t_frac[m]
                mse = float(np.mean((f_valid - t_valid) ** 2))
                ref = float(np.mean(f_valid**2 + t_valid**2))
                if ref <= 0.0:
                    skill = 1.0 if mse == 0.0 else float("nan")
                else:
                    skill = 1.0 - mse / ref
                mat[i_thr, i_scale] = skill

        per_case.append(
            ISSCaseResult(
                thresholds_mm=[float(x) for x in thresholds_mm],
                scales_km=[int(x) for x in scales_km],
                skill_matrix=mat.astype(float).tolist(),
            )
        )
        all_case_mats.append(mat)

    if len(all_case_mats) == 0:
        mean_mat = np.full((len(thresholds_mm), len(scales_km)), np.nan, dtype=np.float64)
    else:
        mean_mat = np.nanmean(np.stack(all_case_mats, axis=0), axis=0)

    return ISSResult(
        per_case=per_case,
        thresholds_mm=[float(x) for x in thresholds_mm],
        scales_km=[int(x) for x in scales_km],
        mean_skill_matrix=mean_mat.astype(float).tolist(),
        num_cases=forecast_bhw.shape[0],
    )


# -----------------------------------------------------------------------------
# SAL helpers
# -----------------------------------------------------------------------------



def _object_mask(field: np.ndarray, threshold: float) -> np.ndarray:
    return field >= threshold



def _center_of_mass(field: np.ndarray) -> np.ndarray:
    total = float(np.sum(field))
    if total <= 0.0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    y_idx, x_idx = np.indices(field.shape)
    y = float(np.sum(y_idx * field) / total)
    x = float(np.sum(x_idx * field) / total)
    return np.array([y, x], dtype=np.float64)



def _domain_diagonal(shape: tuple[int, ...]) -> float:
    ny, nx = shape
    return float(np.sqrt((ny - 1) ** 2 + (nx - 1) ** 2))



def _structure_stat(field: np.ndarray, threshold: float) -> float:
    obj = _object_mask(field, threshold)
    if not np.any(obj):
        return 0.0
    obj_vals = field[obj]
    peak = float(np.max(obj_vals))
    if peak <= 0.0:
        return 0.0
    return float(np.mean(obj_vals) / peak)


# -----------------------------------------------------------------------------
# SAL
# -----------------------------------------------------------------------------



def compute_sal(
    forecast: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    threshold_mm: float = 1.0,
) -> SALResult:
    forecast_bhw, target_bhw, mask_bhw = _normalize_inputs(forecast, target, mask)
    per_case: list[SALCaseResult] = []

    for b in range(forecast_bhw.shape[0]):
        f = np.where(mask_bhw[b], forecast_bhw[b], 0.0)
        t = np.where(mask_bhw[b], target_bhw[b], 0.0)

        mean_f = float(np.mean(f[mask_bhw[b]])) if np.any(mask_bhw[b]) else 0.0
        mean_t = float(np.mean(t[mask_bhw[b]])) if np.any(mask_bhw[b]) else 0.0
        denom_amp = 0.5 * (mean_f + mean_t)
        amplitude = 0.0 if denom_amp == 0.0 else (mean_f - mean_t) / denom_amp

        struct_f = _structure_stat(f, threshold_mm)
        struct_t = _structure_stat(t, threshold_mm)
        denom_struct = 0.5 * (struct_f + struct_t)
        structure = 0.0 if denom_struct == 0.0 else (struct_f - struct_t) / denom_struct

        com_f = _center_of_mass(f)
        com_t = _center_of_mass(t)
        diag = _domain_diagonal(f.shape)
        if not np.all(np.isfinite(com_f)) or not np.all(np.isfinite(com_t)) or diag <= 0.0:
            location = float("nan")
        else:
            location = float(np.linalg.norm(com_f - com_t) / diag)

        per_case.append(
            SALCaseResult(
                structure=float(structure),
                amplitude=float(amplitude),
                location=float(location),
            )
        )

    if len(per_case) == 0:
        return SALResult(
            per_case=[],
            mean_structure=float("nan"),
            mean_amplitude=float("nan"),
            mean_location=float("nan"),
            num_cases=0,
        )

    structure_vals = np.asarray([x.structure for x in per_case], dtype=np.float64)
    amplitude_vals = np.asarray([x.amplitude for x in per_case], dtype=np.float64)
    location_vals = np.asarray([x.location for x in per_case], dtype=np.float64)

    return SALResult(
        per_case=per_case,
        mean_structure=float(np.nanmean(structure_vals)),
        mean_amplitude=float(np.nanmean(amplitude_vals)),
        mean_location=float(np.nanmean(location_vals)),
        num_cases=len(per_case),
    )