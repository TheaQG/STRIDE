"""
Feature-loading utilities for the small DANRA/ERA5 STRIDE adapter.

Responsibilities:
    - Load full-domain target and conditioning fields before any spatial crop
    - Stack requested dynamic variables in a deterministic order
    - Load requested static variables
    - Apply source-specific orientation corrections
    - Apply source-specific unit conversions
    - Optionally enforce simple consistency rules for selected static fields
    - Build day-of-year sin/cos features from YYYYMMDD date strings
    - Keep source-specific file-key handling out of the rest of the adapter

This module decides:
    - Channel order for cond_dynamic
    - Channel order for cond_static
    - Variable names stored in metadata

This module does not decide:
    - Fixed vs shuffled crop anchors
    - Spatial region selection
    - Per-sample crop metadata

Those responsibilities belong to the adapter / region utilities.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from data_adapters.danra_era5_small.unit_conversion import apply_unit_conversion
from data_adapters.danra_era5_small.variable_registry import (
    get_source_flip_config,
    get_source_npz_key,
    validate_variables,
)



DEFAULT_FLOAT_DTYPE = np.float32


def parse_yyyymmdd(date_str: str) -> datetime:
    """
    Parse a compact YYYYMMDD string into a datetime object.

    Parameters
    ----------
    date_str
        Date string in YYYYMMDD format.
    """
    if not isinstance(date_str, str):
        raise TypeError(f"Expected date_str to be a string, got {type(date_str)}")

    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(
            f"Expected date string in YYYYMMDD format, got '{date_str}'"
        )

    return datetime.strptime(date_str, "%Y%m%d")


def is_leap_year(year: int) -> bool:
    """
    Return True if `year` is a leap year in the Gregorian calendar.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def compute_day_of_year(date_str: str) -> int:
    """
    Compute the 1-indexed day-of-year from a YYYYMMDD string.

    Returns
    -------
    int
        Day of year in the interval [1, 365] or [1, 366] for leap years.
    """
    dt = parse_yyyymmdd(date_str)
    return int(dt.timetuple().tm_yday)


def build_doy_sincos(
    date_str: str,
    *,
    use_leap_years: bool = False,
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE,  # type: ignore
) -> np.ndarray:
    """
    Build a continuous day-of-year sin/cos encoding from a YYYYMMDD string.

    Parameters
    ----------
    date_str
        Date string in YYYYMMDD format.
    use_leap_years
        If True, use 366 days for leap years and 365 otherwise.
        If False, fold Feb 29 onto Feb 28 so the embedding always uses a
        fixed 365-day cycle.
    dtype
        Output floating dtype.

    Returns
    -------
    np.ndarray
        Array with shape [2] containing [sin(theta), cos(theta)].
    """
    dt = parse_yyyymmdd(date_str)
    doy = compute_day_of_year(date_str)

    if use_leap_years:
        period = 366 if is_leap_year(dt.year) else 365
    else:
        # Fold Feb 29 onto Feb 28 to keep a stable 365-day cycle.
        if is_leap_year(dt.year) and dt.month == 2 and dt.day == 29:
            doy = 59
        elif is_leap_year(dt.year) and doy > 60:
            doy -= 1
        period = 365

    theta = 2.0 * math.pi * float(doy - 1) / float(period)
    return np.asarray([math.sin(theta), math.cos(theta)], dtype=dtype)


def apply_orientation(
    array: np.ndarray,
    variable: str,
    source: str,
) -> np.ndarray:
    """
    Apply source-specific orientation corrections so all loaded fields share
    the same spatial convention inside STRIDE.
    """
    flip_ud, flip_lr = get_source_flip_config(variable, source)

    corrected = array
    if flip_ud:
        corrected = np.flipud(corrected)
    if flip_lr:
        corrected = np.fliplr(corrected)

    return corrected


def load_npz_array(
    file_path: str | Path,
    variable: str,
    source: str,
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE, # type: ignore
) -> np.ndarray:
    """
    Load a single 2D field from an `.npz` file using the variable registry.

    Parameters
    ----------
    file_path
        Path to the `.npz` file.
    variable
        Canonical STRIDE variable name registered in the variable registry,
        e.g. `prcp`, `temp`, `cape`, `msl`, `z_pl_500`, `lsm`, `topo`.
    source
        Source name used by the variable registry, e.g. `DANRA`, `ERA5`, `STATIC`.
    dtype
        Output dtype. Everything should become float32 for downstream consistency.

    Returns
    -------
    np.ndarray
        A 2D array with shape `[H, W]`.
    """
    validate_variables([variable])

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    npz_key = get_source_npz_key(variable, source)
    if npz_key is None:
        raise ValueError(
            f"No npz key registered for variable='{variable}', source='{source}'"
        )

    with np.load(path, allow_pickle=False) as data:
        if npz_key not in data:
            raise KeyError(
                f"Expected key '{npz_key}' not found in file '{path}'. "
                f"Available keys: {list(data.keys())}"
            )
        array = data[npz_key]

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D field for variable='{variable}', source='{source}', "
            f"but got shape {array.shape} from file '{path}'"
        )
    array = apply_orientation(array=array, variable=variable, source=source)
    array = apply_unit_conversion(array=array, variable=variable, source=source)

    return np.asarray(array, dtype=dtype)


def load_target_field(
    target_path: str | Path,
    target_variable: str,
    target_source: str = "DANRA",
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE, # type: ignore
) -> np.ndarray:
    """
    Load one full-domain target field as a 2D array with shape `[H, W]`.

    Cropping is intentionally not performed here. Spatial region selection is
    handled later by the adapter / region utilities so fixed crops and
    train-time spatial shuffling use the same raw field loading path.
    """
    return load_npz_array(
        file_path=target_path,
        variable=target_variable,
        source=target_source,
        dtype=dtype,
    )


def load_dynamic_conditioning(
    dynamic_paths: dict[str, str | Path],
    variable_order: Iterable[str],
    source: str = "ERA5",
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE, # type: ignore
) -> np.ndarray:
    """
    Load and stack full-domain dynamic conditioning variables in deterministic
    channel order.

    Parameters
    ----------
    dynamic_paths
        Mapping from canonical variable name to file path.
    variable_order
        Ordered iterable specifying channel order, e.g. [`prcp`, `temp`,
        `cape`, `msl`, `z_pl_500`].
    source
        Source name used by the registry, typically `ERA5` for the first experiment.
    dtype
        Output dtype.

    Returns
    -------
    np.ndarray
        Full-domain array with shape `[C_dyn, H, W]`.

    Notes
    -----
    No cropping is performed here. Spatial cropping (fixed or shuffled) is
    applied later in the adapter so all sample fields remain aligned.
    """
    ordered_variables = list(variable_order)
    validate_variables(ordered_variables)

    missing = [var for var in ordered_variables if var not in dynamic_paths]
    if missing:
        raise KeyError(
            f"Missing dynamic conditioning paths for variables: {missing}. "
            f"Available: {list(dynamic_paths.keys())}"
        )

    channels: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None

    for variable in ordered_variables:
        array = load_npz_array(
            file_path=dynamic_paths[variable],
            variable=variable,
            source=source,
            dtype=dtype,
        )

        if reference_shape is None:
            reference_shape = tuple(array.shape) # type: ignore
        elif array.shape != reference_shape:
            raise ValueError(
                f"Dynamic conditioning variables must share the same shape. "
                f"Expected {reference_shape}, got {array.shape} for variable '{variable}'"
            )

        channels.append(array)

    return np.stack(channels, axis=0)


def _find_first_present(
    candidates: Iterable[str],
    variable_order: list[str],
) -> int | None:
    """
    Return the index of the first candidate variable present in `variable_order`.

    This keeps the consistency logic tolerant to small naming differences such as
    `topo` vs `topography`.
    """
    for candidate in candidates:
        if candidate in variable_order:
            return variable_order.index(candidate)
    return None


def enforce_static_consistency(
    static_stack: np.ndarray,
    variable_order: list[str],
) -> np.ndarray:
    """
    Enforce simple physical consistency rules between selected static fields.

    Current rule:
        - If both a land-sea mask and a topography field are present, set
          topography to exactly 0.0 wherever the mask indicates sea.

    Notes
    -----
    This is intentionally conservative. The function should remain safe for
    larger variable sets, so it only applies a correction when the relevant
    fields are explicitly present.
    """
    lsm_idx = _find_first_present(["lsm", "land_sea_mask"], variable_order)
    topo_idx = _find_first_present(["topo", "topography", "orog"], variable_order)

    if lsm_idx is None or topo_idx is None:
        return static_stack

    corrected = static_stack.copy()
    land_mask = corrected[lsm_idx] > 0.5
    corrected[topo_idx] = np.where(land_mask, corrected[topo_idx], 0.0)

    return corrected


def load_static_features(
    static_paths: dict[str, str | Path],
    variable_order: Iterable[str],
    source: str = "STATIC",
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE, # type: ignore
) -> np.ndarray | None:
    """
    Load and stack full-domain static variables in deterministic channel order.

    Parameters
    ----------
    static_paths
        Mapping from canonical static variable name to file path.
    variable_order
        Ordered iterable specifying static channel order, e.g. [`lsm`, `topo`].
        The loader itself is generic and can support additional static fields as
        long as they are registered and have discoverable file paths.
    source
        Source name used by the registry, typically `STATIC`.
    dtype
        Output dtype.

    Returns
    -------
    np.ndarray | None
        Full-domain array with shape `[C_static, H, W]`, or None if no static
        variables are requested.

    Notes
    -----
    No cropping is performed here. Spatial cropping (fixed or shuffled) is
    applied later in the adapter so all sample fields remain aligned.
    """
    ordered_variables = list(variable_order)
    if not ordered_variables:
        return None

    validate_variables(ordered_variables)

    missing = [var for var in ordered_variables if var not in static_paths]
    if missing:
        raise KeyError(
            f"Missing static paths for variables: {missing}. "
            f"Available: {list(static_paths.keys())}"
        )

    channels: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None

    for variable in ordered_variables:
        array = load_npz_array(
            file_path=static_paths[variable],
            variable=variable,
            source=source,
            dtype=dtype,
        )

        if reference_shape is None:
            reference_shape = tuple(array.shape) # type: ignore
        elif array.shape != reference_shape:
            raise ValueError(
                f"Static variables must share the same shape. "
                f"Expected {reference_shape}, got {array.shape} for variable '{variable}'"
            )

        channels.append(array)

    static_stack = np.stack(channels, axis=0)
    static_stack = enforce_static_consistency(
        static_stack=static_stack,
        variable_order=ordered_variables,
    )
    return static_stack



def build_feature_metadata(
    target_variable: str,
    dynamic_variables: Iterable[str],
    static_variables: Iterable[str] | None = None,
) -> dict[str, list[str] | str]:
    """
    Build minimal metadata describing variable ordering for one sample.

    This should stay lightweight. Richer metadata (dates, domains, scaling state,
    transforms, coordinates) belongs in the adapter once full sample assembly exists.
    """
    dynamic_variables = list(dynamic_variables)
    static_variables = list(static_variables) if static_variables is not None else []

    validate_variables([target_variable, *dynamic_variables, *static_variables])

    return {
        "target_var": target_variable,
        "cond_dynamic_vars": dynamic_variables,
        "cond_static_vars": static_variables,
    }


def build_time_feature_metadata(
    date_str: str,
    *,
    use_leap_years: bool = False,
    dtype: np.dtype = DEFAULT_FLOAT_DTYPE,  # type: ignore
) -> dict[str, int | list[float] | str]:
    """
    Build lightweight temporal metadata for one sample.

    Parameters
    ----------
    date_str
        Date string in YYYYMMDD format.
    use_leap_years
        Whether to preserve a 366-day cycle in leap years.
    dtype
        Floating dtype for the returned sin/cos features.

    Returns
    -------
    dict
        Metadata dictionary containing the original date string, the integer
        day-of-year, and a JSON-friendly sin/cos encoding.
    """
    doy = compute_day_of_year(date_str)
    doy_sincos = build_doy_sincos(
        date_str,
        use_leap_years=use_leap_years,
        dtype=dtype,
    )
    return {
        "date": date_str,
        "day_of_year": doy,
        "doy_sin_cos": [float(doy_sincos[0]), float(doy_sincos[1])],
    }
