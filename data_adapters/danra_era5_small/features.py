"""
Feature-loading utilities for the small DANRA/ERA5 STRIDE adapter.

Responsibilities:
    - Stack requested dynamic variables in a deterministic order
    - Load static variables
    - Apply source-specific orientation corrections
    - Apply source-specific unit conversions
    - Enforce static-field consistency between land-sea mask and topography
    - Optionally add day-of-year/seasonality metadata later
    - Keep source-specific file-key handling out of the rest of the adapter

This module decides:
    - Channel order for cond_dynamic
    - Channel order for cond_static
    - Variable names stored in metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from data_adapters.danra_era5_small.unit_conversion import apply_unit_conversion
from stride_core.utils.variable_registry import (
    get_source_flip_config,
    get_source_npz_key,
    validate_variables,
)


DEFAULT_FLOAT_DTYPE = np.float32


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
        Canonical STRIDE variable name, e.g. `prcp`, `temp`, `lsm`, `topo`.
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
    Load one target field as a 2D array with shape `[H, W]`.
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
    Load and stack dynamic conditioning variables in deterministic channel order.

    Parameters
    ----------
    dynamic_paths
        Mapping from canonical variable name to file path.
    variable_order
        Ordered iterable specifying channel order, e.g. [`prcp`, `temp`].
    source
        Source name used by the registry, typically `ERA5` for the first experiment.
    dtype
        Output dtype.

    Returns
    -------
    np.ndarray
        Array with shape `[C_dyn, H, W]`.
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


def enforce_static_consistency(
    static_stack: np.ndarray,
    variable_order: list[str],
) -> np.ndarray:
    """
    Enforce simple physical consistency rules between static fields.

    Current rule:
        - If both `lsm` and `topo` are present, set topography to exactly 0.0
          wherever the land-sea mask indicates sea.

    Notes
    -----
    This is intentionally simple for the first STRIDE setup. Because the current
    LSM is binary, masking topography with the land mask is appropriate and also
    removes small ocean artefacts / interpolation noise in the topo field.
    """
    if "lsm" not in variable_order or "topo" not in variable_order:
        return static_stack

    lsm_idx = variable_order.index("lsm")
    topo_idx = variable_order.index("topo")

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
    Load and stack static variables in deterministic channel order.

    Parameters
    ----------
    static_paths
        Mapping from canonical static variable name to file path.
    variable_order
        Ordered iterable specifying static channel order, e.g. [`lsm`, `topo`].
    source
        Source name used by the registry, typically `STATIC`.
    dtype
        Output dtype.

    Returns
    -------
    np.ndarray | None
        Array with shape `[C_static, H, W]`, or None if no static variables are requested.
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
