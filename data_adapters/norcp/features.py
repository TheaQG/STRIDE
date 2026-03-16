"""
Feature-loading utilities for the NorCP STRIDE adapter.

This module is responsible for loading NorCP fields from NetCDF files at a
specific timestamp, applying source-aware orientation handling and unit
conversion, stacking variables in deterministic channel order, and building
basic temporal feature vectors.

Scope
-----
This module intentionally does not:
- discover files
- build timestamp/sample indexes
- perform spatial cropping
- perform LR->HR resampling

Static fields such as NorCP orography are treated as fixed fields loaded once
per file without any timestamp selection. Static inputs are optional so the
adapter can run for future projections that do not provide orography.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from data_adapters.norcp.unit_conversion import apply_unit_conversion
from data_adapters.norcp.variable_registry import (
    canonicalize_variable_name,
    get_source_flip_config,
    get_source_raw_field_name,
)


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedField:
    """
    Container for one loaded 2D field and lightweight metadata.
    """

    variable: str
    source: str
    timestamp: Optional[datetime]
    array: np.ndarray
    raw_field_name: str
    path: Path


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def _to_python_datetime(value: object) -> datetime:
    """
    Convert decoded xarray/pandas/numpy time values to a Python datetime.
    """
    ts = pd.Timestamp(value) # type: ignore
    if pd.isna(ts):
        raise ValueError(f"Encountered invalid timestamp value: {value!r}")
    return ts.to_pydatetime()



def _apply_orientation(
    array: np.ndarray,
    *,
    flip_ud: bool,
    flip_lr: bool,
) -> np.ndarray:
    """
    Apply source-specific orientation handling.
    """
    out = np.asarray(array)
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return np.asarray(out)



def _ensure_2d(array: np.ndarray, *, variable: str, path: Path) -> np.ndarray:
    """
    Validate that a loaded field is 2D.
    """
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D field for variable='{variable}' from file '{path}', "
            f"got shape {array.shape}"
        )
    return array


def _resolve_file_timestamp(
    timestamp: datetime,
    *,
    offset_hours: float = 0.0,
) -> datetime:
    """
    Convert an aligned sample timestamp back to the raw file timestamp.

    The indexing layer may shift timestamps to align variables with different
    representative times, e.g. precipitation means at 03/09/15/21 shifted by
    -3 hours to align with point variables at 00/06/12/18. When loading from
    file, we therefore need to reverse that shift.

    Example
    -------
    aligned sample timestamp = 00:00
    indexing offset          = -3 hours
    raw file timestamp       = 03:00
    """
    if offset_hours == 0.0:
        return timestamp

    # Reverse the indexing shift. If indexing used -3 hours, loading must use +3 hours.
    return timestamp - timedelta(hours=float(offset_hours))


def _select_time_slice(
    data_array: xr.DataArray,
    timestamp: datetime,
    *,
    time_variable: str = "time",
) -> xr.DataArray:
    """
    Select one timestep by exact decoded timestamp.
    """
    if time_variable not in data_array.coords and time_variable not in data_array.dims:
        raise KeyError(
            f"Time variable '{time_variable}' not present in DataArray coords/dims: "
            f"dims={data_array.dims}, coords={list(data_array.coords)}"
        )

    times = [_to_python_datetime(value) for value in data_array[time_variable].values]
    try:
        index = times.index(timestamp)
    except ValueError as exc:
        preview = [t.isoformat() for t in times[:5]]
        raise KeyError(
            f"Timestamp {timestamp.isoformat()} not found in file time coordinate. "
            f"First available timestamps: {preview}"
        ) from exc

    return data_array.isel({time_variable: index})


# -------------------------------------------------------------------
# Core loaders
# -------------------------------------------------------------------

def load_field_at_timestamp(
    file_path: str | Path,
    variable: str,
    source: str,
    timestamp: datetime,
    *,
    time_variable: str = "time",
    file_time_offset_hours: float = 0.0,
) -> LoadedField:
    """
    Load one 2D field at a specific timestamp and convert it to canonical units.

    Parameters
    ----------
    file_path:
        NetCDF file containing the variable.
    variable:
        Canonical STRIDE variable name or level-specific NorCP name, e.g.
        'prcp', 'temp', 'hus1000', 'ta850'.
    source:
        Source label such as 'NORCP_HR' or 'NORCP_LR'.
    timestamp:
        Timestamp to select.
    time_variable:
        Name of the time coordinate.
    file_time_offset_hours:
        Offset in hours that was applied during indexing for this variable.
        This function reverses that offset when selecting the raw file time.
    """
    path = Path(file_path)
    raw_field_name = get_source_raw_field_name(variable, source)
    if raw_field_name is None:
        raise ValueError(
            f"No raw field name registered for variable='{variable}', source='{source}'"
        )

    flip_ud, flip_lr = get_source_flip_config(variable, source)
    file_timestamp = _resolve_file_timestamp(timestamp, offset_hours=file_time_offset_hours)

    with xr.open_dataset(path, decode_times=True) as ds:
        if raw_field_name not in ds.data_vars:
            raise KeyError(
                f"Raw field '{raw_field_name}' not found in file '{path}'. "
                f"Available variables: {list(ds.data_vars)}"
            )
        da = ds[raw_field_name]
        da = _select_time_slice(da, timestamp=file_timestamp, time_variable=time_variable)
        values = da.values

    values = np.asarray(values, dtype=np.float32)
    values = _ensure_2d(values, variable=variable, path=path)
    values = _apply_orientation(values, flip_ud=flip_ud, flip_lr=flip_lr)
    values = apply_unit_conversion(values, variable=variable, source=source)

    return LoadedField(
        variable=variable,
        source=source,
        timestamp=file_timestamp,
        array=values,
        raw_field_name=raw_field_name,
        path=path,
    )



def load_static_field(
    file_path: str | Path,
    variable: str,
    source: str,
) -> LoadedField:
    """
    Load one static 2D field and convert it to canonical units.
    """
    path = Path(file_path)
    raw_field_name = get_source_raw_field_name(variable, source)
    if raw_field_name is None:
        raise ValueError(
            f"No raw field name registered for variable='{variable}', source='{source}'"
        )

    flip_ud, flip_lr = get_source_flip_config(variable, source)

    with xr.open_dataset(path, decode_times=True) as ds:
        if raw_field_name not in ds.data_vars:
            raise KeyError(
                f"Raw field '{raw_field_name}' not found in file '{path}'. "
                f"Available variables: {list(ds.data_vars)}"
            )
        da = ds[raw_field_name]

        # Some static fields can still carry singleton dimensions such as time/height.
        squeeze_dims = {
            dim: 0
            for dim, size in da.sizes.items()
            if dim not in {"y", "x", "lat", "lon"} and size == 1
        }
        if squeeze_dims:
            da = da.isel(squeeze_dims)

        values = da.values

    values = np.asarray(values, dtype=np.float32)
    values = np.squeeze(values)
    values = _ensure_2d(values, variable=variable, path=path)
    values = _apply_orientation(values, flip_ud=flip_ud, flip_lr=flip_lr)
    values = apply_unit_conversion(values, variable=variable, source=source)

    return LoadedField(
        variable=variable,
        source=source,
        timestamp=None,
        array=values,
        raw_field_name=raw_field_name,
        path=path,
    )


# -------------------------------------------------------------------
# Stack builders
# -------------------------------------------------------------------

def _stack_loaded_fields(fields: Iterable[LoadedField]) -> np.ndarray:
    """
    Stack loaded 2D fields into channel-first [C, H, W] format.
    """
    arrays = [field.array for field in fields]
    if not arrays:
        raise ValueError("Cannot stack zero loaded fields")

    first_shape = arrays[0].shape
    for idx, array in enumerate(arrays[1:], start=1):
        if array.shape != first_shape:
            raise ValueError(
                f"All fields must share the same shape before stacking. "
                f"First shape={first_shape}, field {idx} shape={array.shape}"
            )

    return np.stack(arrays, axis=0).astype(np.float32, copy=False)



def load_target_stack(
    target_files: dict[str, str | Path],
    timestamp: datetime,
    *,
    source: str = "NORCP_HR",
    variable_order: Optional[Iterable[str]] = None,
    time_variable: str = "time",
    variable_time_offsets: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, list[LoadedField]]:
    """
    Load and stack HR target variables as [C, H, W].
    """
    ordered_variables = list(variable_order) if variable_order is not None else list(target_files.keys())
    variable_time_offsets = variable_time_offsets or {}
    fields = [
        load_field_at_timestamp(
            file_path=target_files[variable],
            variable=variable,
            source=source,
            timestamp=timestamp,
            time_variable=time_variable,
            file_time_offset_hours=variable_time_offsets.get(variable, 0.0),
        )
        for variable in ordered_variables
    ]
    return _stack_loaded_fields(fields), fields



def load_dynamic_stack(
    dynamic_files: dict[str, str | Path],
    timestamp: datetime,
    *,
    source: str = "NORCP_LR",
    variable_order: Optional[Iterable[str]] = None,
    time_variable: str = "time",
    variable_time_offsets: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, list[LoadedField]]:
    """
    Load and stack LR dynamic variables as [C, H, W].
    """
    ordered_variables = list(variable_order) if variable_order is not None else list(dynamic_files.keys())
    variable_time_offsets = variable_time_offsets or {}
    fields = [
        load_field_at_timestamp(
            file_path=dynamic_files[variable],
            variable=variable,
            source=source,
            timestamp=timestamp,
            time_variable=time_variable,
            file_time_offset_hours=variable_time_offsets.get(variable, 0.0),
        )
        for variable in ordered_variables
    ]
    return _stack_loaded_fields(fields), fields



def load_static_stack(
    static_files: dict[str, str | Path],
    *,
    source_by_variable: Optional[dict[str, str]] = None,
    variable_order: Optional[Iterable[str]] = None,
) -> tuple[Optional[np.ndarray], list[LoadedField]]:
    """
    Load and stack static variables as [C, H, W].

    Static variables are optional. If `static_files` is empty, this returns
    `(None, [])` so downstream code can run without static conditioning.
    """
    ordered_variables = list(variable_order) if variable_order is not None else list(static_files.keys())
    source_by_variable = source_by_variable or {}

    if not ordered_variables:
        return None, []

    fields = [
        load_static_field(
            file_path=static_files[variable],
            variable=variable,
            source=source_by_variable.get(variable, "NORCP_STATIC"),
        )
        for variable in ordered_variables
    ]
    return _stack_loaded_fields(fields), fields


# -------------------------------------------------------------------
# Time features and metadata
# -------------------------------------------------------------------

def build_day_of_year_features(timestamp: datetime) -> np.ndarray:
    """
    Build sine/cosine day-of-year features.

    Returns
    -------
    np.ndarray
        Shape [2], dtype float32.
    """
    doy = float(timestamp.timetuple().tm_yday)
    angle = 2.0 * np.pi * doy / 365.0
    return np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32)



def build_loaded_field_metadata(fields: Iterable[LoadedField]) -> list[dict[str, object]]:
    """
    Build a lightweight serializable summary for loaded fields.
    """
    rows: list[dict[str, object]] = []
    for field in fields:
        rows.append(
            {
                "variable": field.variable,
                "canonical_variable": canonicalize_variable_name(field.variable)
                if field.variable in {"pr", "tas", "orog"}
                else field.variable,
                "source": field.source,
                "timestamp": None if field.timestamp is None else field.timestamp.isoformat(),
                "raw_field_name": field.raw_field_name,
                "path": str(field.path),
                "shape": tuple(int(v) for v in field.array.shape),
                "dtype": str(field.array.dtype),
                "min": float(np.nanmin(field.array)),
                "max": float(np.nanmax(field.array)),
            }
        )
    return rows