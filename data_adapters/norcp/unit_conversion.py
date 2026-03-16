"""
Unit conversion utilities for the NorCP STRIDE adapter.

This module converts raw NorCP source fields into STRIDE canonical physical units.
Conversions are source-aware and variable-aware.

Verified NorCP conventions from inspected NetCDF headers:
    - pr: precipitation_flux, units kg m-2 s-1, cell_methods = "time: mean"
    - hus*: specific_humidity, units 1, cell_methods = "time: point"
    - ta*: air_temperature, units K, cell_methods = "time: point"
    - tas: near-surface air_temperature, units K, cell_methods = "time: point"
    - ua*: eastward_wind, units m s-1, cell_methods = "time: point"
    - va*: northward_wind, units m s-1, cell_methods = "time: point"
    - zg*: geopotential_height, units m, cell_methods = "time: point"

Notes
-----
NorCP precipitation is stored as precipitation flux rather than accumulated
precipitation. Since 1 kg m-2 equals 1 mm water equivalent, conversion from
kg m-2 s-1 to mm day-1 is obtained by multiplying by 86400.
"""

from __future__ import annotations

import numpy as np

from data_adapters.norcp.variable_registry import (
    get_variable_spec,
    get_source_spec,
    is_positive_definite,
)


PRECIP_FLUX_TO_MM_PER_DAY = 86400.0


def _as_float_array(array: np.ndarray) -> np.ndarray:
    """
    Convert input to floating numpy array without modifying the original in place.
    """
    return np.asarray(array, dtype=np.float32).copy()


def _clip_nonnegative(array: np.ndarray) -> np.ndarray:
    """
    Clip array to be nonnegative.
    """
    return np.maximum(array, 0.0)


def _convert_precipitation(
    array: np.ndarray,
    source: str,
) -> np.ndarray:
    """
    Convert precipitation flux to STRIDE canonical units.

    Current canonical target:
        mm day-1

    Verified NorCP metadata:
        - standard_name = "precipitation_flux"
        - units = "kg m-2 s-1"
        - cell_methods = "time: mean"

    Therefore, the NorCP precipitation fields represent time-mean precipitation
    rate over the aggregation interval, not accumulated precipitation.
    """
    arr = _as_float_array(array)

    if source in {"NORCP_HR", "NORCP_LR"}:
        arr = arr * PRECIP_FLUX_TO_MM_PER_DAY
        return _clip_nonnegative(arr)

    raise KeyError(f"Unsupported precipitation source for conversion: {source}")


def _convert_specific_humidity(
    array: np.ndarray,
    source: str,
) -> np.ndarray:
    """
    Convert specific humidity to STRIDE canonical units.

    Current canonical target:
        unitless mass fraction

    Verified NorCP metadata:
        - standard_name = "specific_humidity"
        - units = "1"
        - cell_methods = "time: point"
    """
    arr = _as_float_array(array)

    if source == "NORCP_LR":
        return _clip_nonnegative(arr)

    raise KeyError(f"Unsupported specific-humidity source for conversion: {source}")


def _convert_temperature(
    array: np.ndarray,
    source: str,
) -> np.ndarray:
    """
    Convert air temperature to STRIDE canonical units.

    Current canonical target:
        degC

    Verified NorCP metadata for both tas and ta*:
        - standard_name = "air_temperature"
        - units = "K"
        - cell_methods = "time: point"
    """
    arr = _as_float_array(array)

    if source in {"NORCP_HR", "NORCP_LR"}:
        return arr - 273.15

    raise KeyError(f"Unsupported temperature source for conversion: {source}")


def _convert_wind_component(
    array: np.ndarray,
    source: str,
    component_name: str,
) -> np.ndarray:
    """
    Convert horizontal wind component to STRIDE canonical units.

    Current canonical target:
        m s-1

    Verified NorCP metadata for ua* and va*:
        - units = "m s-1"
        - cell_methods = "time: point"
    """
    arr = _as_float_array(array)

    if source == "NORCP_LR":
        return arr

    raise KeyError(f"Unsupported {component_name} source for conversion: {source}")


def _convert_geopotential_height(
    array: np.ndarray,
    source: str,
) -> np.ndarray:
    """
    Convert geopotential height to STRIDE canonical units.

    Current canonical target:
        m

    Verified NorCP metadata for zg*:
        - standard_name = "geopotential_height"
        - units = "m"
        - cell_methods = "time: point"

    No gravity-based conversion is needed here because the inspected NorCP zg*
    fields are already geopotential height in meters, not geopotential in m2 s-2.
    """
    arr = _as_float_array(array)

    if source == "NORCP_LR":
        return arr

    raise KeyError(f"Unsupported geopotential-height source for conversion: {source}")


def _convert_topography(
    array: np.ndarray,
    source: str,
) -> np.ndarray:
    """
    Convert topography to STRIDE canonical units.

    Current canonical target:
        meters
    """
    arr = _as_float_array(array)

    if source == "NORCP_STATIC":
        return arr

    raise KeyError(f"Unsupported topography source for conversion: {source}")


def _normalize_variable_name(variable: str) -> str:
    """
    Collapse level-specific NorCP variable names into conversion families.

    Examples
    --------
    - hus1000 -> hus
    - ta850   -> ta
    - ua700   -> ua
    - va500   -> va
    - zg950   -> zg
    - tas     -> temp
    - temp    -> temp
    - prcp    -> prcp
    """
    if variable.startswith("hus"):
        return "hus"
    if variable.startswith("ta") and variable != "tas":
        return "ta"
    if variable == "tas" or variable == "temp":
        return "temp"
    if variable.startswith("ua"):
        return "ua"
    if variable.startswith("va"):
        return "va"
    if variable.startswith("zg"):
        return "zg"
    if variable in {"pr", "prcp"}:
        return "prcp"
    if variable in {"orog", "topo"}:
        return "topo"
    return variable


def _is_positive_definite_family(variable_family: str) -> bool:
    """
    Family-level positivity helper for variables not yet present in the registry.
    """
    return variable_family in {"prcp", "hus"}


def apply_unit_conversion(
    array: np.ndarray,
    variable: str,
    source: str,
) -> np.ndarray:
    """
    Convert a raw source field into STRIDE canonical physical units.

    Parameters
    ----------
    array:
        Raw input field.
    variable:
        Canonical STRIDE variable name or NorCP level-specific variable name,
        e.g. 'prcp', 'temp', 'tas', 'ta850', 'hus1000', 'ua700', 'zg500'.
    source:
        Source label, e.g. 'NORCP_HR', 'NORCP_LR', 'NORCP_STATIC'.

    Returns
    -------
    np.ndarray
        Converted field in canonical STRIDE units.
    """
    variable_family = _normalize_variable_name(variable)

    # Validate early when the registry already knows the variable.
    try:
        get_variable_spec(variable)
        get_source_spec(variable, source)
        positive_definite = is_positive_definite(variable)
    except KeyError:
        positive_definite = _is_positive_definite_family(variable_family)

    if variable_family == "prcp":
        converted = _convert_precipitation(array=array, source=source)
    elif variable_family == "hus":
        converted = _convert_specific_humidity(array=array, source=source)
    elif variable_family in {"temp", "ta"}:
        converted = _convert_temperature(array=array, source=source)
    elif variable_family == "ua":
        converted = _convert_wind_component(array=array, source=source, component_name="eastward-wind")
    elif variable_family == "va":
        converted = _convert_wind_component(array=array, source=source, component_name="northward-wind")
    elif variable_family == "zg":
        converted = _convert_geopotential_height(array=array, source=source)
    elif variable_family == "topo":
        converted = _convert_topography(array=array, source=source)
    else:
        raise KeyError(f"No unit conversion rule implemented for variable '{variable}'")

    if positive_definite:
        converted = _clip_nonnegative(converted)

    return converted