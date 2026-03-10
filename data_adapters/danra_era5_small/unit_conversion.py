"""
Physical unit harmonization for the small DANRA/ERA5 STRIDE adapter.

This module converts source-specific raw units into a common STRIDE physical-space
convention before any transforms or statistics are applied.

Current STRIDE conventions for the first experiment:
    - prcp: mm day-1
    - temp: degC
    - lsm: unitless
    - topo: m

Current source assumptions inherited from the legacy pipeline:
    - DANRA prcp is already in mm day-1
    - ERA5 prcp is in m day-1 and must be converted to mm day-1
    - ERA5 temp is in K and must be converted to degC
    - STATIC fields require no unit conversion

The conversion step belongs in data loading, not in transforms.
"""

from __future__ import annotations

import numpy as np


KELVIN_TO_CELSIUS_OFFSET = 273.15
METERS_TO_MILLIMETERS = 1000.0



def apply_unit_conversion(
    array: np.ndarray,
    variable: str,
    source: str,
) -> np.ndarray:
    """
    Convert a loaded 2D field into STRIDE's canonical physical-space units.

    Parameters
    ----------
    array
        Input 2D array in source-native units.
    variable
        Canonical STRIDE variable name, e.g. `prcp`, `temp`, `lsm`, `topo`.
    source
        Source identifier, e.g. `DANRA`, `ERA5`, `STATIC`.

    Returns
    -------
    np.ndarray
        Converted array in STRIDE's canonical units.
    """
    converted = np.asarray(array, dtype=np.float32)

    # Precipitation
    if variable == "prcp":
        if source == "ERA5":
            # ERA5 precipitation is treated as m/day in the legacy setup.
            converted = converted * METERS_TO_MILLIMETERS
        elif source == "DANRA":
            # DANRA precipitation is already treated as mm/day in the legacy setup.
            pass
        elif source == "STATIC":
            raise ValueError("STATIC source is not valid for variable 'prcp'")

        # Numerical cleanup: precipitation should not be negative.
        converted = np.clip(converted, 0.0, None)
        return converted

    # Temperature
    if variable == "temp":
        if source == "ERA5":
            converted = converted - KELVIN_TO_CELSIUS_OFFSET
        elif source == "DANRA":
            # Legacy setup also converted temperature from K to degC.
            converted = converted - KELVIN_TO_CELSIUS_OFFSET
        elif source == "STATIC":
            raise ValueError("STATIC source is not valid for variable 'temp'")
        return converted

    # Static / already canonical variables
    if variable in {"lsm", "topo"}:
        return converted

    raise ValueError(
        f"No unit conversion rule defined for variable='{variable}', source='{source}'"
    )
