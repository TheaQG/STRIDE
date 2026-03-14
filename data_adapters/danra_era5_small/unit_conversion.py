"""
Physical unit harmonization for the DANRA/ERA5 STRIDE adapter.

This module converts source-specific raw units into a common STRIDE physical-space
convention before any transforms or statistics are applied.

Current STRIDE conventions used by the adapter:
    - prcp: mm day-1
    - temp: degC
    - cape: J kg-1
    - msl: Pa
    - ewvf: kg m-1 s-1
    - nwvf: kg m-1 s-1
    - pev: source-native physical units (currently passed through)
    - z_pl_250: source-native physical units (currently passed through)
    - z_pl_500: source-native physical units (currently passed through)
    - z_pl_850: source-native physical units (currently passed through)
    - z_pl_1000: source-native physical units (currently passed through)
    - theta_e_850: source-native physical units (currently passed through)
    - lsm: unitless
    - topo: m

Current source assumptions inherited from the legacy pipeline:
    - DANRA prcp is already in mm day-1
    - ERA5 prcp is treated as m day-1 and converted to mm day-1
    - ERA5 temp is in K and converted to degC
    - DANRA temp is also converted from K to degC
    - Most additional ERA5 conditioning variables are currently assumed to already
      be in the desired physical units for training and are therefore passed through
      unchanged unless a clear conversion rule is needed.

The conversion step belongs in data loading, not in transforms.
"""

from __future__ import annotations

import numpy as np


KELVIN_TO_CELSIUS_OFFSET = 273.15
METERS_TO_MILLIMETERS = 1000.0


NONNEGATIVE_VARIABLES = {"prcp", "cape"}
PASS_THROUGH_VARIABLES = {
    "cape",
    "msl",
    "ewvf",
    "nwvf",
    "pev",
    "z_pl_250",
    "z_pl_500",
    "z_pl_850",
    "z_pl_1000",
    "theta_e_850",
    "lsm",
    "topo",
}



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
        Canonical STRIDE variable name, e.g. `prcp`, `temp`, `cape`, `msl`,
        `z_pl_500`, `theta_e_850`, `lsm`, `topo`.
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

    # Additional conditioning variables currently passed through unchanged.
    # This is intentional until a variable-specific conversion rule is needed.
    if variable in PASS_THROUGH_VARIABLES:
        if source == "STATIC" and variable not in {"lsm", "topo"}:
            raise ValueError(
                f"STATIC source is not valid for variable '{variable}'"
            )
        if variable in NONNEGATIVE_VARIABLES:
            converted = np.clip(converted, 0.0, None)
        return converted

    raise ValueError(
        "No unit conversion rule defined for "
        f"variable='{variable}', source='{source}'"
    )
