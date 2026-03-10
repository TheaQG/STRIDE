"""
Colormap and variable-visualization helpers for STRIDE.

This module centralizes plotting conventions so they can be reused consistently
across:
- training previews
- post-training generation plots
- evaluation figures
- comparison scripts

The goal is not to encode every possible plotting choice, but to provide a
clean default mapping from variable names to sensible colormaps and labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.colors as mcolors


@dataclass(frozen=True)
class VariablePlotSpec:
    """
    Plotting metadata for a variable.
    """

    cmap: str | mcolors.Colormap
    label: str
    unit: str | None = None

    @property
    def label_with_unit(self) -> str:
        if self.unit is None or self.unit == "":
            return self.label
        return f"{self.label} [{self.unit}]"



def normalize_variable_name(name: str | None) -> str | None:
    if name is None:
        return None
    return str(name).strip().lower()



def build_precip_cmap() -> mcolors.Colormap:
    """
    STRIDE default precipitation colormap.

    This follows the soft white→green/teal→dark tone style you preferred in the
    older setup, while staying lightweight and self-contained.
    """
    precip_colors = [
        "#ffffff",
        "#c9f3df",
        "#66c29a",
        "#2b8c67",
        "#26534A",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "stride_precip_white_tealgray",
        precip_colors,
        N=256,
    )
    cmap = cmap.copy()
    cmap.set_under("#c2c2c2")
    return cmap


STRIDE_PRECIP_CMAP = build_precip_cmap()


_DEFAULT_SPECS: dict[str, VariablePlotSpec] = {
    # Target / surface variables
    "prcp": VariablePlotSpec(
        cmap=STRIDE_PRECIP_CMAP,
        label="Precipitation",
        unit="mm/day",
    ),
    "precip": VariablePlotSpec(
        cmap=STRIDE_PRECIP_CMAP,
        label="Precipitation",
        unit="mm/day",
    ),
    "precipitation": VariablePlotSpec(
        cmap=STRIDE_PRECIP_CMAP,
        label="Precipitation",
        unit="mm/day",
    ),
    "tp": VariablePlotSpec(
        cmap=STRIDE_PRECIP_CMAP,
        label="Precipitation",
        unit="mm/day",
    ),
    "temp": VariablePlotSpec(
        cmap="plasma",
        label="Temperature",
        unit="°C",
    ),
    "temperature": VariablePlotSpec(
        cmap="plasma",
        label="Temperature",
        unit="°C",
    ),
    "t2m": VariablePlotSpec(
        cmap="plasma",
        label="Temperature",
        unit="°C",
    ),
    "tas": VariablePlotSpec(
        cmap="plasma",
        label="Temperature",
        unit="°C",
    ),
    # Static variables
    "lsm": VariablePlotSpec(
        cmap="binary",
        label="Land-sea mask",
        unit=None,
    ),
    "land_sea_mask": VariablePlotSpec(
        cmap="binary",
        label="Land-sea mask",
        unit=None,
    ),
    "mask": VariablePlotSpec(
        cmap="binary",
        label="Land-sea mask",
        unit=None,
    ),
    "topo": VariablePlotSpec(
        cmap="terrain",
        label="Topography",
        unit="m",
    ),
    "orog": VariablePlotSpec(
        cmap="terrain",
        label="Topography",
        unit="m",
    ),
    "elevation": VariablePlotSpec(
        cmap="terrain",
        label="Topography",
        unit="m",
    ),
    "height": VariablePlotSpec(
        cmap="terrain",
        label="Topography",
        unit="m",
    ),
    # Atmospheric / dynamic variables
    "cape": VariablePlotSpec(
        cmap="viridis",
        label="CAPE",
        unit=None,
    ),
    "nwvf": VariablePlotSpec(
        cmap="cividis",
        label="Northward vapor flux",
        unit=None,
    ),
    "ewvf": VariablePlotSpec(
        cmap="magma",
        label="Eastward vapor flux",
        unit=None,
    ),
    "msl": VariablePlotSpec(
        cmap="coolwarm",
        label="Mean sea-level pressure",
        unit=None,
    ),
    "z_pl_250": VariablePlotSpec(
        cmap="coolwarm",
        label="Geopotential @ 250 hPa",
        unit=None,
    ),
    "z_pl_500": VariablePlotSpec(
        cmap="coolwarm",
        label="Geopotential @ 500 hPa",
        unit=None,
    ),
    "z_pl_850": VariablePlotSpec(
        cmap="coolwarm",
        label="Geopotential @ 850 hPa",
        unit=None,
    ),
    "z_pl_1000": VariablePlotSpec(
        cmap="coolwarm",
        label="Geopotential @ 1000 hPa",
        unit=None,
    ),
}


_DEFAULT_FALLBACK = VariablePlotSpec(
    cmap="viridis",
    label="Field",
    unit=None,
)


_ALIAS_MAP: dict[str, str] = {
    "prec": "prcp",
    "rain": "prcp",
    "rainfall": "prcp",
    "orography": "topo",
}



def get_variable_plot_spec(name: str | None) -> VariablePlotSpec:
    normalized = normalize_variable_name(name)
    if normalized is None:
        return _DEFAULT_FALLBACK

    canonical = _ALIAS_MAP.get(normalized, normalized)
    return _DEFAULT_SPECS.get(canonical, VariablePlotSpec(
        cmap=_DEFAULT_FALLBACK.cmap,
        label=str(name),
        unit=None,
    ))



def get_variable_cmap(name: str | None) -> str | mcolors.Colormap:
    return get_variable_plot_spec(name).cmap



def get_variable_label(name: str | None, *, with_unit: bool = False) -> str:
    spec = get_variable_plot_spec(name)
    if with_unit:
        return spec.label_with_unit
    return spec.label



def get_variable_unit(name: str | None) -> str | None:
    return get_variable_plot_spec(name).unit
