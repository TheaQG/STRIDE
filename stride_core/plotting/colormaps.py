"""
Colormap and variable-visualization helpers for STRIDE.

This module centralizes plotting conventions so they can be reused consistently
across:
- training previews
- post-training generation plots
- evaluation figures
- comparison scripts

The goal is not to encode every possible plotting choice, but to provide a
clean default mapping from canonical variable names to sensible colormaps,
labels, and units.
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


PRESSURE_LEVELS = (500, 700, 850, 950, 1000)


def _build_norcp_pressure_level_specs() -> dict[str, VariablePlotSpec]:
    """
    Add plotting specs for NorCP pressure-level variables.
    """
    specs: dict[str, VariablePlotSpec] = {}

    for level in PRESSURE_LEVELS:
        level_str = str(level)

        specs[f"hus{level_str}"] = VariablePlotSpec(
            cmap="viridis",
            label=f"Specific humidity @ {level_str} hPa",
            unit="1",
        )
        specs[f"ta{level_str}"] = VariablePlotSpec(
            cmap="plasma",
            label=f"Air temperature @ {level_str} hPa",
            unit="°C",
        )
        specs[f"ua{level_str}"] = VariablePlotSpec(
            cmap="coolwarm",
            label=f"Eastward wind @ {level_str} hPa",
            unit="m s⁻¹",
        )
        specs[f"va{level_str}"] = VariablePlotSpec(
            cmap="coolwarm",
            label=f"Northward wind @ {level_str} hPa",
            unit="m s⁻¹",
        )
        specs[f"zg{level_str}"] = VariablePlotSpec(
            cmap="cividis",
            label=f"Geopotential height @ {level_str} hPa",
            unit="m",
        )

    return specs


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
        unit="J kg⁻¹",
    ),
    "nwvf": VariablePlotSpec(
        cmap="coolwarm",
        label="Northward vapor flux",
        unit="kg m⁻¹ s⁻¹",
    ),
    "ewvf": VariablePlotSpec(
        cmap="coolwarm",
        label="Eastward vapor flux",
        unit="kg m⁻¹ s⁻¹",
    ),
    "msl": VariablePlotSpec(
        cmap="cividis",
        label="Mean sea-level pressure",
        unit="Pa",
    ),
    "pev": VariablePlotSpec(
        cmap="YlGnBu",
        label="Potential evaporation",
        unit=None,
    ),
    "z_pl_250": VariablePlotSpec(
        cmap="Spectral_r",
        label="Geopotential @ 250 hPa",
        unit="m² s⁻²",
    ),
    "z_pl_500": VariablePlotSpec(
        cmap="Spectral_r",
        label="Geopotential @ 500 hPa",
        unit="m² s⁻²",
    ),
    "z_pl_850": VariablePlotSpec(
        cmap="Spectral_r",
        label="Geopotential @ 850 hPa",
        unit="m² s⁻²",
    ),
    "z_pl_1000": VariablePlotSpec(
        cmap="Spectral_r",
        label="Geopotential @ 1000 hPa",
        unit="m² s⁻²",
    ),
    "theta_e_850": VariablePlotSpec(
        cmap="magma",
        label="Equivalent potential temperature @ 850 hPa",
        unit="K",
    ),
}

_DEFAULT_SPECS.update(_build_norcp_pressure_level_specs())


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
    "topography": "topo",
    "z250": "z_pl_250",
    "z500": "z_pl_500",
    "z850": "z_pl_850",
    "z1000": "z_pl_1000",
    "z_pl250": "z_pl_250",
    "z_pl500": "z_pl_500",
    "z_pl850": "z_pl_850",
    "z_pl1000": "z_pl_1000",
    "thetae_850": "theta_e_850",
    "the_e_850": "theta_e_850",
    "hus500": "hus500",
    "hus700": "hus700",
    "hus850": "hus850",
    "hus950": "hus950",
    "hus1000": "hus1000",
    "ta500": "ta500",
    "ta700": "ta700",
    "ta850": "ta850",
    "ta950": "ta950",
    "ta1000": "ta1000",
    "ua500": "ua500",
    "ua700": "ua700",
    "ua850": "ua850",
    "ua950": "ua950",
    "ua1000": "ua1000",
    "va500": "va500",
    "va700": "va700",
    "va850": "va850",
    "va950": "va950",
    "va1000": "va1000",
    "zg500": "zg500",
    "zg700": "zg700",
    "zg850": "zg850",
    "zg950": "zg950",
    "zg1000": "zg1000",
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
