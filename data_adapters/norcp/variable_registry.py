"""
Canonical variable registry for the NorCP STRIDE adapter.

This module defines the canonical variable names used by the NorCP adapter and maps them to
source-specific raw field names, units, transforms, and loading conventions.

Canonical names should be used everywhere in the adapter and the wider STRIDE codebase.
For pressure-level variables, the canonical names include the level suffix, e.g.:
    - hus500, hus700, hus850, hus950, hus1000
    - ta500, ta700, ta850, ta950, ta1000
    - ua500, ua700, ua850, ua950, ua1000
    - va500, va700, va850, va950, va1000
    - zg500, zg700, zg850, zg950, zg1000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class SourceSpec:
    """
    Dataset/source-specific naming and loading information.
    """

    dataset_name: str
    raw_field_name: Optional[str] = None
    file_prefix: Optional[str] = None
    flip_ud: bool = False
    flip_lr: bool = False
    is_static: bool = False

    # Optional metadata that may become useful for NetCDF-based loading
    level: Optional[str] = None
    native_grid: Optional[str] = None
    time_kind: Optional[str] = None  # e.g. "instantaneous", "point", "mean_rate"


@dataclass(frozen=True)
class VariableSpec:
    """
    Canonical STRIDE variable specification.
    """

    key: str
    long_name: str
    units: str
    cmap: str
    is_positive_definite: bool
    is_static: bool
    default_transform: str

    # Optional plotting / evaluation metadata
    plot_range: Optional[tuple[float, float]] = None
    dry_threshold: Optional[float] = None
    groups: tuple[str, ...] = field(default_factory=tuple)

    # Source-specific aliases and conventions
    sources: Dict[str, SourceSpec] = field(default_factory=dict)


# -------------------------------------------------------------------
# Registry builders
# -------------------------------------------------------------------

PRESSURE_LEVELS: tuple[int, ...] = (500, 700, 850, 950, 1000)


def _build_pressure_level_variable_specs() -> Dict[str, VariableSpec]:
    """
    Build NorCP pressure-level variable specifications.
    """
    registry: Dict[str, VariableSpec] = {}

    for level in PRESSURE_LEVELS:
        level_str = str(level)

        registry[f"hus{level_str}"] = VariableSpec(
            key=f"hus{level_str}",
            long_name=f"Specific Humidity at {level_str} hPa",
            units="1",
            cmap="viridis",
            is_positive_definite=True,
            is_static=False,
            default_transform="zscore",
            plot_range=(0.0, 0.03),
            groups=("dynamic", "thermodynamic", "pressure_level"),
            sources={
                "NORCP_LR": SourceSpec(
                    dataset_name="NORCP_LR",
                    raw_field_name=f"hus{level_str}",
                    is_static=False,
                    level=level_str,
                    native_grid="lr",
                    time_kind="point",
                ),
            },
        )

        registry[f"ta{level_str}"] = VariableSpec(
            key=f"ta{level_str}",
            long_name=f"Air Temperature at {level_str} hPa",
            units="degC",
            cmap="plasma",
            is_positive_definite=False,
            is_static=False,
            default_transform="zscore",
            plot_range=(-60.0, 30.0),
            groups=("dynamic", "thermodynamic", "pressure_level"),
            sources={
                "NORCP_LR": SourceSpec(
                    dataset_name="NORCP_LR",
                    raw_field_name=f"ta{level_str}",
                    is_static=False,
                    level=level_str,
                    native_grid="lr",
                    time_kind="point",
                ),
            },
        )

        registry[f"ua{level_str}"] = VariableSpec(
            key=f"ua{level_str}",
            long_name=f"Eastward Wind at {level_str} hPa",
            units="m s-1",
            cmap="coolwarm",
            is_positive_definite=False,
            is_static=False,
            default_transform="zscore",
            plot_range=(-40.0, 40.0),
            groups=("dynamic", "wind", "pressure_level"),
            sources={
                "NORCP_LR": SourceSpec(
                    dataset_name="NORCP_LR",
                    raw_field_name=f"ua{level_str}",
                    is_static=False,
                    level=level_str,
                    native_grid="lr",
                    time_kind="point",
                ),
            },
        )

        registry[f"va{level_str}"] = VariableSpec(
            key=f"va{level_str}",
            long_name=f"Northward Wind at {level_str} hPa",
            units="m s-1",
            cmap="coolwarm",
            is_positive_definite=False,
            is_static=False,
            default_transform="zscore",
            plot_range=(-40.0, 40.0),
            groups=("dynamic", "wind", "pressure_level"),
            sources={
                "NORCP_LR": SourceSpec(
                    dataset_name="NORCP_LR",
                    raw_field_name=f"va{level_str}",
                    is_static=False,
                    level=level_str,
                    native_grid="lr",
                    time_kind="point",
                ),
            },
        )

        registry[f"zg{level_str}"] = VariableSpec(
            key=f"zg{level_str}",
            long_name=f"Geopotential Height at {level_str} hPa",
            units="m",
            cmap="cividis",
            is_positive_definite=False,
            is_static=False,
            default_transform="zscore",
            plot_range=(0.0, 20000.0),
            groups=("dynamic", "dynamics", "pressure_level"),
            sources={
                "NORCP_LR": SourceSpec(
                    dataset_name="NORCP_LR",
                    raw_field_name=f"zg{level_str}",
                    is_static=False,
                    level=level_str,
                    native_grid="lr",
                    time_kind="point",
                ),
            },
        )

    return registry


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------

VARIABLE_REGISTRY: Dict[str, VariableSpec] = {
    "prcp": VariableSpec(
        key="prcp",
        long_name="Precipitation",
        units="mm day-1",
        cmap="precip_white_tealgray",
        is_positive_definite=True,
        is_static=False,
        default_transform="log_zscore",
        plot_range=(0.0, 60.0),
        dry_threshold=0.1,
        groups=("dynamic", "hydrology", "surface"),
        sources={
            "NORCP_HR": SourceSpec(
                dataset_name="NORCP_HR",
                raw_field_name="pr",
                is_static=False,
                native_grid="hr",
                time_kind="mean_rate",
            ),
            "NORCP_LR": SourceSpec(
                dataset_name="NORCP_LR",
                raw_field_name="pr",
                is_static=False,
                native_grid="lr",
                time_kind="mean_rate",
            ),
        },
    ),
    "temp": VariableSpec(
        key="temp",
        long_name="2 m Temperature",
        units="degC",
        cmap="plasma",
        is_positive_definite=False,
        is_static=False,
        default_transform="zscore",
        plot_range=(-20.0, 40.0),
        groups=("dynamic", "thermodynamic", "surface"),
        sources={
            "NORCP_HR": SourceSpec(
                dataset_name="NORCP_HR",
                raw_field_name="tas",
                is_static=False,
                native_grid="hr",
                time_kind="point",
            ),
            "NORCP_LR": SourceSpec(
                dataset_name="NORCP_LR",
                raw_field_name="tas",
                is_static=False,
                native_grid="lr",
                time_kind="point",
            ),
        },
    ),
    "topo": VariableSpec(
        key="topo",
        long_name="Topography",
        units="m",
        cmap="terrain",
        is_positive_definite=False,
        is_static=True,
        default_transform="zscore",
        plot_range=(-50.0, 2500.0),
        groups=("static", "geography", "orography"),
        sources={
            "NORCP_STATIC": SourceSpec(
                dataset_name="NORCP_STATIC",
                raw_field_name="orog",
                is_static=True,
                native_grid="hr",
            ),
            "NORCP_LR": SourceSpec(
                dataset_name="NORCP_LR",
                raw_field_name="orog",
                is_static=True,
                native_grid="lr",
            ),
        },
    ),
}

VARIABLE_REGISTRY.update(_build_pressure_level_variable_specs())


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def get_variable_spec(variable: str) -> VariableSpec:
    """
    Return the full VariableSpec for a canonical variable name.
    """
    if variable not in VARIABLE_REGISTRY:
        raise KeyError(
            f"Unknown canonical variable '{variable}'. "
            f"Available: {list(VARIABLE_REGISTRY.keys())}"
        )
    return VARIABLE_REGISTRY[variable]


def list_registered_variables() -> list[str]:
    """
    Return all canonical variable names.
    """
    return list(VARIABLE_REGISTRY.keys())


def is_static_variable(variable: str) -> bool:
    """
    Return whether a canonical variable is static.
    """
    return get_variable_spec(variable).is_static


def is_positive_definite(variable: str) -> bool:
    """
    Return whether a canonical variable should be non-negative.
    """
    return get_variable_spec(variable).is_positive_definite


def get_default_transform(variable: str) -> str:
    """
    Return the default transform name for a canonical variable.
    """
    return get_variable_spec(variable).default_transform


def get_units(variable: str) -> str:
    """
    Return the display units for a canonical variable.
    """
    return get_variable_spec(variable).units


def get_long_name(variable: str) -> str:
    """
    Return the long display name for a canonical variable.
    """
    return get_variable_spec(variable).long_name


def get_cmap(variable: str) -> str:
    """
    Return the plotting colormap name for a canonical variable.
    """
    return get_variable_spec(variable).cmap


def get_plot_range(variable: str) -> Optional[tuple[float, float]]:
    """
    Return the default plotting range for a canonical variable.
    """
    return get_variable_spec(variable).plot_range


def get_source_spec(variable: str, source: str) -> SourceSpec:
    """
    Return source-specific naming info for a canonical variable.

    Example:
        get_source_spec("prcp", "NORCP_LR")
    """
    spec = get_variable_spec(variable)
    if source not in spec.sources:
        raise KeyError(
            f"Variable '{variable}' has no source spec for '{source}'. "
            f"Available: {list(spec.sources.keys())}"
        )
    return spec.sources[source]


def get_source_file_prefix(variable: str, source: str) -> Optional[str]:
    """
    Return source-specific file prefix if one is used by the dataset layout.
    """
    return get_source_spec(variable, source).file_prefix


def get_source_raw_field_name(variable: str, source: str) -> Optional[str]:
    """
    Return source-specific raw field name, e.g. NetCDF variable name.
    """
    return get_source_spec(variable, source).raw_field_name


def get_source_flip_config(variable: str, source: str) -> tuple[bool, bool]:
    """
    Return source-specific orientation handling as (flip_ud, flip_lr).
    """
    source_spec = get_source_spec(variable, source)
    return source_spec.flip_ud, source_spec.flip_lr


def canonicalize_variable_name(name: str) -> str:
    """
    Map known aliases to canonical STRIDE names.

    This should only be used as a light compatibility helper.
    Adapters should still be explicit about source-specific naming.
    """
    alias_map = {
        "prcp": "prcp",
        "pr": "prcp",
        "tp": "prcp",
        "precip": "prcp",
        "precipitation": "prcp",
        "temp": "temp",
        "tas": "temp",
        "t2m": "temp",
        "temperature": "temp",
        "topo": "topo",
        "topography": "topo",
        "orog": "topo",
    }

    for level in PRESSURE_LEVELS:
        level_str = str(level)
        alias_map[f"hus{level_str}"] = f"hus{level_str}"
        alias_map[f"ta{level_str}"] = f"ta{level_str}"
        alias_map[f"ua{level_str}"] = f"ua{level_str}"
        alias_map[f"va{level_str}"] = f"va{level_str}"
        alias_map[f"zg{level_str}"] = f"zg{level_str}"

    if name not in alias_map:
        raise KeyError(
            f"Cannot canonicalize variable name '{name}'. "
            f"Known aliases: {list(alias_map.keys())}"
        )
    return alias_map[name]


def validate_variables(variables: Sequence[str]) -> None:
    """
    Raise if any canonical variable is unknown.
    """
    unknown = [v for v in variables if v not in VARIABLE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown canonical variables: {unknown}. "
            f"Available: {list(VARIABLE_REGISTRY.keys())}"
        )