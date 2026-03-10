"""
    Canonical variable registry for STRIDE.

    This module defines the canonical variable names used across the codebase and maps them to
    source-specific aliases, units, transforms, and plotting metadata.

    Canonical names should be used everywhere in the codebase:
        - prcp
        - temp
        - lsm
        - topo
    
        Anything dataset-specific (e.g. DANRA 'tp_tot', ERA5 'tp_589x789') should be resolved
        through this registry, not hard-coded elsewhere.
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
    file_prefix: Optional[str] = None
    npz_key: Optional[str] = None
    flip_ud: bool = False
    flip_lr: bool = False
    is_static: bool = False
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
            # DANRA target files like: tp_tot_19910813.npz
            # Key inside file is left flexible for now; adapter can validate.
            "DANRA": SourceSpec(
                dataset_name="DANRA",
                file_prefix="tp_tot",
                npz_key="data",
                is_static=False,
            ),
            # ERA5 conditioning files like: prcp_589x789_19910813.npz
            # Legacy alias function suggested internal field like tp_589x789.
            "ERA5": SourceSpec(
                dataset_name="ERA5",
                file_prefix="prcp_589x789",
                npz_key="arr_0",
                is_static=False,
            ),
        },
    ),
    "temp": VariableSpec(
        key="temp",
        long_name="2 m Temperature",
        units="K",
        cmap="plasma",
        is_positive_definite=False,
        is_static=False,
        default_transform="zscore",
        plot_range=(250.0, 320.0), # Kelvin
        groups=("dynamic", "thermodynamic", "surface"),
        sources={
            # DANRA target files like: t2m_ave_19910813.npz
            "DANRA": SourceSpec(
                dataset_name="DANRA",
                file_prefix="t2m_ave",
                npz_key="data",
                is_static=False,
            ),
            # ERA5 conditioning files like: temp_589x789_19910813.npz
            # Legacy alias function suggested internal field t2m_589x789.
            "ERA5": SourceSpec(
                dataset_name="ERA5",
                file_prefix="temp_589x789",
                npz_key="arr_0",
                is_static=False,
            ),
        },
    ),
    "lsm": VariableSpec(
        key="lsm",
        long_name="Land-Sea Mask",
        units="1",
        cmap="gray",
        is_positive_definite=True,
        is_static=True,
        default_transform="identity",
        plot_range=(0.0, 1.0),
        groups=("static", "geography", "mask"),
        sources={
            "STATIC": SourceSpec(
                dataset_name="STATIC",
                file_prefix="lsm_full",
                npz_key="data",
                flip_ud=True,
                flip_lr=False,
                is_static=True,
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
        plot_range=(-50.0, 250.0),
        groups=("static", "geography", "orography"),
        sources={
            "STATIC": SourceSpec(
                dataset_name="STATIC",
                file_prefix="topo_full",
                npz_key="data",
                flip_ud=True,
                flip_lr=False,
                is_static=True,
            ),
        },
    ),
}


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
        get_source_spec("prcp", "ERA5")
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
    Return source-specific file prefix, e.g.:
        prcp + DANRA -> tp_tot
        temp + ERA5  -> temp_589x789
    """
    return get_source_spec(variable, source).file_prefix


def get_source_npz_key(variable: str, source: str) -> Optional[str]:
    """
    Return source-specific npz key if known.
    This is allowed to be None when the adapter should infer/validate it.
    """
    return get_source_spec(variable, source).npz_key


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
        "tp": "prcp",
        "tp_tot": "prcp",
        "tp_589x789": "prcp",
        "temp": "temp",
        "t2m": "temp",
        "t2m_ave": "temp",
        "t2m_589x789": "temp",
        "lsm": "lsm",
        "topo": "topo",
    }

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