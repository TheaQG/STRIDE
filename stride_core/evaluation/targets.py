"""
Target-product routing helpers for STRIDE evaluation.

This module defines explicitly which forecast product should be used for each
metric. The goal is to keep product-selection logic out of ``evaluator.py`` so
metric computation, plotting, and summary-table generation all use the same
routing rules.

Terminology
-----------
A "product" refers to one of the forecast representations available from a
loaded evaluation case, for example:

- ``ensemble_members``
- ``ensemble_mean``
- ``pmm``

Design principles
-----------------
- metric-to-product routing should be explicit and centralized
- routing should be easy to inspect and override later
- family-level defaults should exist, but metric-specific routing should win
- this module should stay lightweight and dependency-free
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# -----------------------------------------------------------------------------
# Supported product names
# -----------------------------------------------------------------------------

ENSEMBLE_MEMBERS: Final[str] = "ensemble_members"
ENSEMBLE_MEAN: Final[str] = "ensemble_mean"
PMM: Final[str] = "pmm"

SUPPORTED_PRODUCTS: Final[set[str]] = {
    ENSEMBLE_MEMBERS,
    ENSEMBLE_MEAN,
    PMM,
}


# -----------------------------------------------------------------------------
# Family defaults
# -----------------------------------------------------------------------------

FAMILY_DEFAULT_PRODUCTS: Final[dict[str, str]] = {
    "probabilistic": ENSEMBLE_MEMBERS,
    "spatial": PMM,
    "climatology": PMM,
    "temporal": PMM,
}


# -----------------------------------------------------------------------------
# Metric-specific routing
# -----------------------------------------------------------------------------

# NOTE:
# Metric-level routing always overrides the family default.
#
# These choices reflect the intended scientific use in STRIDE:
# - probabilistic metrics should use the full ensemble
# - spatial structure metrics are primarily evaluated on PMM output
# - SAL is kept on PMM for now to match the main deterministic spatial product
# - climatology and temporal diagnostics default to PMM unless explicitly noted
METRIC_PRODUCTS: Final[dict[str, str]] = {
    # Probabilistic
    "crps": ENSEMBLE_MEMBERS,
    "crps_decomposition": ENSEMBLE_MEMBERS,
    "spread_skill": ENSEMBLE_MEMBERS,
    "pit_histogram": ENSEMBLE_MEMBERS,
    "rank_histogram": ENSEMBLE_MEMBERS,
    "reliability_diagram": ENSEMBLE_MEMBERS,
    # Spatial
    # PSD metrics should be computed from ensemble members so spectral
    # structure can be averaged across the ensemble and spread inspected.
    "psd": ENSEMBLE_MEMBERS,
    "psd_slope": ENSEMBLE_MEMBERS,
    "iss": PMM,
    "sal": PMM,
    # Climatology
    "pixel_value_distribution": PMM,
    "histogram_comparison": PMM,
    "qq_plot": PMM,
    "extremes": PMM,
    "wet_day_frequency": PMM,
    "annual_precipitation_sum": PMM,
    "seasonal_accumulations": PMM,
    # Temporal
    "lag_autocorrelation": PMM,
    "wet_spell_length": PMM,
    "dry_spell_length": PMM,
}


# -----------------------------------------------------------------------------
# Routing result
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricTarget:
    """
    Lightweight resolved routing record for one evaluation metric.

    Attributes
    ----------
    family:
        Metric family name such as ``probabilistic`` or ``spatial``.
    metric:
        Metric key inside the family, e.g. ``crps`` or ``psd``.
    product:
        Forecast product name to request from the loaded evaluation case.
    """

    family: str
    metric: str
    product: str


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def is_supported_product(product: str) -> bool:
    """
    Return whether a forecast product name is supported by this routing layer.
    """
    return product in SUPPORTED_PRODUCTS


# -----------------------------------------------------------------------------
# Public routing API
# -----------------------------------------------------------------------------


def get_family_default_product(family: str) -> str:
    """
    Return the default forecast product for an evaluation family.

    Parameters
    ----------
    family:
        Evaluation family name.

    Returns
    -------
    str
        Product name.

    Raises
    ------
    KeyError
        If the family is unknown.
    """
    if family not in FAMILY_DEFAULT_PRODUCTS:
        raise KeyError(
            f"Unknown evaluation family {family!r}. "
            f"Known families: {sorted(FAMILY_DEFAULT_PRODUCTS)}"
        )

    product = FAMILY_DEFAULT_PRODUCTS[family]
    if not is_supported_product(product):
        raise ValueError(
            f"Family default for {family!r} resolved to unsupported product {product!r}"
        )
    return product



def get_metric_product(metric: str, *, family: str | None = None) -> str:
    """
    Resolve the forecast product for a metric.

    Resolution order
    ----------------
    1. metric-specific routing in ``METRIC_PRODUCTS``
    2. family default in ``FAMILY_DEFAULT_PRODUCTS`` if ``family`` is provided

    Parameters
    ----------
    metric:
        Metric name, e.g. ``crps`` or ``psd``.
    family:
        Optional metric family. Required if the metric is not explicitly routed.

    Returns
    -------
    str
        Forecast product name.

    Raises
    ------
    KeyError
        If the metric is not known and no family fallback is available.
    ValueError
        If the resolved product is unsupported.
    """
    if metric in METRIC_PRODUCTS:
        product = METRIC_PRODUCTS[metric]
    elif family is not None:
        product = get_family_default_product(family)
    else:
        raise KeyError(
            f"No explicit routing defined for metric {metric!r}, and no family "
            "fallback was provided."
        )

    if not is_supported_product(product):
        raise ValueError(
            f"Metric {metric!r} resolved to unsupported product {product!r}"
        )
    return product



def resolve_metric_target(family: str, metric: str) -> MetricTarget:
    """
    Resolve the full routing record for one metric.
    """
    return MetricTarget(
        family=family,
        metric=metric,
        product=get_metric_product(metric, family=family),
    )



def get_probabilistic_product(metric: str) -> str:
    """
    Convenience wrapper for probabilistic metrics.
    """
    return get_metric_product(metric, family="probabilistic")



def get_spatial_product(metric: str) -> str:
    """
    Convenience wrapper for spatial metrics.
    """
    return get_metric_product(metric, family="spatial")



def get_climatology_product(metric: str) -> str:
    """
    Convenience wrapper for climatology metrics.
    """
    return get_metric_product(metric, family="climatology")



def get_temporal_product(metric: str) -> str:
    """
    Convenience wrapper for temporal metrics.
    """
    return get_metric_product(metric, family="temporal")



def summarize_target_routing() -> dict[str, dict[str, str]]:
    """
    Return a nested summary of family defaults and metric overrides.

    This is mainly useful for debugging, documentation, and future summary
    writers that want to record which forecast product each headline metric uses.
    """
    return {
        "family_defaults": dict(FAMILY_DEFAULT_PRODUCTS),
        "metric_overrides": dict(METRIC_PRODUCTS),
    }