

"""
Summary-metric extraction utilities for STRIDE evaluation.

This module converts nested evaluation outputs into a compact, table-friendly
representation that can later be written to CSV / Markdown / LaTeX.

Design goals
------------
- keep summary extraction separate from metric computation
- work directly on the serializable ``Evaluator.metrics`` structure
- provide one compact scalar per headline diagnostic where possible
- preserve enough metadata that later writers can render readable tables

Current scope
-------------
This first version focuses on extracting robust headline summary metrics from
already-computed evaluation results. It does *not* yet write files itself.
That can be added as a thin wrapper later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isnan
from statistics import mean
from typing import Any


# -----------------------------------------------------------------------------
# Summary row
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryMetric:
    """
    Compact summary record for one headline evaluation metric.

    Attributes
    ----------
    family:
        Evaluation family name such as ``probabilistic`` or ``spatial``.
    metric:
        Internal metric key.
    display_name:
        Human-friendly display name for tables.
    value:
        Extracted scalar value.
    direction:
        Interpretation of better/worse values, e.g. ``lower_better``.
    ideal_value:
        Optional ideal reference value when meaningful.
    product:
        Forecast product used for the metric if known.
    notes:
        Optional free-text explanation of what the scalar represents.
    """

    family: str
    metric: str
    display_name: str
    value: float
    direction: str
    ideal_value: float | None = None
    product: str | None = None
    notes: str | None = None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """
    Convert a scalar-like value to float when possible.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        if isnan(out):
            return None
        return out
    return None



def _mean_ignore_none(values: list[float | None]) -> float | None:
    usable = [v for v in values if v is not None]
    if len(usable) == 0:
        return None
    return float(mean(usable))



def _extract_case_entries(metrics: dict[str, Any], family: str) -> dict[str, Any]:
    family_block = metrics.get(family, {})
    if not isinstance(family_block, dict):
        return {}
    case_level = family_block.get("case_level", {})
    if not isinstance(case_level, dict):
        return {}
    return case_level



def _extract_metric_product(case_entry: dict[str, Any], metric: str) -> str | None:
    metric_products = case_entry.get("metric_products", {})
    if not isinstance(metric_products, dict):
        return None
    product = metric_products.get(metric)
    return product if isinstance(product, str) else None



def _collect_case_metric_values(
    metrics: dict[str, Any],
    *,
    family: str,
    metric: str,
    scalar_getter,
) -> tuple[list[float | None], str | None]:
    case_entries = _extract_case_entries(metrics, family)
    values: list[float | None] = []
    resolved_product: str | None = None

    for case_entry in case_entries.values():
        if not isinstance(case_entry, dict):
            continue
        metric_payload = case_entry.get(metric)
        if metric_payload is None:
            continue
        values.append(scalar_getter(metric_payload))
        if resolved_product is None:
            resolved_product = _extract_metric_product(case_entry, metric)

    # Fallback: some families currently store certain metrics only in aggregate.
    if len(values) == 0:
        family_block = metrics.get(family, {})
        if isinstance(family_block, dict):
            aggregate_block = family_block.get("aggregate", {})
            if isinstance(aggregate_block, dict) and metric in aggregate_block:
                values.append(scalar_getter(aggregate_block.get(metric)))

    return values, resolved_product


# -----------------------------------------------------------------------------
# Scalar getters per metric family
# -----------------------------------------------------------------------------


def _get_direct_scalar(payload: Any) -> float | None:
    """
    Extract a scalar directly from a payload when the metric already returns one.
    """
    scalar = _safe_float(payload)
    if scalar is not None:
        return scalar

    if isinstance(payload, dict):
        preferred_keys = (
            "value",
            "score",
            "mean",
            "mean_crps",
            "crps",
            "ks_statistic",
            "ks",
            "distance",
            "wasserstein",
            "l1_distance",
            "l2_distance",
            "mean_abs_deviation",
            "mean_skill_error",
        )
        for key in preferred_keys:
            scalar = _safe_float(payload.get(key))
            if scalar is not None:
                return scalar

    return None


def _get_pit_scalar(payload: Any) -> float | None:
    """
    Extract a PIT calibration scalar.

    Preferred interpretation:
    - use an explicitly provided KS/statistic-like scalar if present
    - else derive a simple deviation score from normalized histogram counts
    """
    scalar = _get_direct_scalar(payload)
    if scalar is not None:
        return scalar

    if not isinstance(payload, dict):
        return None

    counts = payload.get("normalized_counts")
    if counts is None:
        counts = payload.get("counts")
    if counts is None:
        return None

    try:
        arr = [float(v) for v in counts]
    except Exception:
        return None

    if len(arr) == 0:
        return None

    total = sum(arr)
    if total <= 0.0:
        return None

    normalized = [v / total for v in arr]
    uniform = 1.0 / len(normalized)
    return float(mean(abs(v - uniform) for v in normalized))


def _get_spread_skill_scalar(payload: Any) -> float | None:
    """
    Extract a compact spread-skill scalar.

    Preferred interpretation:
    - use an explicitly provided scalar if present
    - else compare mean spread and mean skill from curve values
    """
    scalar = _get_direct_scalar(payload)
    if scalar is not None:
        return scalar

    if not isinstance(payload, dict):
        return None

    spread_values = payload.get("spread_values")
    if spread_values is None:
        spread_values = payload.get("spread")
    skill_values = payload.get("skill_values")
    if skill_values is None:
        skill_values = payload.get("skill")

    try:
        spread = [float(v) for v in spread_values] if spread_values is not None else []
        skill = [float(v) for v in skill_values] if skill_values is not None else []
    except Exception:
        return None

    if len(spread) == 0 or len(skill) == 0:
        return None

    n = min(len(spread), len(skill))
    spread_mean = mean(spread[:n])
    skill_mean = mean(skill[:n])
    if skill_mean == 0.0:
        return None
    return float(spread_mean / skill_mean)


def _get_reliability_scalar(payload: Any) -> float | None:
    """
    Extract a compact reliability scalar.

    Preferred interpretation:
    - use an explicitly provided scalar if present
    - else compute mean absolute calibration gap across all available curves
    """
    scalar = _get_direct_scalar(payload)
    if scalar is not None:
        return scalar

    if not isinstance(payload, dict):
        return None

    curves = payload.get("curves")
    if not isinstance(curves, dict):
        return None

    gaps: list[float] = []
    for curve_payload in curves.values():
        if not isinstance(curve_payload, dict):
            continue
        forecast = curve_payload.get("forecast_probabilities")
        if forecast is None:
            forecast = curve_payload.get("bin_centers")
        observed = curve_payload.get("observed_frequencies")
        if observed is None:
            observed = curve_payload.get("event_frequencies")
        if forecast is None or observed is None:
            continue
        try:
            fx = [float(v) for v in forecast]
            oy = [float(v) for v in observed]
        except Exception:
            continue
        n = min(len(fx), len(oy))
        if n == 0:
            continue
        gaps.extend(abs(fx[i] - oy[i]) for i in range(n))

    if len(gaps) == 0:
        return None
    return float(mean(gaps))



def _get_psd_scalar(payload: Any) -> float | None:
    """
    Extract a single PSD headline scalar.

    Preferred interpretation:
    - if the metric already exposes a slope error / difference, use it
    - else if both slopes exist, use absolute slope difference
    """
    if not isinstance(payload, dict):
        return _safe_float(payload)

    for key in (
        "slope_error",
        "slope_difference",
        "abs_slope_difference",
        "slope_diff",
    ):
        scalar = _safe_float(payload.get(key))
        if scalar is not None:
            return scalar

    forecast_slope = _safe_float(payload.get("forecast_slope"))
    target_slope = _safe_float(payload.get("target_slope"))
    if forecast_slope is not None and target_slope is not None:
        return abs(forecast_slope - target_slope)

    return None



def _get_iss_scalar(payload: Any) -> float | None:
    """
    Extract a single ISS headline scalar.

    Preferred interpretation:
    - use an explicitly provided mean / score if available
    - else take the mean over numeric values in nested arrays / dicts
    """
    if not isinstance(payload, dict):
        return _safe_float(payload)

    for key in ("mean", "score", "iss_mean", "aggregate"):
        scalar = _safe_float(payload.get(key))
        if scalar is not None:
            return scalar

    numeric_values: list[float] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                _collect(value)
        elif isinstance(obj, list):
            for value in obj:
                _collect(value)
        else:
            scalar = _safe_float(obj)
            if scalar is not None:
                numeric_values.append(scalar)

    _collect(payload)
    if len(numeric_values) == 0:
        return None
    return float(mean(numeric_values))



def _get_extreme_scalar(payload: Any) -> float | None:
    """
    Extract a single extremes headline scalar.

    Preferred interpretation:
    - use explicit tail error / bias if exposed
    - else average absolute forecast-target quantile differences over all
      available quantile pairs
    """
    if not isinstance(payload, dict):
        return _safe_float(payload)

    for key in (
        "tail_bias",
        "tail_error",
        "mean_abs_quantile_error",
        "extreme_bias",
    ):
        scalar = _safe_float(payload.get(key))
        if scalar is not None:
            return scalar

    diffs: list[float] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        forecast_q = _safe_float(value.get("forecast"))
        target_q = _safe_float(value.get("target"))
        if forecast_q is not None and target_q is not None:
            diffs.append(abs(forecast_q - target_q))

    if len(diffs) == 0:
        return None
    return float(mean(diffs))



def _get_annual_sum_mean_scalar(payload: Any) -> float | None:
    """
    Extract annual-sum mean bias-like scalar.
    """
    if not isinstance(payload, dict):
        return _safe_float(payload)

    for key in (
        "mean_bias",
        "mean_sum_bias",
        "annual_sum_mean_bias",
        "bias",
    ):
        scalar = _safe_float(payload.get(key))
        if scalar is not None:
            return scalar
    forecast_mean = _safe_float(payload.get("forecast_mean"))
    if forecast_mean is None:
        forecast_mean = _safe_float(payload.get("forecast_mean_sum"))
    target_mean = _safe_float(payload.get("target_mean"))
    if target_mean is None:
        target_mean = _safe_float(payload.get("target_mean_sum"))
    if forecast_mean is not None and target_mean is not None:
        return forecast_mean - target_mean
    return None



def _get_annual_sum_std_scalar(payload: Any) -> float | None:
    """
    Extract annual-sum variability preservation scalar.

    Preferred interpretation:
    - use std ratio if exposed
    - else compute forecast_std / target_std
    """
    if not isinstance(payload, dict):
        return None

    for key in (
        "std_ratio",
        "std_sum_ratio",
        "annual_sum_std_ratio",
        "spatial_std_ratio",
    ):
        scalar = _safe_float(payload.get(key))
        if scalar is not None:
            return scalar
    forecast_std = _safe_float(payload.get("forecast_std"))
    if forecast_std is None:
        forecast_std = _safe_float(payload.get("forecast_std_sum"))
    target_std = _safe_float(payload.get("target_std"))
    if target_std is None:
        target_std = _safe_float(payload.get("target_std_sum"))
    if forecast_std is not None and target_std not in (None, 0.0):
        return forecast_std / target_std
    return None



def _get_sal_scalar(payload: Any) -> float | None:
    """
    Extract a compact SAL summary scalar.

    Preferred interpretation:
    - use explicit aggregate if present
    - else average absolute S/A/L components
    """
    if not isinstance(payload, dict):
        return _safe_float(payload)

    for key in ("aggregate", "sal_norm", "mean_abs_sal"):
        scalar = _safe_float(payload.get(key))
        if scalar is not None:
            return scalar

    components = [
        _safe_float(payload.get("S")),
        _safe_float(payload.get("A")),
        _safe_float(payload.get("L")),
    ]
    usable = [abs(v) for v in components if v is not None]
    if len(usable) == 0:
        return None
    return float(mean(usable))


# -----------------------------------------------------------------------------
# Family summary builders
# -----------------------------------------------------------------------------


def _maybe_append_summary(
    rows: list[SummaryMetric],
    *,
    family: str,
    metric: str,
    display_name: str,
    direction: str,
    ideal_value: float | None,
    notes: str,
    values: list[float | None],
    product: str | None,
) -> None:
    value = _mean_ignore_none(values)
    if value is None:
        return
    rows.append(
        SummaryMetric(
            family=family,
            metric=metric,
            display_name=display_name,
            value=value,
            direction=direction,
            ideal_value=ideal_value,
            product=product,
            notes=notes,
        )
    )



def build_probabilistic_summary(metrics: dict[str, Any]) -> list[SummaryMetric]:
    rows: list[SummaryMetric] = []

    values, product = _collect_case_metric_values(
        metrics,
        family="probabilistic",
        metric="crps",
        scalar_getter=_get_direct_scalar,
    )
    _maybe_append_summary(
        rows,
        family="probabilistic",
        metric="crps",
        display_name="CRPS",
        direction="lower_better",
        ideal_value=0.0,
        notes="Mean CRPS across evaluated cases.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="probabilistic",
        metric="pit_histogram",
        scalar_getter=_get_pit_scalar,
    )
    _maybe_append_summary(
        rows,
        family="probabilistic",
        metric="pit_histogram",
        display_name="PIT KS",
        direction="lower_better",
        ideal_value=0.0,
        notes="Calibration scalar extracted from PIT histogram / ECDF output.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="probabilistic",
        metric="spread_skill",
        scalar_getter=_get_spread_skill_scalar,
    )
    _maybe_append_summary(
        rows,
        family="probabilistic",
        metric="spread_skill",
        display_name="Spread-skill",
        direction="closer_to_one_better",
        ideal_value=1.0,
        notes="Headline spread-skill scalar averaged across cases.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="probabilistic",
        metric="reliability_diagram",
        scalar_getter=_get_reliability_scalar,
    )
    _maybe_append_summary(
        rows,
        family="probabilistic",
        metric="reliability_diagram",
        display_name="Reliability",
        direction="lower_better",
        ideal_value=0.0,
        notes="Calibration-gap style scalar extracted from reliability output.",
        values=values,
        product=product,
    )

    return rows



def build_spatial_summary(metrics: dict[str, Any]) -> list[SummaryMetric]:
    rows: list[SummaryMetric] = []

    values, product = _collect_case_metric_values(
        metrics,
        family="spatial",
        metric="psd",
        scalar_getter=_get_psd_scalar,
    )
    _maybe_append_summary(
        rows,
        family="spatial",
        metric="psd",
        display_name="PSD",
        direction="lower_better",
        ideal_value=0.0,
        notes="Headline multiscale-structure error scalar from PSD output.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="spatial",
        metric="iss",
        scalar_getter=_get_iss_scalar,
    )
    _maybe_append_summary(
        rows,
        family="spatial",
        metric="iss",
        display_name="ISS",
        direction="higher_better",
        ideal_value=None,
        notes="Mean spatial-coherence score extracted from ISS output.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="spatial",
        metric="sal",
        scalar_getter=_get_sal_scalar,
    )
    _maybe_append_summary(
        rows,
        family="spatial",
        metric="sal",
        display_name="SAL",
        direction="lower_better",
        ideal_value=0.0,
        notes="Mean absolute SAL component aggregate across cases.",
        values=values,
        product=product,
    )

    return rows



def build_climatology_summary(metrics: dict[str, Any]) -> list[SummaryMetric]:
    rows: list[SummaryMetric] = []

    values, product = _collect_case_metric_values(
        metrics,
        family="climatology",
        metric="histogram_comparison",
        scalar_getter=_get_direct_scalar,
    )
    _maybe_append_summary(
        rows,
        family="climatology",
        metric="histogram_comparison",
        display_name="Wasserstein",
        direction="lower_better",
        ideal_value=0.0,
        notes="Distribution-shift scalar from histogram comparison output.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="climatology",
        metric="extremes",
        scalar_getter=_get_extreme_scalar,
    )
    _maybe_append_summary(
        rows,
        family="climatology",
        metric="extremes",
        display_name="Extremes",
        direction="lower_better",
        ideal_value=0.0,
        notes="Mean absolute tail discrepancy extracted from extremes output.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="climatology",
        metric="annual_precipitation_sum",
        scalar_getter=_get_annual_sum_mean_scalar,
    )
    _maybe_append_summary(
        rows,
        family="climatology",
        metric="annual_precipitation_sum_mean",
        display_name="Annual sum mean",
        direction="closer_to_zero_better",
        ideal_value=0.0,
        notes="Annual accumulated precipitation mean bias.",
        values=values,
        product=product,
    )

    values, product = _collect_case_metric_values(
        metrics,
        family="climatology",
        metric="annual_precipitation_sum",
        scalar_getter=_get_annual_sum_std_scalar,
    )
    _maybe_append_summary(
        rows,
        family="climatology",
        metric="annual_precipitation_sum_std",
        display_name="Annual sum std",
        direction="closer_to_one_better",
        ideal_value=1.0,
        notes="Annual accumulated precipitation spatial-variability ratio.",
        values=values,
        product=product,
    )

    return rows



def build_temporal_summary(metrics: dict[str, Any]) -> list[SummaryMetric]:
    rows: list[SummaryMetric] = []

    values, product = _collect_case_metric_values(
        metrics,
        family="temporal",
        metric="lag_autocorrelation",
        scalar_getter=_get_direct_scalar,
    )
    _maybe_append_summary(
        rows,
        family="temporal",
        metric="lag_autocorrelation",
        display_name="Lag autocorrelation",
        direction="closer_to_zero_better",
        ideal_value=0.0,
        notes="Temporal-persistence mismatch scalar extracted from lag autocorrelation output.",
        values=values,
        product=product,
    )

    return rows


# -----------------------------------------------------------------------------
# Top-level API
# -----------------------------------------------------------------------------


def build_summary_metrics(metrics: dict[str, Any]) -> list[SummaryMetric]:
    """
    Build all headline summary metrics from a serialized evaluation result.
    """
    rows: list[SummaryMetric] = []
    rows.extend(build_probabilistic_summary(metrics))
    rows.extend(build_spatial_summary(metrics))
    rows.extend(build_climatology_summary(metrics))
    rows.extend(build_temporal_summary(metrics))
    return rows



def summary_metrics_as_dicts(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return summary metrics as a list of plain dictionaries.
    """
    return [asdict(row) for row in build_summary_metrics(metrics)]