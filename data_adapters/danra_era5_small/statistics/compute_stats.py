"""
Split-aware statistics computation for the small DANRA/ERA5 STRIDE adapter.

This module computes variable-wise global statistics from the training split
(or another selected split) using the already standardized STRIDE data path:
    - source-aware loading
    - orientation correction
    - static consistency enforcement
    - optional crop selection

For the first STRIDE setup, statistics are computed by pooling values across all
selected dates for one variable and one domain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_adapters.danra_era5_small.features import (
    load_dynamic_conditioning,
    load_static_features,
    load_target_field,
)
from data_adapters.danra_era5_small.paths import (
    build_date_to_file_map,
    build_experiment_file_index,
)
from data_adapters.danra_era5_small.regions import CropSpec, crop_sample_fields
from data_adapters.danra_era5_small.splits import get_dates_for_split, load_split_manifest
from data_adapters.danra_era5_small.statistics.schemas import (
    CropConfig,
    StatsRequest,
    StatsResult,
    StatsSummary,
)
from data_adapters.danra_era5_small.transforms import (
    compute_log_zscore_stats,
    compute_zscore_stats,
)


DEFAULT_ROOT_DIR = Path("/Users/au728490/Data/Data_DiffMod_small")
DEFAULT_SIZE_TAG = "size_589x789"


STATIC_PATHS = {
    "lsm": DEFAULT_ROOT_DIR / "data_lsm" / "truth_fullDomain" / "lsm_full.npz",
    "topo": DEFAULT_ROOT_DIR / "data_topo" / "truth_fullDomain" / "topo_full.npz",
}


SUPPORTED_VARIABLE_SPECS: dict[str, dict[str, Any]] = {
    "target_prcp": {
        "variable": "prcp",
        "source": "DANRA",
        "transform_name": "log_zscore",
    },
    "era5_prcp": {
        "variable": "prcp",
        "source": "ERA5",
        "transform_name": "log_zscore",
    },
    "era5_temp": {
        "variable": "temp",
        "source": "ERA5",
        "transform_name": "zscore",
    },
    "static_topo": {
        "variable": "topo",
        "source": "STATIC",
        "transform_name": "zscore",
    },
    "static_lsm": {
        "variable": "lsm",
        "source": "STATIC",
        "transform_name": "identity",
    },
}



def _build_crop_spec(crop: CropConfig | None) -> CropSpec | None:
    if crop is None:
        return None
    return CropSpec(
        anchor_y=crop.anchor_y,
        anchor_x=crop.anchor_x,
        height=crop.height,
        width=crop.width,
    )



def _summarize_physical_values(values: np.ndarray, n_dates: int) -> StatsSummary:
    values = np.asarray(values, dtype=np.float32)
    return StatsSummary(
        n_dates=n_dates,
        n_values=int(values.size),
        min=float(np.nanmin(values)),
        max=float(np.nanmax(values)),
        mean=float(np.nanmean(values)),
        std=float(np.nanstd(values)),
    )


def _compute_additional_distribution_stats(values: np.ndarray) -> dict[str, float]:
    """
    Compute additional descriptive statistics in physical space.
    """
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "median": float(np.nanmedian(values)),
        "p25": float(np.nanpercentile(values, 25)),
        "p75": float(np.nanpercentile(values, 75)),
    }


def _compute_transform_stats(values: np.ndarray, transform_name: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)

    if transform_name == "identity":
        stats = _compute_additional_distribution_stats(values)
        return stats

    if transform_name == "zscore":
        stats = compute_zscore_stats(values)
        stats.update(_compute_additional_distribution_stats(values))
        return stats

    if transform_name == "log_zscore":
        stats = compute_log_zscore_stats(values, clip_min=0.0)
        stats.update(_compute_additional_distribution_stats(values))

        values_log = np.log1p(np.clip(values, 0.0, None)).astype(np.float32)
        stats.update(
            {
                "min_log": float(np.nanmin(values_log)),
                "max_log": float(np.nanmax(values_log)),
                "median_log": float(np.nanmedian(values_log)),
                "p25_log": float(np.nanpercentile(values_log, 25)),
                "p75_log": float(np.nanpercentile(values_log, 75)),
            }
        )
        return stats

    raise ValueError(f"Unsupported transform_name '{transform_name}'")



def _load_variable_field_for_date(
    date_str: str,
    request: StatsRequest,
    file_index: dict[str, dict[str, object]],
    crop_spec: CropSpec | None,
    root_dir: Path,
) -> np.ndarray:
    sample_paths = file_index[date_str]

    danra_temp_map = None
    if request.source == "DANRA" and request.variable == "temp":
        danra_temp_map = build_date_to_file_map(
            root_dir=root_dir,
            source="DANRA",
            variable="temp",
            size_tag=DEFAULT_SIZE_TAG,
        )
        if date_str not in danra_temp_map:
            raise ValueError(
                f"DANRA temp file missing for date '{date_str}'"
            )

    target = load_target_field(
        target_path=sample_paths["target"], # type: ignore
        target_variable="prcp",
        target_source="DANRA",
    )
    cond_dynamic = load_dynamic_conditioning(
        dynamic_paths=sample_paths["cond_dynamic"], # type: ignore
        variable_order=list(request.conditioning_variable_order),
        source="ERA5",
    )
    cond_static = load_static_features(
        static_paths={
            "lsm": root_dir / "data_lsm" / "truth_fullDomain" / "lsm_full.npz",
            "topo": root_dir / "data_topo" / "truth_fullDomain" / "topo_full.npz",
        },
        variable_order=list(request.static_variable_order),
        source="STATIC",
    )

    cropped = crop_sample_fields(
        target=target,
        cond_dynamic=cond_dynamic,
        cond_static=cond_static,
        crop_spec=crop_spec,
    )

    if request.source == "DANRA" and request.variable == "prcp":
        return np.asarray(cropped["target"], dtype=np.float32)

    if request.source == "DANRA" and request.variable == "temp":
        danra_temp = load_target_field(
            target_path=danra_temp_map[date_str], # type: ignore
            target_variable="temp",
            target_source="DANRA",
        )

        danra_temp_cropped = crop_sample_fields(
            target=danra_temp,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            crop_spec=crop_spec,
        )["target"]

        return np.asarray(danra_temp_cropped, dtype=np.float32)

    if request.source == "ERA5":
        variable_order = list(request.conditioning_variable_order)
        if request.variable not in variable_order:
            raise ValueError(
                f"Requested ERA5 variable '{request.variable}' not found in conditioning order {variable_order}"
            )
        idx = variable_order.index(request.variable)
        return np.asarray(cropped["cond_dynamic"][idx], dtype=np.float32)

    if request.source == "STATIC":
        variable_order = list(request.static_variable_order)
        if cropped["cond_static"] is None:
            raise ValueError("Requested STATIC field but cond_static is None")
        if request.variable not in variable_order:
            raise ValueError(
                f"Requested STATIC variable '{request.variable}' not found in static order {variable_order}"
            )
        idx = variable_order.index(request.variable)
        return np.asarray(cropped["cond_static"][idx], dtype=np.float32)

    raise ValueError(
        f"Unsupported request combination: source='{request.source}', variable='{request.variable}'"
    )



def compute_statistics_for_request(
    request: StatsRequest,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    size_tag: str = DEFAULT_SIZE_TAG,
) -> StatsResult:
    """
    Compute pooled statistics for one variable/domain/split request.
    """
    root_path = Path(root_dir)
    split_manifest = load_split_manifest(request.split_manifest_path)
    dates = get_dates_for_split(split_manifest, request.split_name)
    if not dates:
        raise ValueError(
            f"Split '{request.split_name}' in manifest '{request.split_manifest_path}' contains no dates"
        )

    file_index = build_experiment_file_index(
        root_dir=root_path,
        size_tag=size_tag,
        target_variable="prcp",
        conditioning_variables=list(request.conditioning_variable_order),
        target_source="DANRA",
        conditioning_source="ERA5",
    )

    missing_dates = [date for date in dates if date not in file_index]
    if missing_dates:
        raise ValueError(
            f"Some split dates are missing from the experiment file index: {missing_dates[:10]}"
        )

    crop_spec = _build_crop_spec(request.crop)
    pooled_fields: list[np.ndarray] = []

    for date_str in dates:
        field = _load_variable_field_for_date(
            date_str=date_str,
            request=request,
            file_index=file_index,
            crop_spec=crop_spec,
            root_dir=root_path,
        )
        pooled_fields.append(field.reshape(-1))

    pooled_values = np.concatenate(pooled_fields, axis=0)
    transform_stats = _compute_transform_stats(
        values=pooled_values,
        transform_name=request.transform_name,
    )
    physical_summary = _summarize_physical_values(
        values=pooled_values,
        n_dates=len(dates),
    )

    metadata = {
        "size_tag": size_tag,
        "conditioning_variable_order": list(request.conditioning_variable_order),
        "static_variable_order": list(request.static_variable_order),
    }

    return StatsResult(
        variable=request.variable,
        source=request.source,
        transform_name=request.transform_name,
        split_name=request.split_name,
        split_manifest_path=str(request.split_manifest_path),
        date_count=len(dates),
        dates_used=list(dates),
        domain_tag=request.domain_tag,
        crop=request.crop.to_dict() if request.crop is not None else None,
        transform_stats=transform_stats,
        physical_summary=physical_summary,
        metadata=metadata,
    )



def build_default_stats_requests(
    split_manifest_path: str | Path,
    split_name: str,
    domain_tag: str,
    crop: CropConfig | None = None,
) -> list[StatsRequest]:
    """
    Build a convenient first-pass set of statistics requests for the small setup.
    """
    requests: list[StatsRequest] = []
    for spec in SUPPORTED_VARIABLE_SPECS.values():
        requests.append(
            StatsRequest(
                split_name=split_name,
                split_manifest_path=str(split_manifest_path),
                variable=spec["variable"],
                source=spec["source"],
                transform_name=spec["transform_name"],
                domain_tag=domain_tag,
                crop=crop,
            )
        )
    return requests
