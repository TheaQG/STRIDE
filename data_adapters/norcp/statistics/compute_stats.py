"""
Split-aware statistics computation for the NorCP STRIDE adapter.

This module computes variable-wise global statistics from a selected split using
already standardized NorCP data loading steps:
    - source-aware loading
    - unit conversion
    - optional timestamp-offset handling
    - optional crop selection
    - static consistency support

Statistics are computed by pooling values across all selected timestamps for one
variable and one domain.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from data_adapters.norcp.features import load_field_at_timestamp, load_static_field
from data_adapters.norcp.regions import CropSpec, maybe_crop_2d
from data_adapters.norcp.splits.splits import get_split_timestamps, load_split_manifest, parse_timestamp
from data_adapters.norcp.statistics.schemas import (
    CropConfig,
    StatsRequest,
    StatsResult,
    StatsSummary,
)
from data_adapters.norcp.statistics.cdo_backend import compute_stats_for_request_cdo
from data_adapters.norcp.variable_registry import get_source_raw_field_name


DEFAULT_LOG_EPSILON = 1e-6


def _build_crop_spec(crop: CropConfig | None) -> CropSpec | None:
    if crop is None:
        return None
    return CropSpec(
        top=crop.top,
        left=crop.left,
        height=crop.height,
        width=crop.width,
    )



def _infer_raw_field_name(variable: str, source: str) -> str:
    raw_field = get_source_raw_field_name(variable, source)
    if raw_field is None:
        raise ValueError(
            f"No raw field mapping registered for variable='{variable}', source='{source}'"
        )
    return raw_field



def _resolve_variable_dir_name(variable: str, spatial_tag: str) -> str:
    if spatial_tag == "3km":
        source = "NORCP_HR"
    elif spatial_tag == "12km":
        source = "NORCP_LR"
    else:
        raise ValueError(f"Unsupported spatial_tag '{spatial_tag}'")

    raw_name = get_source_raw_field_name(variable, source)
    if raw_name is None and variable == "topo":
        raw_name = get_source_raw_field_name(variable, "NORCP_STATIC")
    return raw_name or variable



def _default_static_file_path(root_dir: Path, scenario_name: str, spatial_tag: str, variable: str) -> Path:
    raw_name = _resolve_variable_dir_name(variable, spatial_tag)
    return root_dir / scenario_name / spatial_tag / "fx" / raw_name / f"{raw_name}_{spatial_tag}_fx.nc"



def _default_variable_dir(
    root_dir: Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variable: str,
) -> Path:
    raw_name = _resolve_variable_dir_name(variable, spatial_tag)
    return root_dir / scenario_name / spatial_tag / temporal_tag / raw_name



def _list_candidate_nc_files(variable_dir: Path) -> list[Path]:
    if not variable_dir.exists():
        raise FileNotFoundError(f"NorCP variable directory does not exist: {variable_dir}")
    files = sorted(variable_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in NorCP variable directory: {variable_dir}")
    return files



def _to_python_datetimes(values: Any) -> list[datetime]:
    return [pd.Timestamp(v).to_pydatetime() for v in values]



def _build_file_time_cache(file_paths: list[Path], raw_field_name: str) -> dict[Path, tuple[datetime, datetime]]:
    cache: dict[Path, tuple[datetime, datetime]] = {}
    for path in file_paths:
        with xr.open_dataset(path, decode_times=True) as ds:
            if raw_field_name not in ds.data_vars:
                raise KeyError(
                    f"Raw field '{raw_field_name}' not found in file '{path}'. Available: {list(ds.data_vars)}"
                )
            da = ds[raw_field_name]
            if "time" not in da.coords and "time" not in da.dims:
                raise KeyError(f"Expected time dimension for file '{path}'")
            times = _to_python_datetimes(da["time"].values)
            if not times:
                raise ValueError(f"No decoded times found in file '{path}'")
            cache[path] = (times[0], times[-1])
    return cache



def _reverse_aligned_timestamp(timestamp: datetime, offset_hours: float) -> datetime:
    return timestamp - pd.Timedelta(hours=float(offset_hours)).to_pytimedelta()



def _select_file_for_timestamp(
    file_paths: list[Path],
    file_time_cache: dict[Path, tuple[datetime, datetime]],
    *,
    aligned_timestamp: datetime,
    variable_time_offset_hours: float,
) -> Path:
    file_timestamp = _reverse_aligned_timestamp(aligned_timestamp, variable_time_offset_hours)
    for path in file_paths:
        start, end = file_time_cache[path]
        if start <= file_timestamp <= end:
            return path
    raise FileNotFoundError(
        f"No NorCP file covers aligned_timestamp={aligned_timestamp.isoformat()} "
        f"(file timestamp {file_timestamp.isoformat()})"
    )



def _summarize_physical_values(values: np.ndarray, n_samples: int) -> StatsSummary:
    values = np.asarray(values, dtype=np.float32)
    return StatsSummary(
        n_samples=n_samples,
        n_values=int(values.size),
        min=float(np.nanmin(values)),
        max=float(np.nanmax(values)),
        mean=float(np.nanmean(values)),
        std=float(np.nanstd(values)),
    )



def _compute_additional_distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "median": float(np.nanmedian(values)),
        "p25": float(np.nanpercentile(values, 25)),
        "p75": float(np.nanpercentile(values, 75)),
    }



def _compute_zscore_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    stats = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
    }
    stats.update(_compute_additional_distribution_stats(values))
    return stats



def _compute_log_zscore_stats(values: np.ndarray, epsilon: float = DEFAULT_LOG_EPSILON) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    clipped = np.clip(values, 0.0, None)
    logged = np.log(clipped + epsilon).astype(np.float32)
    stats = {
        "log_mean": float(np.nanmean(logged)),
        "log_std": float(np.nanstd(logged)),
        "epsilon": float(epsilon),
    }
    stats.update(_compute_additional_distribution_stats(values))
    stats.update(
        {
            "min_log": float(np.nanmin(logged)),
            "max_log": float(np.nanmax(logged)),
            "median_log": float(np.nanmedian(logged)),
            "p25_log": float(np.nanpercentile(logged, 25)),
            "p75_log": float(np.nanpercentile(logged, 75)),
        }
    )
    return stats



def _compute_transform_stats(values: np.ndarray, transform_name: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)

    if transform_name == "identity":
        return _compute_additional_distribution_stats(values)
    if transform_name == "zscore":
        return _compute_zscore_stats(values)
    if transform_name == "log_zscore":
        return _compute_log_zscore_stats(values)

    raise ValueError(f"Unsupported transform_name '{transform_name}'")



def _load_static_values(request: StatsRequest, crop_spec: CropSpec | None) -> tuple[np.ndarray, list[str]]:
    root_dir = Path(request.root_dir)
    static_path = Path(request.static_file_path) if request.static_file_path is not None else _default_static_file_path(
        root_dir=root_dir,
        scenario_name=request.scenario_name,
        spatial_tag=request.spatial_tag,
        variable=request.variable,
    )
    field = load_static_field(
        file_path=static_path,
        variable=request.variable,
        source=request.source,
    )
    array = maybe_crop_2d(field.array, crop_spec)
    return np.asarray(array, dtype=np.float32).reshape(-1), [str(static_path)]



def _load_temporal_values(request: StatsRequest, crop_spec: CropSpec | None) -> tuple[np.ndarray, list[str], list[str]]:
    manifest = load_split_manifest(request.split_manifest_path)
    split_timestamps = get_split_timestamps(manifest, request.split_name)
    aligned_datetimes = [parse_timestamp(ts) for ts in split_timestamps]

    if request.temporal_tag is None:
        raise ValueError("Temporal statistics require request.temporal_tag to be set")

    root_dir = Path(request.root_dir)
    variable_dir = _default_variable_dir(
        root_dir=root_dir,
        scenario_name=request.scenario_name,
        spatial_tag=request.spatial_tag,
        temporal_tag=request.temporal_tag,
        variable=request.variable,
    )
    file_paths = _list_candidate_nc_files(variable_dir)
    raw_field_name = _infer_raw_field_name(request.variable, request.source)
    file_time_cache = _build_file_time_cache(file_paths, raw_field_name)

    pooled_values: list[np.ndarray] = []
    used_timestamps: list[str] = []
    used_files: set[str] = set()

    for timestamp_str, aligned_timestamp in zip(split_timestamps, aligned_datetimes):
        selected_file = _select_file_for_timestamp(
            file_paths,
            file_time_cache,
            aligned_timestamp=aligned_timestamp,
            variable_time_offset_hours=request.variable_time_offset_hours,
        )
        field = load_field_at_timestamp(
            file_path=selected_file,
            variable=request.variable,
            source=request.source,
            timestamp=aligned_timestamp,
            file_time_offset_hours=request.variable_time_offset_hours,
        )
        array = maybe_crop_2d(field.array, crop_spec)
        pooled_values.append(np.asarray(array, dtype=np.float32).reshape(-1))
        used_timestamps.append(timestamp_str)
        used_files.add(str(selected_file))

    if not pooled_values:
        raise ValueError(
            f"No temporal values were loaded for variable='{request.variable}', source='{request.source}'"
        )

    return np.concatenate(pooled_values, axis=0), used_timestamps, sorted(used_files)



def compute_stats_for_request_python(request: StatsRequest) -> StatsResult:
    """
    Compute one NorCP statistics result from a declarative request.
    """
    print(
        "[stats/python] Starting statistics computation | "
        f"variable={request.variable} | source={request.source} | "
        f"split={request.split_name} | domain={request.domain_tag}"
    )
    crop_spec = _build_crop_spec(request.crop)

    if request.source == "NORCP_STATIC":
        print("[stats/python] Loading static field")
        values, files_used = _load_static_values(request, crop_spec)
        timestamps_used = ["static"]
        sample_count = 1
    else:
        print("[stats/python] Loading temporal values for selected split")
        values, timestamps_used, files_used = _load_temporal_values(request, crop_spec)
        sample_count = len(timestamps_used)
        print(
            "[stats/python] Loaded temporal pool | "
            f"n_timestamps={sample_count} | n_values={int(values.size)}"
        )

    print("[stats/python] Computing pooled summary and transform statistics")
    physical_summary = _summarize_physical_values(values, sample_count)
    transform_stats = _compute_transform_stats(values, request.transform_name)

    metadata = dict(request.metadata)
    metadata.update(
        {
            "backend": "python",
            "root_dir": request.root_dir,
            "files_used": files_used,
            "variable_time_offset_hours": float(request.variable_time_offset_hours),
        }
    )

    print(
        "[stats/python] Finished statistics computation | "
        f"variable={request.variable} | source={request.source}"
    )
    return StatsResult(
        scenario_name=request.scenario_name,
        variable=request.variable,
        source=request.source,
        transform_name=request.transform_name,
        split_name=request.split_name,
        split_manifest_path=request.split_manifest_path,
        sample_count=sample_count,
        timestamps_used=timestamps_used,
        domain_tag=request.domain_tag,
        spatial_tag=request.spatial_tag,
        temporal_tag=request.temporal_tag,
        crop=None if request.crop is None else request.crop.to_dict(),
        transform_stats=transform_stats,
        physical_summary=physical_summary,
        metadata=metadata,
    )


def compute_stats_for_request(request: StatsRequest) -> StatsResult:
    """
    Compute one NorCP statistics result using the configured backend.

    Supported backends:
    - "python": iterate through timestamps in Python and pool values directly
    - "cdo": use CDO for temporal/spatial NetCDF subsetting, then compute exact
      pooled transform stats in Python from the reduced file

    Notes
    -----
    The current NorCP default should be:
    - full spatial domain
    - training split only
    - crop=None

    Crop-aware statistics are still supported for later setups where training is
    performed on a fixed subdomain.
    """
    backend = str(getattr(request, "backend", "python")).lower()

    if backend == "python":
        return compute_stats_for_request_python(request)
    if backend == "cdo":
        return compute_stats_for_request_cdo(request)

    raise ValueError(
        f"Unsupported statistics backend '{backend}'. Expected one of ['python', 'cdo']"
    )