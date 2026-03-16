

"""
CDO-backed statistics utilities for the NorCP STRIDE adapter.

This backend uses CDO for the heavy NetCDF slicing work:
- selecting variables
- selecting split-aligned time ranges
- optionally cropping with index-space boxes
- optionally merging multiple NetCDF files over time

After the reduced subset is produced, Python performs the final pooled
statistics exactly over all retained values. This hybrid approach keeps the
results trustworthy while moving the expensive IO-heavy work out of Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from data_adapters.norcp.regions import CropSpec
from data_adapters.norcp.splits.splits import get_split_timestamps, load_split_manifest, parse_timestamp
from data_adapters.norcp.statistics.schemas import CropConfig, StatsRequest, StatsResult, StatsSummary
from data_adapters.norcp.variable_registry import get_source_raw_field_name


DEFAULT_CDO_EXECUTABLE = "cdo"
DEFAULT_LOG_EPSILON = 1e-6


@dataclass(frozen=True)
class CDOSelectionSummary:
    """
    Lightweight summary of the CDO selection that was applied.
    """

    raw_start_time: str | None
    raw_end_time: str | None
    input_files: list[str]
    selected_variable: str
    crop: dict[str, int] | None
    reduced_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_start_time": self.raw_start_time,
            "raw_end_time": self.raw_end_time,
            "input_files": self.input_files,
            "selected_variable": self.selected_variable,
            "crop": self.crop,
            "reduced_file": self.reduced_file,
        }


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


def _assert_regular_spacing(timestamps: Sequence[datetime]) -> None:
    if len(timestamps) <= 2:
        return
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    first_delta = deltas[0]
    for delta in deltas[1:]:
        if delta != first_delta:
            raise ValueError(
                "CDO backend currently expects regularly spaced timestamps within each split. "
                f"Found deltas {sorted({str(d) for d in deltas})}"
            )


def _assert_contiguous_coverage(timestamps: Sequence[datetime], *, offset_hours: float) -> tuple[datetime, datetime]:
    if not timestamps:
        raise ValueError("No timestamps available for CDO selection")
    _assert_regular_spacing(timestamps)
    raw_times = [_reverse_aligned_timestamp(ts, offset_hours) for ts in timestamps]
    return raw_times[0], raw_times[-1]


def _select_overlapping_files(
    file_paths: list[Path],
    file_time_cache: dict[Path, tuple[datetime, datetime]],
    *,
    raw_start: datetime,
    raw_end: datetime,
) -> list[Path]:
    selected: list[Path] = []
    for path in file_paths:
        start, end = file_time_cache[path]
        if end < raw_start or start > raw_end:
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError(
            f"No NorCP files overlap requested raw time range {raw_start.isoformat()} -> {raw_end.isoformat()}"
        )
    return selected


def _ensure_cdo_available(cdo_executable: str = DEFAULT_CDO_EXECUTABLE) -> str:
    resolved = shutil.which(cdo_executable)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find CDO executable '{cdo_executable}' on PATH"
        )
    return resolved


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "CDO command failed\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def _format_cdo_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _crop_to_selindexbox(crop: CropSpec | None) -> str | None:
    if crop is None:
        return None
    x1 = crop.left + 1
    x2 = crop.left + crop.width
    y1 = crop.top + 1
    y2 = crop.top + crop.height
    return f"{x1},{x2},{y1},{y2}"


def _merge_if_needed(
    *,
    cdo_executable: str,
    input_files: list[Path],
    work_dir: Path,
) -> Path:
    if len(input_files) == 1:
        return input_files[0]

    merged_path = work_dir / "merged.nc"
    command = [cdo_executable, "-O", "-L", "mergetime", *[str(path) for path in input_files], str(merged_path)]
    _run_command(command)
    return merged_path


def _subset_temporal_file_with_cdo(
    *,
    cdo_executable: str,
    merged_input_path: Path,
    output_path: Path,
    raw_field_name: str,
    raw_start: datetime,
    raw_end: datetime,
    crop: CropSpec | None,
) -> None:
    current_input = merged_input_path

    selected_name_path = output_path.parent / "selected_name.nc"
    command = [
        cdo_executable,
        "-O",
        "-L",
        f"selname,{raw_field_name}",
        str(current_input),
        str(selected_name_path),
    ]
    _run_command(command)
    current_input = selected_name_path

    selected_time_path = output_path.parent / "selected_time.nc"
    command = [
        cdo_executable,
        "-O",
        "-L",
        f"seldate,{_format_cdo_datetime(raw_start)},{_format_cdo_datetime(raw_end)}",
        str(current_input),
        str(selected_time_path),
    ]
    _run_command(command)
    current_input = selected_time_path

    if crop is not None:
        crop_spec = _crop_to_selindexbox(crop)
        if crop_spec is None:
            raise RuntimeError("Unexpected missing crop spec")
        command = [
            cdo_executable,
            "-O",
            "-L",
            f"selindexbox,{crop_spec}",
            str(current_input),
            str(output_path),
        ]
        _run_command(command)
    else:
        shutil.copyfile(current_input, output_path)


def _subset_static_file_with_cdo(
    *,
    cdo_executable: str,
    input_path: Path,
    output_path: Path,
    raw_field_name: str,
    crop: CropSpec | None,
) -> None:
    current_input = input_path

    selected_name_path = output_path.parent / "selected_name_static.nc"
    command = [
        cdo_executable,
        "-O",
        "-L",
        f"selname,{raw_field_name}",
        str(current_input),
        str(selected_name_path),
    ]
    _run_command(command)
    current_input = selected_name_path

    if crop is not None:
        crop_spec = _crop_to_selindexbox(crop)
        if crop_spec is None:
            raise RuntimeError("Unexpected missing crop spec")
        command = [
            cdo_executable,
            "-O",
            "-L",
            f"selindexbox,{crop_spec}",
            str(current_input),
            str(output_path),
        ]
        _run_command(command)
    else:
        shutil.copyfile(current_input, output_path)


def _extract_values_from_reduced_file(reduced_file: Path, raw_field_name: str) -> np.ndarray:
    with xr.open_dataset(reduced_file, decode_times=True) as ds:
        if raw_field_name not in ds.data_vars:
            raise KeyError(
                f"Raw field '{raw_field_name}' not found in reduced file '{reduced_file}'. "
                f"Available variables: {list(ds.data_vars)}"
            )
        da = ds[raw_field_name]
        values = np.asarray(da.values, dtype=np.float32)
    return values.reshape(-1)


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


def build_cdo_selection_summary(
    request: StatsRequest,
    *,
    raw_start: datetime | None,
    raw_end: datetime | None,
    input_files: Iterable[Path],
    raw_field_name: str,
    crop: CropSpec | None,
    reduced_file: Path,
) -> CDOSelectionSummary:
    return CDOSelectionSummary(
        raw_start_time=None if raw_start is None else raw_start.isoformat(),
        raw_end_time=None if raw_end is None else raw_end.isoformat(),
        input_files=[str(path) for path in input_files],
        selected_variable=raw_field_name,
        crop=None if crop is None else crop.to_dict(),
        reduced_file=str(reduced_file),
    )


def compute_stats_for_request_cdo(
    request: StatsRequest,
    *,
    cdo_executable: str = DEFAULT_CDO_EXECUTABLE,
) -> StatsResult:
    """
    Compute one NorCP statistics result using a CDO-backed subset workflow.

    CDO performs the expensive temporal/spatial selection on NetCDF files. The
    final pooled statistics are then computed exactly in Python from the reduced
    subset.
    """
    print(
        "[stats/cdo] Starting statistics computation | "
        f"variable={request.variable} | source={request.source} | "
        f"split={request.split_name} | domain={request.domain_tag}"
    )
    cdo_path = _ensure_cdo_available(cdo_executable)
    crop_spec = _build_crop_spec(request.crop)
    root_dir = Path(request.root_dir)
    raw_field_name = _infer_raw_field_name(request.variable, request.source)

    with tempfile.TemporaryDirectory(prefix="norcp_stats_cdo_") as tmpdir:
        work_dir = Path(tmpdir)
        reduced_file = work_dir / "reduced_subset.nc"

        if request.source == "NORCP_STATIC":
            print("[stats/cdo] Using static-field CDO subset path")
            static_path = (
                Path(request.static_file_path)
                if request.static_file_path is not None
                else _default_static_file_path(
                    root_dir=root_dir,
                    scenario_name=request.scenario_name,
                    spatial_tag=request.spatial_tag,
                    variable=request.variable,
                )
            )
            _subset_static_file_with_cdo(
                cdo_executable=cdo_path,
                input_path=static_path,
                output_path=reduced_file,
                raw_field_name=raw_field_name,
                crop=crop_spec,
            )
            pooled_values = _extract_values_from_reduced_file(reduced_file, raw_field_name)
            timestamps_used = ["static"]
            sample_count = 1
            selection_summary = build_cdo_selection_summary(
                request,
                raw_start=None,
                raw_end=None,
                input_files=[static_path],
                raw_field_name=raw_field_name,
                crop=crop_spec,
                reduced_file=reduced_file,
            )
        else:
            print("[stats/cdo] Using temporal-field CDO subset path")
            if request.temporal_tag is None:
                raise ValueError("Temporal CDO statistics require request.temporal_tag to be set")

            manifest = load_split_manifest(request.split_manifest_path)
            split_timestamps = get_split_timestamps(manifest, request.split_name)
            aligned_datetimes = [parse_timestamp(ts) for ts in split_timestamps]
            raw_start, raw_end = _assert_contiguous_coverage(
                aligned_datetimes,
                offset_hours=request.variable_time_offset_hours,
            )

            variable_dir = _default_variable_dir(
                root_dir=root_dir,
                scenario_name=request.scenario_name,
                spatial_tag=request.spatial_tag,
                temporal_tag=request.temporal_tag,
                variable=request.variable,
            )
            file_paths = _list_candidate_nc_files(variable_dir)
            file_time_cache = _build_file_time_cache(file_paths, raw_field_name)
            selected_files = _select_overlapping_files(
                file_paths,
                file_time_cache,
                raw_start=raw_start,
                raw_end=raw_end,
            )
            print(
                "[stats/cdo] Selected overlapping files | "
                f"n_files={len(selected_files)} | "
                f"raw_start={raw_start.isoformat()} | raw_end={raw_end.isoformat()}"
            )

            print("[stats/cdo] Running CDO merge/subset operations")
            merged_input = _merge_if_needed(
                cdo_executable=cdo_path,
                input_files=selected_files,
                work_dir=work_dir,
            )
            _subset_temporal_file_with_cdo(
                cdo_executable=cdo_path,
                merged_input_path=merged_input,
                output_path=reduced_file,
                raw_field_name=raw_field_name,
                raw_start=raw_start,
                raw_end=raw_end,
                crop=crop_spec,
            )
            pooled_values = _extract_values_from_reduced_file(reduced_file, raw_field_name)
            timestamps_used = list(split_timestamps)
            sample_count = len(timestamps_used)
            selection_summary = build_cdo_selection_summary(
                request,
                raw_start=raw_start,
                raw_end=raw_end,
                input_files=selected_files,
                raw_field_name=raw_field_name,
                crop=crop_spec,
                reduced_file=reduced_file,
            )

        print(
            "[stats/cdo] Reduced subset ready | "
            f"sample_count={sample_count} | n_values={int(pooled_values.size)}"
        )
        print("[stats/cdo] Computing pooled summary and transform statistics")
        physical_summary = _summarize_physical_values(pooled_values, sample_count)
        transform_stats = _compute_transform_stats(pooled_values, request.transform_name)

        metadata = dict(request.metadata)
        metadata.update(
            {
                "backend": "cdo",
                "cdo_executable": cdo_path,
                "selection": selection_summary.to_dict(),
                "variable_time_offset_hours": float(request.variable_time_offset_hours),
            }
        )

        print(
            "[stats/cdo] Finished statistics computation | "
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