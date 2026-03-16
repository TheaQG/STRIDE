"""
Temporal indexing utilities for the NorCP STRIDE adapter.

This module sits on top of `paths.py` and turns NorCP file-range discovery into
sample-level timestamp indexing. It is responsible for:

- reading time coordinates from merged NetCDF files
- building per-variable timestamp -> file lookups
- resolving duplicate timestamp coverage across overlapping file families
- intersecting timestamps across required variables
- producing a lightweight sample index that the adapter can later consume

Design notes
------------
- `paths.py` discovers candidate files and parses filename metadata.
- `indexing.py` inspects the actual `time` coordinate inside those files.
- Duplicate timestamp coverage can happen when plain files and infix-tagged
  files coexist in the same variable directory. By default, this module prefers
  files without an infix tag over infix-tagged variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import xarray as xr

from data_adapters.norcp.paths import (
    NorcpFileInfo,
    build_multi_variable_time_range_index,
)


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TimestampedFileRecord:
    """
    Mapping from one timestamp to the file that contains it.
    """

    timestamp: datetime
    file_info: NorcpFileInfo


@dataclass(frozen=True)
class NorcpSampleIndexEntry:
    """
    One aligned sample entry for the NorCP adapter.
    """

    timestamp: datetime
    scenario_name: str
    target_files: dict[str, Path] = field(default_factory=dict)
    dynamic_files: dict[str, Path] = field(default_factory=dict)
    static_files: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


# -------------------------------------------------------------------
# Time parsing helpers
# -------------------------------------------------------------------

def _to_python_datetime(value: object) -> datetime:
    """
    Convert a decoded xarray/pandas/numpy time value to a Python datetime.
    """
    # Accept only types supported by pd.Timestamp
    if isinstance(value, (int, float, str, datetime)):
        ts = pd.Timestamp(value)
    elif hasattr(value, "item") and callable(getattr(value, "item", None)):
        ts = pd.Timestamp(value.item()) # type: ignore
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)}")
    if pd.isna(ts):
        raise ValueError(f"Encountered invalid timestamp value: {value!r}")
    return ts.to_pydatetime()


def apply_timestamp_offset(
    timestamps: Iterable[datetime],
    offset_hours: float = 0.0,
) -> list[datetime]:
    """
    Shift timestamps by a fixed offset in hours.

    This is useful when different variables encode the representative time of an
    aggregation window differently. For example, NorCP 6-hour precipitation
    means can appear at 03, 09, 15, 21, while point variables are available at
    00, 06, 12, 18. In that case, using `offset_hours=-3` for precipitation
    aligns the timestamps to the interval start.
    """
    if offset_hours == 0:
        return list(timestamps)
    delta = timedelta(hours=offset_hours)
    return [timestamp + delta for timestamp in timestamps]


def read_file_timestamps(
    file_path: str | Path,
    time_variable: str = "time",
) -> list[datetime]:
    """
    Read and decode all timestamps from one NorCP NetCDF file.
    """
    path = Path(file_path)
    with xr.open_dataset(path, decode_times=True) as ds:
        if time_variable not in ds.variables and time_variable not in ds.coords:
            raise KeyError(
                f"Time variable '{time_variable}' not found in file: {path}"
            )
        values = ds[time_variable].values

    if values.ndim != 1:
        raise ValueError(
            f"Expected 1D time coordinate in file {path}, got shape {values.shape}"
        )

    return [_to_python_datetime(value) for value in values]


# -------------------------------------------------------------------
# Duplicate handling
# -------------------------------------------------------------------

def _file_priority(file_info: NorcpFileInfo) -> tuple[int, str, str]:
    """
    Ranking used to resolve duplicate timestamp coverage.

    Preference order:
    1. Plain files (no infix tag)
    2. Lexicographically smaller infix tag
    3. Lexicographically smaller path name
    """
    infix_rank = 0 if file_info.infix_tag in {None, ""} else 1
    infix_value = file_info.infix_tag or ""
    return (infix_rank, infix_value, file_info.path.name)



def choose_preferred_file(
    file_infos: Iterable[NorcpFileInfo],
) -> NorcpFileInfo:
    """
    Choose the preferred file among overlapping candidates.
    """
    file_infos = list(file_infos)
    if not file_infos:
        raise ValueError("choose_preferred_file received an empty file list")
    return sorted(file_infos, key=_file_priority)[0]


# -------------------------------------------------------------------
# Per-variable timestamp indexing
# -------------------------------------------------------------------

def build_timestamp_to_file_index(
    file_infos: Iterable[NorcpFileInfo],
    time_variable: str = "time",
    offset_hours: float = 0.0,
) -> dict[datetime, NorcpFileInfo]:
    """
    Build a timestamp -> file lookup for one variable.

    If multiple files contain the same timestamp, a deterministic preference
    rule is applied via `choose_preferred_file`.
    A fixed timestamp offset can optionally be applied before indexing.
    """
    file_infos = list(file_infos)
    timestamp_candidates: dict[datetime, list[NorcpFileInfo]] = {}

    for file_info in file_infos:
        timestamps = read_file_timestamps(file_info.path, time_variable=time_variable)
        timestamps = apply_timestamp_offset(timestamps, offset_hours=offset_hours)
        for timestamp in timestamps:
            timestamp_candidates.setdefault(timestamp, []).append(file_info)

    resolved: dict[datetime, NorcpFileInfo] = {}
    for timestamp, candidates in timestamp_candidates.items():
        resolved[timestamp] = choose_preferred_file(candidates)

    return resolved



def build_multi_variable_timestamp_index(
    variable_file_infos: dict[str, list[NorcpFileInfo]],
    time_variable: str = "time",
    variable_time_offsets: Optional[dict[str, float]] = None,
) -> dict[str, dict[datetime, NorcpFileInfo]]:
    """
    Build timestamp -> file lookups for several variables.
    """
    variable_time_offsets = variable_time_offsets or {}
    return {
        variable: build_timestamp_to_file_index(
            file_infos=file_infos,
            time_variable=time_variable,
            offset_hours=variable_time_offsets.get(variable, 0.0),
        )
        for variable, file_infos in variable_file_infos.items()
    }


# -------------------------------------------------------------------
# Timestamp intersection
# -------------------------------------------------------------------

def intersect_variable_timestamps(
    variable_timestamp_index: dict[str, dict[datetime, NorcpFileInfo]],
) -> list[datetime]:
    """
    Compute the sorted intersection of timestamps across all variables.
    """
    if not variable_timestamp_index:
        return []

    sets = [set(timestamp_map.keys()) for timestamp_map in variable_timestamp_index.values()]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


# -------------------------------------------------------------------
# High-level builders
# -------------------------------------------------------------------

def build_aligned_timestamp_index(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variables: Iterable[str],
    time_variable: str = "time",
    variable_time_offsets: Optional[dict[str, float]] = None,
) -> tuple[dict[str, dict[datetime, NorcpFileInfo]], list[datetime]]:
    """
    Discover files for several variables and compute their common timestamps.
    """
    variable_file_infos = build_multi_variable_time_range_index(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=spatial_tag,
        temporal_tag=temporal_tag,
        variables=variables,
    )
    variable_timestamp_index = build_multi_variable_timestamp_index(
        variable_file_infos=variable_file_infos,
        time_variable=time_variable,
        variable_time_offsets=variable_time_offsets,
    )
    common_timestamps = intersect_variable_timestamps(variable_timestamp_index)
    return variable_timestamp_index, common_timestamps



def build_norcp_sample_index(
    root_dir: str | Path,
    scenario_name: str,
    target_variables: Iterable[str],
    dynamic_variables: Iterable[str],
    target_spatial_tag: str,
    dynamic_spatial_tag: str,
    temporal_tag: str,
    static_files: Optional[dict[str, str | Path]] = None,
    time_variable: str = "time",
    target_time_offsets: Optional[dict[str, float]] = None,
    dynamic_time_offsets: Optional[dict[str, float]] = None,
) -> list[NorcpSampleIndexEntry]:
    """
    Build a sample index for paired target and dynamic NorCP variables.

    Parameters
    ----------
    root_dir:
        Top-level NorCP root.
    scenario_name:
        Scenario directory name, e.g. 'ECMWF-ERAINT'.
    target_variables:
        Canonical/raw variable names to use on the HR grid.
    dynamic_variables:
        Canonical/raw variable names to use on the LR grid.
    target_spatial_tag:
        Usually '3km'.
    dynamic_spatial_tag:
        Usually '12km'.
    temporal_tag:
        Usually '3hr' or '6hr'.
    static_files:
        Optional mapping from static variable name to already-resolved file path.
    time_variable:
        Time coordinate name inside NetCDF files.
    target_time_offsets:
        Optional per-target-variable timestamp shifts in hours.
    dynamic_time_offsets:
        Optional per-dynamic-variable timestamp shifts in hours.
    """
    target_variables = list(target_variables)
    dynamic_variables = list(dynamic_variables)
    static_files_dict: dict[str, Path] = {
        key: Path(value)
        for key, value in (static_files or {}).items()
    }

    target_index, target_common = build_aligned_timestamp_index(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=target_spatial_tag,
        temporal_tag=temporal_tag,
        variables=target_variables,
        time_variable=time_variable,
        variable_time_offsets=target_time_offsets,
    )
    dynamic_index, dynamic_common = build_aligned_timestamp_index(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=dynamic_spatial_tag,
        temporal_tag=temporal_tag,
        variables=dynamic_variables,
        time_variable=time_variable,
        variable_time_offsets=dynamic_time_offsets,
    )

    common_timestamps = sorted(set(target_common).intersection(dynamic_common))

    sample_index: list[NorcpSampleIndexEntry] = []
    for timestamp in common_timestamps:
        target_files = {
            variable: target_index[variable][timestamp].path
            for variable in target_variables
        }
        dynamic_files = {
            variable: dynamic_index[variable][timestamp].path
            for variable in dynamic_variables
        }
        metadata = {
            "target_spatial_tag": target_spatial_tag,
            "dynamic_spatial_tag": dynamic_spatial_tag,
            "temporal_tag": temporal_tag,
            "target_time_offsets": str(target_time_offsets or {}),
            "dynamic_time_offsets": str(dynamic_time_offsets or {}),
        }
        sample_index.append(
            NorcpSampleIndexEntry(
                timestamp=timestamp,
                scenario_name=scenario_name,
                target_files=target_files,
                dynamic_files=dynamic_files,
                static_files=static_files_dict,
                metadata=metadata,
            )
        )

    return sample_index


# -------------------------------------------------------------------
# Small summaries for debugging
# -------------------------------------------------------------------

def describe_timestamp_index(
    timestamp_index: dict[datetime, NorcpFileInfo],
    limit: int = 5,
) -> list[dict[str, str]]:
    """
    Convert a timestamp -> file index to a compact debug summary.
    """
    rows: list[dict[str, str]] = []
    for idx, (timestamp, file_info) in enumerate(sorted(timestamp_index.items())):
        if idx >= limit:
            break
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "path": str(file_info.path),
                "variable": file_info.variable,
                "spatial_tag": file_info.spatial_tag,
                "temporal_tag": file_info.temporal_tag,
                "infix_tag": file_info.infix_tag or "",
            }
        )
    return rows



def describe_sample_index(
    sample_index: Iterable[NorcpSampleIndexEntry],
    limit: int = 5,
) -> list[dict[str, object]]:
    """
    Convert a sample index to a compact debug summary.
    """
    rows: list[dict[str, object]] = []
    for idx, entry in enumerate(sample_index):
        if idx >= limit:
            break
        rows.append(
            {
                "timestamp": entry.timestamp.isoformat(),
                "scenario_name": entry.scenario_name,
                "target_files": {k: str(v) for k, v in entry.target_files.items()},
                "dynamic_files": {k: str(v) for k, v in entry.dynamic_files.items()},
                "static_files": {k: str(v) for k, v in entry.static_files.items()},
                "metadata": entry.metadata,
            }
        )
    return rows