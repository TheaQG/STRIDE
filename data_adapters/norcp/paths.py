"""
Path and file-discovery utilities for the NorCP STRIDE adapter.

This module is responsible for discovering NorCP NetCDF files, parsing their
spatial/temporal tags and optional infix tags from filenames, and building timestamp-aware lookup maps
that later adapter components can consume.

Design goals
------------
- Keep file discovery separate from loading and unit conversion.
- Support both HR (3 km) and LR (12 km) NorCP products.
- Support scenario-aware directory structures.
- Work with merged multi-timestep NetCDF files whose filenames encode a start
  and end timestamp range rather than one file per timestep.
- Accept both plain range filenames and range filenames with an extra infix token,
  e.g. 'July', inserted before the timestamp range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Iterable, Optional
from data_adapters.norcp.variable_registry import get_source_raw_field_name


# -------------------------------------------------------------------
# Filename parsing
# -------------------------------------------------------------------

NORCP_FILENAME_PATTERN = re.compile(
    r"^(?P<variable>[A-Za-z0-9]+)"
    r"_(?P<spatial_tag>3km|12km)"
    r"_(?P<temporal_tag>1hr|3hr|6hr)"
    r"(?:_(?P<infix>[A-Za-z][A-Za-z0-9_-]*))?"
    r"_(?P<start>\d{10,14})-(?P<end>\d{10,14})"
    r"(?:_[A-Za-z0-9_-]+)?"
    r"\.nc$"
)

TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
    "%Y%m%d%H",
    "%Y%m%d",
)


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class NorcpFileInfo:
    """
    Parsed metadata for one NorCP NetCDF file.
    """

    path: Path
    variable: str
    spatial_tag: str
    temporal_tag: str
    infix_tag: str | None
    start_time: datetime
    end_time: datetime

    @property
    def source_grid(self) -> str:
        """
        Return a simple grid label inferred from the spatial tag.
        """
        if self.spatial_tag == "3km":
            return "hr"
        if self.spatial_tag == "12km":
            return "lr"
        raise ValueError(f"Unsupported spatial tag: {self.spatial_tag}")


# -------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------

def _ensure_existing_dir(path: str | Path, label: str) -> Path:
    """
    Resolve and validate a directory path.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved



def parse_norcp_timestamp(token: str) -> datetime:
    """
    Parse a NorCP timestamp token.

    Supports tokens such as:
        - YYYYMMDD
        - YYYYMMDDHH
        - YYYYMMDDHHMM
        - YYYYMMDDHHMMSS
    """
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse NorCP timestamp token: {token}")



def parse_norcp_filename(path: str | Path) -> NorcpFileInfo:
    """
    Parse NorCP metadata encoded in a NetCDF filename.

    Expected pattern examples:
        pr_12km_6hr_199801010300-201812312100.nc
        ta1000_12km_6hr_199801010000-201812311800.nc
        tas_3km_3hr_198601010000-200512312100.nc
        pr_12km_6hr_July_199801010300-201812312100.nc
    """
    file_path = Path(path)
    match = NORCP_FILENAME_PATTERN.match(file_path.name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected NorCP pattern: {file_path.name}"
        )

    data = match.groupdict()
    start_time = parse_norcp_timestamp(data["start"])
    end_time = parse_norcp_timestamp(data["end"])

    return NorcpFileInfo(
        path=file_path,
        variable=data["variable"],
        spatial_tag=data["spatial_tag"],
        temporal_tag=data["temporal_tag"],
        infix_tag=data.get("infix"),
        start_time=start_time,
        end_time=end_time,
    )



def file_covers_timestamp(file_info: NorcpFileInfo, timestamp: datetime) -> bool:
    """
    Return whether a parsed NorCP file covers a given timestamp.
    """
    return file_info.start_time <= timestamp <= file_info.end_time


def resolve_variable_dir_name(variable: str, spatial_tag: str) -> str:
    """
    Resolve the on-disk NorCP variable directory name from a canonical variable name.

    Examples
    --------
    - prcp on 3km/12km -> pr
    - temp on 3km/12km -> tas
    - hus1000 on 12km  -> hus1000
    - topo on 3km      -> orog

    Notes
    -----
    This maps STRIDE canonical names to raw NorCP variable directory names.
    For NorCP, the source is inferred from the spatial tag:
        - 3km  -> NORCP_HR
        - 12km -> NORCP_LR
    with a special fallback for topography to NORCP_STATIC when needed.
    """
    if spatial_tag == "3km":
        source = "NORCP_HR"
    elif spatial_tag == "12km":
        source = "NORCP_LR"
    else:
        raise ValueError(f"Unsupported spatial tag: {spatial_tag}")

    try:
        raw_name = get_source_raw_field_name(variable, source)
    except KeyError:
        raw_name = None

    if raw_name is None and variable == "topo":
        raw_name = get_source_raw_field_name(variable, "NORCP_STATIC")

    return raw_name or variable


# -------------------------------------------------------------------
# Root and directory resolution
# -------------------------------------------------------------------

def get_dataset_root(root_dir: str | Path) -> Path:
    """
    Resolve the top-level NorCP dataset root.
    """
    return _ensure_existing_dir(root_dir, label="NorCP dataset root")



def get_scenario_root(root_dir: str | Path, scenario_name: str) -> Path:
    """
    Resolve a scenario root under the NorCP dataset root.

    Expected layout example:
        <root>/<scenario_name>/
    """
    dataset_root = get_dataset_root(root_dir)
    scenario_root = dataset_root / scenario_name
    return _ensure_existing_dir(scenario_root, label=f"NorCP scenario root '{scenario_name}'")



def get_resolution_root(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
) -> Path:
    """
    Resolve a resolution/frequency root.

    Expected layout example:
        <root>/<scenario_name>/<spatial_tag>/<temporal_tag>/

    Example:
        .../ECMWF-ERAINT/12km/6hr/
        .../ECMWF-ERAINT/3km/6hr/
    """
    scenario_root = get_scenario_root(root_dir=root_dir, scenario_name=scenario_name)
    resolution_root = scenario_root / spatial_tag / temporal_tag
    return _ensure_existing_dir(
        resolution_root,
        label=(
            f"NorCP resolution root for scenario='{scenario_name}', "
            f"spatial_tag='{spatial_tag}', temporal_tag='{temporal_tag}'"
        ),
    )



def get_variable_dir(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variable: str,
) -> Path:
    """
    Resolve the directory containing one variable.

    Expected layout example:
        <root>/<scenario_name>/<spatial_tag>/<temporal_tag>/<variable>/
    """
    resolution_root = get_resolution_root(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=spatial_tag,
        temporal_tag=temporal_tag,
    )
    resolved_variable = resolve_variable_dir_name(variable, spatial_tag)
    variable_dir = resolution_root / resolved_variable
    return _ensure_existing_dir(
        variable_dir,
        label=(
            f"NorCP variable directory for scenario='{scenario_name}', "
            f"spatial_tag='{spatial_tag}', temporal_tag='{temporal_tag}', variable='{variable}', resolved_variable='{resolved_variable}'"
        ),
    )


# -------------------------------------------------------------------
# File discovery
# -------------------------------------------------------------------

def list_variable_files(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variable: str,
) -> list[Path]:
    """
    List all NetCDF files for one NorCP variable directory.
    """
    variable_dir = get_variable_dir(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=spatial_tag,
        temporal_tag=temporal_tag,
        variable=variable,
    )
    return sorted(variable_dir.glob("*.nc"))



def list_variable_file_infos(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variable: str,
) -> list[NorcpFileInfo]:
    """
    Discover and parse all files for one NorCP variable.
    """
    infos: list[NorcpFileInfo] = []
    for path in list_variable_files(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=spatial_tag,
        temporal_tag=temporal_tag,
        variable=variable,
    ):
        info = parse_norcp_filename(path)
        resolved_variable = resolve_variable_dir_name(variable, spatial_tag)
        if info.variable != resolved_variable:
            raise ValueError(
                f"Variable directory/file mismatch for {path}: "
                f"requested variable='{variable}', resolved directory variable='{resolved_variable}', "
                f"parsed filename variable='{info.variable}'"
            )
        if info.spatial_tag != spatial_tag:
            raise ValueError(
                f"Spatial tag mismatch for {path}: "
                f"expected '{spatial_tag}', parsed '{info.spatial_tag}'"
            )
        if info.temporal_tag != temporal_tag:
            raise ValueError(
                f"Temporal tag mismatch for {path}: "
                f"expected '{temporal_tag}', parsed '{info.temporal_tag}'"
            )
        infos.append(info)
    return infos



def build_variable_time_range_index(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variable: str,
) -> list[NorcpFileInfo]:
    """
    Return the parsed file list for one variable, sorted by start time.

    This is a range index, not a per-timestep index. It is the appropriate first
    abstraction for NorCP because each file can contain many timestamps.
    """
    infos = list_variable_file_infos(
        root_dir=root_dir,
        scenario_name=scenario_name,
        spatial_tag=spatial_tag,
        temporal_tag=temporal_tag,
        variable=variable,
    )
    return sorted(infos, key=lambda info: (info.start_time, info.end_time, info.path.name))



def build_multi_variable_time_range_index(
    root_dir: str | Path,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
    variables: Iterable[str],
) -> dict[str, list[NorcpFileInfo]]:
    """
    Build a parsed file-range index for several variables.
    """
    return {
        variable: build_variable_time_range_index(
            root_dir=root_dir,
            scenario_name=scenario_name,
            spatial_tag=spatial_tag,
            temporal_tag=temporal_tag,
            variable=variable,
        )
        for variable in variables
    }


# -------------------------------------------------------------------
# Lookup helpers
# -------------------------------------------------------------------

def find_file_covering_timestamp(
    file_infos: Iterable[NorcpFileInfo],
    timestamp: datetime,
) -> Optional[NorcpFileInfo]:
    """
    Find the first file whose encoded time range covers the requested timestamp.

    Returns None if no file covers the timestamp.
    """
    for info in file_infos:
        if file_covers_timestamp(info, timestamp):
            return info
    return None



def validate_nonempty_file_index(
    file_infos: Iterable[NorcpFileInfo],
    variable: str,
    scenario_name: str,
    spatial_tag: str,
    temporal_tag: str,
) -> None:
    """
    Raise a helpful error if file discovery returned nothing.
    """
    file_infos = list(file_infos)
    if not file_infos:
        raise FileNotFoundError(
            "No NorCP files found for "
            f"scenario='{scenario_name}', spatial_tag='{spatial_tag}', "
            f"temporal_tag='{temporal_tag}', variable='{variable}'"
        )



def describe_file_infos(file_infos: Iterable[NorcpFileInfo]) -> list[dict[str, str]]:
    """
    Convert parsed file metadata to a small serializable summary.
    """
    summary: list[dict[str, str]] = []
    for info in file_infos:
        summary.append(
            {
                "path": str(info.path),
                "variable": info.variable,
                "spatial_tag": info.spatial_tag,
                "temporal_tag": info.temporal_tag,
                "infix_tag": info.infix_tag if info.infix_tag is not None else "",
                "start_time": info.start_time.isoformat(),
                "end_time": info.end_time.isoformat(),
            }
        )
    return summary