
"""
Path and file-index utilities for the small DANRA/ERA5 STRIDE adapter.

Responsibilities:
    - Locate all files under all/
    - Parse filenames into (variable, date)
    - Build date-indexed lookup tables
    - Intersect available dates across HR/LR variables
    
Answers questions like:
    - What dates exist for DANRA precip?
    - What dates exist for ERA5 precip and temp?
    - Which dates are common across all requested variables?
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, TypedDict

from stride_core.utils.variable_registry import get_source_file_prefix, validate_variables

CANONICAL_SOURCES = {"DANRA", "ERA5"}
DATE_LENGTH = 8


class ExperimentSamplePaths(TypedDict):
    target: Path
    cond_dynamic: dict[str, str | Path]


def get_source_variable_dir(
    root_dir: str | Path,
    source: str,
    variable: str,
    size_tag: str,
) -> Path:
    """
    Return the canonical directory containing `all/` files for a given source/variable.

    Examples
    --------
    DANRA + prcp + size_589x789 ->
        <root>/data_DANRA/size_589x789/prcp_589x789/all

    ERA5 + temp + size_589x789 ->
        <root>/data_ERA5/size_589x789/temp_589x789/all
    """
    validate_variables([variable])

    if source not in CANONICAL_SOURCES:
        raise ValueError(
            f"Unsupported source '{source}'. Supported sources: {sorted(CANONICAL_SOURCES)}"
        )

    root_path = Path(root_dir)
    size_suffix = size_tag.replace("size_", "")

    variable_dir = (
        root_path
        / f"data_{source}"
        / size_tag
        / f"{variable}_{size_suffix}"
        / "all"
    )

    if not variable_dir.exists():
        raise FileNotFoundError(
            f"Expected directory does not exist: {variable_dir}"
        )

    return variable_dir


def list_npz_files(variable_dir: str | Path) -> list[Path]:
    """
    Return all `.npz` files in a variable directory, sorted by filename.
    """
    variable_path = Path(variable_dir)
    if not variable_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {variable_path}")

    files = sorted(variable_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {variable_path}")

    return files


def extract_date_from_filename(filename: str | Path, source: str, variable: str) -> str:
    """
    Extract the YYYYMMDD date from a filename using the source-specific prefix.

    Expected examples
    -----------------
    DANRA prcp: tp_tot_19910813.npz
    ERA5  prcp: prcp_589x789_19910813.npz
    ERA5  temp: temp_589x789_19910813.npz
    """
    if source not in CANONICAL_SOURCES:
        raise ValueError(
            f"Unsupported source '{source}'. Supported sources: {sorted(CANONICAL_SOURCES)}"
        )

    path = Path(filename)
    prefix = get_source_file_prefix(variable, source)
    if prefix is None:
        raise ValueError(
            f"No source file prefix registered for variable='{variable}', source='{source}'"
        )

    stem = path.stem
    expected_prefix = f"{prefix}_"
    if not stem.startswith(expected_prefix):
        raise ValueError(
            f"Filename '{path.name}' does not match expected prefix '{expected_prefix}'"
        )

    date_str = stem[len(expected_prefix):]
    if len(date_str) != DATE_LENGTH or not date_str.isdigit():
        raise ValueError(
            f"Could not parse YYYYMMDD date from filename '{path.name}'"
        )

    return date_str


def build_date_to_file_map(
    root_dir: str | Path,
    source: str,
    variable: str,
    size_tag: str,
) -> dict[str, Path]:
    """
    Build a mapping from date string (YYYYMMDD) to file path for one source/variable.
    """
    variable_dir = get_source_variable_dir(
        root_dir=root_dir,
        source=source,
        variable=variable,
        size_tag=size_tag,
    )
    files = list_npz_files(variable_dir)

    date_to_file: dict[str, Path] = {}
    for file_path in files:
        date_str = extract_date_from_filename(file_path.name, source=source, variable=variable)
        if date_str in date_to_file:
            raise ValueError(
                f"Duplicate date '{date_str}' found for source='{source}', variable='{variable}'"
            )
        date_to_file[date_str] = file_path

    return date_to_file


def intersect_available_dates(*date_maps: Dict[str, Path]) -> list[str]:
    """
    Return the sorted intersection of dates across one or more date->file maps.
    """
    if not date_maps:
        raise ValueError("At least one date map must be provided")

    common_dates = set(date_maps[0].keys())
    for mapping in date_maps[1:]:
        common_dates &= set(mapping.keys())

    return sorted(common_dates)


def build_dynamic_date_maps(
    root_dir: str | Path,
    source: str,
    variables: Iterable[str],
    size_tag: str,
) -> dict[str, dict[str, Path]]:
    """
    Build date->file maps for multiple dynamic variables from the same source.
    """
    variables = list(variables)
    validate_variables(variables)

    return {
        variable: build_date_to_file_map(
            root_dir=root_dir,
            source=source,
            variable=variable,
            size_tag=size_tag,
        )
        for variable in variables
    }


def build_experiment_file_index(
    root_dir: str | Path,
    size_tag: str,
    target_variable: str,
    conditioning_variables: list[str],
    target_source: str = "DANRA",
    conditioning_source: str = "ERA5",
) -> dict[str, dict[str, object]]:
    """
    Build a per-date file index for the first STRIDE experiment.

    First experiment
    ----------------
    - Target: DANRA precipitation
    - Conditioning: ERA5 precipitation + temperature
    - Statics handled elsewhere (not per-date)

    Returns
    -------
    dict
        Example structure:
        {
            "19910813": {
                "target": Path(...),
                "cond_dynamic": {
                    "prcp": Path(...),
                    "temp": Path(...),
                },
            },
            ...
        }
    """
    validate_variables([target_variable, *conditioning_variables])

    target_map = build_date_to_file_map(
        root_dir=root_dir,
        source=target_source,
        variable=target_variable,
        size_tag=size_tag,
    )

    conditioning_maps = build_dynamic_date_maps(
        root_dir=root_dir,
        source=conditioning_source,
        variables=conditioning_variables,
        size_tag=size_tag,
    )

    common_dates = intersect_available_dates(target_map, *conditioning_maps.values())

    file_index: dict[str, dict[str, object]] = {}
    for date_str in common_dates:
        file_index[date_str] = {
            "target": target_map[date_str],
            "cond_dynamic": {
                variable: conditioning_maps[variable][date_str]
                for variable in conditioning_variables
            },
        }

    return file_index