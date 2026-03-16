

"""
Split utilities for the NorCP STRIDE adapter.

This module provides timestamp-based split handling for NorCP. In contrast to
simpler date-only split setups, NorCP samples are indexed by full timestamps,
which is a better fit for temporal and scenario-aware climate experiments.

Design goals
------------
- represent splits using full timestamps in ISO-8601 format
- support temporal train/val/test splits based on time ranges
- support saving/loading split manifests as JSON
- validate manifests aggressively so downstream adapter code can trust them

Recommended usage
-----------------
For climate-research workflows, temporal splits are usually the most meaningful
choice because they test whether models generalize to later time periods and,
more broadly, to different climate regimes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
VALID_SPLIT_NAMES = ("train", "val", "test")


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class TimeRange:
    """
    Inclusive timestamp range for one split segment.
    """

    start: str
    end: str

    def start_datetime(self) -> datetime:
        return parse_timestamp(self.start)

    def end_datetime(self) -> datetime:
        return parse_timestamp(self.end)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SplitManifest:
    """
    JSON-serializable split manifest for NorCP.

    Fields
    ------
    dataset_name:
        Human-readable dataset label, e.g. "NorCP".
    scenario_name:
        Scenario identifier, e.g. "ECMWF-ERAINT".
    split_type:
        Manifest type, currently expected to be "temporal".
    timestamp_format:
        Serialization format used for timestamps.
    splits:
        Mapping from split name to ordered timestamp strings.
    metadata:
        Free-form metadata for provenance/debugging.
    """

    dataset_name: str
    scenario_name: str
    split_type: str
    timestamp_format: str
    splits: dict[str, list[str]]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "scenario_name": self.scenario_name,
            "split_type": self.split_type,
            "timestamp_format": self.timestamp_format,
            "splits": self.splits,
            "metadata": self.metadata,
        }


# -------------------------------------------------------------------
# Timestamp helpers
# -------------------------------------------------------------------


def parse_timestamp(value: str) -> datetime:
    """
    Parse one ISO timestamp string.
    """
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp '{value}'. Expected format {TIMESTAMP_FORMAT}"
        ) from exc



def format_timestamp(value: datetime) -> str:
    """
    Format one datetime as canonical NorCP split timestamp string.
    """
    return value.strftime(TIMESTAMP_FORMAT)



def normalize_timestamps(timestamps: Iterable[datetime | str]) -> list[str]:
    """
    Convert timestamps to canonical ISO strings and sort them.
    """
    normalized: list[datetime] = []
    for value in timestamps:
        if isinstance(value, datetime):
            normalized.append(value)
        elif isinstance(value, str):
            normalized.append(parse_timestamp(value))
        else:
            raise TypeError(
                f"Unsupported timestamp type {type(value)}. Expected datetime or str."
            )
    return [format_timestamp(ts) for ts in sorted(normalized)]



def validate_timestamp_strings(timestamps: Sequence[str]) -> None:
    """
    Validate timestamp strings and ensure they are unique and sorted.
    """
    parsed = [parse_timestamp(value) for value in timestamps]

    if len(parsed) != len(set(parsed)):
        raise ValueError("Split timestamps must be unique")

    if parsed != sorted(parsed):
        raise ValueError("Split timestamps must be sorted in ascending order")


# -------------------------------------------------------------------
# Range filtering
# -------------------------------------------------------------------


def timestamps_in_range(
    timestamps: Iterable[datetime | str],
    time_range: TimeRange,
) -> list[str]:
    """
    Select timestamps within an inclusive time range.
    """
    start_dt = time_range.start_datetime()
    end_dt = time_range.end_datetime()

    if end_dt < start_dt:
        raise ValueError(
            f"Invalid time range: end {time_range.end} is earlier than start {time_range.start}"
        )

    selected: list[datetime] = []
    for value in timestamps:
        ts = value if isinstance(value, datetime) else parse_timestamp(value)
        if start_dt <= ts <= end_dt:
            selected.append(ts)

    return [format_timestamp(ts) for ts in sorted(selected)]



def subtract_timestamps(
    timestamps: Iterable[str],
    timestamps_to_remove: Iterable[str],
) -> list[str]:
    """
    Remove one timestamp set from another while preserving sorted order.
    """
    remove_set = set(timestamps_to_remove)
    remaining = [ts for ts in timestamps if ts not in remove_set]
    validate_timestamp_strings(remaining)
    return remaining


# -------------------------------------------------------------------
# Manifest builders
# -------------------------------------------------------------------


def build_temporal_split_manifest(
    *,
    available_timestamps: Sequence[datetime | str],
    scenario_name: str,
    dataset_name: str = "NorCP",
    val_range: Optional[TimeRange] = None,
    test_range: Optional[TimeRange] = None,
    train_range: Optional[TimeRange] = None,
    metadata: Optional[dict[str, object]] = None,
) -> SplitManifest:
    """
    Build a timestamp-based temporal split manifest.

    Rules
    -----
    - `val_range` and `test_range` are selected first.
    - If `train_range` is provided, train is selected explicitly from that range.
    - Otherwise, train is all remaining timestamps not used by val/test.

    This design is well suited for climate experiments where validation/test are
    defined as later time periods.
    """
    normalized_all = normalize_timestamps(available_timestamps)
    validate_timestamp_strings(normalized_all)

    val_timestamps = timestamps_in_range(normalized_all, val_range) if val_range is not None else []
    test_timestamps = timestamps_in_range(normalized_all, test_range) if test_range is not None else []

    overlap = set(val_timestamps).intersection(test_timestamps)
    if overlap:
        raise ValueError(
            f"Validation and test ranges overlap for timestamps: {sorted(overlap)[:5]}"
        )

    occupied = set(val_timestamps).union(test_timestamps)

    if train_range is not None:
        train_timestamps = timestamps_in_range(normalized_all, train_range)
        train_overlap = set(train_timestamps).intersection(occupied)
        if train_overlap:
            raise ValueError(
                f"Train range overlaps validation/test timestamps: {sorted(train_overlap)[:5]}"
            )
    else:
        train_timestamps = [ts for ts in normalized_all if ts not in occupied]

    validate_timestamp_strings(train_timestamps)
    validate_timestamp_strings(val_timestamps)
    validate_timestamp_strings(test_timestamps)

    manifest_metadata = dict(metadata or {})
    manifest_metadata.setdefault("n_available", len(normalized_all))
    manifest_metadata.setdefault("n_train", len(train_timestamps))
    manifest_metadata.setdefault("n_val", len(val_timestamps))
    manifest_metadata.setdefault("n_test", len(test_timestamps))
    if train_range is not None:
        manifest_metadata["train_range"] = train_range.to_dict()
    if val_range is not None:
        manifest_metadata["val_range"] = val_range.to_dict()
    if test_range is not None:
        manifest_metadata["test_range"] = test_range.to_dict()

    return SplitManifest(
        dataset_name=dataset_name,
        scenario_name=scenario_name,
        split_type="temporal",
        timestamp_format=TIMESTAMP_FORMAT,
        splits={
            "train": train_timestamps,
            "val": val_timestamps,
            "test": test_timestamps,
        },
        metadata=manifest_metadata,
    )


# -------------------------------------------------------------------
# Manifest validation and IO
# -------------------------------------------------------------------


def validate_split_manifest(manifest: SplitManifest) -> None:
    """
    Validate a split manifest thoroughly.
    """
    if manifest.timestamp_format != TIMESTAMP_FORMAT:
        raise ValueError(
            f"Unsupported timestamp_format '{manifest.timestamp_format}'. "
            f"Expected '{TIMESTAMP_FORMAT}'"
        )

    missing = [name for name in VALID_SPLIT_NAMES if name not in manifest.splits]
    if missing:
        raise ValueError(f"Split manifest is missing required splits: {missing}")

    all_seen: list[str] = []
    for split_name in VALID_SPLIT_NAMES:
        timestamps = manifest.splits[split_name]
        if not isinstance(timestamps, list):
            raise TypeError(
                f"Split '{split_name}' must contain a list of timestamp strings"
            )
        validate_timestamp_strings(timestamps)
        all_seen.extend(timestamps)

    if len(all_seen) != len(set(all_seen)):
        raise ValueError("Split manifest contains overlapping timestamps across splits")



def save_split_manifest(
    manifest: SplitManifest,
    output_path: str | Path,
) -> None:
    """
    Save a split manifest as JSON.
    """
    validate_split_manifest(manifest)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)



def load_split_manifest(manifest_path: str | Path) -> SplitManifest:
    """
    Load and validate a split manifest from JSON.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = SplitManifest(
        dataset_name=str(data["dataset_name"]),
        scenario_name=str(data["scenario_name"]),
        split_type=str(data["split_type"]),
        timestamp_format=str(data["timestamp_format"]),
        splits={
            str(key): list(value)
            for key, value in dict(data["splits"]).items()
        },
        metadata=dict(data.get("metadata", {})),
    )
    validate_split_manifest(manifest)
    return manifest


# -------------------------------------------------------------------
# Convenience helpers
# -------------------------------------------------------------------


def get_split_timestamps(
    manifest: SplitManifest,
    split_name: str,
) -> list[str]:
    """
    Return timestamps for one split.
    """
    if split_name not in manifest.splits:
        raise KeyError(
            f"Unknown split '{split_name}'. Available: {list(manifest.splits.keys())}"
        )
    return list(manifest.splits[split_name])



def summarize_split_manifest(manifest: SplitManifest) -> dict[str, object]:
    """
    Build a lightweight summary for logging/debugging.
    """
    summary: dict[str, object] = {
        "dataset_name": manifest.dataset_name,
        "scenario_name": manifest.scenario_name,
        "split_type": manifest.split_type,
        "timestamp_format": manifest.timestamp_format,
        "counts": {
            split_name: len(manifest.splits.get(split_name, []))
            for split_name in VALID_SPLIT_NAMES
        },
    }

    for split_name in VALID_SPLIT_NAMES:
        timestamps = manifest.splits.get(split_name, [])
        if timestamps:
            summary[f"{split_name}_start"] = timestamps[0]
            summary[f"{split_name}_end"] = timestamps[-1]
        else:
            summary[f"{split_name}_start"] = None
            summary[f"{split_name}_end"] = None

    return summary