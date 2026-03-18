"""
Split utilities for the small DANRA/ERA5 STRIDE adapter.

This module supports two split strategies:
    1. Random date-based split (deterministic via seed)
    2. Year-based split using explicit year lists per subset

Design principles
-----------------
- Splits are derived from the common-date index built from `all/`
- Splits are defined on date strings (YYYYMMDD), not file paths
- Split manifests should be saved and reused rather than recomputed ad hoc
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


VALID_SPLIT_NAMES = {"train", "val", "test"}


def extract_year_from_date(date_str: str) -> int:
    """
    Extract the year from a date string in YYYYMMDD format.
    """
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Expected date in YYYYMMDD format, got '{date_str}'")
    return int(date_str[:4])


def validate_dates(dates: list[str]) -> None:
    """
    Validate that all dates are unique and follow YYYYMMDD format.
    """
    if len(dates) != len(set(dates)):
        raise ValueError("Date list contains duplicates")

    invalid = [date for date in dates if len(date) != 8 or not date.isdigit()]
    if invalid:
        raise ValueError(f"Invalid date strings found: {invalid[:10]}")


def build_random_split_manifest(
    dates: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    split_name: str = "random_split",
) -> dict[str, Any]:
    """
    Build a deterministic random split manifest from a list of common dates.
    """
    validate_dates(dates)

    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_fraction}"
        )

    shuffled_dates = list(sorted(dates))
    rng = random.Random(seed)
    rng.shuffle(shuffled_dates)

    n_total = len(shuffled_dates)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_test = n_total - n_train - n_val

    train_dates = sorted(shuffled_dates[:n_train])
    val_dates = sorted(shuffled_dates[n_train:n_train + n_val])
    test_dates = sorted(shuffled_dates[n_train + n_val:])

    manifest = {
        "split_strategy": "random",
        "split_name": split_name,
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": test_fraction,
        },
        "n_total": n_total,
        "n_train": len(train_dates),
        "n_val": len(val_dates),
        "n_test": len(test_dates),
        "dates": {
            "train": train_dates,
            "val": val_dates,
            "test": test_dates,
        },
    }

    validate_split_manifest(manifest, expected_all_dates=dates)
    return manifest


def build_year_based_split_manifest(
    dates: list[str],
    train_years: list[int],
    valid_years: list[int],
    test_years: list[int],
    split_name: str = "year_based_split",
) -> dict[str, Any]:
    """
    Build a split manifest using explicit year assignments.
    """
    validate_dates(dates)

    year_sets = {
        "train": set(train_years),
        "val": set(valid_years),
        "test": set(test_years),
    }

    overlaps = []
    split_names = ["train", "val", "test"]
    for i, split_a in enumerate(split_names):
        for split_b in split_names[i + 1:]:
            overlap = year_sets[split_a] & year_sets[split_b]
            if overlap:
                overlaps.append((split_a, split_b, sorted(overlap)))
    if overlaps:
        raise ValueError(f"Year-based split sets overlap: {overlaps}")

    split_dates = {"train": [], "val": [], "test": []}
    unassigned_dates: list[str] = []

    for date_str in sorted(dates):
        year = extract_year_from_date(date_str)
        assigned = False
        for split_name_key in split_dates:
            if year in year_sets[split_name_key]:
                split_dates[split_name_key].append(date_str)
                assigned = True
                break
        if not assigned:
            unassigned_dates.append(date_str)

    if unassigned_dates:
        raise ValueError(
            "Some dates were not assigned to any split in the year-based split: "
            f"{unassigned_dates[:10]}"
        )

    manifest = {
        "split_strategy": "year_based",
        "split_name": split_name,
        "year_assignment": {
            "train": sorted(train_years),
            "val": sorted(valid_years),
            "test": sorted(test_years),
        },
        "n_total": len(dates),
        "n_train": len(split_dates["train"]),
        "n_val": len(split_dates["val"]),
        "n_test": len(split_dates["test"]),
        "dates": {
            "train": sorted(split_dates["train"]),
            "val": sorted(split_dates["val"]),
            "test": sorted(split_dates["test"]),
        },
    }

    validate_split_manifest(manifest, expected_all_dates=dates)
    return manifest


def validate_split_manifest(
    manifest: dict[str, Any],
    expected_all_dates: list[str] | None = None,
) -> None:
    """
    Validate split manifest structure and subset consistency.
    """
    if "dates" not in manifest:
        raise ValueError("Manifest must contain a 'dates' key")

    split_dates = manifest["dates"]
    missing_splits = VALID_SPLIT_NAMES - set(split_dates.keys())
    if missing_splits:
        raise ValueError(f"Manifest missing required splits: {sorted(missing_splits)}")

    train_dates = split_dates["train"]
    val_dates = split_dates["val"]
    test_dates = split_dates["test"]

    for subset_name, subset_dates in split_dates.items():
        if subset_name not in VALID_SPLIT_NAMES:
            raise ValueError(f"Unknown split subset '{subset_name}'")
        validate_dates(list(subset_dates))

    train_set = set(train_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)

    if train_set & val_set:
        raise ValueError("Train and val splits overlap")
    if train_set & test_set:
        raise ValueError("Train and test splits overlap")
    if val_set & test_set:
        raise ValueError("Val and test splits overlap")

    union_dates = sorted(train_set | val_set | test_set)

    if expected_all_dates is not None:
        expected_sorted = sorted(expected_all_dates)
        if union_dates != expected_sorted:
            missing = sorted(set(expected_sorted) - set(union_dates))
            extra = sorted(set(union_dates) - set(expected_sorted))
            raise ValueError(
                "Split manifest union does not match expected dates. "
                f"Missing={missing[:10]}, Extra={extra[:10]}"
            )

    n_total = len(union_dates)
    if manifest.get("n_total") != n_total:
        raise ValueError(
            f"Manifest n_total={manifest.get('n_total')} does not match actual total={n_total}"
        )
    if manifest.get("n_train") != len(train_dates):
        raise ValueError("Manifest n_train does not match actual train count")
    if manifest.get("n_val") != len(val_dates):
        raise ValueError("Manifest n_val does not match actual val count")
    if manifest.get("n_test") != len(test_dates):
        raise ValueError("Manifest n_test does not match actual test count")


def save_split_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    """
    Save a split manifest to JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_split_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a split manifest from JSON.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Expected split manifest JSON to contain a dict, got {type(manifest)}")

    validate_split_manifest(manifest)
    return manifest


def get_dates_for_split(manifest: dict[str, Any], split_name: str) -> list[str]:
    """
    Return the date list for one split subset.
    """
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(
            f"Unknown split_name '{split_name}'. Supported: {sorted(VALID_SPLIT_NAMES)}"
        )
    return list(manifest["dates"][split_name])
