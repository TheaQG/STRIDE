

"""
I/O helpers for split-aware statistics in the NorCP STRIDE adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_adapters.norcp.statistics.schemas import StatsResult


DEFAULT_STATS_ROOT = Path(__file__).resolve().parents[1] / "saved" / "statistics"


def sanitize_tag(tag: str) -> str:
    """
    Convert a free-form tag into a filename-safe token.
    """
    safe = tag.replace("/", "_").replace(" ", "_")
    return safe.replace("__", "_")



def build_stats_output_path(
    split_name: str,
    domain_tag: str,
    scenario_name: str,
    variable: str,
    source: str,
    transform_name: str,
    root_dir: str | Path | None = None,
) -> Path:
    """
    Build a canonical output path for one NorCP statistics JSON file.
    """
    root = Path(root_dir) if root_dir is not None else DEFAULT_STATS_ROOT
    split_tag = sanitize_tag(split_name)
    domain_tag = sanitize_tag(domain_tag)
    scenario_tag = sanitize_tag(scenario_name)
    source_tag = sanitize_tag(source)
    filename = f"{variable}__{source_tag}__{transform_name}.json"
    return root / scenario_tag / split_tag / domain_tag / filename



def save_stats_result(
    stats_result: StatsResult,
    output_path: str | Path,
) -> None:
    """
    Save a `StatsResult` object to JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats_result.to_dict(), f, indent=2, sort_keys=True)



def load_stats_json(stats_path: str | Path) -> dict[str, Any]:
    """
    Load a saved statistics JSON file.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Statistics file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if not isinstance(stats, dict):
        raise ValueError(f"Expected statistics JSON to contain a dict, got {type(stats)}")
    return stats



def load_transform_stats(stats_path: str | Path) -> dict[str, float]:
    """
    Load only the transform statistics block from a saved stats JSON file.
    """
    stats = load_stats_json(stats_path)
    transform_stats = stats.get("transform_stats")
    if not isinstance(transform_stats, dict):
        raise ValueError(f"Expected 'transform_stats' dict in statistics file: {stats_path}")
    return {key: float(value) for key, value in transform_stats.items()}