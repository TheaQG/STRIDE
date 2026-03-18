"""
CLI launcher for building NorCP temporal split manifests.

This creates a stable saved split JSON under:
    data_adapters/norcp/saved/splits/

The current intended default use is temporal splitting over the full available
sample index, with train/val/test defined by explicit timestamp ranges.


For running the data splits creation scirpt:
`
python -m data_adapters.norcp.launch_splits \
  --root-dir /Users/au728490/Data/NorCP/cropped \
  --scenario-name ECMWF-ERAINT \
  --val-start 2010-01-01T00:00:00 \
  --val-end 2012-12-31T18:00:00 \
  --test-start 2013-01-01T00:00:00 \
  --test-end 2018-12-31T18:00:00
`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_adapters.norcp.indexing import build_aligned_timestamp_index
from data_adapters.norcp.splits.splits import (
    TimeRange,
    build_temporal_split_manifest,
    save_split_manifest,
    summarize_split_manifest,
)


DEFAULT_SPLITS_ROOT = Path(__file__).resolve().parent / "saved" / "splits"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and save a NorCP temporal split manifest."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory of the cropped NorCP dataset.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        required=True,
        help="Scenario name, e.g. ECMWF-ERAINT.",
    )
    parser.add_argument(
        "--spatial-tag",
        type=str,
        default="12km",
        help="Spatial tag used to build the aligned timestamp index. Default: 12km.",
    )
    parser.add_argument(
        "--temporal-tag",
        type=str,
        default="6hr",
        help="Temporal tag used to build the aligned timestamp index. Default: 6hr.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["prcp", "temp", "hus1000"],
        help="Variables used to define the common aligned timestamp index.",
    )
    parser.add_argument(
        "--prcp-offset-hours",
        type=float,
        default=-3.0,
        help="Aligned timestamp offset for precipitation-like fields. Default: -3.0.",
    )
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-start", type=str, required=True)
    parser.add_argument("--val-end", type=str, required=True)
    parser.add_argument("--test-start", type=str, required=True)
    parser.add_argument("--test-end", type=str, required=True)
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional explicit output filename. If omitted, a canonical name is built.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPLITS_ROOT,
        help="Directory to save split manifests under.",
    )
    return parser



def _sanitize_tag(tag: str) -> str:
    return tag.replace(" ", "_").replace("/", "_").replace(":", "-")



def _default_output_name(args: argparse.Namespace) -> str:
    return (
        f"temporal__{_sanitize_tag(args.scenario_name)}"
        f"__train_{_sanitize_tag(args.train_start or 'auto')}_{_sanitize_tag(args.train_end or 'auto')}"
        f"__val_{_sanitize_tag(args.val_start)}_{_sanitize_tag(args.val_end)}"
        f"__test_{_sanitize_tag(args.test_start)}_{_sanitize_tag(args.test_end)}.json"
    )



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    variable_time_offsets = {
        variable: args.prcp_offset_hours
        for variable in args.variables
        if variable == "prcp"
    }

    print("[launch_splits] Building aligned timestamp index")
    _, common_timestamps = build_aligned_timestamp_index(
        root_dir=args.root_dir,
        scenario_name=args.scenario_name,
        spatial_tag=args.spatial_tag,
        temporal_tag=args.temporal_tag,
        variables=args.variables,
        variable_time_offsets=variable_time_offsets,
    )

    if not common_timestamps:
        raise RuntimeError("No common aligned timestamps found. Cannot build split manifest.")

    train_range = None
    if args.train_start is not None or args.train_end is not None:
        if args.train_start is None or args.train_end is None:
            raise ValueError("Both --train-start and --train-end must be provided together.")
        train_range = TimeRange(start=args.train_start, end=args.train_end)

    val_range = TimeRange(start=args.val_start, end=args.val_end)
    test_range = TimeRange(start=args.test_start, end=args.test_end)

    print("[launch_splits] Building temporal split manifest")
    manifest = build_temporal_split_manifest(
        available_timestamps=common_timestamps,
        scenario_name=args.scenario_name,
        dataset_name="NorCP",
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        metadata={
            "root_dir": str(args.root_dir),
            "spatial_tag": args.spatial_tag,
            "temporal_tag": args.temporal_tag,
            "variables": list(args.variables),
            "variable_time_offsets": variable_time_offsets,
            "launcher": "data_adapters.norcp.launch_splits",
        },
    )

    output_name = args.output_name or _default_output_name(args)
    output_path = args.output_root / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[launch_splits] Saving split manifest")
    save_split_manifest(manifest, output_path)

    print("[launch_splits] Done")
    print(summarize_split_manifest(manifest))
    print(f"Saved split manifest: {output_path}")


if __name__ == "__main__":
    main()
