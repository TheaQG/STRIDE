"""
CLI launcher for batch-computing NorCP statistics.

This creates stable statistics JSON files under:
    data_adapters/norcp/saved/statistics/

Current intended default use:
- full spatial domain
- training split only
- CDO backend
- one statistics JSON per variable/source/transform combination
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from data_adapters.norcp.statistics.compute_stats import compute_stats_for_request
from data_adapters.norcp.statistics.io import build_stats_output_path, save_stats_result
from data_adapters.norcp.statistics.schemas import CropConfig, StatsRequest
from data_adapters.norcp.variable_registry import get_default_transform


DEFAULT_STATS_ROOT = Path(__file__).resolve().parent / "saved" / "statistics"


PRESSURE_LEVELS = ["500", "700", "850", "950", "1000"]
PRESSURE_LEVEL_VARIABLE_PREFIXES = ["hus", "ta", "ua", "va", "zg"]


TARGET_VARIABLES = [
    ("prcp", "NORCP_HR", "3km", "6hr", -3.0),
    ("temp", "NORCP_HR", "3km", "6hr", 0.0),
]

BASE_DYNAMIC_VARIABLES = [
    ("prcp", "NORCP_LR", "12km", "6hr", -3.0),
    ("temp", "NORCP_LR", "12km", "6hr", 0.0),
]

DYNAMIC_VARIABLES = BASE_DYNAMIC_VARIABLES + [
    (f"{prefix}{level}", "NORCP_LR", "12km", "6hr", 0.0)
    for level in PRESSURE_LEVELS
    for prefix in PRESSURE_LEVEL_VARIABLE_PREFIXES
]

STATIC_VARIABLES = [
    ("topo", "NORCP_STATIC", "3km", None, 0.0),
]



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and save NorCP statistics for a split/domain setup."
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
        "--split-manifest-path",
        type=Path,
        required=True,
        help="Path to a saved NorCP split manifest JSON.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="train",
        help="Split to compute statistics over. Default: train.",
    )
    parser.add_argument(
        "--domain-tag",
        type=str,
        default="full_domain",
        help="Name for the statistics domain. Default: full_domain.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cdo",
        choices=["python", "cdo"],
        help="Statistics backend. Default: cdo.",
    )
    parser.add_argument(
        "--include-static",
        action="store_true",
        help="Also compute static variable statistics when files exist.",
    )
    parser.add_argument(
        "--crop-top",
        type=int,
        default=None,
        help="Optional crop top index for future cropped-domain stats.",
    )
    parser.add_argument(
        "--crop-left",
        type=int,
        default=None,
        help="Optional crop left index for future cropped-domain stats.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=None,
        help="Optional crop height for future cropped-domain stats.",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=None,
        help="Optional crop width for future cropped-domain stats.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_STATS_ROOT,
        help="Directory to save statistics JSONs under.",
    )
    return parser



def _maybe_build_crop(args: argparse.Namespace) -> CropConfig | None:
    crop_values = [args.crop_top, args.crop_left, args.crop_height, args.crop_width]
    if all(value is None for value in crop_values):
        return None
    if any(value is None for value in crop_values):
        raise ValueError(
            "Crop arguments must either all be omitted or all be provided: "
            "--crop-top --crop-left --crop-height --crop-width"
        )
    return CropConfig(
        top=int(args.crop_top),
        left=int(args.crop_left),
        height=int(args.crop_height),
        width=int(args.crop_width),
    )




def _split_tag_from_manifest(path: Path) -> str:
    return path.stem


def _summarize_request_specs(request_specs: Sequence[tuple[str, str, str, str | None, float]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for variable, source, _spatial_tag, _temporal_tag, _offset_hours in request_specs:
        grouped.setdefault(source, []).append(variable)
    return grouped



def _build_request(
    *,
    args: argparse.Namespace,
    variable: str,
    source: str,
    spatial_tag: str,
    temporal_tag: str | None,
    variable_time_offset_hours: float,
    crop: CropConfig | None,
) -> StatsRequest:
    static_file_path = None
    if source == "NORCP_STATIC":
        static_file_path = str(
            args.root_dir / args.scenario_name / spatial_tag / "fx" / "orog" / f"orog_{spatial_tag}_fx.nc"
        )

    return StatsRequest(
        scenario_name=args.scenario_name,
        root_dir=str(args.root_dir),
        split_name=args.split_name,
        split_manifest_path=str(args.split_manifest_path),
        variable=variable,
        source=source,
        transform_name=get_default_transform(variable),
        domain_tag=args.domain_tag,
        spatial_tag=spatial_tag,
        temporal_tag=temporal_tag,
        variable_time_offset_hours=variable_time_offset_hours,
        crop=crop,
        static_file_path=static_file_path,
        backend=args.backend,
        metadata={
            "launcher": "data_adapters.norcp.launch_statistics",
            "split_manifest_tag": _split_tag_from_manifest(args.split_manifest_path),
        },
    )



def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    crop = _maybe_build_crop(args)
    split_tag = _split_tag_from_manifest(args.split_manifest_path)
    output_root = args.output_root / split_tag
    output_root.mkdir(parents=True, exist_ok=True)

    request_specs = list(TARGET_VARIABLES) + list(DYNAMIC_VARIABLES)
    if args.include_static:
        request_specs += list(STATIC_VARIABLES)

    total = len(request_specs)
    request_summary = _summarize_request_specs(request_specs)
    print(
        "[launch_statistics] Starting batch statistics computation | "
        f"scenario={args.scenario_name} | split={args.split_name} | "
        f"domain={args.domain_tag} | backend={args.backend} | n_requests={total}"
    )
    print("[launch_statistics] Variable summary by source:")
    for source, variables in request_summary.items():
        print(f"  - {source}: {len(variables)} variables")
        print(f"    {variables}")

    saved_paths: list[str] = []

    for idx, (variable, source, spatial_tag, temporal_tag, offset_hours) in enumerate(request_specs, start=1):
        print(
            "[launch_statistics] Computing request "
            f"{idx}/{total} | variable={variable} | source={source} | "
            f"spatial_tag={spatial_tag} | temporal_tag={temporal_tag}"
        )
        request = _build_request(
            args=args,
            variable=variable,
            source=source,
            spatial_tag=spatial_tag,
            temporal_tag=temporal_tag,
            variable_time_offset_hours=offset_hours,
            crop=crop,
        )
        result = compute_stats_for_request(request)
        output_path = build_stats_output_path(
            split_name=request.split_name,
            domain_tag=request.domain_tag,
            scenario_name=request.scenario_name,
            variable=request.variable,
            source=request.source,
            transform_name=request.transform_name,
            root_dir=output_root,
        )
        save_stats_result(result, output_path)
        saved_paths.append(str(output_path))
        print(f"[launch_statistics] Saved: {output_path}")

    print("[launch_statistics] Done")
    print({
        "n_saved": len(saved_paths),
        "saved_paths": saved_paths,
    })


if __name__ == "__main__":
    main()
