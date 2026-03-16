import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.statistics.compute_stats import compute_stats_for_request
from data_adapters.norcp.statistics.io import (
    build_stats_output_path,
    load_stats_json,
    load_transform_stats,
    save_stats_result,
)
from data_adapters.norcp.statistics.schemas import CropConfig, StatsRequest


ROOT_DIR = Path("/Users/au728490/Data/NorCP/cropped")
SCENARIO_NAME = "ECMWF-ERAINT"
SPLIT_MANIFEST_PATH = (
    REPO_ROOT / "test_scripts" / "norcp" / "saved_split_tests" / "norcp_temporal_split_manifest.json"
)
OUTPUT_ROOT = REPO_ROOT / "test_scripts" / "norcp" / "saved_stats_tests"


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_stats_result(label: str, stats_result) -> None:
    print_section(label)
    print(
        {
            "scenario_name": stats_result.scenario_name,
            "variable": stats_result.variable,
            "source": stats_result.source,
            "transform_name": stats_result.transform_name,
            "split_name": stats_result.split_name,
            "sample_count": stats_result.sample_count,
            "timestamps_used_preview": stats_result.timestamps_used[:3],
            "domain_tag": stats_result.domain_tag,
            "spatial_tag": stats_result.spatial_tag,
            "temporal_tag": stats_result.temporal_tag,
            "crop": stats_result.crop,
        }
    )
    print("physical_summary:", stats_result.physical_summary.to_dict())
    print("transform_stats:", stats_result.transform_stats)
    print("metadata:", stats_result.metadata)


if not SPLIT_MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Split manifest not found: {SPLIT_MANIFEST_PATH}. "
        "Run test_scripts/splits_test.py first."
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1) HR precipitation statistics over full domain
# -----------------------------------------------------------------------------
print_section("Preparing HR precipitation full-domain requests")

# hr_prcp_request_python = StatsRequest(
#     scenario_name=SCENARIO_NAME,
#     root_dir=str(ROOT_DIR),
#     split_name="train",
#     split_manifest_path=str(SPLIT_MANIFEST_PATH),
#     variable="prcp",
#     source="NORCP_HR",
#     transform_name="log_zscore",
#     domain_tag="full_domain",
#     spatial_tag="3km",
#     temporal_tag="6hr",
#     variable_time_offset_hours=-3.0,
#     crop=None,
#     backend="python",
#     metadata={
#         "test_case": "hr_precip_full_domain_python",
#         "note": "HR precipitation should use log_zscore stats.",
#     },
# )

hr_prcp_request_cdo = StatsRequest(
    scenario_name=SCENARIO_NAME,
    root_dir=str(ROOT_DIR),
    split_name="train",
    split_manifest_path=str(SPLIT_MANIFEST_PATH),
    variable="prcp",
    source="NORCP_HR",
    transform_name="log_zscore",
    domain_tag="full_domain",
    spatial_tag="3km",
    temporal_tag="6hr",
    variable_time_offset_hours=-3.0,
    crop=None,
    backend="cdo",
    metadata={
        "test_case": "hr_precip_full_domain_cdo",
        "note": "HR precipitation should use log_zscore stats.",
    },
)

# print_section("Running HR precipitation full-domain stats with Python backend")
# hr_prcp_result_python = compute_stats_for_request(hr_prcp_request_python)
# print_stats_result("HR precipitation full-domain stats | python", hr_prcp_result_python)

# hr_prcp_python_output_path = build_stats_output_path(
#     split_name=hr_prcp_request_python.split_name,
#     domain_tag=hr_prcp_request_python.domain_tag,
#     scenario_name=hr_prcp_request_python.scenario_name,
#     variable=hr_prcp_request_python.variable,
#     source=hr_prcp_request_python.source,
#     transform_name=hr_prcp_request_python.transform_name,
#     root_dir=OUTPUT_ROOT,
# )
# save_stats_result(hr_prcp_result_python, hr_prcp_python_output_path)
# print(f"Saved to: {hr_prcp_python_output_path}")
# print("Reloaded JSON keys:", list(load_stats_json(hr_prcp_python_output_path).keys()))
# print("Reloaded transform stats:", load_transform_stats(hr_prcp_python_output_path))

print_section("Running HR precipitation full-domain stats with CDO backend")
hr_prcp_result_cdo = compute_stats_for_request(hr_prcp_request_cdo)
print_stats_result("HR precipitation full-domain stats | cdo", hr_prcp_result_cdo)

hr_prcp_cdo_output_path = build_stats_output_path(
    split_name=hr_prcp_request_cdo.split_name,
    domain_tag=hr_prcp_request_cdo.domain_tag,
    scenario_name=hr_prcp_request_cdo.scenario_name,
    variable=hr_prcp_request_cdo.variable,
    source=hr_prcp_request_cdo.source,
    transform_name=f"{hr_prcp_request_cdo.transform_name}__cdo",
    root_dir=OUTPUT_ROOT,
)
save_stats_result(hr_prcp_result_cdo, hr_prcp_cdo_output_path)
print(f"Saved to: {hr_prcp_cdo_output_path}")

# print_section("Python vs CDO comparison | HR precipitation full domain")
# print(
#     {
#         "python_mean": hr_prcp_result_python.physical_summary.mean,
#         "cdo_mean": hr_prcp_result_cdo.physical_summary.mean,
#         "python_std": hr_prcp_result_python.physical_summary.std,
#         "cdo_std": hr_prcp_result_cdo.physical_summary.std,
#         "python_log_mean": hr_prcp_result_python.transform_stats.get("log_mean"),
#         "cdo_log_mean": hr_prcp_result_cdo.transform_stats.get("log_mean"),
#         "python_log_std": hr_prcp_result_python.transform_stats.get("log_std"),
#         "cdo_log_std": hr_prcp_result_cdo.transform_stats.get("log_std"),
#     }
# )


# -----------------------------------------------------------------------------
# 2) LR humidity statistics over full domain
# -----------------------------------------------------------------------------
# lr_hus_request = StatsRequest(
#     scenario_name=SCENARIO_NAME,
#     root_dir=str(ROOT_DIR),
#     split_name="train",
#     split_manifest_path=str(SPLIT_MANIFEST_PATH),
#     variable="hus1000",
#     source="NORCP_LR",
#     transform_name="zscore",
#     domain_tag="full_domain",
#     spatial_tag="12km",
#     temporal_tag="6hr",
#     variable_time_offset_hours=0.0,
#     crop=None,
#     backend="python",
#     metadata={
#         "test_case": "lr_hus1000_full_domain",
#         "note": "LR humidity should use standard zscore stats.",
#     },
# )
#
# lr_hus_result = compute_stats_for_request(lr_hus_request)
# print_stats_result("LR humidity full-domain stats", lr_hus_result)
#
# lr_hus_output_path = build_stats_output_path(
#     split_name=lr_hus_request.split_name,
#     domain_tag=lr_hus_request.domain_tag,
#     scenario_name=lr_hus_request.scenario_name,
#     variable=lr_hus_request.variable,
#     source=lr_hus_request.source,
#     transform_name=lr_hus_request.transform_name,
#     root_dir=OUTPUT_ROOT,
# )
# save_stats_result(lr_hus_result, lr_hus_output_path)
# print(f"Saved to: {lr_hus_output_path}")
#
#
# -----------------------------------------------------------------------------
# 3) Static topography statistics over full domain
# -----------------------------------------------------------------------------
#
# static_topo_request = StatsRequest(
#     scenario_name=SCENARIO_NAME,
#     root_dir=str(ROOT_DIR),
#     split_name="train",
#     split_manifest_path=str(SPLIT_MANIFEST_PATH),
#     variable="topo",
#     source="NORCP_STATIC",
#     transform_name="zscore",
#     domain_tag="full_domain",
#     spatial_tag="3km",
#     temporal_tag=None,
#     variable_time_offset_hours=0.0,
#     crop=None,
#     static_file_path=str(
#         ROOT_DIR / SCENARIO_NAME / "3km" / "fx" / "orog" / "orog_3km_fx.nc"
#     ),
#     backend="python",
#     metadata={
#         "test_case": "static_topography_full_domain",
#         "note": "Static topography should not depend on split timestamps.",
#     },
# )
#
# static_topo_result = compute_stats_for_request(static_topo_request)
# print_stats_result("Static topography full-domain stats", static_topo_result)
#
# static_topo_output_path = build_stats_output_path(
#     split_name=static_topo_request.split_name,
#     domain_tag=static_topo_request.domain_tag,
#     scenario_name=static_topo_request.scenario_name,
#     variable=static_topo_request.variable,
#     source=static_topo_request.source,
#     transform_name=static_topo_request.transform_name,
#     root_dir=OUTPUT_ROOT,
# )
# save_stats_result(static_topo_result, static_topo_output_path)
# print(f"Saved to: {static_topo_output_path}")
#
#
# -----------------------------------------------------------------------------
# 4) HR precipitation statistics on a cropped domain
# -----------------------------------------------------------------------------
#
# crop_config = CropConfig(top=8, left=4, height=64, width=48)
#
# hr_prcp_crop_request = StatsRequest(
#     scenario_name=SCENARIO_NAME,
#     root_dir=str(ROOT_DIR),
#     split_name="train",
#     split_manifest_path=str(SPLIT_MANIFEST_PATH),
#     variable="prcp",
#     source="NORCP_HR",
#     transform_name="log_zscore",
#     domain_tag="cropped_domain",
#     spatial_tag="3km",
#     temporal_tag="6hr",
#     variable_time_offset_hours=-3.0,
#     crop=crop_config,
#     backend="python",
#     metadata={
#         "test_case": "hr_precip_cropped_domain",
#         "note": "Cropped HR precipitation stats for future training-domain checks.",
#     },
# )
#
# hr_prcp_crop_result = compute_stats_for_request(hr_prcp_crop_request)
# print_stats_result("HR precipitation cropped-domain stats", hr_prcp_crop_result)
#
# hr_prcp_crop_output_path = build_stats_output_path(
#     split_name=hr_prcp_crop_request.split_name,
#     domain_tag=hr_prcp_crop_request.domain_tag,
#     scenario_name=hr_prcp_crop_request.scenario_name,
#     variable=hr_prcp_crop_request.variable,
#     source=hr_prcp_crop_request.source,
#     transform_name=hr_prcp_crop_request.transform_name,
#     root_dir=OUTPUT_ROOT,
# )
# save_stats_result(hr_prcp_crop_result, hr_prcp_crop_output_path)
# print(f"Saved to: {hr_prcp_crop_output_path}")


# -----------------------------------------------------------------------------
# Final quick sanity summary
# -----------------------------------------------------------------------------

print_section("Final quick sanity summary")
print(
    {
        "hr_prcp_full_cdo_sample_count": hr_prcp_result_cdo.sample_count,
        "saved_outputs": [
            str(hr_prcp_cdo_output_path),
        ],
    }
)