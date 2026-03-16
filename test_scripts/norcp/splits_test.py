

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.splits.splits import (
    TimeRange,
    build_temporal_split_manifest,
    get_split_timestamps,
    load_split_manifest,
    save_split_manifest,
    summarize_split_manifest,
)
from data_adapters.norcp.indexing import build_aligned_timestamp_index


ROOT_DIR = Path("/Users/au728490/Data/NorCP/cropped")
SCENARIO_NAME = "ECMWF-ERAINT"
SPATIAL_TAG = "12km"
TEMPORAL_TAG = "6hr"
VARIABLES = ["prcp", "temp", "hus1000"]
VARIABLE_TIME_OFFSETS = {"prcp": -3.0}

OUTPUT_DIR = REPO_ROOT / "test_scripts" / "saved_split_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUTPUT_DIR / "norcp_temporal_split_manifest.json"


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# Build the available timestamp set from real NorCP files
# -----------------------------------------------------------------------------

variable_index, common_timestamps = build_aligned_timestamp_index(
    root_dir=ROOT_DIR,
    scenario_name=SCENARIO_NAME,
    spatial_tag=SPATIAL_TAG,
    temporal_tag=TEMPORAL_TAG,
    variables=VARIABLES,
    variable_time_offsets=VARIABLE_TIME_OFFSETS,
)

print_section("Common timestamp index summary")
print(f"n_common_timestamps = {len(common_timestamps)}")
print(f"first_5 = {[ts.isoformat() for ts in common_timestamps[:5]]}")
print(f"last_5 = {[ts.isoformat() for ts in common_timestamps[-5:]]}")

if not common_timestamps:
    raise RuntimeError("No common timestamps found. Cannot build split manifest.")


# -----------------------------------------------------------------------------
# Build a temporal split manifest
# Adjust these ranges if you want a different historical holdout setup.
# -----------------------------------------------------------------------------

val_range = TimeRange(start="2010-01-01T00:00:00", end="2012-12-31T18:00:00")
test_range = TimeRange(start="2013-01-01T00:00:00", end="2018-12-31T18:00:00")

manifest = build_temporal_split_manifest(
    available_timestamps=common_timestamps,
    scenario_name=SCENARIO_NAME,
    dataset_name="NorCP",
    val_range=val_range,
    test_range=test_range,
    metadata={
        "root_dir": str(ROOT_DIR),
        "spatial_tag": SPATIAL_TAG,
        "temporal_tag": TEMPORAL_TAG,
        "variables": VARIABLES,
        "variable_time_offsets": VARIABLE_TIME_OFFSETS,
        "note": "Temporal split test manifest built from common aligned NorCP timestamps.",
    },
)

print_section("In-memory manifest summary")
print(summarize_split_manifest(manifest))


# -----------------------------------------------------------------------------
# Inspect per-split counts and boundaries
# -----------------------------------------------------------------------------

for split_name in ("train", "val", "test"):
    split_timestamps = get_split_timestamps(manifest, split_name)
    print_section(f"Split inspection: {split_name}")
    print(f"count = {len(split_timestamps)}")
    print(f"first_3 = {split_timestamps[:3]}")
    print(f"last_3 = {split_timestamps[-3:] if split_timestamps else []}")


# -----------------------------------------------------------------------------
# Save and reload to test JSON roundtrip
# -----------------------------------------------------------------------------

save_split_manifest(manifest, MANIFEST_PATH)
reloaded_manifest = load_split_manifest(MANIFEST_PATH)

print_section("Reloaded manifest summary")
print(summarize_split_manifest(reloaded_manifest))


# -----------------------------------------------------------------------------
# Roundtrip equality checks
# -----------------------------------------------------------------------------

assert manifest.to_dict() == reloaded_manifest.to_dict(), (
    "Saved and reloaded manifests do not match"
)

print_section("Roundtrip check")
print("Manifest save/load roundtrip succeeded.")
print(f"Saved manifest path: {MANIFEST_PATH}")


# -----------------------------------------------------------------------------
# Final concise summary
# -----------------------------------------------------------------------------

print_section("Final quick sanity summary")
print(
    {
        "scenario_name": SCENARIO_NAME,
        "n_common_timestamps": len(common_timestamps),
        "n_train": len(get_split_timestamps(reloaded_manifest, "train")),
        "n_val": len(get_split_timestamps(reloaded_manifest, "val")),
        "n_test": len(get_split_timestamps(reloaded_manifest, "test")),
        "manifest_path": str(MANIFEST_PATH),
    }
)