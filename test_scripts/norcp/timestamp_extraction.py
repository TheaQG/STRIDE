import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.paths import build_variable_time_range_index
from data_adapters.norcp.indexing import build_timestamp_to_file_index, describe_timestamp_index
from data_adapters.norcp.indexing import build_aligned_timestamp_index



infos = build_variable_time_range_index(
    root_dir='/Users/au728490/Data/NorCP/cropped',
    scenario_name="ECMWF-ERAINT",
    spatial_tag="12km",
    temporal_tag="6hr",
    variable="pr",
)

timestamp_index = build_timestamp_to_file_index(infos)
print(f"n_timestamps = {len(timestamp_index)}")
print(describe_timestamp_index(timestamp_index, limit=5))

variable_index, common_timestamps = build_aligned_timestamp_index(
    root_dir='/Users/au728490/Data/NorCP/cropped',
    scenario_name="ECMWF-ERAINT",
    spatial_tag="12km",
    temporal_tag="6hr",
    variables=["pr", "tas", "hus1000"],
    variable_time_offsets={"pr": -3.0},
)

print(len(common_timestamps))
print(common_timestamps[:5])
