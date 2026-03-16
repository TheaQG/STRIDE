from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.paths import (
    parse_norcp_filename,
    build_variable_time_range_index,
    describe_file_infos,
)

example = "pr_12km_6hr_199801010300-201812312100.nc"
print(parse_norcp_filename(example))

root_dir = "/Users/au728490/Data/NorCP/cropped"
scenario_name = "ECMWF-ERAINT"

infos = build_variable_time_range_index(
    root_dir=root_dir,
    scenario_name=scenario_name,
    spatial_tag="12km",
    temporal_tag="6hr",
    variable="pr",
)

print(f"n_files = {len(infos)}")
print(describe_file_infos(infos[:3]))