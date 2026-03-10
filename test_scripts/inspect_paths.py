


from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/inspect_paths.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.danra_era5_small.paths import (
    build_date_to_file_map,
    build_experiment_file_index,
    intersect_available_dates,
)


def main() -> None:
    root_dir = Path("/Users/au728490/Data/Data_DiffMod_small")
    size_tag = "size_589x789"

    target_variable = "prcp"
    conditioning_variables = ["prcp", "temp"]

    print("\n=== Building date maps for first STRIDE experiment ===")

    danra_prcp_map = build_date_to_file_map(
        root_dir=root_dir,
        source="DANRA",
        variable="prcp",
        size_tag=size_tag,
    )
    era5_prcp_map = build_date_to_file_map(
        root_dir=root_dir,
        source="ERA5",
        variable="prcp",
        size_tag=size_tag,
    )
    era5_temp_map = build_date_to_file_map(
        root_dir=root_dir,
        source="ERA5",
        variable="temp",
        size_tag=size_tag,
    )

    print(f"DANRA prcp files: {len(danra_prcp_map)}")
    print(f"ERA5 prcp files:  {len(era5_prcp_map)}")
    print(f"ERA5 temp files:  {len(era5_temp_map)}")

    common_dates = intersect_available_dates(
        danra_prcp_map,
        era5_prcp_map,
        era5_temp_map,
    )

    print(f"\nCommon dates across target + conditioning variables: {len(common_dates)}")
    print(f"First 5 common dates: {common_dates[:5]}")

    file_index = build_experiment_file_index(
        root_dir=root_dir,
        size_tag=size_tag,
        target_variable=target_variable,
        conditioning_variables=conditioning_variables,
        target_source="DANRA",
        conditioning_source="ERA5",
    )

    print(f"\nIndexed samples: {len(file_index)}")

    if not file_index:
        raise RuntimeError("File index is empty")

    first_date = sorted(file_index.keys())[0]
    first_sample = file_index[first_date]

    print(f"\nExample indexed sample for date {first_date}:")
    print(f"  target: {first_sample['target']}")
    print("  cond_dynamic:")
    for variable, path in first_sample["cond_dynamic"].items(): # type: ignore
        print(f"    - {variable}: {path}")


if __name__ == "__main__":
    main()