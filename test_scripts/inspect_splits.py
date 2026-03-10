from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/inspect_splits.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.danra_era5_small.paths import build_experiment_file_index
from data_adapters.danra_era5_small.splits import (
    build_random_split_manifest,
    build_year_based_split_manifest,
    get_dates_for_split,
    save_split_manifest,
)


def print_split_summary(title: str, manifest: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"split_strategy: {manifest['split_strategy']}")
    print(f"split_name:     {manifest['split_name']}")
    print(f"n_total:        {manifest['n_total']}")
    print(f"n_train:        {manifest['n_train']}")
    print(f"n_valid:        {manifest['n_valid']}")
    print(f"n_test:         {manifest['n_test']}")

    for split_name in ["train", "valid", "test"]:
        split_dates = get_dates_for_split(manifest, split_name)
        first_five = split_dates[:5]
        years = sorted({int(date[:4]) for date in split_dates})
        print(f"\n{split_name}:")
        print(f"  count:      {len(split_dates)}")
        print(f"  first five: {first_five}")
        print(f"  years:      {years}")



def main() -> None:
    root_dir = Path("/Users/au728490/Data/Data_DiffMod_small")
    size_tag = "size_589x789"

    file_index = build_experiment_file_index(
        root_dir=root_dir,
        size_tag=size_tag,
        target_variable="prcp",
        conditioning_variables=["prcp", "temp"],
        target_source="DANRA",
        conditioning_source="ERA5",
    )

    common_dates = sorted(file_index.keys())
    print(f"Found {len(common_dates)} common dates in experiment file index.")
    print(f"First 10 common dates: {common_dates[:10]}")

    # 1) Random split example
    random_manifest = build_random_split_manifest(
        dates=common_dates,
        train_fraction=0.70,
        valid_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        split_name="random_seed42",
    )
    print_split_summary("Random split", random_manifest)

    random_output = (
        REPO_ROOT
        / "data_adapters"
        / "danra_era5_small"
        / "saved"
        / "splits"
        / "random_seed42.json"
    )
    save_split_manifest(random_manifest, random_output)
    print(f"\nSaved random split manifest to: {random_output}")

    # 2) Year-based split example
    year_manifest = build_year_based_split_manifest(
        dates=common_dates,
        train_years=list(range(1991, 2017)),
        valid_years=[2017, 2018],
        test_years=[2019, 2020],
        split_name="year_based_1991_2020",
    )
    print_split_summary("Year-based split", year_manifest)

    year_output = (
        REPO_ROOT
        / "data_adapters"
        / "danra_era5_small"
        / "saved"
        / "splits"
        / "year_based_1991_2020.json"
    )
    save_split_manifest(year_manifest, year_output)
    print(f"\nSaved year-based split manifest to: {year_output}")


if __name__ == "__main__":
    main()
