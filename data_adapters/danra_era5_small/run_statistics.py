"""
Run statistics computation for the DANRA/ERA5 STRIDE adapter.

This script computes pooled global statistics for variables used in the
first STRIDE experiment and saves them to the adapter statistics directory.

Three domains are computed:

1) Full domain (589x789)
2) Denmark crop (128x128)
3) Shuffle patch (180x180) used for random spatial shuffling

Statistics are computed only from the selected split subset (typically "train").
"""

from pathlib import Path
import sys

# Allow direct execution via:
# python data_adapters/danra_era5_small/run_statistics.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.danra_era5_small.statistics.compute_stats import (
    build_default_stats_requests,
    compute_statistics_for_request,
)
from data_adapters.danra_era5_small.statistics.schemas import CropConfig, StatsRequest
from data_adapters.danra_era5_small.statistics.io import (
    build_stats_output_path,
    save_stats_result,
)


# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------

ROOT_DATA_DIR = Path("/Users/au728490/Data/Data_DiffMod_small")
SIZE_TAG = "size_589x789"

SPLIT_MANIFEST_PATHS = [
    Path(__file__).resolve().parent / "saved" / "splits" / "random_seed42.json",
    Path(__file__).resolve().parent / "saved" / "splits" / "year_based_1991_2020.json",
]

SPLIT_NAME = "train"


# ---------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------

# 1) Full domain
FULL_DOMAIN_TAG = "full_589x789"
FULL_DOMAIN_CROP = None

# 2) Denmark evaluation crop
DK_DOMAIN_TAG = "dk_128x128"
DK_DOMAIN_CROP = CropConfig(
    anchor_y=200,
    anchor_x=300,
    height=128,
    width=128,
)

# 3) Shuffle patch (larger region used for random spatial crop during training)
SHUFFLE_DOMAIN_TAG = "shuffle_180x180"
SHUFFLE_DOMAIN_CROP = CropConfig(
    anchor_y=340,
    anchor_x=170,
    height=180,
    width=180,
)


DOMAINS = [
    (FULL_DOMAIN_TAG, FULL_DOMAIN_CROP),
    (DK_DOMAIN_TAG, DK_DOMAIN_CROP),
    (SHUFFLE_DOMAIN_TAG, SHUFFLE_DOMAIN_CROP),
]


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------


def compute_domain_statistics(
    split_manifest_path: Path,
    domain_tag: str,
    crop: CropConfig | None,
) -> None:
    """
    Compute statistics for all default variables for one spatial domain.
    """

    print("\n============================================================")
    print(f"Computing statistics for domain: {domain_tag}")
    print(f"Using split manifest: {split_manifest_path.name}")
    print("============================================================")

    split_tag = split_manifest_path.stem

    requests = build_default_stats_requests(
        split_manifest_path=split_manifest_path,
        split_name=SPLIT_NAME,
        domain_tag=domain_tag,
        crop=crop,
    )

    # Additional statistics request for DANRA temperature
    requests.append(
        StatsRequest(
            split_name=SPLIT_NAME,
            split_manifest_path=str(split_manifest_path),
            variable="temp",
            source="DANRA",
            transform_name="zscore",
            domain_tag=domain_tag,
            crop=crop,
        )
    )

    for request in requests:

        result = compute_statistics_for_request(
            request=request,
            root_dir=ROOT_DATA_DIR,
            size_tag=SIZE_TAG,
        )

        output_path = build_stats_output_path(
            split_name=split_tag,
            domain_tag=domain_tag,
            source=request.source,
            variable=request.variable,
            transform_name=request.transform_name,
        )

        save_stats_result(result, output_path)

        summary = result.physical_summary

        print(
            f"Saved stats: {request.source:>5} {request.variable:>6} | {request.transform_name:<10}"
            f" | n_dates={summary.n_dates:3d}"
            f" | mean={summary.mean:.4f}"
            f" | std={summary.std:.4f}"
        )



def main() -> None:

    print("\n=============================")
    print("STRIDE statistics computation")
    print("=============================\n")

    print(f"Split subset:   {SPLIT_NAME}")
    print("Split manifests:")
    for split_manifest_path in SPLIT_MANIFEST_PATHS:
        print(f"  - {split_manifest_path}")

    for split_manifest_path in SPLIT_MANIFEST_PATHS:
        for domain_tag, crop in DOMAINS:
            compute_domain_statistics(split_manifest_path, domain_tag, crop)

    print("\nAll statistics computed successfully.")


if __name__ == "__main__":
    main()