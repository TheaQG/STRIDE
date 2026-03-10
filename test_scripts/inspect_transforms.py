

from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/inspect_transforms.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from data_adapters.danra_era5_small.features import (
    load_dynamic_conditioning,
    load_static_features,
    load_target_field,
)
from data_adapters.danra_era5_small.paths import build_experiment_file_index
from data_adapters.danra_era5_small.regions import CropSpec, crop_sample_fields
from data_adapters.danra_era5_small.transforms import (
    build_transform,
    compute_log_zscore_stats,
    compute_zscore_stats,
)


PLOT_CMAPS = {
    "prcp": "Blues",
    "temp": "plasma",
}


def print_array_info(name: str, array: np.ndarray) -> None:
    print(f"\n{name}:")
    print(f"  shape: {array.shape}")
    print(f"  dtype: {array.dtype}")
    print(f"  min:   {np.nanmin(array)}")
    print(f"  max:   {np.nanmax(array)}")
    print(f"  mean:  {np.nanmean(array)}")


def plot_transform_triptych(
    raw_field: np.ndarray,
    transformed_field: np.ndarray,
    backtransformed_field: np.ndarray,
    variable_name: str,
    transform_name: str,
    date_str: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(
        f"STRIDE transform inspection | date={date_str} | var={variable_name} | transform={transform_name}",
        fontsize=11,
    )

    panels = [
        ("raw physical space", raw_field),
        ("transformed model space", transformed_field),
        ("back-transformed physical space", backtransformed_field),
    ]

    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=PLOT_CMAPS.get(variable_name, "viridis"))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    plt.show()


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

    if not file_index:
        raise RuntimeError("File index is empty")

    first_date = sorted(file_index.keys())[0]
    sample_paths = file_index[first_date]

    static_paths = {
        "lsm": root_dir / "data_lsm" / "truth_fullDomain" / "lsm_full.npz",
        "topo": root_dir / "data_topo" / "truth_fullDomain" / "topo_full.npz",
    }

    target = load_target_field(
        target_path=sample_paths["target"], # type: ignore
        target_variable="prcp",
        target_source="DANRA",
    )
    cond_dynamic = load_dynamic_conditioning(
        dynamic_paths=sample_paths["cond_dynamic"], # type: ignore
        variable_order=["prcp", "temp"],
        source="ERA5",
    )
    cond_static = load_static_features(
        static_paths=static_paths, # type: ignore
        variable_order=["lsm", "topo"],
        source="STATIC",
    )

    crop_spec = CropSpec(anchor_y=200, anchor_x=300, height=128, width=128)
    cropped = crop_sample_fields(
        target=target,
        cond_dynamic=cond_dynamic,
        cond_static=cond_static,
        crop_spec=crop_spec,
    )

    # Choose which variable to inspect here.
    # Supported first-pass options:
    #   1) target precipitation: cropped["target"] with log_zscore
    #   2) ERA5 precipitation:   cropped["cond_dynamic"][0] with log_zscore
    #   3) ERA5 temperature:     cropped["cond_dynamic"][1] with zscore
    inspection_mode = "target_prcp"

    if inspection_mode == "target_prcp":
        variable_name = "prcp"
        transform_name = "log_zscore"
        raw_field = cropped["target"]
        stats = compute_log_zscore_stats(raw_field, clip_min=0.0)
    elif inspection_mode == "era5_prcp":
        variable_name = "prcp"
        transform_name = "log_zscore"
        raw_field = cropped["cond_dynamic"][0]
        stats = compute_log_zscore_stats(raw_field, clip_min=0.0)
    elif inspection_mode == "era5_temp":
        variable_name = "temp"
        transform_name = "zscore"
        raw_field = cropped["cond_dynamic"][1]
        stats = compute_zscore_stats(raw_field)
    else:
        raise ValueError(f"Unknown inspection_mode '{inspection_mode}'")

    transform = build_transform(
        transform_name=transform_name,
        stats=stats,
        clip_min=0.0,
    )

    transformed = transform.forward(raw_field)
    backtransformed = transform.inverse(transformed)

    print(f"\n=== Inspecting transform for date {first_date} ===")
    print(f"inspection_mode: {inspection_mode}")
    print(f"variable_name: {variable_name}")
    print(f"transform_name: {transform_name}")
    print(f"transform_metadata: {transform.get_metadata()}")

    print_array_info("raw_field", raw_field)
    print_array_info("transformed", transformed)
    print_array_info("backtransformed", backtransformed)

    if variable_name == "prcp":
        reference = np.clip(raw_field, 0.0, None)
    else:
        reference = raw_field

    reconstruction_error = np.abs(backtransformed - reference)
    print(f"\nmax reconstruction abs error:  {np.nanmax(reconstruction_error)}")
    print(f"mean reconstruction abs error: {np.nanmean(reconstruction_error)}")

    plot_transform_triptych(
        raw_field=raw_field,
        transformed_field=transformed,
        backtransformed_field=backtransformed,
        variable_name=variable_name,
        transform_name=transform_name,
        date_str=first_date,
    )


if __name__ == "__main__":
    main()