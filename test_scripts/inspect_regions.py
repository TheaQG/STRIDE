from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/inspect_region_crop.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from data_adapters.danra_era5_small.features import (
    load_dynamic_conditioning,
    load_static_features,
    load_target_field,
)
from data_adapters.danra_era5_small.paths import build_experiment_file_index
from data_adapters.danra_era5_small.regions import CropSpec, crop_sample_fields


PLOT_CMAPS = {
    "prcp": "Blues",
    "temp": "plasma",
    "lsm": "gray",
    "topo": "terrain",
}



def print_array_info(name: str, array: np.ndarray | None) -> None:
    print(f"\n{name}:")
    if array is None:
        print("  None")
        return
    print(f"  shape: {array.shape}")
    print(f"  dtype: {array.dtype}")
    print(f"  min:   {np.nanmin(array)}")
    print(f"  max:   {np.nanmax(array)}")
    print(f"  mean:  {np.nanmean(array)}")



def plot_full_domain_with_crop(
    target: np.ndarray,
    crop_spec: CropSpec,
    date_str: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(target, origin="lower", cmap=PLOT_CMAPS["prcp"])
    rect = patches.Rectangle(
        (crop_spec.anchor_x, crop_spec.anchor_y),
        crop_spec.width,
        crop_spec.height,
        linewidth=2.5,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(f"Full-domain DANRA prcp with crop outline | date={date_str}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.9)  # type: ignore
    plt.tight_layout()
    plt.show()



def plot_cropped_panels(
    date_str: str,
    cropped_target: np.ndarray,
    cropped_cond_dynamic: np.ndarray,
    cropped_cond_static: np.ndarray | None,
) -> None:
    panels: list[tuple[str, np.ndarray, str]] = [("cropped target: prcp", cropped_target, "prcp")]
    panels.append(("cropped cond_dynamic: prcp", cropped_cond_dynamic[0], "prcp"))
    panels.append(("cropped cond_dynamic: temp", cropped_cond_dynamic[1], "temp"))

    if cropped_cond_static is not None:
        panels.append(("cropped cond_static: lsm", cropped_cond_static[0], "lsm"))
        panels.append(("cropped cond_static: topo", cropped_cond_static[1], "topo"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    if len(panels) == 1:
        axes = [axes]

    fig.suptitle(f"Cropped STRIDE sample panels | date={date_str}", fontsize=11)

    for ax, (title, field, var_name) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=PLOT_CMAPS.get(var_name, "viridis"))  # type: ignore
        ax.set_title(title, fontsize=10) # type: ignore
        ax.set_xlabel("x") # type: ignore
        ax.set_ylabel("y") # type: ignore
        plt.colorbar(im, ax=ax, shrink=0.85) # type: ignore

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

    # Placeholder crop. Update anchor once you choose the Denmark cutout.
    crop_spec = CropSpec(anchor_y=200, anchor_x=380, height=128, width=128)

    cropped = crop_sample_fields(
        target=target,
        cond_dynamic=cond_dynamic,
        cond_static=cond_static,
        crop_spec=crop_spec,
    )

    print(f"\n=== Inspecting cropped sample for date {first_date} ===")
    print_array_info("cropped target", cropped["target"])
    print_array_info("cropped cond_dynamic", cropped["cond_dynamic"])
    print_array_info("cropped cond_static", cropped["cond_static"])

    print("\nRegion info:")
    for key, value in cropped["region_info"].items():
        print(f"  {key}: {value}")

    plot_full_domain_with_crop(target=target, crop_spec=crop_spec, date_str=first_date)
    plot_cropped_panels(
        date_str=first_date,
        cropped_target=cropped["target"],
        cropped_cond_dynamic=cropped["cond_dynamic"],
        cropped_cond_static=cropped["cond_static"],
    )

if __name__ == "__main__":
    main()