from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/inspect_features.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from data_adapters.danra_era5_small.features import (
    build_feature_metadata,
    load_dynamic_conditioning,
    load_static_features,
    load_target_field,
)
from data_adapters.danra_era5_small.paths import build_experiment_file_index


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



def print_channel_info(prefix: str, array: np.ndarray, channel_names: list[str]) -> None:
    for idx, channel_name in enumerate(channel_names):
        print_array_info(f"{prefix}[{channel_name}]", array[idx])



def plot_feature_panels(
    date_str: str,
    target: np.ndarray,
    cond_dynamic: np.ndarray,
    cond_dynamic_names: list[str],
    cond_static: np.ndarray | None,
    cond_static_names: list[str],
) -> None:
    panels: list[tuple[str, np.ndarray, str]] = [("target: prcp (DANRA)", target, "prcp")]

    for idx, name in enumerate(cond_dynamic_names):
        panels.append((f"cond_dynamic: {name} (ERA5)", cond_dynamic[idx], name))

    if cond_static is not None:
        for idx, name in enumerate(cond_static_names):
            panels.append((f"cond_static: {name} (STATIC)", cond_static[idx], name))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"STRIDE feature inspection | date={date_str} | target=DANRA prcp | cond=ERA5 prcp,temp | static=lsm,topo",
        fontsize=11,
    )

    for ax, (title, field, var_name) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=PLOT_CMAPS.get(var_name, "viridis")) # type: ignore
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

    if not file_index:
        raise RuntimeError("File index is empty")

    first_date = sorted(file_index.keys())[0]
    sample_paths = file_index[first_date]

    static_paths = {
        "lsm": root_dir / "data_lsm" / "truth_fullDomain" / "lsm_full.npz",
        "topo": root_dir / "data_topo" / "truth_fullDomain" / "topo_full.npz",
    }

    dynamic_names = ["prcp", "temp"]
    static_names = ["lsm", "topo"]

    target = load_target_field(
        target_path=sample_paths["target"], # type: ignore
        target_variable="prcp",
        target_source="DANRA",
    )
    cond_dynamic = load_dynamic_conditioning(
        dynamic_paths=sample_paths["cond_dynamic"], # type: ignore
        variable_order=dynamic_names,
        source="ERA5",
    )
    cond_static = load_static_features(
        static_paths=static_paths, # type: ignore
        variable_order=static_names,
        source="STATIC",
    )
    metadata = build_feature_metadata(
        target_variable="prcp",
        dynamic_variables=dynamic_names,
        static_variables=static_names,
    )

    print(f"\n=== Inspecting features for date {first_date} ===")
    print_array_info("target", target)
    print_array_info("cond_dynamic", cond_dynamic)
    print_channel_info("cond_dynamic", cond_dynamic, dynamic_names)

    if cond_static is not None:
        print_array_info("cond_static", cond_static)
        print_channel_info("cond_static", cond_static, static_names)
    else:
        print_array_info("cond_static", None)

    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    plot_feature_panels(
        date_str=first_date,
        target=target,
        cond_dynamic=cond_dynamic,
        cond_dynamic_names=dynamic_names,
        cond_static=cond_static,
        cond_static_names=static_names,
    )


if __name__ == "__main__":
    main()