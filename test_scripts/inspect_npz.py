from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CMAP = "viridis"


def inspect_npz(file_path: str | Path) -> None:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    print("\n=====================================")
    print(f"Inspecting file: {file_path}")
    print("=====================================")

    with np.load(file_path, allow_pickle=False) as data:
        keys = list(data.keys())
        print(f"\nKeys in file: {keys}")

        for key in keys:
            arr = data[key]

            print("\n-------------------------------------")
            print(f"Key: {key}")
            print(f"Type: {type(arr)}")

            if isinstance(arr, np.ndarray):
                print(f"Shape: {arr.shape}")
                print(f"Dtype: {arr.dtype}")

                try:
                    print(f"Min: {np.nanmin(arr)}")
                    print(f"Max: {np.nanmax(arr)}")
                    print(f"Mean: {np.nanmean(arr)}")
                except Exception:
                    print("Could not compute statistics.")

                if arr.ndim == 2:
                    plot_full_field(arr, title=f"{file_path.name} | key={key}")
                else:
                    print("Skipping plot because array is not 2D.")
            else:
                print("Non-array object:")
                print(arr)

    print("\nDone.\n")


def plot_full_field(array: np.ndarray, title: str, cmap: str = DEFAULT_CMAP) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(array, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    files_to_check = [
        "/Users/au728490/Data/Data_DiffMod_small/data_DANRA/size_589x789/prcp_589x789/all/tp_tot_19910813.npz",
        "/Users/au728490/Data/Data_DiffMod_small/data_ERA5/size_589x789/prcp_589x789/all/prcp_589x789_19910813.npz",
        "/Users/au728490/Data/Data_DiffMod_small/data_ERA5/size_589x789/temp_589x789/all/temp_589x789_19910813.npz",
        "/Users/au728490/Data/Data_DiffMod_small/data_lsm/truth_fullDomain/lsm_full.npz",
        "/Users/au728490/Data/Data_DiffMod_small/data_topo/truth_fullDomain/topo_full.npz",
    ]

    for f in files_to_check:
        try:
            inspect_npz(f)
        except Exception as e:
            print(f"\nFailed to inspect {f}")
            print(e)