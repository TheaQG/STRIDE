from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/inspect_adapter.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch

from data_adapters.danra_era5_small.adapter import DanraEra5SmallAdapter
from stride_core.configs.adapter_config import AdapterConfig


PLOT_CMAPS = {
    "prcp": "Blues",
    "temp": "plasma",
    "lsm": "gray",
    "topo": "terrain",
}



def print_tensor_info(name: str, tensor: torch.Tensor | None) -> None:
    print(f"\n{name}:")
    if tensor is None:
        print("  None")
        return

    arr = tensor.detach().cpu().numpy()
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  min:   {np.nanmin(arr)}")
    print(f"  max:   {np.nanmax(arr)}")
    print(f"  mean:  {np.nanmean(arr)}")



def plot_adapter_sample(
    sample: dict,
    title_prefix: str,
) -> None:
    target = sample["target"].detach().cpu().numpy()
    cond_dynamic = sample["cond_dynamic"].detach().cpu().numpy()
    cond_static = (
        sample["cond_static"].detach().cpu().numpy()
        if sample["cond_static"] is not None
        else None
    )
    meta = sample["meta"]

    dynamic_names = meta["cond_dynamic_vars"]
    static_names = meta["cond_static_vars"]

    panels: list[tuple[str, np.ndarray, str]] = [
        (f"target: {meta['target_var']}", target[0], meta["target_var"])
    ]

    for idx, name in enumerate(dynamic_names):
        panels.append((f"cond_dynamic: {name}", cond_dynamic[idx], name))

    if cond_static is not None:
        for idx, name in enumerate(static_names):
            panels.append((f"cond_static: {name}", cond_static[idx], name))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))

    # Help static type checkers understand the axes type
    if isinstance(axes, Axes):
        axes_list: list[Axes] = [axes]
    else:
        axes_list = list(axes)


    fig.suptitle(title_prefix, fontsize=11)

    for ax, (panel_title, field, var_name) in zip(axes_list, panels):
        im = ax.imshow(field, origin="lower", cmap=PLOT_CMAPS.get(var_name, "viridis"))
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    plt.show()



def main() -> None:
    cfg = AdapterConfig.from_yaml(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "datasets"
        / "danra_era5_small.yaml"
    )

    adapter = DanraEra5SmallAdapter(cfg)
    dataset = adapter.build_dataset()

    print("\n=== Inspecting adapter dataset ===")
    print(f"dataset length: {len(dataset)}")
    print(f"split name:     {cfg.split_name}")
    print(f"domain tag:     {cfg.domain_tag}")
    print(f"stats split tag:{cfg.split_tag}")

    sample = dataset[0]

    print_tensor_info("target", sample["target"])
    print_tensor_info("cond_dynamic", sample["cond_dynamic"])
    print_tensor_info("cond_static", sample["cond_static"])

    print("\ncond_coord:")
    for key, value in sample["cond_coord"].items():
        print(f"  {key}: {value}")

    print("\nmeta:")
    for key, value in sample["meta"].items():
        if key == "transform_metadata":
            print(f"  {key}: ...")
        else:
            print(f"  {key}: {value}")

    print("\ntransform_metadata:")
    print(sample["meta"]["transform_metadata"])

    print("\nDOY check:")
    time_features = sample.get("time_features", None)
    if time_features is None:
        print("  time_features: None")
    else:
        tf = time_features.detach().cpu().numpy()
        print(f"  time_features shape: {tuple(time_features.shape)}")
        print(f"  time_features dtype: {time_features.dtype}")
        print(f"  sin(DOY): {float(tf[0])}")
        print(f"  cos(DOY): {float(tf[1])}")

    print(f"  meta['day_of_year']: {sample['meta'].get('day_of_year', None)}")
    print(f"  meta['doy_sin_cos']: {sample['meta'].get('doy_sin_cos', None)}")

    plot_adapter_sample(
        sample,
        title_prefix=(
            f"STRIDE adapter inspection | split={cfg.split_name} | "
            f"domain={cfg.domain_tag} | transformed={cfg.apply_transforms}"
        ),
    )


if __name__ == "__main__":
    main()
