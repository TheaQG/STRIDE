from pathlib import Path
import sys

# Allow direct execution via: python test_scripts/norcp/inspect_adapter_norcp.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch

from data_adapters.norcp.adapter import NorCPAdapter
from stride_core.configs.adapter_config import AdapterConfig
from stride_core.plotting.colormaps import get_variable_cmap



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

    n_cols = min(4, max(1, len(panels)))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    axes_list = list(axes.flat)

    fig.suptitle(title_prefix, fontsize=11)

    for ax, (panel_title, field, var_name) in zip(axes_list, panels):
        im = ax.imshow(field, origin="lower", cmap=get_variable_cmap(var_name))
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes_list[len(panels):]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # save
    fig.savefig("norcp_adapter_sample_inspection.png", dpi=300)
    plt.show()



def main() -> None:
    cfg = AdapterConfig.from_yaml(
        Path(__file__).resolve().parents[2]
        / "configs"
        / "datasets"
        / "norcp.yaml"
    )

    adapter = NorCPAdapter(cfg)
    dataset = adapter.build_dataset()

    print("\n=== Inspecting adapter dataset ===")
    print(f"dataset length: {len(dataset)}")
    print(f"split name:     {cfg.split_name}")
    print(f"domain tag:     {cfg.domain_tag}")
    print(f"stats split tag:{cfg.split_stats_tag}")

    sample = dataset[0]
    meta = sample["meta"]

    print_tensor_info("target", sample["target"])
    print_tensor_info("cond_dynamic", sample["cond_dynamic"])
    print_tensor_info("cond_static", sample["cond_static"])

    print("\ncond_coord:")
    for key, value in sample["cond_coord"].items():
        print(f"  {key}: {value}")

    print("\nmeta:")
    for key, value in meta.items():
        if key in {"transform_metadata", "field_metadata"}:
            print(f"  {key}: ...")
        else:
            print(f"  {key}: {value}")

    print("\nfield_metadata preview:")
    field_metadata = meta.get("field_metadata", {})
    print(
        {
            "target_preview": field_metadata.get("target", [])[:2],
            "dynamic_preview": field_metadata.get("dynamic", [])[:3],
            "static_preview": field_metadata.get("static", [])[:2],
        }
    )

    print("\ntransform_metadata:")
    print(meta["transform_metadata"])

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

    print(f"  meta['day_of_year']: {meta.get('day_of_year', None)}")
    print(f"  meta['doy_sin_cos']: {meta.get('doy_sin_cos', None)}")

    plot_adapter_sample(
        sample,
        title_prefix=(
            f"STRIDE adapter inspection | split={cfg.split_name} | "
            f"domain={cfg.domain_tag} | transformed={cfg.apply_transforms}"
        ),
    )


if __name__ == "__main__":
    main()