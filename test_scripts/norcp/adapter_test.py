

"""
Smoke / inspection test for the NorCP adapter.

Purpose
-------
- Build the NorCP adapter with a lightweight config object.
- Instantiate one dataset split.
- Inspect one sample from the real adapter output.
- Print key tensor shapes and metadata.
- Optionally plot all target / dynamic / static channels in one compact figure.

This is intentionally similar in spirit to the DANRA/ERA5 adapter inspection
workflow, but tailored to the current NorCP adapter contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.adapter import NorCPAdapter
from stride_core.plotting.colormaps import (
    get_variable_cmap,
    get_variable_label,
)


# -----------------------------------------------------------------------------
# User-facing test configuration
# -----------------------------------------------------------------------------

ROOT_DIR = Path("/Users/au728490/Data/NorCP/cropped")
SCENARIO_NAME = "ECMWF-ERAINT"
SPLIT_NAME = "train"
DOMAIN_TAG = "full_domain"
TEMPORAL_TAG = "6hr"
TARGET_SPATIAL_TAG = "3km"
DYNAMIC_SPATIAL_TAG = "12km"

TARGET_VARIABLES = ["prcp"] #, "temp"
DYNAMIC_VARIABLES = [
    "prcp",
    "temp",
    "hus500",
    "ta500",
    "ua500",
    "va500",
    "zg500",
    "hus700",
    "ta700",
    "ua700",
    "va700",
    "zg700",
    "hus850",
    "ta850",
    "ua850",
    "va850",
    "zg850",
    "hus950",
    "ta950",
    "ua950",
    "va950",
    "zg950",
    "hus1000",
    "ta1000",
    "ua1000",
    "va1000",
    "zg1000",
]
STATIC_VARIABLES = ["topo"]

TARGET_TIME_OFFSETS = {"prcp": -3.0}
DYNAMIC_TIME_OFFSETS = {"prcp": -3.0}

APPLY_TRANSFORMS = True
PLOT_SAMPLE = True
SAMPLE_INDEX = 0
FIGSIZE = (16, 10)
MAX_DYNAMIC_PANELS = None  # set to an int to cap the number of dynamic panels


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)



def find_latest_split_manifest() -> Path:
    splits_root = Path(__file__).resolve().parents[2] / "data_adapters" / "norcp" / "saved" / "splits"
    candidates = sorted(splits_root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No split manifest JSON found under: {splits_root}\n"
            "Run the NorCP split launcher first."
        )
    return candidates[0]



def build_adapter_cfg(split_manifest_path: Path) -> SimpleNamespace:
    split_manifest_tag = split_manifest_path.stem
    stats_root = (
        Path(__file__).resolve().parents[2]
        / "data_adapters"
        / "norcp"
        / "saved"
        / "statistics"
        / split_manifest_tag
    )

    return SimpleNamespace(
        root_dir=str(ROOT_DIR),
        scenario_name=SCENARIO_NAME,
        split_name=SPLIT_NAME,
        split_manifest_path=str(split_manifest_path),
        domain_tag=DOMAIN_TAG,
        temporal_tag=TEMPORAL_TAG,
        target_spatial_tag=TARGET_SPATIAL_TAG,
        dynamic_spatial_tag=DYNAMIC_SPATIAL_TAG,
        target_variables=list(TARGET_VARIABLES),
        dynamic_variables=list(DYNAMIC_VARIABLES),
        static_variables=list(STATIC_VARIABLES),
        target_source="NORCP_HR",
        dynamic_source="NORCP_LR",
        static_source="NORCP_STATIC",
        target_time_offsets=dict(TARGET_TIME_OFFSETS),
        dynamic_time_offsets=dict(DYNAMIC_TIME_OFFSETS),
        apply_transforms=APPLY_TRANSFORMS,
        stats_root=str(stats_root),
        crop=None,
    )



def tensor_to_numpy(x: torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    return x.detach().cpu().numpy()



def summarize_tensor(name: str, tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    arr = tensor_to_numpy(tensor)
    if arr is None:
        return None
    return {
        "name": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }



def _variable_colormap(variable: str):
    try:
        return get_variable_cmap(variable)
    except Exception:
        return None


def _variable_title(prefix: str, variable: str) -> str:
    try:
        label = get_variable_label(variable, with_unit=True)
    except Exception:
        label = variable
    return f"{prefix}: {label}"



def plot_sample_panels(
    *,
    target: np.ndarray,
    target_variables: list[str],
    dynamic: np.ndarray,
    dynamic_variables: list[str],
    static: np.ndarray | None,
    static_variables: list[str],
    timestamp: str,
) -> None:
    panel_specs: list[tuple[str, np.ndarray, str]] = []

    for idx, variable in enumerate(target_variables):
        panel_specs.append((_variable_title("target", variable), target[idx], variable))

    dynamic_variables_to_plot = dynamic_variables
    dynamic_to_plot = dynamic
    if MAX_DYNAMIC_PANELS is not None:
        dynamic_variables_to_plot = dynamic_variables[:MAX_DYNAMIC_PANELS]
        dynamic_to_plot = dynamic[:MAX_DYNAMIC_PANELS]

    for idx, variable in enumerate(dynamic_variables_to_plot):
        panel_specs.append((_variable_title("dynamic", variable), dynamic_to_plot[idx], variable))

    if static is not None:
        for idx, variable in enumerate(static_variables[: static.shape[0]]):
            panel_specs.append((_variable_title("static", variable), static[idx], variable))

    n_panels = len(panel_specs)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax, (title, arr, variable) in zip(axes.flat, panel_specs):
        cmap = _variable_colormap(variable)
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"NorCP adapter sample inspection | {timestamp}", fontsize=12)
    fig.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print_section("Resolve split manifest")
    split_manifest_path = find_latest_split_manifest()
    print(f"Using split manifest: {split_manifest_path}")

    print_section("Build adapter")
    cfg = build_adapter_cfg(split_manifest_path)
    adapter = NorCPAdapter(cfg) # type: ignore
    dataset = adapter.build_dataset()
    print(
        {
            "dataset_len": len(dataset),
            "split_name": cfg.split_name,
            "scenario_name": cfg.scenario_name,
            "target_variables": cfg.target_variables,
            "n_dynamic_variables": len(cfg.dynamic_variables),
            "static_variables": cfg.static_variables,
            "apply_transforms": cfg.apply_transforms,
            "stats_root": cfg.stats_root,
        }
    )

    print_section("Fetch one adapter sample")
    sample = dataset[SAMPLE_INDEX]
    print(f"Fetched sample index: {SAMPLE_INDEX}")
    print(f"Returned keys: {list(sample.keys())}")

    target = sample["target"]
    cond_dynamic = sample["cond_dynamic"]
    cond_static = sample["cond_static"]
    time_features = sample["time_features"]
    cond_coord = sample["cond_coord"]
    meta = sample["meta"]

    print_section("Tensor summaries")
    print(summarize_tensor("target", target))
    print(summarize_tensor("cond_dynamic", cond_dynamic))
    print(summarize_tensor("cond_static", cond_static))
    print(
        {
            "time_features_shape": tuple(time_features.shape),
            "time_features": time_features.detach().cpu().numpy().tolist(),
        }
    )

    print_section("cond_coord")
    print(cond_coord)

    print_section("meta summary")
    print(
        {
            "timestamp": meta.get("timestamp"),
            "scenario_name": meta.get("scenario_name"),
            "split_name": meta.get("split_name"),
            "domain_tag": meta.get("domain_tag"),
            "target_variables": meta.get("target_variables"),
            "n_dynamic_variables": len(meta.get("dynamic_variables", [])),
            "static_variables": meta.get("static_variables"),
            "target_source": meta.get("target_source"),
            "dynamic_source": meta.get("dynamic_source"),
            "static_source": meta.get("static_source"),
            "apply_transforms": meta.get("apply_transforms"),
        }
    )

    print_section("Field metadata preview")
    field_metadata = meta.get("field_metadata", {})
    print(
        {
            "target_preview": field_metadata.get("target", [])[:2],
            "dynamic_preview": field_metadata.get("dynamic", [])[:3],
            "static_preview": field_metadata.get("static", [])[:2],
        }
    )

    if PLOT_SAMPLE:
        print_section("Plot adapter sample")
        target_np = tensor_to_numpy(target)
        dynamic_np = tensor_to_numpy(cond_dynamic)
        static_np = tensor_to_numpy(cond_static)
        if target_np is None or dynamic_np is None:
            raise RuntimeError("Expected non-null target and cond_dynamic tensors for plotting")
        plot_sample_panels(
            target=target_np,
            target_variables=list(cfg.target_variables),
            dynamic=dynamic_np,
            dynamic_variables=list(cfg.dynamic_variables),
            static=static_np,
            static_variables=list(cfg.static_variables),
            timestamp=str(meta.get("timestamp")),
        )

    print_section("Final quick sanity summary")
    print(
        {
            "sample_index": SAMPLE_INDEX,
            "timestamp": meta.get("timestamp"),
            "target_shape": tuple(target.shape),
            "cond_dynamic_shape": tuple(cond_dynamic.shape),
            "cond_static_shape": None if cond_static is None else tuple(cond_static.shape),
            "time_features_shape": tuple(time_features.shape),
        }
    )


if __name__ == "__main__":
    main()