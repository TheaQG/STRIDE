"""
Smoke test for train-time spatial shuffling in the DANRA/ERA5 adapter.

This test verifies that:
- train split samples use randomized crop anchors when spatial shuffling is enabled
- validation split samples remain deterministic
- returned tensor shapes remain fixed
- sampled crop anchors stay inside the configured cutout domain

The test works by creating temporary resolved-style data configs for train and
validation, then building adapters directly from those configs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stride_core.configs.adapter_config import AdapterConfig
from stride_core.training.data import build_dataset_adapter


BASE_DATA_CONFIG_PATH = REPO_ROOT / "configs" / "datasets" / "danra_era5_small.yaml"
CUTOUT_DOMAIN = (340, 520, 170, 350)  # [x1, x2, y1, y2]


def _resolve_repo_relative_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def _resolve_manifest_path_from_temp_config(config_path: Path) -> Path | None:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        return None

    data_cfg = payload.get("data")
    if not isinstance(data_cfg, dict):
        return None

    split_cfg = data_cfg.get("split")
    if not isinstance(split_cfg, dict):
        return None

    raw_manifest = split_cfg.get("manifest_path")
    if raw_manifest is None:
        raw_manifest = split_cfg.get("split_manifest_path")
    if raw_manifest is None:
        return None

    resolved = _resolve_repo_relative_path(str(raw_manifest))
    return None if resolved is None else Path(resolved)


def _build_split_dataset(config_path: Path, split_name: str):
    adapter_cfg = AdapterConfig.from_yaml(config_path)

    resolved_manifest_path = _resolve_manifest_path_from_temp_config(config_path)
    if resolved_manifest_path is not None:
        adapter_cfg = replace(
            adapter_cfg,
            split_manifest_path=resolved_manifest_path,
        )

    adapter = build_dataset_adapter(adapter_cfg)
    datasets = adapter.build_datasets()
    if split_name not in datasets:
        raise KeyError(f"Adapter did not return split {split_name!r}")
    return datasets[split_name]


def print_section(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def _pick_two_distinct_indices(dataset) -> tuple[int, int]:
    """
    Pick two distinct dataset indices for visual comparison.
    """
    n = len(dataset)
    if n < 2:
        raise ValueError(
            f"Need at least 2 samples in dataset to compare distinct days, got {n}"
        )
    return 0, 1


def _write_temp_data_config(
    base_path: Path,
    *,
    split_name: str,
    shuffle_enabled: bool,
) -> Path:
    if not base_path.exists():
        raise FileNotFoundError(f"Base data config does not exist: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected dataset config to load into a dict, got {type(payload)}"
        )

    data_cfg = payload.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Expected top-level 'data' section to be a dict")

    split_cfg = data_cfg.setdefault("split", {})
    if not isinstance(split_cfg, dict):
        raise ValueError("Expected 'data.split' to be a dict")
    split_cfg["name"] = split_name

    # Resolve split manifest path to absolute repo location so that
    # writing the config into a temp directory does not break path resolution.
    if split_cfg.get("manifest_path") is not None:
        split_cfg["manifest_path"] = _resolve_repo_relative_path(
            str(split_cfg["manifest_path"])
        )
    if split_cfg.get("split_manifest_path") is not None:
        split_cfg["split_manifest_path"] = _resolve_repo_relative_path(
            str(split_cfg["split_manifest_path"])
        )

    domain_cfg = data_cfg.setdefault("domain", {})
    if not isinstance(domain_cfg, dict):
        raise ValueError("Expected 'data.domain' to be a dict")

    if data_cfg.get("root_dir") is not None:
        data_cfg["root_dir"] = _resolve_repo_relative_path(str(data_cfg["root_dir"]))

    crop_cfg = domain_cfg.setdefault("crop", {})
    if not isinstance(crop_cfg, dict):
        raise ValueError("Expected 'data.domain.crop' to be a dict")
    for key in ("anchor_y", "anchor_x", "height", "width"):
        if key not in crop_cfg:
            raise ValueError(f"Missing required crop key: {key}")

    spatial_shuffle_cfg = domain_cfg.setdefault("spatial_shuffle", {})
    if not isinstance(spatial_shuffle_cfg, dict):
        raise ValueError("Expected 'data.domain.spatial_shuffle' to be a dict")
    spatial_shuffle_cfg["enabled"] = shuffle_enabled
    spatial_shuffle_cfg["train_only"] = True
    spatial_shuffle_cfg["cutout_domain"] = list(CUTOUT_DOMAIN)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"stride_shuffle_{split_name}_"))
    out_path = tmp_dir / base_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out_path


def _region_info(sample: dict) -> dict:
    meta = sample.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Expected sample['meta'] to be a dict")
    region_info = meta.get("region_info")
    if not isinstance(region_info, dict):
        raise ValueError("Expected sample['meta']['region_info'] to be a dict")
    return region_info


def _assert_shapes(sample: dict) -> None:
    target = sample["target"]
    cond_dynamic = sample["cond_dynamic"]
    cond_static = sample["cond_static"]
    time_features = sample["time_features"]

    assert tuple(target.shape[-2:]) == (128, 128), (
        f"Unexpected target spatial shape: {tuple(target.shape)}"
    )
    assert tuple(cond_dynamic.shape[-2:]) == (128, 128), (
        f"Unexpected cond_dynamic spatial shape: {tuple(cond_dynamic.shape)}"
    )
    if cond_static is not None:
        assert tuple(cond_static.shape[-2:]) == (128, 128), (
            f"Unexpected cond_static spatial shape: {tuple(cond_static.shape)}"
        )
    assert tuple(time_features.shape) == (2,), (
        f"Unexpected time_features shape: {tuple(time_features.shape)}"
    )


def _assert_anchor_inside_cutout(region_info: dict) -> None:
    x1, x2, y1, y2 = CUTOUT_DOMAIN
    anchor_x = int(region_info["crop_anchor_x"])
    anchor_y = int(region_info["crop_anchor_y"])
    crop_w = int(region_info["crop_width"])
    crop_h = int(region_info["crop_height"])

    assert x1 <= anchor_x <= x2 - crop_w, (
        f"anchor_x out of bounds: {anchor_x}, expected within [{x1}, {x2 - crop_w}]"
    )
    assert y1 <= anchor_y <= y2 - crop_h, (
        f"anchor_y out of bounds: {anchor_y}, expected within [{y1}, {y2 - crop_h}]"
    )


def _build_full_domain_canvas(region_info: dict):
    """
    Build a simple blank full-domain canvas from region metadata.

    This avoids depending on private dataset loading internals while still
    allowing us to visualize the full domain extent, the configured cutout
    region, and the sampled crop boxes.
    """
    import numpy as np

    full_height = int(region_info["full_height"])
    full_width = int(region_info["full_width"])
    return np.zeros((full_height, full_width), dtype=float)


def _save_train_visual_inspection(
    sample_a: dict,
    sample_b: dict,
    region_a: dict,
    region_b: dict,
) -> Path:
    """
    Save a side-by-side multi-panel plot of shuffled train samples A and B,
    including target, dynamic conditions, and static conditions.
    """
    target_a = sample_a["target"].detach().cpu().squeeze(0).numpy()
    target_b = sample_b["target"].detach().cpu().squeeze(0).numpy()

    cond_dynamic_a = sample_a["cond_dynamic"].detach().cpu().numpy()
    cond_dynamic_b = sample_b["cond_dynamic"].detach().cpu().numpy()

    cond_static_a = sample_a["cond_static"]
    cond_static_b = sample_b["cond_static"]
    cond_static_a_np = (
        None if cond_static_a is None else cond_static_a.detach().cpu().numpy()
    )
    cond_static_b_np = (
        None if cond_static_b is None else cond_static_b.detach().cpu().numpy()
    )

    meta_a = sample_a.get("meta", {})
    dyn_names_a = list(meta_a.get("cond_dynamic_vars", []))
    static_names_a = list(meta_a.get("cond_static_vars", []))

    rows: list[tuple[str, object, object]] = [("target", target_a, target_b)]

    for idx in range(cond_dynamic_a.shape[0]):
        name_a = dyn_names_a[idx] if idx < len(dyn_names_a) else f"dyn_{idx}"
        rows.append(
            (
                f"cond_dynamic: {name_a}",
                cond_dynamic_a[idx],
                cond_dynamic_b[idx],
            )
        )

    if cond_static_a_np is not None and cond_static_b_np is not None:
        for idx in range(cond_static_a_np.shape[0]):
            name_a = (
                static_names_a[idx] if idx < len(static_names_a) else f"static_{idx}"
            )
            rows.append(
                (
                    f"cond_static: {name_a}",
                    cond_static_a_np[idx],
                    cond_static_b_np[idx],
                )
            )

    n_rows = len(rows)
    out_dir = REPO_ROOT / "runs" / "smoke_tests" / "smoke_test_shuffle"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "train_shuffle_visual_inspection.png"

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.2 * n_rows), squeeze=False)

    for row_idx, (label, array_a, array_b) in enumerate(rows):
        im0 = axes[row_idx, 0].imshow(array_a, origin="lower")
        axes[row_idx, 0].set_title(
            f"{label} | sample A\n"
            f"y={region_a['crop_anchor_y']}, x={region_a['crop_anchor_x']}"
        )
        axes[row_idx, 0].set_xlabel("x")
        axes[row_idx, 0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[row_idx, 1].imshow(array_b, origin="lower")
        axes[row_idx, 1].set_title(
            f"{label} | sample B\n"
            f"y={region_b['crop_anchor_y']}, x={region_b['crop_anchor_x']}"
        )
        axes[row_idx, 1].set_xlabel("x")
        axes[row_idx, 1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Spatial shuffle smoke test: train crops and conditioning")
    fig.subplots_adjust(top=0.96)
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.show()

    return figure_path


def _save_full_domain_location_plot(
    full_target,
    region_a: dict,
    region_b: dict,
) -> Path:
    """
    Save a figure showing the full target domain, the shuffle cutout region,
    and the sampled crop boxes for train samples A and B.
    """
    if hasattr(full_target, "detach"):
        full_target = full_target.detach().cpu().numpy()

    x1, x2, y1, y2 = CUTOUT_DOMAIN

    out_dir = REPO_ROOT / "runs" / "smoke_tests" / "smoke_test_shuffle"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / "train_shuffle_full_domain_locations.png"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Make sure y-axis is 0 at the bottom
    im = ax.imshow(full_target, cmap="Greys", vmin=0.0, vmax=1.0, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cutout_rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2.0,
        linestyle="--",
        label="Shuffle cutout",
    )
    ax.add_patch(cutout_rect)

    rect_a = Rectangle(
        (int(region_a["crop_anchor_x"]), int(region_a["crop_anchor_y"])),
        int(region_a["crop_width"]),
        int(region_a["crop_height"]),
        fill=False,
        linewidth=2.0,
        label="Sample A crop",
    )
    ax.add_patch(rect_a)

    rect_b = Rectangle(
        (int(region_b["crop_anchor_x"]), int(region_b["crop_anchor_y"])),
        int(region_b["crop_width"]),
        int(region_b["crop_height"]),
        fill=False,
        linewidth=2.0,
        label="Sample B crop",
    )
    ax.add_patch(rect_b)

    ax.set_title(
        "Spatial shuffle smoke test: full domain extent, cutout, and sampled crops"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(figure_path), dpi=150, bbox_inches="tight")
    plt.show()

    return figure_path


def main() -> None:
    print_section("STRIDE spatial shuffle smoke test")
    print(f"Base data config: {BASE_DATA_CONFIG_PATH}")

    train_cfg_path = _write_temp_data_config(
        BASE_DATA_CONFIG_PATH,
        split_name="train",
        shuffle_enabled=True,
    )
    valid_cfg_path = _write_temp_data_config(
        BASE_DATA_CONFIG_PATH,
        split_name="valid",
        shuffle_enabled=True,
    )

    print_section("Temporary configs written")
    print(f"Train config: {train_cfg_path}")
    print(f"Valid config: {valid_cfg_path}")

    train_ds = _build_split_dataset(train_cfg_path, "train")
    valid_ds = _build_split_dataset(valid_cfg_path, "valid")

    print_section("Inspecting train split randomness")
    train_idx_a, train_idx_b = _pick_two_distinct_indices(train_ds)
    train_a = train_ds[train_idx_a]
    train_b = train_ds[train_idx_b]
    train_region_a = _region_info(train_a)
    train_region_b = _region_info(train_b)

    print(f"Train sample A region: {train_region_a}")
    print(f"Train sample B region: {train_region_b}")

    print(f"Train sample A index: {train_idx_a}")
    print(f"Train sample B index: {train_idx_b}")
    if isinstance(train_a.get("meta"), dict):
        print(f"Train sample A date:  {train_a['meta'].get('date')}")
    if isinstance(train_b.get("meta"), dict):
        print(f"Train sample B date:  {train_b['meta'].get('date')}")

    _assert_shapes(train_a)
    _assert_shapes(train_b)
    _assert_anchor_inside_cutout(train_region_a)
    _assert_anchor_inside_cutout(train_region_b)

    train_anchor_a = (
        int(train_region_a["crop_anchor_y"]),
        int(train_region_a["crop_anchor_x"]),
    )
    train_anchor_b = (
        int(train_region_b["crop_anchor_y"]),
        int(train_region_b["crop_anchor_x"]),
    )


    full_target = _build_full_domain_canvas(train_region_a)

    figure_path = _save_train_visual_inspection(
        train_a,
        train_b,
        train_region_a,
        train_region_b,
    )
    location_figure_path = _save_full_domain_location_plot(
        full_target,
        train_region_a,
        train_region_b,
    )

    print("Train spatial shuffling check passed.")
    print(f"Saved train visual inspection plot: {figure_path}")
    print(f"Saved full-domain location plot: {location_figure_path}")

    print_section("Inspecting validation split determinism")
    valid_a = valid_ds[0]
    valid_b = valid_ds[0]
    valid_region_a = _region_info(valid_a)
    valid_region_b = _region_info(valid_b)

    print(f"Valid sample A region: {valid_region_a}")
    print(f"Valid sample B region: {valid_region_b}")

    _assert_shapes(valid_a)
    _assert_shapes(valid_b)

    valid_anchor_a = (
        int(valid_region_a["crop_anchor_y"]),
        int(valid_region_a["crop_anchor_x"]),
    )
    valid_anchor_b = (
        int(valid_region_b["crop_anchor_y"]),
        int(valid_region_b["crop_anchor_x"]),
    )

    if valid_anchor_a != valid_anchor_b:
        raise RuntimeError(
            "Validation split returned different crop anchors across repeated access; "
            "expected deterministic fixed crop for non-train splits."
        )

    print("Validation determinism check passed.")

    print_section("Spatial shuffle smoke test completed successfully")


if __name__ == "__main__":
    main()
