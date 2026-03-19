from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_adapters.norcp.adapter import NorCPDataset
from stride_core.configs.adapter_config import AdapterConfig


def _load_adapter_config(path: Path) -> AdapterConfig:
    """
    Load an AdapterConfig robustly from YAML.
    """
    if hasattr(AdapterConfig, "from_yaml"):
        return AdapterConfig.from_yaml(path)  # type: ignore[attr-defined]

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AdapterConfig(**data)



def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _entry_get(sample_entry: Any, *names: str) -> Any:
    """
    Read a field from either a dataclass/object-style sample entry or a dict-style entry.
    """
    if isinstance(sample_entry, dict):
        for name in names:
            if name in sample_entry:
                return sample_entry[name]
    else:
        for name in names:
            if hasattr(sample_entry, name):
                return getattr(sample_entry, name)

    available = list(sample_entry.keys()) if isinstance(sample_entry, dict) else sorted(vars(sample_entry).keys())
    raise KeyError(f"Could not find any of {names} on sample entry. Available fields: {available}")



def _summary_stats(raw: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = reconstructed - raw
    denom = np.maximum(np.abs(raw), 1e-8)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_err": float(np.max(np.abs(diff))),
        "mean_bias": float(np.mean(diff)),
        "mean_rel_abs_err": float(np.mean(np.abs(diff) / denom)),
    }



def _print_stats(name: str, raw: np.ndarray, transformed: np.ndarray, reconstructed: np.ndarray) -> None:
    stats = _summary_stats(raw, reconstructed)
    print(f"\n=== {name} ===")
    print(f"raw shape:              {tuple(raw.shape)}")
    print(f"transformed shape:      {tuple(transformed.shape)}")
    print(f"reconstructed shape:    {tuple(reconstructed.shape)}")
    print(f"raw min/max:            {float(np.min(raw)):.6f} / {float(np.max(raw)):.6f}")
    print(f"transformed min/max:    {float(np.min(transformed)):.6f} / {float(np.max(transformed)):.6f}")
    print(f"reconstructed min/max:  {float(np.min(reconstructed)):.6f} / {float(np.max(reconstructed)):.6f}")
    print(f"MAE:                    {stats['mae']:.10f}")
    print(f"RMSE:                   {stats['rmse']:.10f}")
    print(f"Max abs err:            {stats['max_abs_err']:.10f}")
    print(f"Mean bias:              {stats['mean_bias']:.10f}")
    print(f"Mean rel abs err:       {stats['mean_rel_abs_err']:.10f}")



def _plot_triplet(
    raw: np.ndarray,
    transformed: np.ndarray,
    reconstructed: np.ndarray,
    *,
    title_prefix: str,
    outpath: Path,
) -> None:
    error = reconstructed - raw
    vmax_err = float(np.max(np.abs(error)))
    if vmax_err == 0.0:
        vmax_err = 1e-12

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(raw)
    axes[0].set_title(f"{title_prefix}\nraw physical")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(transformed)
    axes[1].set_title(f"{title_prefix}\nmodel space")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(reconstructed)
    axes[2].set_title(f"{title_prefix}\nback to physical")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error, vmin=-vmax_err, vmax=vmax_err)
    axes[3].set_title(f"{title_prefix}\nreconstruction error")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=180)
    plt.close(fig)
    print(f"Saved figure: {outpath}")



def run_transform_investigation(adapter_config_path: Path, sample_index: int, outdir: Path) -> None:
    cfg = _load_adapter_config(adapter_config_path)
    dataset = NorCPDataset(cfg)

    if dataset.spatial_shuffle:
        raise ValueError(
            "This investigation script expects deterministic crops. "
            "Please disable spatial_shuffle in the adapter config before running it."
        )

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} is out of range for dataset of length {len(dataset)}")

    timestamp = dataset.timestamps[sample_index]
    sample_entry = dataset._sample_lookup[timestamp]  # noqa: SLF001 - intentional debug access
    sample = dataset[sample_index]

    target_file_mapping = _entry_get(sample_entry, "target_files", "target_file_mapping")
    dynamic_file_mapping = _entry_get(sample_entry, "dynamic_files", "dynamic_file_mapping")

    raw_target, _ = dataset._load_temporal_stack(  # noqa: SLF001 - intentional debug access
        file_mapping=target_file_mapping,
        variable_order=dataset.target_variables,
        source=dataset.target_source,
        aligned_timestamp=timestamp,
        time_offsets=dataset.target_time_offsets,
        crop_spec=dataset.hr_crop_spec,
    )
    raw_dynamic, _ = dataset._load_temporal_stack(  # noqa: SLF001 - intentional debug access
        file_mapping=dynamic_file_mapping,
        variable_order=dataset.dynamic_variables,
        source=dataset.dynamic_source,
        aligned_timestamp=timestamp,
        time_offsets=dataset.dynamic_time_offsets,
        crop_spec=dataset.lr_crop_spec,
    )
    raw_static, _ = dataset._load_static_stack(  # noqa: SLF001 - intentional debug access
        file_mapping={k: Path(v) for k, v in dataset.static_file_paths.items()},
        variable_order=dataset.static_variables,
        source=dataset.static_source,
        crop_spec=dataset.hr_crop_spec,
    )

    transformed_target = _to_numpy(sample["target"])
    transformed_dynamic = _to_numpy(sample["cond_dynamic"])
    transformed_static = None if sample["cond_static"] is None else _to_numpy(sample["cond_static"])

    print("\n====================================")
    print("NorCP transform investigation script")
    print("====================================")
    print(f"Adapter config: {adapter_config_path}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample index:   {sample_index}")
    print(f"Timestamp:      {timestamp}")
    print(f"Output dir:     {outdir}")

    # Target variables
    for channel_idx, variable_name in enumerate(dataset.target_variables):
        raw = np.asarray(raw_target[channel_idx], dtype=np.float32)
        transformed = np.asarray(transformed_target[channel_idx], dtype=np.float32)
        reconstructed = dataset.target_transforms[variable_name].inverse(transformed)
        reconstructed = np.asarray(reconstructed, dtype=np.float32)

        _print_stats(f"target::{variable_name}", raw, transformed, reconstructed)
        _plot_triplet(
            raw,
            transformed,
            reconstructed,
            title_prefix=f"target::{variable_name}",
            outpath=outdir / f"target__{variable_name}.png",
        )

    # Dynamic conditioning variables
    for channel_idx, variable_name in enumerate(dataset.dynamic_variables):
        raw = np.asarray(raw_dynamic[channel_idx], dtype=np.float32)
        transformed = np.asarray(transformed_dynamic[channel_idx], dtype=np.float32)
        reconstructed = dataset.dynamic_transforms[variable_name].inverse(transformed)
        reconstructed = np.asarray(reconstructed, dtype=np.float32)

        _print_stats(f"cond_dynamic::{variable_name}", raw, transformed, reconstructed)
        _plot_triplet(
            raw,
            transformed,
            reconstructed,
            title_prefix=f"cond_dynamic::{variable_name}",
            outpath=outdir / f"cond_dynamic__{variable_name}.png",
        )

    # Static conditioning variables
    if raw_static is not None and transformed_static is not None:
        for channel_idx, variable_name in enumerate(dataset.static_variables):
            raw = np.asarray(raw_static[channel_idx], dtype=np.float32)
            transformed = np.asarray(transformed_static[channel_idx], dtype=np.float32)
            reconstructed = dataset.static_transforms[variable_name].inverse(transformed)
            reconstructed = np.asarray(reconstructed, dtype=np.float32)

            _print_stats(f"cond_static::{variable_name}", raw, transformed, reconstructed)
            _plot_triplet(
                raw,
                transformed,
                reconstructed,
                title_prefix=f"cond_static::{variable_name}",
                outpath=outdir / f"cond_static__{variable_name}.png",
            )
    else:
        print("\nNo static variables were loaded for this dataset/config.")

    print("\nDone. Inspect the printed reconstruction errors and saved plots.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect NorCP transforms by comparing raw physical fields, transformed model-space "
            "fields, and inverse-transformed physical fields."
        )
    )
    parser.add_argument(
        "adapter_config",
        type=Path,
        help="Path to the NorCP adapter config YAML, typically a compiled data_resolved.yaml",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index to inspect.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "test_scripts" / "norcp" / "saved_transform_checks",
        help="Directory where diagnostic plots are saved.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    run_transform_investigation(
        adapter_config_path=args.adapter_config,
        sample_index=args.sample_index,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()