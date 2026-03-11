
"""
Output-saving helpers for STRIDE generation runs.

This module centralizes file and metadata writing so `generator.py` does not grow
into a path-building and serialization-heavy orchestration file.

Current scope
-------------
- create per-case output directories
- save individual ensemble members as compressed `.npz`
- save aggregate products such as ensemble mean / PMM as compressed `.npz`
- save lightweight JSON metadata
- save run-level metadata

Important design note
---------------------
The current implementation keeps the simple and transparent layout:

runs/<train_run>/generation/<generation_run>/
    generation_metadata.json
    samples/
        <date>/
            member_0000.npz
            member_0000.json
            member_0001.npz
            ...
            ensemble_mean.npz
            ensemble_mean.json
            pmm.npz
            pmm.json

This is good for debugging and early development, but for large ensembles
(50-100 members per case) file count may become excessive. That should be
handled in a later iteration via a more compact backend, for example:
- one `.npz` containing the full ensemble per case
- Zarr storage
- NetCDF/HDF5 storage
- optional "compact mode" controlled from config

For now, we preserve the simple file-per-product layout because it is easy to
inspect and robust during early pipeline development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from stride_core.generation.generation_utils import build_preview_payload, tensor_to_numpy


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


TensorKeySet = tuple[str, ...]


PHYSICAL_TENSOR_KEYS: TensorKeySet = (
    "target_physical",
    "generated_physical",
    "cond_dynamic_physical",
    "cond_static_physical",
)

MODELSPACE_TENSOR_KEYS: TensorKeySet = (
    "target",
    "generated",
    "cond_dynamic",
    "cond_static",
)



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def case_output_dir(samples_dir: str | Path, *, date: Any, fallback_name: str) -> Path:
    samples_dir = ensure_dir(samples_dir)
    if date is not None:
        return ensure_dir(samples_dir / str(date))
    return ensure_dir(samples_dir / fallback_name)



def _select_tensor_keys(save_physical: bool) -> TensorKeySet:
    return PHYSICAL_TENSOR_KEYS if save_physical else MODELSPACE_TENSOR_KEYS



def _build_array_payload(
    batch: dict[str, Any],
    generated: torch.Tensor,
    *,
    save_physical: bool,
) -> dict[str, np.ndarray]:
    payload = build_preview_payload(batch, generated)

    generated_array = tensor_to_numpy(generated)
    if generated_array is None:
        raise ValueError("Failed to convert generated tensor to numpy array.")

    save_payload: dict[str, np.ndarray] = {
        "generated": generated_array,
    }

    tensor_keys = _select_tensor_keys(save_physical)
    for key in tensor_keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            array = tensor_to_numpy(value)
            if array is not None:
                save_payload[key] = array

    return save_payload



def _build_common_metadata(
    payload: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    save_physical: bool,
) -> dict[str, Any]:
    return {
        "date": payload.get("date"),
        "domain_tag": payload.get("domain_tag"),
        "variable_names": payload.get("variable_names"),
        "checkpoint_path": str(checkpoint_path),
        "save_physical": bool(save_physical),
    }


# -----------------------------------------------------------------------------
# Product saving
# -----------------------------------------------------------------------------



def save_member(
    batch: dict[str, Any],
    generated: torch.Tensor,
    *,
    case_output_dir: str | Path,
    member_idx: int,
    checkpoint_path: str | Path,
    save_physical: bool,
    save_metadata: bool,
) -> tuple[Path, Path | None]:
    case_output_dir = ensure_dir(case_output_dir)
    payload = build_preview_payload(batch, generated)
    array_payload = _build_array_payload(
        batch,
        generated,
        save_physical=save_physical,
    )

    npz_path = case_output_dir / f"member_{member_idx:04d}.npz"
    np.savez_compressed(npz_path, **array_payload)

    json_path: Path | None = None
    if save_metadata:
        metadata = _build_common_metadata(
            payload,
            checkpoint_path=checkpoint_path,
            save_physical=save_physical,
        )
        metadata["member_idx"] = int(member_idx)
        json_path = case_output_dir / f"member_{member_idx:04d}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return npz_path, json_path



def save_member_bundle(
    batch: dict[str, Any],
    generated_members: list[torch.Tensor],
    *,
    case_output_dir: str | Path,
    checkpoint_path: str | Path,
    save_physical: bool,
    save_metadata: bool,
) -> tuple[Path, Path | None]:
    if len(generated_members) == 0:
        raise ValueError("generated_members must contain at least one tensor")

    case_output_dir = ensure_dir(case_output_dir)

    member_arrays = [tensor_to_numpy(member) for member in generated_members]
    member_arrays = [arr for arr in member_arrays if arr is not None]
    if not member_arrays:
        raise ValueError("All members failed to convert to numpy arrays.")
    stacked_generated = np.stack(member_arrays, axis=0)
    if stacked_generated.ndim >= 2 and stacked_generated.shape[1] == 1:
        stacked_generated = stacked_generated[:, 0]

    reference_generated = generated_members[0]
    payload = build_preview_payload(batch, reference_generated)
    array_payload = _build_array_payload(
        batch,
        reference_generated,
        save_physical=save_physical,
    )

    array_payload["generated_members"] = stacked_generated
    array_payload.pop("generated", None)
    if save_physical:
        array_payload.pop("generated_physical", None)

    npz_path = case_output_dir / "ensemble_members.npz"
    np.savez_compressed(npz_path, **array_payload)

    json_path: Path | None = None
    if save_metadata:
        metadata = _build_common_metadata(
            payload,
            checkpoint_path=checkpoint_path,
            save_physical=save_physical,
        )
        metadata["aggregate_name"] = "ensemble_members"
        metadata["ensemble_size"] = int(len(generated_members))
        json_path = case_output_dir / "ensemble_members.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return npz_path, json_path



def save_aggregate(
    batch: dict[str, Any],
    generated: torch.Tensor,
    *,
    case_output_dir: str | Path,
    name: str,
    checkpoint_path: str | Path,
    save_physical: bool,
    save_metadata: bool,
) -> tuple[Path, Path | None]:
    case_output_dir = ensure_dir(case_output_dir)
    payload = build_preview_payload(batch, generated)
    array_payload = _build_array_payload(
        batch,
        generated,
        save_physical=save_physical,
    )

    npz_path = case_output_dir / f"{name}.npz"
    np.savez_compressed(npz_path, **array_payload)

    json_path: Path | None = None
    if save_metadata:
        metadata = _build_common_metadata(
            payload,
            checkpoint_path=checkpoint_path,
            save_physical=save_physical,
        )
        metadata["aggregate_name"] = str(name)
        json_path = case_output_dir / f"{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return npz_path, json_path


# -----------------------------------------------------------------------------
# Run-level metadata
# -----------------------------------------------------------------------------



def save_run_metadata(
    output_path: str | Path,
    *,
    run_name: str,
    generation_run_config: str | Path,
    dataset_config: str | Path,
    model_config: str | Path,
    generation_config: str | Path,
    training_config: str | Path | None,
    checkpoint_mode: str,
    checkpoint_path: str | Path,
    use_ema_weights: bool,
    split: str,
    batch_size: int,
    ensemble_size: int,
    max_cases: int | None,
    device: str,
    save_physical: bool,
    save_ensemble_mean: bool,
    save_pmm: bool,
    save_plots: bool,
    output_format: str,
    storage_mode: str,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    metadata = {
        "run_name": run_name,
        "generation_run_config": str(generation_run_config),
        "dataset_config": str(dataset_config),
        "model_config": str(model_config),
        "generation_config": str(generation_config),
        "training_config": None if training_config is None else str(training_config),
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_path": str(checkpoint_path),
        "use_ema_weights": bool(use_ema_weights),
        "split": split,
        "batch_size": int(batch_size),
        "ensemble_size": int(ensemble_size),
        "max_cases": None if max_cases is None else int(max_cases),
        "device": str(device),
        "save_physical": bool(save_physical),
        "save_ensemble_mean": bool(save_ensemble_mean),
        "save_pmm": bool(save_pmm),
        "save_plots": bool(save_plots),
        "output_format": str(output_format),
        "storage_mode": str(storage_mode),
        "storage_note": (
            "Current generation outputs use a simple file-per-product layout. "
            "For large ensembles, consider a future compact backend such as per-case "
            "ensemble bundles, Zarr, or NetCDF/HDF5."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_path