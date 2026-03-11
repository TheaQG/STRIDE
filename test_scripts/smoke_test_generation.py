"""
Smoke test for the STRIDE post-training generation pipeline.

This script verifies the first end-to-end generation milestone:
- load a generation-run config
- initialize the Generator
- run generation on a very small number of cases
- confirm the expected files were written
- inspect saved ensemble-member outputs, one saved ensemble mean, and one PMM product

The goal is not scientific validation yet. It is only a structural test that
checkpoint loading, dataset construction, sampling, PMM computation, and output
saving all work coherently together.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import numpy as np


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.generation.generator import Generator, GenerationRunConfig


DEFAULT_CONFIG = "configs/generation_runs/generate_test_best.yaml"


# -----------------------------------------------------------------------------
# Small printing helpers
# -----------------------------------------------------------------------------


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")



def summarize_array(name: str, array: np.ndarray) -> None:
    print(f"{name}:")
    print(f"  shape: {array.shape}")
    print(f"  dtype: {array.dtype}")
    print(f"  min:   {float(np.nanmin(array)):.6f}")
    print(f"  max:   {float(np.nanmax(array)):.6f}")
    print(f"  mean:  {float(np.nanmean(array)):.6f}")



def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file was not created: {path}")


# -----------------------------------------------------------------------------
# Inspection helpers
# -----------------------------------------------------------------------------


def inspect_npz(npz_path: Path) -> dict[str, np.ndarray]:
    require_file(npz_path)
    loaded = np.load(npz_path)
    arrays: dict[str, np.ndarray] = {key: loaded[key] for key in loaded.files}

    print(f"Loaded NPZ: {npz_path}")
    print(f"Keys: {sorted(arrays.keys())}")

    for key, value in arrays.items():
        summarize_array(key, value)

    return arrays



def inspect_json(json_path: Path) -> dict[str, Any]:
    require_file(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print(f"Loaded JSON: {json_path}")
    print(f"Keys: {sorted(payload.keys())}")
    return payload


# -----------------------------------------------------------------------------
# Main smoke test
# -----------------------------------------------------------------------------


def main() -> None:
    config_path = Path(DEFAULT_CONFIG).resolve()

    print_header("STRIDE generation smoke test")
    print(f"Generation config: {config_path}")

    cfg = GenerationRunConfig.from_yaml(config_path)

    print_header("Initializing generator")
    generator = Generator(cfg)
    print("Generator initialized successfully.")

    print_header("Running generation")
    generator.run()
    print("Generation run completed successfully.")

    print_header("Checking expected outputs")
    output_dir = generator.output_dir
    samples_dir = generator.samples_dir
    metadata_path = generator.metadata_path

    require_file(metadata_path)
    print(f"Run metadata exists: {metadata_path}")

    case_dirs = sorted([path for path in samples_dir.iterdir() if path.is_dir()])
    print(f"Found {len(case_dirs)} case directories in: {samples_dir}")

    expected_cases = cfg.limits.max_cases if cfg.limits.max_cases is not None else len(case_dirs)
    if len(case_dirs) != expected_cases:
        raise AssertionError(
            f"Expected {expected_cases} case directories, found {len(case_dirs)}"
        )

    first_case_dir = case_dirs[0]
    print(f"Inspecting first case directory: {first_case_dir}")

    storage_mode = cfg.outputs.storage_mode
    if storage_mode == "per_member":
        member_path = first_case_dir / "member_0000.npz"
        member_json_path = first_case_dir / "member_0000.json"
    elif storage_mode == "per_member_bundle":
        member_path = first_case_dir / "ensemble_members.npz"
        member_json_path = first_case_dir / "ensemble_members.json"
    else:
        raise ValueError(f"Unsupported storage_mode in smoke test: {storage_mode!r}")

    ensemble_mean_path = first_case_dir / "ensemble_mean.npz"
    ensemble_mean_json_path = first_case_dir / "ensemble_mean.json"
    pmm_path = first_case_dir / "pmm.npz"
    pmm_json_path = first_case_dir / "pmm.json"

    member_arrays = inspect_npz(member_path)
    member_meta = inspect_json(member_json_path)

    if cfg.outputs.save_ensemble_mean:
        ensemble_arrays = inspect_npz(ensemble_mean_path)
        ensemble_meta = inspect_json(ensemble_mean_json_path)
    else:
        ensemble_arrays = None
        ensemble_meta = None

    if cfg.outputs.save_pmm:
        pmm_arrays = inspect_npz(pmm_path)
        pmm_meta = inspect_json(pmm_json_path)
    else:
        pmm_arrays = None
        pmm_meta = None

    print_header("Running structural assertions")
    if storage_mode == "per_member":
        required_member_keys = {
            "generated",
            "target_physical",
            "generated_physical",
            "cond_dynamic_physical",
            "cond_static_physical",
        }
        missing_member_keys = required_member_keys - set(member_arrays.keys())
        if missing_member_keys:
            raise AssertionError(
                f"Saved member NPZ is missing expected keys: {sorted(missing_member_keys)}"
            )

        generated = member_arrays["generated"]
        generated_physical = member_arrays["generated_physical"]
        target_physical = member_arrays["target_physical"]

        if generated.shape[0] != 1:
            raise AssertionError(f"Expected member batch dimension 1, got {generated.shape}")
        if generated_physical.shape[0] != 1:
            raise AssertionError(
                f"Expected generated_physical batch dimension 1, got {generated_physical.shape}"
            )
        if target_physical.shape[0] != 1:
            raise AssertionError(
                f"Expected target_physical batch dimension 1, got {target_physical.shape}"
            )

        if member_meta.get("save_physical") is not True:
            raise AssertionError("Expected member metadata to record save_physical=True")

    elif storage_mode == "per_member_bundle":
        required_member_keys = {
            "generated_members",
            "target_physical",
            "cond_dynamic_physical",
            "cond_static_physical",
        }
        missing_member_keys = required_member_keys - set(member_arrays.keys())
        if missing_member_keys:
            raise AssertionError(
                "Saved ensemble bundle NPZ is missing expected keys: "
                f"{sorted(missing_member_keys)}"
            )

        generated_members = member_arrays["generated_members"]
        target_physical = member_arrays["target_physical"]

        if generated_members.shape[0] != cfg.sampling.ensemble_size:
            raise AssertionError(
                "Expected generated_members leading dimension to equal ensemble size, got "
                f"{generated_members.shape}"
            )
        if target_physical.shape[0] != 1:
            raise AssertionError(
                f"Expected target_physical batch dimension 1, got {target_physical.shape}"
            )

        if member_meta.get("aggregate_name") != "ensemble_members":
            raise AssertionError(
                "Expected ensemble bundle metadata aggregate_name='ensemble_members'"
            )
        if member_meta.get("ensemble_size") != cfg.sampling.ensemble_size:
            raise AssertionError(
                "Expected ensemble bundle metadata ensemble_size to match config"
            )
        if member_meta.get("save_physical") is not True:
            raise AssertionError("Expected bundle metadata to record save_physical=True")

    else:
        raise ValueError(f"Unsupported storage_mode in structural assertions: {storage_mode!r}")

    if cfg.outputs.save_ensemble_mean:
        assert ensemble_arrays is not None
        assert ensemble_meta is not None
        if "generated_physical" not in ensemble_arrays:
            raise AssertionError("ensemble_mean.npz missing generated_physical")
        if ensemble_meta.get("aggregate_name") != "ensemble_mean":
            raise AssertionError(
                "ensemble_mean metadata missing aggregate_name='ensemble_mean'"
            )

    if cfg.outputs.save_pmm:
        assert pmm_arrays is not None
        assert pmm_meta is not None
        if "generated_physical" not in pmm_arrays:
            raise AssertionError("pmm.npz missing generated_physical")
        if pmm_meta.get("aggregate_name") != "pmm":
            raise AssertionError("pmm metadata missing aggregate_name='pmm'")
        pmm_generated = pmm_arrays["generated_physical"]
        if pmm_generated.shape[0] != 1:
            raise AssertionError(
                f"Expected pmm generated_physical batch dimension 1, got {pmm_generated.shape}"
            )

    print_header("Generation smoke test completed successfully")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
