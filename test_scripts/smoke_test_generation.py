"""
Smoke test for the STRIDE post-training generation pipeline.

This script now follows the new experiment-driven configuration structure.
It starts from an experiment config, compiles a tiny smoke-version of that
experiment, runs training + generation (but skips evaluation), and then checks
that expected generation artifacts were written.

The goal is not scientific validation. It is only a structural test that:
- experiment config compilation works
- training produces a checkpoint
- generation loads that checkpoint
- dataset construction works
- sampling runs
- PMM / ensemble outputs are saved coherently
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import yaml


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.configs.config_compiler import ConfigCompiler
from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.training.trainer import Trainer
from stride_core.generation.generator import Generator, GenerationRunConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "train_generate_evaluate_test.yaml"
)
EXPERIMENT_CONFIG_DIR = REPO_ROOT / "configs" / "experiments"
SMOKE_RUN_PARENT = REPO_ROOT / "runs" / "smoke_tests"
SMOKE_EXPERIMENT_NAME = "smoke_test_generation"


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



def _abs_from_root(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the STRIDE generation smoke test from an experiment config."
        )
    )
    parser.add_argument(
        "experiment_config",
        nargs="?",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        help=(
            "Path to the experiment YAML to use. Defaults to "
            "configs/experiments/train_generate_evaluate_test.yaml"
        ),
    )
    return parser.parse_args()


def resolve_experiment_config(path_arg: str) -> Path:
    requested_path = Path(path_arg)
    if not requested_path.is_absolute():
        requested_path = (REPO_ROOT / requested_path).resolve()

    if not requested_path.exists():
        raise FileNotFoundError(
            f"Experiment config does not exist: {requested_path}"
        )
    if not requested_path.is_file():
        raise ValueError(
            f"Expected a YAML experiment config file, got: {requested_path}"
        )
    if requested_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Expected a YAML experiment config file, got: {requested_path}"
        )
    return requested_path


# -----------------------------------------------------------------------------
# Smoke-config preparation
# -----------------------------------------------------------------------------


def build_smoke_experiment_config(config_path: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected experiment config to load into a dict, got {type(payload)}"
        )

    experiment_cfg = payload.get("experiment")
    stages_cfg = payload.get("stages")
    bases_cfg = payload.get("bases")

    if not isinstance(experiment_cfg, dict):
        raise ValueError("Expected top-level 'experiment' section to be a dict")
    if not isinstance(stages_cfg, dict):
        raise ValueError("Expected top-level 'stages' section to be a dict")
    if not isinstance(bases_cfg, dict):
        raise ValueError("Expected top-level 'bases' section to be a dict")

    experiment_cfg["name"] = SMOKE_EXPERIMENT_NAME
    experiment_cfg["output_root"] = str(SMOKE_RUN_PARENT.resolve())

    # Training + generation smoke test, but skip evaluation.
    stages_cfg["training"] = True
    stages_cfg["generation"] = True
    stages_cfg["evaluation"] = False

    for key in ("model", "training", "generation", "sampler", "evaluation", "data"):
        if key in bases_cfg and bases_cfg.get(key) is not None:
            bases_cfg[key] = _abs_from_root(bases_cfg.get(key))

    training_cfg = payload.setdefault("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Expected top-level 'training' section to be a dict")
    training_overrides = training_cfg.setdefault("overrides", {})
    if not isinstance(training_overrides, dict):
        raise ValueError("Expected 'training.overrides' to be a dict")

    generation_cfg = payload.setdefault("generation", {})
    if not isinstance(generation_cfg, dict):
        raise ValueError("Expected top-level 'generation' section to be a dict")
    generation_overrides = generation_cfg.setdefault("overrides", {})
    if not isinstance(generation_overrides, dict):
        raise ValueError("Expected 'generation.overrides' to be a dict")

    # Keep the smoke test tiny and deterministic.
    training_overrides.update(
        {
            "training": {
                "loop": {
                    "max_epochs": 1,
                    "max_train_batches": 2,
                    "max_val_batches": 1,
                    "validate_every_n_epochs": 1,
                },
                "checkpointing": {
                    "enabled": True,
                    "save_every_n_epochs": 1,
                    "save_best": True,
                    "resume_from": None,
                },
                "data": {
                    "num_workers": 0,
                },
                "validation": {
                    "enabled": True,
                },
            }
        }
    )

    # Ensure generation stays tiny.
    generation_overrides.setdefault("generation_run", {})
    if not isinstance(generation_overrides["generation_run"], dict):
        raise ValueError("Expected 'generation.overrides.generation_run' to be a dict")
    generation_run_overrides = generation_overrides["generation_run"]

    generation_run_overrides.setdefault("data", {})
    if not isinstance(generation_run_overrides["data"], dict):
        raise ValueError("Expected 'generation.overrides.generation_run.data' to be a dict")
    generation_run_overrides["data"]["split"] = "test"
    generation_run_overrides["data"]["batch_size"] = 1
    generation_run_overrides["data"]["num_workers"] = 0
    generation_run_overrides["data"]["pin_memory"] = False
    generation_run_overrides["data"]["shuffle"] = False

    generation_run_overrides.setdefault("sampling", {})
    if not isinstance(generation_run_overrides["sampling"], dict):
        raise ValueError(
            "Expected 'generation.overrides.generation_run.sampling' to be a dict"
        )
    generation_run_overrides["sampling"]["ensemble_size"] = 3
    generation_run_overrides["sampling"]["use_fixed_seed"] = True
    generation_run_overrides["sampling"]["base_seed"] = 42

    generation_run_overrides.setdefault("limits", {})
    if not isinstance(generation_run_overrides["limits"], dict):
        raise ValueError(
            "Expected 'generation.overrides.generation_run.limits' to be a dict"
        )
    generation_run_overrides["limits"]["max_cases"] = 2

    generation_run_overrides.setdefault("outputs", {})
    if not isinstance(generation_run_overrides["outputs"], dict):
        raise ValueError(
            "Expected 'generation.overrides.generation_run.outputs' to be a dict"
        )
    generation_run_overrides["outputs"]["save_members"] = True
    generation_run_overrides["outputs"]["save_pmm"] = True
    generation_run_overrides["outputs"]["save_ensemble_mean"] = True
    generation_run_overrides["outputs"]["save_plots"] = False
    generation_run_overrides["outputs"]["save_physical"] = True
    generation_run_overrides["outputs"]["output_format"] = "npz"
    generation_run_overrides["outputs"]["storage_mode"] = "per_member_bundle"

    temp_dir = Path(tempfile.mkdtemp(prefix="stride_generation_smoke_"))
    temp_config_path = temp_dir / config_path.name
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return temp_config_path


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    experiment_config_path = resolve_experiment_config(args.experiment_config)

    print_header("STRIDE generation smoke test")
    print(f"Experiment config: {experiment_config_path}")

    smoke_config_path = build_smoke_experiment_config(experiment_config_path)
    print_header("Smoke experiment config written")
    print(f"Smoke config: {smoke_config_path}")

    exp_cfg = ExperimentConfig.from_yaml(smoke_config_path)
    compiler = ConfigCompiler(exp_cfg)
    compiled = compiler.compile()

    experiment_root = compiled.experiment_root
    training_output_dir = experiment_root / "training"
    generation_output_dir = experiment_root / "generation"

    if experiment_root.exists():
        shutil.rmtree(experiment_root)
        compiled = compiler.compile()
        experiment_root = compiled.experiment_root
        training_output_dir = experiment_root / "training"
        generation_output_dir = experiment_root / "generation"

    print_header("Compiled smoke configs")
    print(f"Training config:   {compiled.training_run_config_path}")
    print(f"Generation config: {compiled.generation_run_config_path}")
    print(f"Experiment root:   {experiment_root}")

    print_header("Initializing trainer")
    trainer = Trainer(compiled.training_run_config_path)
    print("Trainer initialized successfully.")

    print_header("Running smoke training")
    trainer.fit()
    print("Smoke training completed successfully.")

    checkpoint_dir = training_output_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "checkpoint_latest.pt"
    best_checkpoint = checkpoint_dir / "checkpoint_best.pt"
    epoch_checkpoint = checkpoint_dir / "checkpoint_epoch_0001.pt"

    require_file(latest_checkpoint)
    require_file(epoch_checkpoint)
    require_file(best_checkpoint)

    print_header("Loading generation config")
    gen_cfg = GenerationRunConfig.from_yaml(compiled.generation_run_config_path)
    print(f"Generation config loaded from: {compiled.generation_run_config_path}")

    print_header("Initializing generator")
    generator = Generator(gen_cfg)
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

    if len(case_dirs) == 0:
        raise AssertionError("No case directories were generated.")

    if gen_cfg.limits.max_cases is not None:
        if len(case_dirs) > gen_cfg.limits.max_cases:
            raise AssertionError(
                f"Generated more cases ({len(case_dirs)}) than max_cases ({gen_cfg.limits.max_cases})"
            )

    first_case_dir = case_dirs[0]
    print(f"Inspecting first case directory: {first_case_dir}")

    storage_mode = gen_cfg.outputs.storage_mode
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

    if gen_cfg.outputs.save_ensemble_mean:
        ensemble_arrays = inspect_npz(ensemble_mean_path)
        ensemble_meta = inspect_json(ensemble_mean_json_path)
    else:
        ensemble_arrays = None
        ensemble_meta = None

    if gen_cfg.outputs.save_pmm:
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

        if generated_members.shape[0] != gen_cfg.sampling.ensemble_size:
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
        if member_meta.get("ensemble_size") != gen_cfg.sampling.ensemble_size:
            raise AssertionError(
                "Expected ensemble bundle metadata ensemble_size to match config"
            )
        if member_meta.get("save_physical") is not True:
            raise AssertionError("Expected bundle metadata to record save_physical=True")

    else:
        raise ValueError(f"Unsupported storage_mode in structural assertions: {storage_mode!r}")

    if gen_cfg.outputs.save_ensemble_mean:
        assert ensemble_arrays is not None
        assert ensemble_meta is not None
        if "generated_physical" not in ensemble_arrays:
            raise AssertionError("ensemble_mean.npz missing generated_physical")
        if ensemble_meta.get("aggregate_name") != "ensemble_mean":
            raise AssertionError(
                "ensemble_mean metadata missing aggregate_name='ensemble_mean'"
            )

    if gen_cfg.outputs.save_pmm:
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
