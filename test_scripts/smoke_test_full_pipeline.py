"""
Smoke test for the STRIDE full pipeline.

This script performs a lightweight end-to-end structural check of:

    training -> generation -> evaluation

using the experiment-driven pipeline runner.

The goal is to validate orchestration, config resolution, stage execution,
and expected output artifacts. It is not intended to validate scientific
correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.configs.experiment_config import ExperimentConfig
from stride_core.pipeline.experiment_runner import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "pipeline_norcp.yaml"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")


def _abs_from_root(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight full-pipeline smoke test from an experiment config."
        )
    )
    parser.add_argument(
        "experiment_config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help=(
            "Path to the experiment YAML to use. Defaults to "
            "configs/experiments/pipeline_norcp.yaml"
        ),
    )
    return parser.parse_args()


def resolve_experiment_config(path_arg: str) -> Path:
    requested_path = Path(path_arg)
    if not requested_path.is_absolute():
        requested_path = (ROOT / requested_path).resolve()

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
    require_file(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected experiment config root to be a dict, got {type(payload)}"
        )

    experiment_cfg = payload.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(
            f"Expected 'experiment' to be a dict, got {type(experiment_cfg)}"
        )

    bases_cfg = payload.get("bases")
    if not isinstance(bases_cfg, dict):
        raise ValueError(f"Expected 'bases' to be a dict, got {type(bases_cfg)}")

    experiment_name = experiment_cfg.get("name", "pipeline_smoke_test")
    smoke_name = f"{experiment_name}_smoke"
    experiment_cfg["name"] = smoke_name
    experiment_cfg["output_root"] = str((ROOT / "runs" / smoke_name).resolve())

    for key in ("model", "training", "generation", "sampler", "evaluation", "data"):
        if key in bases_cfg and bases_cfg.get(key) is not None:
            bases_cfg[key] = _abs_from_root(bases_cfg.get(key))

    stages_cfg = payload.setdefault("stages", {})
    if not isinstance(stages_cfg, dict):
        raise ValueError(f"Expected 'stages' to be a dict, got {type(stages_cfg)}")
    stages_cfg["training"] = True
    stages_cfg["generation"] = True
    stages_cfg["evaluation"] = True

    training_cfg = payload.setdefault("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError(f"Expected 'training' to be a dict, got {type(training_cfg)}")
    training_cfg["run_name"] = "train_smoke"
    training_overrides = training_cfg.setdefault("overrides", {})
    if not isinstance(training_overrides, dict):
        raise ValueError(
            f"Expected 'training.overrides' to be a dict, got {type(training_overrides)}"
        )
    loop_overrides = training_overrides.setdefault("loop", {})
    if not isinstance(loop_overrides, dict):
        raise ValueError(
            f"Expected 'training.overrides.loop' to be a dict, got {type(loop_overrides)}"
        )
    loop_overrides["max_epochs"] = 1
    loop_overrides["max_train_batches"] = 2
    loop_overrides["max_val_batches"] = 1
    loop_overrides["validate_every_n_epochs"] = 1

    logging_overrides = training_overrides.setdefault("logging", {})
    if not isinstance(logging_overrides, dict):
        raise ValueError(
            f"Expected 'training.overrides.logging' to be a dict, got {type(logging_overrides)}"
        )
    logging_overrides["print_every_n_steps"] = 1
    logging_overrides["print_step_progress"] = True
    logging_overrides["log_batch_shapes_once"] = True

    generation_preview_overrides = training_overrides.setdefault("generation_preview", {})
    if not isinstance(generation_preview_overrides, dict):
        raise ValueError(
            "Expected 'training.overrides.generation_preview' to be a dict, "
            f"got {type(generation_preview_overrides)}"
        )
    generation_preview_overrides["enabled"] = False

    monitoring_overrides = training_overrides.setdefault("monitoring", {})
    if not isinstance(monitoring_overrides, dict):
        raise ValueError(
            f"Expected 'training.overrides.monitoring' to be a dict, got {type(monitoring_overrides)}"
        )
    monitoring_overrides["enabled"] = False

    generation_cfg = payload.setdefault("generation", {})
    if not isinstance(generation_cfg, dict):
        raise ValueError(f"Expected 'generation' to be a dict, got {type(generation_cfg)}")
    generation_cfg["run_name"] = "generate_smoke"
    generation_overrides = generation_cfg.setdefault("overrides", {})
    if not isinstance(generation_overrides, dict):
        raise ValueError(
            f"Expected 'generation.overrides' to be a dict, got {type(generation_overrides)}"
        )
    generation_run_overrides = generation_overrides.setdefault("generation_run", {})
    if not isinstance(generation_run_overrides, dict):
        raise ValueError(
            "Expected 'generation.overrides.generation_run' to be a dict, "
            f"got {type(generation_run_overrides)}"
        )
    generation_data_overrides = generation_run_overrides.setdefault("data", {})
    if not isinstance(generation_data_overrides, dict):
        raise ValueError(
            "Expected 'generation.overrides.generation_run.data' to be a dict, "
            f"got {type(generation_data_overrides)}"
        )
    generation_data_overrides["split"] = "test"
    generation_limits_overrides = generation_run_overrides.setdefault("limits", {})
    if not isinstance(generation_limits_overrides, dict):
        raise ValueError(
            "Expected 'generation.overrides.generation_run.limits' to be a dict, "
            f"got {type(generation_limits_overrides)}"
        )
    generation_limits_overrides["max_cases"] = 2

    evaluation_cfg = payload.setdefault("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        raise ValueError(f"Expected 'evaluation' to be a dict, got {type(evaluation_cfg)}")
    evaluation_cfg["run_name"] = "evaluate_smoke"
    evaluation_overrides = evaluation_cfg.setdefault("overrides", {})
    if not isinstance(evaluation_overrides, dict):
        raise ValueError(
            f"Expected 'evaluation.overrides' to be a dict, got {type(evaluation_overrides)}"
        )
    evaluation_run_overrides = evaluation_overrides.setdefault("evaluation_run", {})
    if not isinstance(evaluation_run_overrides, dict):
        raise ValueError(
            "Expected 'evaluation.overrides.evaluation_run' to be a dict, "
            f"got {type(evaluation_run_overrides)}"
        )
    evaluation_data_overrides = evaluation_run_overrides.setdefault("data", {})
    if not isinstance(evaluation_data_overrides, dict):
        raise ValueError(
            "Expected 'evaluation.overrides.evaluation_run.data' to be a dict, "
            f"got {type(evaluation_data_overrides)}"
        )
    evaluation_data_overrides["split"] = "test"
    evaluation_data_overrides["use_case_limit"] = 2

    temp_dir = Path(tempfile.mkdtemp(prefix="stride_pipeline_smoke_"))
    temp_config_path = temp_dir / config_path.name
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return temp_config_path


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def read_last_lines(path: Path, num_lines: int = 40) -> list[str]:
    require_file(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-num_lines:]]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config_path = resolve_experiment_config(args.experiment_config)

    print_header("STRIDE full pipeline smoke test")
    print(f"Experiment config: {config_path}")

    print_header("Preparing smoke config")
    smoke_config_path = build_smoke_experiment_config(config_path)
    print(f"Smoke config: {smoke_config_path}")

    print_header("Initializing pipeline runner")
    cfg = ExperimentConfig.from_yaml(smoke_config_path)
    runner = ExperimentRunner(cfg)
    print("Pipeline runner initialized successfully.")

    print_header("Running full pipeline")
    runner.run()
    print("Pipeline run completed.")

    print_header("Checking expected outputs")
    logs_dir = ROOT / "runs" / "pipeline_logs" / cfg.meta.name
    summary_path = logs_dir / "experiment_summary.json"

    require_file(summary_path)
    print(f"Logs dir: {logs_dir}")
    print(f"Experiment summary exists: {summary_path}")

    loaded_summary = load_json(summary_path)
    print(f"Summary keys: {sorted(loaded_summary.keys())}")

    if "stages" not in loaded_summary:
        raise AssertionError("Expected experiment summary to contain 'stages'")

    stages = loaded_summary["stages"]
    if not isinstance(stages, dict):
        raise AssertionError("Expected 'stages' in experiment summary to be a dict")

    expected_stage_order = ("training", "generation", "evaluation")
    seen_failure = False

    for stage_name in expected_stage_order:
        if stage_name not in stages:
            if seen_failure:
                print(
                    f"{stage_name.capitalize():<10} | not reached (pipeline stopped earlier)"
                )
                continue
            raise AssertionError(
                f"Missing stage summary for '{stage_name}' before any failure was recorded"
            )

        stage_info = stages[stage_name]
        if not isinstance(stage_info, dict):
            raise AssertionError(f"Stage summary for '{stage_name}' must be a dict")

        print(
            f"{stage_name.capitalize():<10} | "
            f"success={stage_info.get('success')} "
            f"skipped={stage_info.get('skipped')} "
            f"return_code={stage_info.get('return_code')}"
        )

        if stage_info.get("enabled", False) and not stage_info.get("skipped", False):
            stdout_path = stage_info.get("stdout_path")
            stderr_path = stage_info.get("stderr_path")
            if stdout_path is not None:
                require_file(Path(stdout_path))
            if stderr_path is not None:
                require_file(Path(stderr_path))

        if not stage_info.get("success", False) and not stage_info.get("skipped", False):
            seen_failure = True
            stderr_path = stage_info.get("stderr_path")
            if stderr_path is not None:
                stderr_path = Path(stderr_path)
                print_header(f"Last lines from failing stage stderr: {stage_name}")
                for line in read_last_lines(stderr_path, num_lines=40):
                    print(line)
            break

    if not loaded_summary.get("success", False):
        raise AssertionError(
            "Expected full pipeline smoke test to succeed. See the failing stage stderr excerpt above and the full logs in runs/pipeline_logs/."
        )

    output_root = cfg.meta.output_root
    if output_root is None:
        raise AssertionError("Expected cfg.meta.output_root to be set after pipeline run")
    compiled_dir = Path(output_root) / cfg.meta.name / "compiled_configs"
    require_file(compiled_dir / "compiled_manifest.json")
    require_file(compiled_dir / "training_run_resolved.yaml")
    require_file(compiled_dir / "generation_run_resolved.yaml")
    require_file(compiled_dir / "evaluation_run_resolved.yaml")

    print_header("Full pipeline smoke test completed successfully")


if __name__ == "__main__":
    main()
