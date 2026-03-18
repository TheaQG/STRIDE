"""
Smoke test for the STRIDE evaluation pipeline.

This script performs a minimal end-to-end structural check:
1. Load an experiment config.
2. Rewrite relevant experiment overrides into a temporary smoke config.
3. Compile the experiment into resolved stage configs.
4. Initialize EvaluationRunConfig and Evaluator from the compiled evaluation config.
5. Run evaluation.
6. Verify that the expected output file exists and has the expected top-level structure.

The goal is to validate orchestration and file plumbing, not scientific correctness.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
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
from stride_core.evaluation.evaluation_config import EvaluationRunConfig
from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.evaluation.evaluator import Evaluator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_CONFIG = (
    ROOT / "configs" / "experiments" / "train_generate_evaluate_test.yaml"
)


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the STRIDE evaluation smoke test from an experiment config."
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


def build_smoke_config(config_path: Path) -> Path:
    require_file(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected experiment config root to be a dict, got {type(payload)}"
        )

    experiment_cfg = payload.setdefault("experiment", {})
    if not isinstance(experiment_cfg, dict):
        raise ValueError(
            f"Expected 'experiment' to be a dict, got {type(experiment_cfg)}"
        )
    experiment_cfg["name"] = "smoke_test_evaluation"
    experiment_cfg["output_root"] = str((ROOT / "runs" / "smoke_tests").resolve())

    bases_cfg = payload.get("bases")
    if not isinstance(bases_cfg, dict):
        raise ValueError(f"Expected 'bases' to be a dict, got {type(bases_cfg)}")

    for key in ("model", "training", "generation", "sampler", "evaluation", "data"):
        if key in bases_cfg and bases_cfg.get(key) is not None:
            path = Path(str(bases_cfg.get(key)))
            if not path.is_absolute():
                bases_cfg[key] = str((ROOT / path).resolve())

    stages_cfg = payload.get("stages")
    if not isinstance(stages_cfg, dict):
        raise ValueError(f"Expected 'stages' to be a dict, got {type(stages_cfg)}")
    stages_cfg["training"] = False
    stages_cfg["generation"] = False
    stages_cfg["evaluation"] = True

    evaluation_cfg = payload.setdefault("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        raise ValueError(
            f"Expected 'evaluation' to be a dict, got {type(evaluation_cfg)}"
        )

    overrides_cfg = evaluation_cfg.setdefault("overrides", {})
    if not isinstance(overrides_cfg, dict):
        raise ValueError(
            f"Expected 'evaluation.overrides' to be a dict, got {type(overrides_cfg)}"
        )

    evaluation_run_cfg = overrides_cfg.setdefault("evaluation_run", {})
    if not isinstance(evaluation_run_cfg, dict):
        raise ValueError(
            "Expected 'evaluation.overrides.evaluation_run' to be a dict, "
            f"got {type(evaluation_run_cfg)}"
        )

    data_cfg = evaluation_run_cfg.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError(
            "Expected 'evaluation.overrides.evaluation_run.data' to be a dict, "
            f"got {type(data_cfg)}"
        )

    paths_cfg = evaluation_run_cfg.setdefault("paths", {})
    if not isinstance(paths_cfg, dict):
        raise ValueError(
            "Expected 'evaluation.overrides.evaluation_run.paths' to be a dict, "
            f"got {type(paths_cfg)}"
        )

    smoke_generation_root = (ROOT / "runs" / "smoke_tests" / "smoke_test_generation").resolve()
    smoke_generation_compiled = smoke_generation_root / "compiled_configs"

    paths_cfg["generation_output_dir"] = str(
        (smoke_generation_root / "generation").resolve()
    )
    paths_cfg["dataset_config"] = str(
        (smoke_generation_compiled / "data_resolved.yaml").resolve()
    )
    paths_cfg["training_config"] = str(
        (smoke_generation_compiled / "training_run_resolved.yaml").resolve()
    )
    paths_cfg["generation_run_config"] = str(
        (smoke_generation_compiled / "generation_run_resolved.yaml").resolve()
    )

    use_case_limit = data_cfg.get("use_case_limit")
    if use_case_limit is None or int(use_case_limit) > 3:
        data_cfg["use_case_limit"] = 3

    temp_dir = Path(tempfile.mkdtemp(prefix="stride_eval_smoke_"))
    temp_config_path = temp_dir / config_path.name
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return temp_config_path



def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def load_jsonable_yaml(path: Path) -> dict[str, Any]:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(payload)}")
    return payload


def _fmt_float(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return "n/a"


def print_metric_summary(metrics: dict[str, Any]) -> None:
    print_header("Metric summary")

    probabilistic = metrics.get("probabilistic", {})
    if isinstance(probabilistic, dict):
        aggregate = probabilistic.get("aggregate", {})
        if isinstance(aggregate, dict):
            crps = aggregate.get("crps", {})
            spread_skill = aggregate.get("spread_skill", {})
            reliability = aggregate.get("reliability_diagram", {})
            if isinstance(crps, dict):
                print(f"Probabilistic | CRPS: {_fmt_float(crps.get('mean_crps'))}")
            if isinstance(spread_skill, dict):
                print(
                    "Probabilistic | Spread-skill corr: "
                    f"{_fmt_float(spread_skill.get('correlation'))}"
                )
            if isinstance(reliability, dict):
                thresholds = reliability.get("thresholds_mm", [])
                print(
                    "Probabilistic | Reliability: done"
                    + (f" (thresholds={thresholds})" if thresholds else "")
                )

    spatial = metrics.get("spatial", {})
    if isinstance(spatial, dict):
        aggregate = spatial.get("aggregate", {})
        if isinstance(aggregate, dict):
            psd = aggregate.get("psd", {})
            psd_slope = aggregate.get("psd_slope", {})
            iss = aggregate.get("iss", {})
            sal = aggregate.get("sal", {})
            if isinstance(psd, dict):
                mean_wavelengths = psd.get("mean_wavelengths_km", [])
                print(
                    "Spatial       | PSD: done"
                    + (
                        f" (n_wavelengths={len(mean_wavelengths)})"
                        if isinstance(mean_wavelengths, list)
                        else ""
                    )
                )
            if isinstance(psd_slope, dict):
                print(
                    "Spatial       | PSD slope Δ: "
                    f"{_fmt_float(psd_slope.get('slope_difference'))}"
                )
            if isinstance(iss, dict):
                thresholds = iss.get("thresholds_mm", [])
                scales = iss.get("scales_km", [])
                print(
                    "Spatial       | ISS: done"
                    + (
                        f" (thresholds={thresholds}, scales={scales})"
                        if thresholds or scales
                        else ""
                    )
                )
            if isinstance(sal, dict):
                print(
                    "Spatial       | SAL: "
                    f"S={_fmt_float(sal.get('mean_structure'))}, "
                    f"A={_fmt_float(sal.get('mean_amplitude'))}, "
                    f"L={_fmt_float(sal.get('mean_location'))}"
                )

    climatology = metrics.get("climatology", {})
    if isinstance(climatology, dict):
        aggregate = climatology.get("aggregate", {})
        if isinstance(aggregate, dict):
            wet = aggregate.get("wet_day_frequency", {})
            annual = aggregate.get("annual_precipitation_sum", {})
            extremes = aggregate.get("extremes", {})
            if isinstance(wet, dict):
                print(
                    "Climatology   | Wet-day bias: "
                    f"{_fmt_float(wet.get('frequency_bias'))}"
                )
            if isinstance(annual, dict):
                print(
                    "Climatology   | Annual sum bias: "
                    f"{_fmt_float(annual.get('mean_sum_bias'))}"
                )
            if isinstance(extremes, dict):
                qlevels = extremes.get("quantile_levels", [])
                print(
                    "Climatology   | Extremes: done"
                    + (f" (quantiles={qlevels})" if qlevels else "")
                )

    temporal = metrics.get("temporal", {})
    if isinstance(temporal, dict):
        aggregate = temporal.get("aggregate", {})
        if isinstance(aggregate, dict):
            lag = aggregate.get("lag_autocorrelation", {})
            wet = aggregate.get("wet_spell_length", {})
            dry = aggregate.get("dry_spell_length", {})
            if isinstance(lag, dict):
                lags = lag.get("lags", [])
                target_autocorr = lag.get("target_autocorr", [])
                n_lags = len(lags) if isinstance(lags, list) else 0
                first_target = target_autocorr[0] if isinstance(target_autocorr, list) and len(target_autocorr) > 0 else None
                print(
                    "Temporal      | Lag autocorr: done"
                    + (f" (n_lags={n_lags}, first_target={_fmt_float(first_target)})" if n_lags > 0 else "")
                )
            if isinstance(wet, dict):
                print(
                    "Temporal      | Wet spells: done"
                    + (
                        f" (max_length={wet.get('max_length')}, n_target_spells={len(wet.get('target_lengths', []))})"
                        if "max_length" in wet
                        else ""
                    )
                )
            if isinstance(dry, dict):
                print(
                    "Temporal      | Dry spells: done"
                    + (
                        f" (max_length={dry.get('max_length')}, n_target_spells={len(dry.get('target_lengths', []))})"
                        if "max_length" in dry
                        else ""
                    )
                )

    sigma_star = metrics.get("sigma_star", {})
    if isinstance(sigma_star, dict):
        print(
            "Sigma-star    | "
            + ("Configured" if len(sigma_star) > 0 else "Not yet populated")
        )


def check_probabilistic_shapes(cfg: EvaluationRunConfig) -> None:
    loader = EvaluationDataLoader(cfg)
    first_case = next(loader.iter_cases())
    product = loader.get_probabilistic_product(first_case)
    forecast = loader.get_forecast_array(product)
    target = loader.get_target_array(product)

    print_header("Checking probabilistic input shapes")
    print(f"Case id: {first_case.case_id}")
    print(f"Probabilistic product: {product.name}")
    print(f"Forecast shape: {tuple(forecast.shape)}")
    print(f"Target shape: {tuple(target.shape)}")

    if forecast.ndim not in (3, 4):
        raise AssertionError(
            "Probabilistic forecast must have shape [M,H,W] or [B,M,H,W], "
            f"got {tuple(forecast.shape)}"
        )
    if target.ndim not in (2, 3):
        raise AssertionError(
            "Probabilistic target must have shape [H,W] or [B,H,W], "
            f"got {tuple(target.shape)}"
        )

    if forecast.ndim == 3 and target.ndim != 2:
        raise AssertionError(
            "Case-level probabilistic forecast [M,H,W] should pair with target [H,W], "
            f"got target shape {tuple(target.shape)}"
        )
    if forecast.ndim == 4 and target.ndim != 3:
        raise AssertionError(
            "Batch probabilistic forecast [B,M,H,W] should pair with target [B,H,W], "
            f"got target shape {tuple(target.shape)}"
        )


def check_spatial_shapes(cfg: EvaluationRunConfig) -> None:
    loader = EvaluationDataLoader(cfg)
    first_case = next(loader.iter_cases())
    product = loader.get_spatial_product(first_case)
    forecast = loader.get_forecast_array(product)
    target = loader.get_target_array(product)

    print_header("Checking spatial input shapes")
    print(f"Case id: {first_case.case_id}")
    print(f"Spatial product: {product.name}")
    print(f"Forecast shape: {tuple(forecast.shape)}")
    print(f"Target shape: {tuple(target.shape)}")

    if forecast.ndim not in (2, 3):
        raise AssertionError(
            "Spatial forecast must have shape [H,W] or [B,H,W], "
            f"got {tuple(forecast.shape)}"
        )
    if target.ndim not in (2, 3):
        raise AssertionError(
            "Spatial target must have shape [H,W] or [B,H,W], "
            f"got {tuple(target.shape)}"
        )

    if forecast.ndim != target.ndim:
        raise AssertionError(
            "Spatial forecast and target must have the same rank, got "
            f"forecast {tuple(forecast.shape)} vs target {tuple(target.shape)}"
        )


def check_climatology_shapes(cfg: EvaluationRunConfig) -> None:
    loader = EvaluationDataLoader(cfg)
    first_case = next(loader.iter_cases())
    product = loader.get_climatology_product(first_case)
    forecast = loader.get_forecast_array(product)
    target = loader.get_target_array(product)

    print_header("Checking climatology input shapes")
    print(f"Case id: {first_case.case_id}")
    print(f"Climatology product: {product.name}")
    print(f"Forecast shape: {tuple(forecast.shape)}")
    print(f"Target shape: {tuple(target.shape)}")

    if forecast.ndim not in (2, 3, 4):
        raise AssertionError(
            "Climatology forecast must have shape [H,W], [M,H,W], or [B,M,H,W], "
            f"got {tuple(forecast.shape)}"
        )
    if target.ndim not in (2, 3):
        raise AssertionError(
            "Climatology target must have shape [H,W] or [B,H,W], "
            f"got {tuple(target.shape)}"
        )

    if forecast.ndim == 2 and target.ndim != 2:
        raise AssertionError(
            "Deterministic climatology forecast [H,W] should pair with target [H,W], "
            f"got target shape {tuple(target.shape)}"
        )
    if forecast.ndim == 3 and target.ndim != 2:
        raise AssertionError(
            "Case-level ensemble climatology forecast [M,H,W] should pair with target [H,W], "
            f"got target shape {tuple(target.shape)}"
        )
    if forecast.ndim == 4 and target.ndim != 3:
        raise AssertionError(
            "Batch climatology forecast [B,M,H,W] should pair with target [B,H,W], "
            f"got target shape {tuple(target.shape)}"
        )


def evaluator_stack_temporal_forecasts(forecasts: list[Any]):
    normalized = []
    for forecast in forecasts:
        arr = np.asarray(forecast)
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 4:
            raise ValueError(
                "Each temporal forecast must have shape [H,W], [M,H,W], or [T,M,H,W]-like sequence input, "
                f"got {tuple(arr.shape)}"
            )
        normalized.append(arr)
    return np.concatenate(normalized, axis=0)


def evaluator_stack_temporal_targets(targets: list[Any]):
    normalized = []
    for target in targets:
        arr = np.asarray(target)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        elif arr.ndim != 3:
            raise ValueError(
                "Each temporal target must have shape [H,W], [T,H,W], or [T,1,H,W], "
                f"got {tuple(arr.shape)}"
            )
        normalized.append(arr)
    return np.concatenate(normalized, axis=0)


def check_temporal_shapes(cfg: EvaluationRunConfig) -> None:
    loader = EvaluationDataLoader(cfg)
    cases = []
    for idx, case in enumerate(loader.iter_cases()):
        cases.append(case)
        if idx >= 2:
            break
    if len(cases) == 0:
        raise AssertionError("No cases available for temporal shape check")

    product = loader.get_temporal_product(cases[0])
    forecast_list = []
    target_list = []
    for case in cases:
        case_product = loader.get_temporal_product(case)
        forecast_list.append(loader.get_forecast_array(case_product))
        target_list.append(loader.get_target_array(case_product))

    stacked_forecast = evaluator_stack_temporal_forecasts(forecast_list)
    stacked_target = evaluator_stack_temporal_targets(target_list)

    print_header("Checking temporal input shapes")
    print(f"Temporal product: {product.name}")
    print(f"Stacked forecast shape: {tuple(stacked_forecast.shape)}")
    print(f"Stacked target shape: {tuple(stacked_target.shape)}")

    if stacked_forecast.ndim != 4:
        raise AssertionError(
            "Temporal forecast must have shape [T, M, H, W], "
            f"got {tuple(stacked_forecast.shape)}"
        )
    if stacked_target.ndim != 3:
        raise AssertionError(
            "Temporal target must have shape [T, H, W], "
            f"got {tuple(stacked_target.shape)}"
        )
    if stacked_forecast.shape[0] != stacked_target.shape[0]:
        raise AssertionError(
            "Temporal forecast and target must have matching time length, got "
            f"forecast {tuple(stacked_forecast.shape)} vs target {tuple(stacked_target.shape)}"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    experiment_config_path = resolve_experiment_config(args.experiment_config)

    print_header("STRIDE evaluation smoke test")
    print(f"Experiment config: {experiment_config_path}")

    print_header("Preparing smoke config")
    smoke_config_path = build_smoke_config(experiment_config_path)
    print(f"Smoke config: {smoke_config_path}")

    exp_cfg = ExperimentConfig.from_yaml(smoke_config_path)
    compiler = ConfigCompiler(exp_cfg)
    compiled = compiler.compile()

    experiment_root = compiled.experiment_root
    if experiment_root.exists():
        shutil.rmtree(experiment_root)
        compiled = compiler.compile()
        experiment_root = compiled.experiment_root

    evaluation_config_path = compiled.evaluation_run_config_path
    if evaluation_config_path is None:
        raise RuntimeError(
            f"Compiler did not produce an evaluation config for experiment: {experiment_config_path}"
        )

    print_header("Compiled smoke configs")
    print(f"Evaluation config: {evaluation_config_path}")
    print(f"Experiment root:   {experiment_root}")

    smoke_generation_root = (ROOT / "runs" / "smoke_tests" / "smoke_test_generation").resolve()
    smoke_generation_compiled = smoke_generation_root / "compiled_configs"

    compiled_eval_payload = load_jsonable_yaml(evaluation_config_path)
    run_cfg = compiled_eval_payload.get("evaluation_run")
    if not isinstance(run_cfg, dict):
        raise ValueError(
            f"Expected compiled evaluation config to contain 'evaluation_run', got {type(run_cfg)}"
        )

    paths_cfg = run_cfg.setdefault("paths", {})
    if not isinstance(paths_cfg, dict):
        raise ValueError(
            f"Expected compiled evaluation config paths to be a dict, got {type(paths_cfg)}"
        )

    paths_cfg["generation_output_dir"] = str((smoke_generation_root / "generation").resolve())
    paths_cfg["dataset_config"] = str((smoke_generation_compiled / "data_resolved.yaml").resolve())
    paths_cfg["training_config"] = str((smoke_generation_compiled / "training_run_resolved.yaml").resolve())
    paths_cfg["generation_run_config"] = str((smoke_generation_compiled / "generation_run_resolved.yaml").resolve())

    with open(evaluation_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(compiled_eval_payload, f, sort_keys=False)

    print_header("Initializing evaluator")
    cfg = EvaluationRunConfig.from_yaml(evaluation_config_path)
    evaluator = Evaluator(cfg)
    print("Evaluator initialized successfully.")

    check_probabilistic_shapes(cfg)
    check_spatial_shapes(cfg)
    check_climatology_shapes(cfg)
    check_temporal_shapes(cfg)

    print_header("Running evaluation")
    evaluator.run()
    print("Evaluation run completed successfully.")

    print_header("Checking expected outputs")
    output_dir = cfg.outputs.output_dir
    metrics_path = output_dir / "evaluation_metrics.json"

    require_file(metrics_path)
    print(f"Evaluation output dir: {output_dir}")
    print(f"Metrics file exists: {metrics_path}")

    metrics = load_json(metrics_path)
    print(f"Top-level metric families: {sorted(metrics.keys())}")
    print_metric_summary(metrics)

    required_top_level = {
        "probabilistic",
        "spatial",
        "climatology",
        "temporal",
        "sigma_star",
    }
    missing = required_top_level - set(metrics.keys())
    if missing:
        raise AssertionError(
            f"Missing expected top-level metric families: {sorted(missing)}"
        )

    probabilistic = metrics.get("probabilistic")
    if not isinstance(probabilistic, dict):
        raise AssertionError("Expected 'probabilistic' metrics to be a dict")
    if "case_level" not in probabilistic:
        raise AssertionError("Expected probabilistic metrics to include 'case_level'")
    if "num_cases" not in probabilistic:
        raise AssertionError("Expected probabilistic metrics to include 'num_cases'")

    spatial = metrics.get("spatial")
    if not isinstance(spatial, dict):
        raise AssertionError("Expected 'spatial' metrics to be a dict")
    if "case_level" not in spatial:
        raise AssertionError("Expected spatial metrics to include 'case_level'")
    if "num_cases" not in spatial:
        raise AssertionError("Expected spatial metrics to include 'num_cases'")

    climatology = metrics.get("climatology")
    if not isinstance(climatology, dict):
        raise AssertionError("Expected 'climatology' metrics to be a dict")
    if "case_level" not in climatology:
        raise AssertionError("Expected climatology metrics to include 'case_level'")
    if "num_cases" not in climatology:
        raise AssertionError("Expected climatology metrics to include 'num_cases'")

    temporal = metrics.get("temporal")
    if not isinstance(temporal, dict):
        raise AssertionError("Expected 'temporal' metrics to be a dict")
    if "case_level" not in temporal:
        raise AssertionError("Expected temporal metrics to include 'case_level'")
    if "num_cases" not in temporal:
        raise AssertionError("Expected temporal metrics to include 'num_cases'")

    print_header("Evaluation smoke test completed successfully")


if __name__ == "__main__":
    main()