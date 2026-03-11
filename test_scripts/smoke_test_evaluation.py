"""
Smoke test for the STRIDE evaluation pipeline.

This script performs a minimal end-to-end structural check:
1. Load the run-level evaluation config.
2. Rewrite relevant config paths to absolute paths in a temporary smoke config.
3. Initialize EvaluationRunConfig and Evaluator.
4. Run evaluation.
5. Verify that the expected output file exists and has the expected top-level structure.

The goal is to validate orchestration and file plumbing, not scientific correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any


import numpy as np
import yaml


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from stride_core.evaluation.evaluation_config import EvaluationRunConfig
from stride_core.evaluation.data_loading import EvaluationDataLoader
from stride_core.evaluation.evaluator import Evaluator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "evaluation_runs" / "evaluate_test_best.yaml"


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


def build_smoke_config(config_path: Path) -> Path:
    require_file(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected evaluation run config root to be a dict, got {type(payload)}"
        )

    run_cfg = payload.get("evaluation_run")
    if not isinstance(run_cfg, dict):
        raise ValueError(
            f"Expected 'evaluation_run' to be a dict, got {type(run_cfg)}"
        )

    paths_cfg = run_cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise ValueError(
            f"Expected 'evaluation_run.paths' to be a dict, got {type(paths_cfg)}"
        )

    data_cfg = run_cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError(
            f"Expected 'evaluation_run.data' to be a dict, got {type(data_cfg)}"
        )

    outputs_cfg = run_cfg.get("outputs")
    if not isinstance(outputs_cfg, dict):
        raise ValueError(
            f"Expected 'evaluation_run.outputs' to be a dict, got {type(outputs_cfg)}"
        )

    # Rewrite all relevant paths to absolute paths anchored at repo root.
    for key in (
        "evaluation_config",
        "generation_output_dir",
        "dataset_config",
        "training_config",
        "generation_run_config",
    ):
        if key in paths_cfg:
            paths_cfg[key] = _abs_from_root(paths_cfg.get(key))

    if "output_dir" in outputs_cfg:
        outputs_cfg["output_dir"] = _abs_from_root(outputs_cfg.get("output_dir"))

    # Keep the smoke test small even if the run config forgot to limit cases.
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
    cases = list(loader.iter_cases())
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
    config_path = DEFAULT_CONFIG.resolve()

    print_header("STRIDE evaluation smoke test")
    print(f"Evaluation config: {config_path}")

    print_header("Preparing smoke config")
    smoke_config_path = build_smoke_config(config_path)
    print(f"Smoke config: {smoke_config_path}")

    print_header("Initializing evaluator")
    cfg = EvaluationRunConfig.from_yaml(smoke_config_path)
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