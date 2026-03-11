"""
Typed configuration objects for STRIDE evaluation.

This module mirrors the configuration discipline used in training and generation:
- one top-level evaluation-run config
- small typed sub-configs
- explicit validation during parsing
- path resolution anchored to the evaluation config location

The evaluation pipeline is intentionally ensemble-first. The configuration therefore
supports separate switches for probabilistic, spatial, climatological, temporal,
and sigma-star diagnostics while keeping the run orchestration clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


ALLOWED_FORECAST_PRODUCTS = {
    "ensemble_members",
    "ensemble_mean",
    "pmm",
}



def _resolve_path(value: str | Path | None, *, config_path: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path



def _require_dict(section: Any, name: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError(f"Expected '{name}' to be a dict, got {type(section)}")
    return section



def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Expected '{name}' to be a bool, got {type(value)}")
    return value



def _coerce_bool_dict(section: dict[str, Any], name: str) -> dict[str, bool]:
    return {key: _require_bool(value, f"{name}.{key}") for key, value in section.items()}



def _coerce_float_list(value: Any, name: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected '{name}' to be a list, got {type(value)}")
    return [float(x) for x in value]



def _coerce_int_list(value: Any, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected '{name}' to be a list, got {type(value)}")
    return [int(x) for x in value]


# -----------------------------------------------------------------------------
# Typed sub-configs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationPathsConfig:
    evaluation_config_path: Path
    generation_run_config_path: Path | None
    generation_output_dir: Path
    dataset_config_path: Path | None
    training_config_path: Path | None


@dataclass(frozen=True)
class EvaluationDataConfig:
    split: str
    use_case_limit: int | None
    forecast_product_for_spatial: str
    forecast_product_for_climatology: str
    forecast_product_for_temporal: str


@dataclass(frozen=True)
class EvaluationProbabilisticConfig:
    crps: bool
    crps_decomposition: bool
    spread_skill: bool
    pit_histogram: bool
    rank_histogram: bool
    reliability_diagram: bool


@dataclass(frozen=True)
class EvaluationSpatialConfig:
    psd: bool
    psd_slope: bool
    iss: bool
    sal: bool


@dataclass(frozen=True)
class EvaluationClimatologyConfig:
    pixel_value_distribution: bool
    histogram_comparison: bool
    qq_plot: bool
    extremes: bool
    wet_day_frequency: bool
    annual_precipitation_sum: bool
    seasonal_accumulations: bool


@dataclass(frozen=True)
class EvaluationTemporalConfig:
    lag_autocorrelation: bool
    wet_spell_length: bool
    dry_spell_length: bool


@dataclass(frozen=True)
class EvaluationSigmaStarConfig:
    enabled: bool
    values: list[float]
    metrics: dict[str, bool]


@dataclass(frozen=True)
class EvaluationMaskConfig:
    apply_land_mask: bool
    apply_coastline_buffer: bool
    coastline_buffer_pixels: int


@dataclass(frozen=True)
class EvaluationThresholdsConfig:
    wet_day_threshold_mm: float
    reliability_thresholds_mm: list[float]
    iss_thresholds_mm: list[float]
    spell_threshold_mm: float
    extreme_quantiles: list[float]


@dataclass(frozen=True)
class EvaluationScalesConfig:
    iss_scales_km: list[int]
    psd_slope_fit_range_km: list[float]
    autocorrelation_lags: list[int]


@dataclass(frozen=True)
class EvaluationPlottingConfig:
    enabled: bool
    families: dict[str, bool]
    save_quicklooks: bool


@dataclass(frozen=True)
class EvaluationOutputConfig:
    output_dir: Path
    save_metrics_json: bool
    save_arrays_npz: bool
    save_figures: bool


# -----------------------------------------------------------------------------
# Top-level config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRunConfig:
    config_path: Path
    run_name: str
    paths: EvaluationPathsConfig
    data: EvaluationDataConfig
    probabilistic: EvaluationProbabilisticConfig
    spatial: EvaluationSpatialConfig
    climatology: EvaluationClimatologyConfig
    temporal: EvaluationTemporalConfig
    sigma_star: EvaluationSigmaStarConfig
    masks: EvaluationMaskConfig
    thresholds: EvaluationThresholdsConfig
    scales: EvaluationScalesConfig
    plotting: EvaluationPlottingConfig
    outputs: EvaluationOutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationRunConfig":
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Evaluation YAML config does not exist: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            run_payload = yaml.safe_load(f)

        if not isinstance(run_payload, dict):
            raise ValueError(
                f"Expected evaluation run config root to be a dict, got {type(run_payload)}"
            )

        run_cfg = _require_dict(run_payload.get("evaluation_run", {}), "evaluation_run")
        paths_cfg = _require_dict(run_cfg.get("paths", {}), "evaluation_run.paths")

        raw_evaluation_config = paths_cfg.get("evaluation_config")
        evaluation_config_path = _resolve_path(
            raw_evaluation_config, config_path=config_path
        )
        if evaluation_config_path is not None and not evaluation_config_path.exists():
            raw_eval_str = str(raw_evaluation_config)
            if not Path(raw_eval_str).is_absolute():
                repo_root = config_path.parents[2]
                repo_relative_candidate = (repo_root / raw_eval_str).resolve()
                if repo_relative_candidate.exists():
                    evaluation_config_path = repo_relative_candidate
        if evaluation_config_path is None:
            raise ValueError(
                "evaluation_run.paths.evaluation_config must be provided"
            )
        if not evaluation_config_path.exists():
            raise FileNotFoundError(
                f"Base evaluation YAML config does not exist: {evaluation_config_path}"
            )

        with open(evaluation_config_path, "r", encoding="utf-8") as f:
            base_payload = yaml.safe_load(f)

        if not isinstance(base_payload, dict):
            raise ValueError(
                f"Expected base evaluation config root to be a dict, got {type(base_payload)}"
            )

        combined_payload = dict(run_payload)
        combined_payload["evaluation_base"] = base_payload
        return cls.from_dict(combined_payload, config_path=config_path)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        config_path: Path,
    ) -> "EvaluationRunConfig":
        run_cfg = _require_dict(payload.get("evaluation_run", {}), "evaluation_run")
        base_cfg = _require_dict(payload.get("evaluation_base", {}), "evaluation_base")

        run_name = str(run_cfg.get("run_name", config_path.stem))

        paths_cfg = _require_dict(run_cfg.get("paths", {}), "evaluation_run.paths")
        data_cfg = _require_dict(run_cfg.get("data", {}), "evaluation_run.data")
        outputs_cfg = _require_dict(run_cfg.get("outputs", {}), "evaluation_run.outputs")

        eval_cfg = _require_dict(base_cfg.get("evaluation", {}), "evaluation_base.evaluation")
        masks_cfg = _require_dict(base_cfg.get("masks", {}), "evaluation_base.masks")
        thresholds_cfg = _require_dict(
            base_cfg.get("thresholds", {}), "evaluation_base.thresholds"
        )
        scales_cfg = _require_dict(base_cfg.get("scales", {}), "evaluation_base.scales")
        plotting_cfg = _require_dict(base_cfg.get("plotting", {}), "evaluation_base.plotting")

        probabilistic_cfg = _require_dict(
            eval_cfg.get("probabilistic", {}), "evaluation_base.evaluation.probabilistic"
        )
        spatial_cfg = _require_dict(
            eval_cfg.get("spatial", {}), "evaluation_base.evaluation.spatial"
        )
        climatology_cfg = _require_dict(
            eval_cfg.get("climatology", {}), "evaluation_base.evaluation.climatology"
        )
        temporal_cfg = _require_dict(
            eval_cfg.get("temporal", {}), "evaluation_base.evaluation.temporal"
        )
        sigma_star_cfg = _require_dict(
            eval_cfg.get("sigma_star", {}), "evaluation_base.evaluation.sigma_star"
        )

        evaluation_config_path = _resolve_path(
            paths_cfg.get("evaluation_config"), config_path=config_path
        )
        if evaluation_config_path is None:
            raise ValueError("evaluation_run.paths.evaluation_config must be provided")

        generation_output_dir = _resolve_path(
            paths_cfg.get("generation_output_dir"), config_path=config_path
        )
        if generation_output_dir is None:
            raise ValueError("evaluation_run.paths.generation_output_dir must be provided")

        output_dir = _resolve_path(outputs_cfg.get("output_dir"), config_path=config_path)
        if output_dir is None:
            raise ValueError("evaluation_run.outputs.output_dir must be provided")

        use_case_limit_raw = data_cfg.get("use_case_limit", None)
        use_case_limit = None if use_case_limit_raw is None else int(use_case_limit_raw)
        if use_case_limit is not None and use_case_limit <= 0:
            raise ValueError(
                f"evaluation_run.data.use_case_limit must be > 0 when set, got {use_case_limit}"
            )

        forecast_product_for_spatial = str(
            data_cfg.get("forecast_product_for_spatial", "pmm")
        )
        forecast_product_for_climatology = str(
            data_cfg.get("forecast_product_for_climatology", "pmm")
        )
        forecast_product_for_temporal = str(
            data_cfg.get("forecast_product_for_temporal", "pmm")
        )
        for name, value in {
            "forecast_product_for_spatial": forecast_product_for_spatial,
            "forecast_product_for_climatology": forecast_product_for_climatology,
            "forecast_product_for_temporal": forecast_product_for_temporal,
        }.items():
            if value not in ALLOWED_FORECAST_PRODUCTS:
                raise ValueError(
                    f"evaluation_run.data.{name} must be one of {sorted(ALLOWED_FORECAST_PRODUCTS)}, got {value!r}"
                )

        sigma_star_enabled = bool(sigma_star_cfg.get("enabled", False))
        sigma_star_values = _coerce_float_list(
            sigma_star_cfg.get("values", []),
            "evaluation_base.evaluation.sigma_star.values",
        )
        sigma_star_metrics = _coerce_bool_dict(
            _require_dict(
                sigma_star_cfg.get("metrics", {}),
                "evaluation_base.evaluation.sigma_star.metrics",
            ),
            "evaluation_base.evaluation.sigma_star.metrics",
        )

        psd_slope_fit_range_km = _coerce_float_list(
            scales_cfg.get("psd_slope_fit_range_km", [5.0, 20.0]),
            "evaluation_base.scales.psd_slope_fit_range_km",
        )
        if len(psd_slope_fit_range_km) != 2:
            raise ValueError(
                "evaluation_base.scales.psd_slope_fit_range_km must contain exactly two values"
            )

        return cls(
            config_path=config_path,
            run_name=run_name,
            paths=EvaluationPathsConfig(
                evaluation_config_path=evaluation_config_path,
                generation_run_config_path=_resolve_path(
                    paths_cfg.get("generation_run_config"), config_path=config_path
                ),
                generation_output_dir=generation_output_dir,
                dataset_config_path=_resolve_path(
                    paths_cfg.get("dataset_config"), config_path=config_path
                ),
                training_config_path=_resolve_path(
                    paths_cfg.get("training_config"), config_path=config_path
                ),
            ),
            data=EvaluationDataConfig(
                split=str(data_cfg.get("split", "test")),
                use_case_limit=use_case_limit,
                forecast_product_for_spatial=forecast_product_for_spatial,
                forecast_product_for_climatology=forecast_product_for_climatology,
                forecast_product_for_temporal=forecast_product_for_temporal,
            ),
            probabilistic=EvaluationProbabilisticConfig(
                **_coerce_bool_dict(
                    probabilistic_cfg,
                    "evaluation_base.evaluation.probabilistic",
                )
            ),
            spatial=EvaluationSpatialConfig(
                **_coerce_bool_dict(spatial_cfg, "evaluation_base.evaluation.spatial")
            ),
            climatology=EvaluationClimatologyConfig(
                **_coerce_bool_dict(
                    climatology_cfg,
                    "evaluation_base.evaluation.climatology",
                )
            ),
            temporal=EvaluationTemporalConfig(
                **_coerce_bool_dict(temporal_cfg, "evaluation_base.evaluation.temporal")
            ),
            sigma_star=EvaluationSigmaStarConfig(
                enabled=sigma_star_enabled,
                values=sigma_star_values,
                metrics=sigma_star_metrics,
            ),
            masks=EvaluationMaskConfig(
                apply_land_mask=bool(masks_cfg.get("apply_land_mask", True)),
                apply_coastline_buffer=bool(
                    masks_cfg.get("apply_coastline_buffer", False)
                ),
                coastline_buffer_pixels=int(
                    masks_cfg.get("coastline_buffer_pixels", 0)
                ),
            ),
            thresholds=EvaluationThresholdsConfig(
                wet_day_threshold_mm=float(
                    thresholds_cfg.get("wet_day_threshold_mm", 1.0)
                ),
                reliability_thresholds_mm=_coerce_float_list(
                    thresholds_cfg.get("reliability_thresholds_mm", [1.0, 5.0, 10.0, 20.0]),
                    "evaluation_base.thresholds.reliability_thresholds_mm",
                ),
                iss_thresholds_mm=_coerce_float_list(
                    thresholds_cfg.get("iss_thresholds_mm", [1.0, 5.0, 10.0, 20.0, 50.0]),
                    "evaluation_base.thresholds.iss_thresholds_mm",
                ),
                spell_threshold_mm=float(thresholds_cfg.get("spell_threshold_mm", 1.0)),
                extreme_quantiles=_coerce_float_list(
                    thresholds_cfg.get("extreme_quantiles", [0.95, 0.99, 0.999, 0.9999]),
                    "evaluation_base.thresholds.extreme_quantiles",
                ),
            ),
            scales=EvaluationScalesConfig(
                iss_scales_km=_coerce_int_list(
                    scales_cfg.get("iss_scales_km", [5, 10, 20, 40]),
                    "evaluation_base.scales.iss_scales_km",
                ),
                psd_slope_fit_range_km=psd_slope_fit_range_km,
                autocorrelation_lags=_coerce_int_list(
                    scales_cfg.get("autocorrelation_lags", [1, 2, 3, 5, 7, 10]),
                    "evaluation_base.scales.autocorrelation_lags",
                ),
            ),
            plotting=EvaluationPlottingConfig(
                enabled=bool(plotting_cfg.get("enabled", True)),
                families=_coerce_bool_dict(
                    _require_dict(
                        plotting_cfg.get("families", {}),
                        "evaluation_base.plotting.families",
                    ),
                    "evaluation_base.plotting.families",
                ),
                save_quicklooks=bool(plotting_cfg.get("save_quicklooks", False)),
            ),
            outputs=EvaluationOutputConfig(
                output_dir=output_dir,
                save_metrics_json=bool(outputs_cfg.get("save_metrics_json", True)),
                save_arrays_npz=bool(outputs_cfg.get("save_arrays_npz", True)),
                save_figures=bool(outputs_cfg.get("save_figures", True)),
            ),
        )