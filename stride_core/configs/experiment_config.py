"""
Experiment configuration for the STRIDE full pipeline.

This module defines the human-authored *single source of truth* for a full
experiment. The intention is that one experiment config should control:

    training -> generation -> evaluation

while still referencing stable reusable base configs for model, generation,
evaluation, and data defaults.

Design principles
-----------------
- Keep one experiment config as the main file a researcher edits.
- Keep stable reusable defaults in separate base configs.
- Allow stage toggles.
- Allow stage-specific overrides.
- Allow model-specific overrides at the experiment level.
- Make data a first-class part of experiment definition.
- Keep the schema strict enough to catch mistakes early, but flexible enough
  that detailed knobs can be extended later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{name}' to be a dict, got {type(value)}")
    return value


# Expand environment variables in a value, error if unresolved
def _expand_env_vars(raw_value: Any) -> Any:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        return raw_value

    expanded = os.path.expandvars(raw_value)
    if "$" in expanded:
        raise ValueError(
            "Unresolved environment variable in path value: "
            f"{raw_value!r} -> {expanded!r}"
        )
    return expanded


def _resolve_path(raw_path: Any, *, config_path: Path) -> Path | None:
    raw_path = _expand_env_vars(raw_path)
    if raw_path is None:
        return None

    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()

    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative

    repo_root = config_path.parents[2]
    repo_relative = (repo_root / path).resolve()
    return repo_relative



def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"Expected '{name}' to be a boolean, got {value!r}")



def _coerce_str_list(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected '{name}' to be a list, got {type(value)}")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"Expected '{name}[{i}]' to be a string, got {type(item)}"
            )
        out.append(item)
    return out



def _coerce_int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected '{name}' to be a list, got {type(value)}")
    out: list[int] = []
    for i, item in enumerate(value):
        if not isinstance(item, int):
            raise ValueError(
                f"Expected '{name}[{i}]' to be an int, got {type(item)}"
            )
        out.append(int(item))
    return out


# -----------------------------------------------------------------------------
# Experiment metadata / toggles / bases
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentMetaConfig:
    name: str
    output_root: Path | None
    seed: int | None


@dataclass(frozen=True)
class ExperimentStageToggleConfig:
    training: bool = True
    generation: bool = True
    evaluation: bool = True


@dataclass(frozen=True)
class ExperimentBasesConfig:
    model_config_path: Path | None
    training_config_path: Path | None
    generation_config_path: Path | None
    sampler_config_path: Path | None
    evaluation_config_path: Path | None
    data_config_path: Path | None


# -----------------------------------------------------------------------------
# Model override config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentModelConfig:
    overrides: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Data config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentTargetConfig:
    variable: str | None
    transform: str | None
    output_shape: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentConditioningConfig:
    dynamic_variables: list[str] = field(default_factory=list)
    static_variables: list[str] = field(default_factory=list)
    input_shape: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentDomainConfig:
    hr_size: list[int] = field(default_factory=list)
    lr_size: list[int] = field(default_factory=list)
    large_domain: bool = False


@dataclass(frozen=True)
class ExperimentStatisticsSplitConfig:
    train: str | None = None
    val: str | None = None
    test: str | None = None


@dataclass(frozen=True)
class ExperimentSplitConfig:
    train: str | None = None
    val: str | None = None
    test: str | None = None
    statistics: ExperimentStatisticsSplitConfig = field(
        default_factory=ExperimentStatisticsSplitConfig
    )


@dataclass(frozen=True)
class ExperimentDataConfig:
    target: ExperimentTargetConfig
    conditioning: ExperimentConditioningConfig
    domain: ExperimentDomainConfig
    split: ExperimentSplitConfig
    overrides: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Stage-specific config blocks
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentTrainingConfig:
    run_name: str | None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentGenerationConfig:
    run_name: str | None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentEvaluationConfig:
    run_name: str | None
    overrides: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Main config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    meta: ExperimentMetaConfig
    stages: ExperimentStageToggleConfig
    bases: ExperimentBasesConfig
    model: ExperimentModelConfig
    data: ExperimentDataConfig
    training: ExperimentTrainingConfig
    generation: ExperimentGenerationConfig
    evaluation: ExperimentEvaluationConfig
    config_path: Path | None = None

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)

        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected experiment config root to be a dict, got {type(payload)}"
            )

        return cls.from_dict(payload, config_path=config_path)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        config_path: Path,
    ) -> "ExperimentConfig":
        experiment_cfg = _require_dict(payload.get("experiment"), "experiment")
        stages_cfg = _require_dict(payload.get("stages"), "stages")
        bases_cfg = _require_dict(payload.get("bases"), "bases")
        model_cfg = _require_dict(payload.get("model"), "model")
        data_cfg = _require_dict(payload.get("data"), "data")
        training_cfg = _require_dict(payload.get("training"), "training")
        generation_cfg = _require_dict(payload.get("generation"), "generation")
        evaluation_cfg = _require_dict(payload.get("evaluation"), "evaluation")

        experiment_name = experiment_cfg.get("name")
        if not isinstance(experiment_name, str) or experiment_name.strip() == "":
            raise ValueError("experiment.name must be provided and non-empty")

        output_root = _resolve_path(
            experiment_cfg.get("output_root"), config_path=config_path
        )

        seed = experiment_cfg.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"experiment.seed must be an int or null, got {type(seed)}")

        target_cfg = _require_dict(data_cfg.get("target"), "data.target")
        conditioning_cfg = _require_dict(
            data_cfg.get("conditioning"), "data.conditioning"
        )
        domain_cfg = _require_dict(data_cfg.get("domain"), "data.domain")
        split_cfg = _require_dict(data_cfg.get("split"), "data.split")
        statistics_split_cfg = _require_dict(
            split_cfg.get("statistics"),
            "data.split.statistics",
        )

        target_variable = target_cfg.get("variable")
        if target_variable is not None and not isinstance(target_variable, str):
            raise ValueError("data.target.variable must be a string or null")

        target_transform = target_cfg.get("transform")
        if target_transform is not None and not isinstance(target_transform, str):
            raise ValueError("data.target.transform must be a string or null")

        training_run_name = training_cfg.get("run_name")
        if training_run_name is not None and not isinstance(training_run_name, str):
            raise ValueError("training.run_name must be a string or null")

        generation_run_name = generation_cfg.get("run_name")
        if generation_run_name is not None and not isinstance(generation_run_name, str):
            raise ValueError("generation.run_name must be a string or null")

        evaluation_run_name = evaluation_cfg.get("run_name")
        if evaluation_run_name is not None and not isinstance(evaluation_run_name, str):
            raise ValueError("evaluation.run_name must be a string or null")

        model_overrides = (
            _require_dict(model_cfg.get("overrides"), "model.overrides")
            if "overrides" in model_cfg
            else model_cfg
        )

        return cls(
            meta=ExperimentMetaConfig(
                name=experiment_name,
                output_root=output_root,
                seed=seed,
            ),
            stages=ExperimentStageToggleConfig(
                training=_coerce_bool(stages_cfg.get("training", True), name="stages.training"),
                generation=_coerce_bool(stages_cfg.get("generation", True), name="stages.generation"),
                evaluation=_coerce_bool(stages_cfg.get("evaluation", True), name="stages.evaluation"),
            ),
            bases=ExperimentBasesConfig(
                model_config_path=_resolve_path(
                    bases_cfg.get("model"), config_path=config_path
                ),
                training_config_path=_resolve_path(
                    bases_cfg.get("training"), config_path=config_path
                ),
                generation_config_path=_resolve_path(
                    bases_cfg.get("generation"), config_path=config_path
                ),
                sampler_config_path=_resolve_path(
                    bases_cfg.get("sampler"), config_path=config_path
                ),
                evaluation_config_path=_resolve_path(
                    bases_cfg.get("evaluation"), config_path=config_path
                ),
                data_config_path=_resolve_path(
                    bases_cfg.get("data"), config_path=config_path
                ),
            ),
            model=ExperimentModelConfig(
                overrides=model_overrides,
            ),
            data=ExperimentDataConfig(
                target=ExperimentTargetConfig(
                    variable=target_variable,
                    transform=target_transform,
                    output_shape=_coerce_int_list(
                        target_cfg.get("output_shape"),
                        name="data.target.output_shape",
                    ),
                ),
                conditioning=ExperimentConditioningConfig(
                    dynamic_variables=_coerce_str_list(
                        conditioning_cfg.get("dynamic_variables"),
                        name="data.conditioning.dynamic_variables",
                    ),
                    static_variables=_coerce_str_list(
                        conditioning_cfg.get("static_variables"),
                        name="data.conditioning.static_variables",
                    ),
                    input_shape=_coerce_int_list(
                        conditioning_cfg.get("input_shape"),
                        name="data.conditioning.input_shape",
                    ),
                ),
                domain=ExperimentDomainConfig(
                    hr_size=_coerce_int_list(
                        domain_cfg.get("hr_size"),
                        name="data.domain.hr_size",
                    ),
                    lr_size=_coerce_int_list(
                        domain_cfg.get("lr_size"),
                        name="data.domain.lr_size",
                    ),
                    large_domain=_coerce_bool(
                        domain_cfg.get("large_domain", False),
                        name="data.domain.large_domain",
                    ),
                ),
                split=ExperimentSplitConfig(
                    train=split_cfg.get("train"),
                    val=split_cfg.get("val"),
                    test=split_cfg.get("test"),
                    statistics=ExperimentStatisticsSplitConfig(
                        train=statistics_split_cfg.get("train"),
                        val=statistics_split_cfg.get("val"),
                        test=statistics_split_cfg.get("test"),
                    ),
                ),
                overrides=_require_dict(data_cfg.get("overrides"), "data.overrides"),
            ),
            training=ExperimentTrainingConfig(
                run_name=training_run_name,
                overrides=_require_dict(training_cfg.get("overrides"), "training.overrides"),
            ),
            generation=ExperimentGenerationConfig(
                run_name=generation_run_name,
                overrides=_require_dict(generation_cfg.get("overrides"), "generation.overrides"),
            ),
            evaluation=ExperimentEvaluationConfig(
                run_name=evaluation_run_name,
                overrides=_require_dict(evaluation_cfg.get("overrides"), "evaluation.overrides"),
            ),
            config_path=config_path,
        )
