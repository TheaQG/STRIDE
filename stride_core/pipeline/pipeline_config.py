"""
Configuration for the STRIDE full experiment pipeline.

This config orchestrates:

    training -> generation -> evaluation

The pipeline does NOT duplicate configuration details from the
individual stages. Instead it simply references the stage configs.

Example YAML
------------

experiment:
  name: edm_downscaling_test

training:
  enabled: true
  config: configs/training_runs/train_edm_small.yaml

generation:
  enabled: true
  config: configs/generation_runs/generate_test_best.yaml

evaluation:
  enabled: true
  config: configs/evaluation_runs/evaluate_test_best.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------
# Stage config container
# ---------------------------------------------------------------------


@dataclass
class PipelineStageConfig:
    enabled: bool
    config_path: Path | None


# ---------------------------------------------------------------------
# Main pipeline config
# ---------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Top-level experiment configuration.
    """

    experiment_name: str

    training: PipelineStageConfig
    generation: PipelineStageConfig
    evaluation: PipelineStageConfig

    config_path: Path | None = None

    # -----------------------------------------------------------------
    # YAML loading
    # -----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":

        config_path = Path(path).expanduser().resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)

        if not isinstance(payload, dict):
            raise ValueError("Pipeline config root must be a dictionary")

        # -----------------------------------------------------------------
        # experiment
        # -----------------------------------------------------------------

        experiment_cfg = payload.get("experiment")
        if not isinstance(experiment_cfg, dict):
            raise ValueError("Missing 'experiment' section in pipeline config")

        experiment_name = experiment_cfg.get("name")
        if not isinstance(experiment_name, str):
            raise ValueError("experiment.name must be defined")

        # -----------------------------------------------------------------
        # stages
        # -----------------------------------------------------------------

        training = cls._parse_stage(payload.get("training"))
        generation = cls._parse_stage(payload.get("generation"))
        evaluation = cls._parse_stage(payload.get("evaluation"))

        return cls(
            experiment_name=experiment_name,
            training=training,
            generation=generation,
            evaluation=evaluation,
            config_path=config_path,
        )

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_stage(stage_cfg: Any) -> PipelineStageConfig:

        if stage_cfg is None:
            return PipelineStageConfig(enabled=False, config_path=None)

        if not isinstance(stage_cfg, dict):
            raise ValueError("Stage configuration must be a dictionary")

        enabled = bool(stage_cfg.get("enabled", True))

        config_path = stage_cfg.get("config")

        if config_path is not None:
            config_path = Path(config_path)

        return PipelineStageConfig(
            enabled=enabled,
            config_path=config_path,
        )