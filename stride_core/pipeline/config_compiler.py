

"""
Config compiler for STRIDE experiments.

This module turns a single human-authored `ExperimentConfig` into a set of
resolved stage configs for:

    training -> generation -> evaluation

The purpose is to let researchers edit one experiment config while the pipeline
materializes explicit stage configs that can be passed to the existing stage
entrypoints unchanged.

Important design note
---------------------
This is a first compiler implementation. It is intentionally conservative:
- it preserves reusable base configs,
- it applies experiment-level overrides,
- it writes fully resolved YAML files to disk,
- and it keeps the stage mains decoupled from experiment-config internals.

The exact data-field mapping can evolve later as the experiment schema becomes
more detailed. The compiler is therefore structured to make those updates easy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import copy
import json

import yaml

from stride_core.pipeline.experiment_config import ExperimentConfig


# -----------------------------------------------------------------------------
# Typed outputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledConfigPaths:
    experiment_root: Path
    compiled_dir: Path
    experiment_config_copy_path: Path
    model_config_path: Path | None
    training_base_config_path: Path | None
    data_config_path: Path | None
    generation_base_config_path: Path | None
    evaluation_base_config_path: Path | None
    training_run_config_path: Path
    generation_run_config_path: Path
    evaluation_run_config_path: Path
    manifest_path: Path

# -----------------------------------------------------------------------------
# Compiler
# -----------------------------------------------------------------------------


class ConfigCompiler:
    """Compile one `ExperimentConfig` into resolved stage configs on disk."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.repo_root = self._infer_repo_root()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self) -> CompiledConfigPaths:
        experiment_root = self._resolve_experiment_root()
        compiled_dir = experiment_root / "compiled_configs"
        compiled_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Load bases
        # ------------------------------------------------------------------

        base_model = self._load_yaml_file(self.cfg.bases.model_config_path)
        base_training = self._load_yaml_file(self.cfg.bases.training_config_path)
        base_generation = self._load_yaml_file(self.cfg.bases.generation_config_path)
        base_evaluation = self._load_yaml_file(self.cfg.bases.evaluation_config_path)
        base_data = self._load_yaml_file(self.cfg.bases.data_config_path)

        # ------------------------------------------------------------------
        # Compile reusable resolved base configs
        # ------------------------------------------------------------------

        resolved_model = copy.deepcopy(base_model) if base_model is not None else None
        resolved_training_base = (
            copy.deepcopy(base_training) if base_training is not None else {}
        )
        resolved_generation_base = (
            copy.deepcopy(base_generation) if base_generation is not None else {}
        )
        resolved_evaluation_base = (
            copy.deepcopy(base_evaluation) if base_evaluation is not None else {}
        )
        resolved_data = self._compile_data_config(
            base_data,
            base_data_config_path=self.cfg.bases.data_config_path,
        )

        model_config_path = self._write_yaml_if_not_none(
            resolved_model,
            compiled_dir / "model_resolved.yaml",
        )
        training_base_config_path = self._write_yaml_if_not_none(
            resolved_training_base,
            compiled_dir / "training_base_resolved.yaml",
        )
        generation_base_config_path = self._write_yaml_if_not_none(
            resolved_generation_base,
            compiled_dir / "generation_base_resolved.yaml",
        )
        evaluation_base_config_path = self._write_yaml_if_not_none(
            resolved_evaluation_base,
            compiled_dir / "evaluation_base_resolved.yaml",
        )
        data_config_path = self._write_yaml_if_not_none(
            resolved_data,
            compiled_dir / "data_resolved.yaml",
        )

        # ------------------------------------------------------------------
        # Compile stage run configs
        # ------------------------------------------------------------------

        training_run_cfg = self._compile_training_run_config(
            training_base_config_path=training_base_config_path,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            generation_base_config_path=generation_base_config_path,
            experiment_root=experiment_root,
        )
        training_run_config_path = self._write_yaml(
            training_run_cfg,
            compiled_dir / "training_run_resolved.yaml",
        )

        generation_run_cfg = self._compile_generation_run_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            generation_base_config_path=generation_base_config_path,
            training_run_config_path=training_run_config_path,
            experiment_root=experiment_root,
        )
        generation_run_config_path = self._write_yaml(
            generation_run_cfg,
            compiled_dir / "generation_run_resolved.yaml",
        )

        evaluation_run_cfg = self._compile_evaluation_run_config(
            data_config_path=data_config_path,
            evaluation_base_config_path=evaluation_base_config_path,
            training_run_config_path=training_run_config_path,
            generation_run_cfg=generation_run_cfg,
            experiment_root=experiment_root,
        )
        evaluation_run_config_path = self._write_yaml(
            evaluation_run_cfg,
            compiled_dir / "evaluation_run_resolved.yaml",
        )

        # ------------------------------------------------------------------
        # Save experiment copy + manifest
        # ------------------------------------------------------------------

        experiment_config_copy_path = self._write_yaml(
            self._to_serializable(self.cfg),
            compiled_dir / "experiment_config_resolved_copy.yaml",
        )

        manifest = {
            "experiment_name": self.cfg.meta.name,
            "experiment_root": str(experiment_root),
            "compiled_dir": str(compiled_dir),
            "model_config_path": str(model_config_path) if model_config_path is not None else None,
            "data_config_path": str(data_config_path) if data_config_path is not None else None,
            "training_base_config_path": (
                str(training_base_config_path)
                if training_base_config_path is not None
                else None
            ),
            "generation_base_config_path": (
                str(generation_base_config_path)
                if generation_base_config_path is not None
                else None
            ),
            "evaluation_base_config_path": (
                str(evaluation_base_config_path)
                if evaluation_base_config_path is not None
                else None
            ),
            "training_run_config_path": str(training_run_config_path),
            "generation_run_config_path": str(generation_run_config_path),
            "evaluation_run_config_path": str(evaluation_run_config_path),
        }
        manifest_path = compiled_dir / "compiled_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return CompiledConfigPaths(
            experiment_root=experiment_root,
            compiled_dir=compiled_dir,
            experiment_config_copy_path=experiment_config_copy_path,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            training_base_config_path=training_base_config_path,
            generation_base_config_path=generation_base_config_path,
            evaluation_base_config_path=evaluation_base_config_path,
            training_run_config_path=training_run_config_path,
            generation_run_config_path=generation_run_config_path,
            evaluation_run_config_path=evaluation_run_config_path,
            manifest_path=manifest_path,
        )

    # ------------------------------------------------------------------
    # Data compilation
    # ------------------------------------------------------------------

    def _compile_data_config(
        self,
        base_data: dict[str, Any] | None,
        *,
        base_data_config_path: Path | None,
    ) -> dict[str, Any]:
        """
        Compile a resolved data config.

        The guiding principle is:
        - preserve the base data config,
        - inject experiment-level data metadata in a structured way,
        - and apply explicit data overrides last.

        The exact adapter-specific field mapping can be expanded later.
        """
        resolved = copy.deepcopy(base_data) if base_data is not None else {}
        if base_data_config_path is not None:
            self._resolve_relative_paths_inplace(
                resolved,
                base_dir=base_data_config_path.parent,
            )

        experiment_data_block = {
            "target": {
                "variable": self.cfg.data.target.variable,
                "transform": self.cfg.data.target.transform,
                "output_shape": self.cfg.data.target.output_shape,
            },
            "conditioning": {
                "dynamic_variables": self.cfg.data.conditioning.dynamic_variables,
                "static_variables": self.cfg.data.conditioning.static_variables,
                "input_shape": self.cfg.data.conditioning.input_shape,
            },
            "domain": {
                "hr_size": self.cfg.data.domain.hr_size,
                "lr_size": self.cfg.data.domain.lr_size,
                "large_domain": self.cfg.data.domain.large_domain,
            },
            "split": {
                "train": self.cfg.data.split.train,
                "valid": self.cfg.data.split.valid,
                "test": self.cfg.data.split.test,
            },
        }

        # Always keep an explicit experiment-owned data block.
        resolved["experiment_data"] = experiment_data_block

        # Best-effort mapping onto common direct fields. This keeps the compiler
        # useful immediately, while still allowing later refinement.
        self._set_if_not_none(resolved, ["target_variable"], self.cfg.data.target.variable)
        self._set_if_not_none(resolved, ["target_transform"], self.cfg.data.target.transform)
        self._set_if_not_none(
            resolved,
            ["dynamic_variables"],
            self.cfg.data.conditioning.dynamic_variables,
        )
        self._set_if_not_none(
            resolved,
            ["static_variables"],
            self.cfg.data.conditioning.static_variables,
        )
        self._set_if_not_none(resolved, ["hr_size"], self.cfg.data.domain.hr_size)
        self._set_if_not_none(resolved, ["lr_size"], self.cfg.data.domain.lr_size)
        self._set_if_not_none(
            resolved,
            ["large_domain"],
            self.cfg.data.domain.large_domain,
        )
        self._set_if_not_none(
            resolved,
            ["input_shape"],
            self.cfg.data.conditioning.input_shape,
        )
        self._set_if_not_none(
            resolved,
            ["output_shape"],
            self.cfg.data.target.output_shape,
        )

        self._deep_update(resolved, copy.deepcopy(self.cfg.data.overrides))
        return resolved
    def _resolve_relative_paths_inplace(
        self,
        obj: Any,
        *,
        base_dir: Path,
    ) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and self._looks_like_path_key(key):
                    path = Path(value)
                    if not path.is_absolute():
                        config_relative = (base_dir / path).resolve()
                        if config_relative.exists():
                            obj[key] = str(config_relative)
                        else:
                            repo_relative = (self.repo_root / path).resolve()
                            obj[key] = str(repo_relative)
                else:
                    self._resolve_relative_paths_inplace(value, base_dir=base_dir)
        elif isinstance(obj, list):
            for item in obj:
                self._resolve_relative_paths_inplace(item, base_dir=base_dir)

    def _looks_like_path_key(self, key: str) -> bool:
        return key.endswith("_path") or key.endswith("_dir") or key in {
            "root_dir",
            "split_manifest_path",
        }

    # ------------------------------------------------------------------
    # Stage run config compilation
    # ------------------------------------------------------------------

    def _compile_training_run_config(
        self,
        *,
        training_base_config_path: Path | None,
        model_config_path: Path | None,
        data_config_path: Path | None,
        generation_base_config_path: Path | None,
        experiment_root: Path,
    ) -> dict[str, Any]:
        run_name = self.cfg.training.run_name or f"train_{self.cfg.meta.name}"

        if training_base_config_path is None:
            raise ValueError("A training base config is required to compile the training stage.")

        config = self._load_yaml_file(training_base_config_path)
        if config is None:
            config = {}
        config = copy.deepcopy(config)

        if "training" not in config or not isinstance(config["training"], dict):
            raise KeyError("Expected training base config to contain a top-level 'training' section")

        self._deep_update(config, copy.deepcopy(self.cfg.training.overrides))

        training_cfg = config["training"]
        training_cfg.setdefault("run", {})
        training_cfg.setdefault("configs", {})
        training_cfg["run"]["name"] = run_name
        training_cfg["run"]["output_dir"] = str((experiment_root / "training").resolve())
        training_cfg["run"]["seed"] = self.cfg.meta.seed
        training_cfg["configs"]["model_config"] = (
            str(model_config_path.resolve()) if model_config_path is not None else None
        )
        training_cfg["configs"]["dataset_config"] = (
            str(data_config_path.resolve()) if data_config_path is not None else None
        )
        training_cfg["configs"]["generation_config"] = (
            str(generation_base_config_path.resolve())
            if generation_base_config_path is not None
            else None
        )
        return config

    def _compile_generation_run_config(
        self,
        *,
        model_config_path: Path | None,
        data_config_path: Path | None,
        generation_base_config_path: Path | None,
        training_run_config_path: Path,
        experiment_root: Path,
    ) -> dict[str, Any]:
        run_name = self.cfg.generation.run_name or f"generate_{self.cfg.meta.name}"
        training_output_dir = experiment_root / "training"
        checkpoint_path = training_output_dir / "checkpoints" / "checkpoint_best.pt"
        generation_output_dir = experiment_root / "generation"

        config: dict[str, Any] = {
            "generation_run": {
                "name": run_name,
                "paths": {
                    "model_config": str(model_config_path)
                    if model_config_path is not None
                    else None,
                    "dataset_config": str(data_config_path)
                    if data_config_path is not None
                    else None,
                    "generation_config": (
                        str(generation_base_config_path)
                        if generation_base_config_path is not None
                        else None
                    ),
                    "training_config": str(training_run_config_path),
                    "checkpoint_path": str(checkpoint_path),
                },
                "data": {
                    "split": "test",
                },
                "outputs": {
                    "output_dir": str(generation_output_dir),
                },
            }
        }

        self._deep_update(config, copy.deepcopy(self.cfg.generation.overrides))

        run_cfg = config["generation_run"]
        run_cfg["name"] = run_name
        run_cfg.setdefault("paths", {})
        run_cfg.setdefault("data", {})
        run_cfg.setdefault("outputs", {})
        run_cfg["paths"]["model_config"] = (
            str(model_config_path.resolve()) if model_config_path is not None else None
        )
        run_cfg["paths"]["dataset_config"] = (
            str(data_config_path.resolve()) if data_config_path is not None else None
        )
        run_cfg["paths"]["generation_config"] = (
            str(generation_base_config_path.resolve())
            if generation_base_config_path is not None
            else None
        )
        run_cfg["paths"]["training_config"] = str(training_run_config_path.resolve())
        run_cfg["paths"]["checkpoint_path"] = str(checkpoint_path.resolve())
        run_cfg["outputs"]["output_dir"] = str((experiment_root / "generation").resolve())
        return config

    def _compile_evaluation_run_config(
        self,
        *,
        data_config_path: Path | None,
        evaluation_base_config_path: Path | None,
        training_run_config_path: Path,
        generation_run_cfg: dict[str, Any],
        experiment_root: Path,
    ) -> dict[str, Any]:
        run_name = self.cfg.evaluation.run_name or f"evaluate_{self.cfg.meta.name}"

        generation_output_dir = self._nested_get(
            generation_run_cfg,
            ["generation_run", "outputs", "output_dir"],
            default=str(experiment_root / "generation"),
        )

        config: dict[str, Any] = {
            "evaluation_run": {
                "run_name": run_name,
                "paths": {
                    "generation_output_dir": generation_output_dir,
                    "dataset_config": str(data_config_path) if data_config_path is not None else None,
                    "evaluation_config": (
                        str(evaluation_base_config_path)
                        if evaluation_base_config_path is not None
                        else None
                    ),
                    "training_config": str(training_run_config_path),
                },
                "data": {
                    "split": "test",
                    "forecast_product_for_spatial": "pmm",
                    "forecast_product_for_climatology": "pmm",
                    "forecast_product_for_temporal": "pmm",
                },
                "outputs": {
                    "output_dir": str(experiment_root / "evaluation"),
                    "save_metrics_json": True,
                    "save_arrays_npz": True,
                    "save_figures": True,
                },
            }
        }

        self._deep_update(config, copy.deepcopy(self.cfg.evaluation.overrides))

        run_cfg = config["evaluation_run"]
        run_cfg["run_name"] = run_name
        run_cfg.setdefault("paths", {})
        run_cfg.setdefault("data", {})
        run_cfg.setdefault("outputs", {})
        run_cfg["paths"]["generation_output_dir"] = str((experiment_root / "generation").resolve())
        run_cfg["paths"]["dataset_config"] = (
            str(data_config_path.resolve()) if data_config_path is not None else None
        )
        run_cfg["paths"]["evaluation_config"] = (
            str(evaluation_base_config_path.resolve())
            if evaluation_base_config_path is not None
            else None
        )
        run_cfg["paths"]["training_config"] = str(training_run_config_path.resolve())
        run_cfg["outputs"]["output_dir"] = str((experiment_root / "evaluation").resolve())
        return config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_experiment_root(self) -> Path:
        if self.cfg.meta.output_root is not None:
            output_root = self.cfg.meta.output_root.resolve()
            return (output_root / self.cfg.meta.name).resolve()
        return (self.repo_root / "runs" / self.cfg.meta.name).resolve()

    def _load_yaml_file(self, path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Base config does not exist: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError(f"Expected YAML root dict in {path}, got {type(payload)}")
        return payload

    def _write_yaml_if_not_none(
        self,
        payload: dict[str, Any] | None,
        path: Path,
    ) -> Path | None:
        if payload is None:
            return None
        return self._write_yaml(payload, path)

    def _write_yaml(self, payload: dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        return path

    def _nested_get(
        self,
        obj: dict[str, Any],
        keys: list[str],
        *,
        default: Any = None,
    ) -> Any:
        cur: Any = obj
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _set_if_not_none(
        self,
        obj: dict[str, Any],
        keys: list[str],
        value: Any,
    ) -> None:
        if value is None:
            return
        if isinstance(value, list) and len(value) == 0:
            return
        cur = obj
        for key in keys[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[keys[-1]] = copy.deepcopy(value)

    def _deep_update(self, base: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    def _to_serializable(self, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return self._to_serializable(asdict(value))
        if isinstance(value, dict):
            return {key: self._to_serializable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value