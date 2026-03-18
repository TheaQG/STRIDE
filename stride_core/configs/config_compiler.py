

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
import logging

import yaml

from stride_core.configs.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


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
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self) -> CompiledConfigPaths:
        experiment_root = self._resolve_experiment_root()
        compiled_dir = experiment_root / "compiled_configs"
        compiled_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Compiling experiment '{self.cfg.meta.name}'")
        self._log(f"Experiment root: {experiment_root}")
        self._log(f"Compiled config directory: {compiled_dir}")

        # ------------------------------------------------------------------
        # Load bases
        # ------------------------------------------------------------------

        base_model = self._load_yaml_file(self.cfg.bases.model_config_path)
        base_training = self._load_yaml_file(self.cfg.bases.training_config_path)
        base_generation = self._load_yaml_file(self.cfg.bases.generation_config_path)
        base_sampler = self._load_yaml_file(self.cfg.bases.sampler_config_path)
        base_evaluation = self._load_yaml_file(self.cfg.bases.evaluation_config_path)
        base_data = self._load_yaml_file(self.cfg.bases.data_config_path)
        self._log("Loaded base configs successfully")
        self._log(f"  model base: {self.cfg.bases.model_config_path}")
        self._log(f"  training base: {self.cfg.bases.training_config_path}")
        self._log(f"  generation base: {self.cfg.bases.generation_config_path}")
        self._log(f"  sampler base: {self.cfg.bases.sampler_config_path}")
        self._log(f"  evaluation base: {self.cfg.bases.evaluation_config_path}")
        self._log(f"  data base: {self.cfg.bases.data_config_path}")

        # ------------------------------------------------------------------
        # Compile reusable resolved base configs
        # ------------------------------------------------------------------

        resolved_training_base = (
            copy.deepcopy(base_training) if base_training is not None else {}
        )
        resolved_generation_base = self._compile_generation_base_config(
            base_generation=base_generation,
            base_sampler=base_sampler,
        )
        resolved_evaluation_base = (
            copy.deepcopy(base_evaluation) if base_evaluation is not None else {}
        )
        resolved_data = self._compile_data_config(
            base_data,
            base_data_config_path=self.cfg.bases.data_config_path,
        )
        resolved_model = self._compile_model_config(
            base_model,
            resolved_data=resolved_data,
        )
        self._validate_resolved_data_config(resolved_data)
        self._validate_data_selection_against_base(
            base_data=base_data,
            resolved_data=resolved_data,
        )
        self._validate_model_data_alignment(
            resolved_data=resolved_data,
            resolved_model=resolved_model,
        )
        self._log("Resolved data/model configs passed validation")

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
            compiled_dir / "generation_resolved.yaml",
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
        self._validate_stage_run_configs(
            training_run_cfg=training_run_cfg,
            generation_run_cfg=generation_run_cfg,
            evaluation_run_cfg=evaluation_run_cfg,
        )
        self._validate_stage_dependencies()
        self._log("Resolved stage configs passed validation")
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
    def _compile_model_config(
        self,
        base_model: dict[str, Any] | None,
        *,
        resolved_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Compile a resolved model config.

        The resolved model config is built by:
        1) copying the base model config,
        2) applying experiment-level model overrides inside the top-level
           `model` section,
        3) overwriting derived channel counts from the resolved data config.

        This keeps experiment-driven model ablations possible while ensuring
        data-dependent channel counts remain the source of truth.
        """
        if base_model is None:
            return None

        resolved = copy.deepcopy(base_model)

        # Apply experiment-level model overrides inside the top-level `model`
        # section, then overwrite derived channel counts from resolved data
        # below.
        resolved.setdefault("model", {})
        self._deep_update(
            resolved["model"],
            copy.deepcopy(self.cfg.model.overrides),
        )

        dynamic_variables = self._nested_get(
            resolved_data,
            ["data", "conditioning", "dynamic", "variables"],
            default=self._nested_get(resolved_data, ["dynamic_variables"], default=[]),
        )
        static_variables = self._nested_get(
            resolved_data,
            ["data", "conditioning", "static", "variables"],
            default=self._nested_get(resolved_data, ["static_variables"], default=[]),
        )
        target_variable = self._nested_get(
            resolved_data,
            ["data", "target", "variable"],
            default=self._nested_get(resolved_data, ["target_variable"], default=None),
        )

        dynamic_count = len(dynamic_variables) if isinstance(dynamic_variables, list) else 0
        static_count = len(static_variables) if isinstance(static_variables, list) else 0
        target_count = 1 if target_variable is not None else 1

        self._set_if_not_none(
            resolved,
            ["model", "in_dynamic_channels"],
            dynamic_count,
        )
        self._set_if_not_none(
            resolved,
            ["model", "in_static_channels"],
            static_count,
        )
        self._set_if_not_none(
            resolved,
            ["model", "out_channels"],
            target_count,
        )

        hr_size = self._nested_get(
            resolved_data,
            ["data", "target", "output_shape"],
            default=self._nested_get(
                resolved_data,
                ["data", "domain", "hr_size"],
                default=self._nested_get(resolved_data, ["output_shape"], default=None),
            ),
        )
        lr_size = self._nested_get(
            resolved_data,
            ["data", "conditioning", "input_shape"],
            default=self._nested_get(
                resolved_data,
                ["data", "domain", "lr_size"],
                default=self._nested_get(resolved_data, ["input_shape"], default=None),
            ),
        )

        hr_hw = self._normalize_shape(hr_size, field_name="resolved_data.hr_size")
        lr_hw = self._normalize_shape(lr_size, field_name="resolved_data.lr_size")

        if hr_hw is not None:
            self._set_if_not_none(resolved, ["model", "spatial", "target_height"], hr_hw[0])
            self._set_if_not_none(resolved, ["model", "spatial", "target_width"], hr_hw[1])
            self._log(f"Resolved model target spatial size from data: {hr_hw}")
        if lr_hw is not None:
            self._set_if_not_none(resolved, ["model", "spatial", "cond_height"], lr_hw[0])
            self._set_if_not_none(resolved, ["model", "spatial", "cond_width"], lr_hw[1])
            self._log(f"Resolved model conditioning spatial size from data: {lr_hw}")

        current_align = self._nested_get(
            resolved,
            ["model", "spatial", "align_cond_to_target"],
            default=None,
        )
        if hr_hw is not None and lr_hw is not None and hr_hw != lr_hw and current_align is None:
            self._set_if_not_none(
                resolved,
                ["model", "spatial", "align_cond_to_target"],
                True,
            )
            self._warn(
                "Conditioning and target grids differ; assuming align_cond_to_target=True "
                f"for model baseline ({lr_hw} -> {hr_hw})."
            )

        return resolved

    def _extract_resolved_interface(
        self,
        *,
        resolved_data: dict[str, Any],
        resolved_model: dict[str, Any] | None,
    ) -> dict[str, Any]:
        target_variable = self._nested_get(
            resolved_data,
            ["data", "target", "variable"],
            default=self._nested_get(resolved_data, ["target_variable"], default=None),
        )
        dynamic_variables = self._nested_get(
            resolved_data,
            ["data", "conditioning", "dynamic", "variables"],
            default=self._nested_get(resolved_data, ["dynamic_variables"], default=[]),
        )
        static_variables = self._nested_get(
            resolved_data,
            ["data", "conditioning", "static", "variables"],
            default=self._nested_get(resolved_data, ["static_variables"], default=[]),
        )
        hr_size = self._nested_get(
            resolved_data,
            ["data", "target", "output_shape"],
            default=self._nested_get(
                resolved_data,
                ["data", "domain", "hr_size"],
                default=self._nested_get(resolved_data, ["output_shape"], default=None),
            ),
        )
        lr_size = self._nested_get(
            resolved_data,
            ["data", "conditioning", "input_shape"],
            default=self._nested_get(
                resolved_data,
                ["data", "domain", "lr_size"],
                default=self._nested_get(resolved_data, ["input_shape"], default=None),
            ),
        )

        model_target_hw = None
        model_cond_hw = None
        model_in_dynamic = None
        model_in_static = None
        model_out = None
        align_cond_to_target = None
        if resolved_model is not None:
            model_target_hw = self._normalize_shape(
                [
                    self._nested_get(resolved_model, ["model", "spatial", "target_height"], default=None),
                    self._nested_get(resolved_model, ["model", "spatial", "target_width"], default=None),
                ],
                field_name="model.spatial.target_hw",
                allow_none_pair=True,
            )
            model_cond_hw = self._normalize_shape(
                [
                    self._nested_get(resolved_model, ["model", "spatial", "cond_height"], default=None),
                    self._nested_get(resolved_model, ["model", "spatial", "cond_width"], default=None),
                ],
                field_name="model.spatial.cond_hw",
                allow_none_pair=True,
            )
            model_in_dynamic = self._nested_get(resolved_model, ["model", "in_dynamic_channels"], default=None)
            model_in_static = self._nested_get(resolved_model, ["model", "in_static_channels"], default=None)
            model_out = self._nested_get(resolved_model, ["model", "out_channels"], default=None)
            align_cond_to_target = self._nested_get(
                resolved_model,
                ["model", "spatial", "align_cond_to_target"],
                default=None,
            )

        return {
            "target_variable": target_variable,
            "dynamic_variables": list(dynamic_variables) if isinstance(dynamic_variables, list) else [],
            "static_variables": list(static_variables) if isinstance(static_variables, list) else [],
            "hr_size": self._normalize_shape(hr_size, field_name="resolved_data.hr_size", allow_none_pair=True),
            "lr_size": self._normalize_shape(lr_size, field_name="resolved_data.lr_size", allow_none_pair=True),
            "model_target_hw": model_target_hw,
            "model_cond_hw": model_cond_hw,
            "model_in_dynamic": model_in_dynamic,
            "model_in_static": model_in_static,
            "model_out": model_out,
            "align_cond_to_target": align_cond_to_target,
        }

    def _validate_resolved_data_config(
        self,
        resolved_data: dict[str, Any],
    ) -> None:
        interface = self._extract_resolved_interface(
            resolved_data=resolved_data,
            resolved_model=None,
        )
        target_variable = interface["target_variable"]
        dynamic_variables = interface["dynamic_variables"]
        static_variables = interface["static_variables"]
        hr_size = interface["hr_size"]
        lr_size = interface["lr_size"]

        if target_variable is None or str(target_variable).strip() == "":
            raise ValueError("Resolved data config is missing 'data.target.variable'")
        if len(dynamic_variables) == 0:
            self._warn("Resolved data config has zero dynamic conditioning variables")
        if len(set(dynamic_variables)) != len(dynamic_variables):
            raise ValueError("Resolved data config contains duplicate dynamic variables")
        if len(set(static_variables)) != len(static_variables):
            raise ValueError("Resolved data config contains duplicate static variables")
        if hr_size is not None and (hr_size[0] <= 0 or hr_size[1] <= 0):
            raise ValueError(f"Resolved HR size must be positive, got {hr_size}")
        if lr_size is not None and (lr_size[0] <= 0 or lr_size[1] <= 0):
            raise ValueError(f"Resolved LR size must be positive, got {lr_size}")

        self._log(f"Resolved target variable: {target_variable}")
        self._log(f"Resolved dynamic variables ({len(dynamic_variables)}): {dynamic_variables}")
        self._log(f"Resolved static variables ({len(static_variables)}): {static_variables}")
        if hr_size is not None:
            self._log(f"Resolved HR/output size: {hr_size}")
        if lr_size is not None:
            self._log(f"Resolved LR/input size: {lr_size}")

    def _validate_data_selection_against_base(
        self,
        *,
        base_data: dict[str, Any] | None,
        resolved_data: dict[str, Any],
    ) -> None:
        if base_data is None:
            return

        resolved_interface = self._extract_resolved_interface(
            resolved_data=resolved_data,
            resolved_model=None,
        )
        base_interface = self._extract_resolved_interface(
            resolved_data=base_data,
            resolved_model=None,
        )

        base_target = base_interface["target_variable"]
        resolved_target = resolved_interface["target_variable"]
        if base_target is not None and resolved_target is not None and base_target != resolved_target:
            self._warn(
                f"Resolved target variable '{resolved_target}' differs from base dataset target '{base_target}'. "
                "Ensure this dataset truly supports target override for the selected variable."
            )

        base_dynamic = set(base_interface["dynamic_variables"])
        resolved_dynamic = set(resolved_interface["dynamic_variables"])
        if len(base_dynamic) > 0 and not resolved_dynamic.issubset(base_dynamic):
            missing = sorted(resolved_dynamic.difference(base_dynamic))
            raise ValueError(
                "Resolved dynamic variable selection is not a subset of the base dataset config: "
                f"{missing}"
            )

        base_static = set(base_interface["static_variables"])
        resolved_static = set(resolved_interface["static_variables"])
        if len(base_static) > 0 and not resolved_static.issubset(base_static):
            missing = sorted(resolved_static.difference(base_static))
            raise ValueError(
                "Resolved static variable selection is not a subset of the base dataset config: "
                f"{missing}"
            )

        self._log("Resolved data selections are compatible with the base dataset config")

    def _validate_model_data_alignment(
        self,
        *,
        resolved_data: dict[str, Any],
        resolved_model: dict[str, Any] | None,
    ) -> None:
        if resolved_model is None:
            self._warn("No resolved model config available; skipping model/data alignment checks")
            return

        interface = self._extract_resolved_interface(
            resolved_data=resolved_data,
            resolved_model=resolved_model,
        )

        dynamic_count = len(interface["dynamic_variables"])
        static_count = len(interface["static_variables"])
        target_count = 1 if interface["target_variable"] is not None else 1

        if interface["model_in_dynamic"] != dynamic_count:
            raise ValueError(
                "Resolved model/data mismatch: "
                f"model.in_dynamic_channels={interface['model_in_dynamic']} but "
                f"resolved data selects {dynamic_count} dynamic variables"
            )
        if interface["model_in_static"] != static_count:
            raise ValueError(
                "Resolved model/data mismatch: "
                f"model.in_static_channels={interface['model_in_static']} but "
                f"resolved data selects {static_count} static variables"
            )
        if interface["model_out"] != target_count:
            raise ValueError(
                "Resolved model/data mismatch: "
                f"model.out_channels={interface['model_out']} but target count resolves to {target_count}"
            )
        if interface["hr_size"] is not None and interface["model_target_hw"] != interface["hr_size"]:
            raise ValueError(
                "Resolved model/data mismatch: "
                f"model target size {interface['model_target_hw']} vs resolved HR size {interface['hr_size']}"
            )
        if interface["lr_size"] is not None and interface["model_cond_hw"] != interface["lr_size"]:
            raise ValueError(
                "Resolved model/data mismatch: "
                f"model conditioning size {interface['model_cond_hw']} vs resolved LR size {interface['lr_size']}"
            )
        if (
            interface["hr_size"] is not None
            and interface["lr_size"] is not None
            and interface["hr_size"] != interface["lr_size"]
            and interface["align_cond_to_target"] is not True
        ):
            raise ValueError(
                "Resolved model/data mismatch: target and conditioning grids differ "
                f"({interface['lr_size']} -> {interface['hr_size']}) but "
                "model.spatial.align_cond_to_target is not True"
            )

        self._log("Resolved model/data interface is aligned")

    def _validate_stage_run_configs(
        self,
        *,
        training_run_cfg: dict[str, Any],
        generation_run_cfg: dict[str, Any],
        evaluation_run_cfg: dict[str, Any],
    ) -> None:
        training_paths = self._nested_get(training_run_cfg, ["training", "configs"], default={})
        generation_paths = self._nested_get(generation_run_cfg, ["generation_run", "paths"], default={})
        evaluation_paths = self._nested_get(evaluation_run_cfg, ["evaluation_run", "paths"], default={})

        if not self._nested_get(training_run_cfg, ["training", "run", "name"], default=None):
            raise ValueError("Resolved training run config is missing training.run.name")
        if not training_paths.get("model_config"):
            raise ValueError("Resolved training run config is missing training.configs.model_config")
        if not training_paths.get("dataset_config"):
            raise ValueError("Resolved training run config is missing training.configs.dataset_config")
        if not generation_paths.get("model_config"):
            raise ValueError("Resolved generation run config is missing generation_run.paths.model_config")
        if not generation_paths.get("dataset_config"):
            raise ValueError("Resolved generation run config is missing generation_run.paths.dataset_config")
        if not generation_paths.get("checkpoint_path"):
            raise ValueError("Resolved generation run config is missing generation_run.paths.checkpoint_path")
        if not evaluation_paths.get("generation_output_dir"):
            raise ValueError("Resolved evaluation run config is missing evaluation_run.paths.generation_output_dir")
        if not evaluation_paths.get("dataset_config"):
            raise ValueError("Resolved evaluation run config is missing evaluation_run.paths.dataset_config")

        self._log("Resolved stage run configs contain the required paths")

    def _validate_stage_dependencies(self) -> None:
        stages = self.cfg.stages
        if not stages.training and not stages.generation and not stages.evaluation:
            raise ValueError("Experiment enables no stages; at least one of training/generation/evaluation must be True")
        if stages.generation and not stages.training:
            self._warn(
                "Generation stage is enabled while training stage is disabled. "
                "Ensure the compiled checkpoint path points to an existing trained model."
            )
        if stages.evaluation and not stages.generation:
            self._warn(
                "Evaluation stage is enabled while generation stage is disabled. "
                "Ensure the expected generation outputs already exist."
            )
        self._log(
            f"Stage selection looks valid: training={stages.training}, "
            f"generation={stages.generation}, evaluation={stages.evaluation}"
        )

    def _normalize_shape(
        self,
        raw: Any,
        *,
        field_name: str,
        allow_none_pair: bool = False,
    ) -> tuple[int, int] | None:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            if len(raw) != 2:
                raise ValueError(f"Expected {field_name} to have length 2, got {raw}")
            if allow_none_pair and raw[0] is None and raw[1] is None:
                return None
            if raw[0] is None or raw[1] is None:
                raise ValueError(f"Expected {field_name} to be fully specified, got {raw}")
            return (int(raw[0]), int(raw[1]))
        raise ValueError(f"Expected {field_name} to be a list/tuple of length 2, got {type(raw)}")

    def _log(self, message: str) -> None:
        self._logger.info(message)

    def _warn(self, message: str) -> None:
        self._logger.warning(message)

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

        `data.overrides` are interpreted relative to the adapter-facing top-level
        `data:` block. For backward compatibility, a legacy override payload that
        already includes a top-level `data` key is also accepted.

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
                "dynamic": {
                    "variables": self.cfg.data.conditioning.dynamic_variables,
                },
                "static": {
                    "variables": self.cfg.data.conditioning.static_variables,
                },
                "input_shape": self.cfg.data.conditioning.input_shape,
            },
            "domain": {
                "hr_size": self.cfg.data.domain.hr_size,
                "lr_size": self.cfg.data.domain.lr_size,
                "large_domain": self.cfg.data.domain.large_domain,
            },
            "split": {
                "train": self.cfg.data.split.train,
                "val": self.cfg.data.split.val,
                "test": self.cfg.data.split.test,
                "statistics": {
                    "train": self.cfg.data.split.statistics.train,
                    "val": self.cfg.data.split.statistics.val,
                    "test": self.cfg.data.split.statistics.test,
                },
            },
        }

        # Always keep an explicit experiment-owned data block.
        resolved["experiment_data"] = experiment_data_block

        # Write the experiment selections directly into the adapter-facing
        # nested data config structure.
        self._set_if_not_none(
            resolved,
            ["data", "target", "variable"],
            self.cfg.data.target.variable,
        )
        self._set_if_not_none(
            resolved,
            ["data", "conditioning", "dynamic", "variables"],
            self.cfg.data.conditioning.dynamic_variables,
        )
        self._set_if_not_none(
            resolved,
            ["data", "conditioning", "static", "variables"],
            self.cfg.data.conditioning.static_variables,
        )
        self._set_if_not_none(
            resolved,
            ["data", "split", "train"],
            self.cfg.data.split.train,
        )
        self._set_if_not_none(
            resolved,
            ["data", "split", "val"],
            self.cfg.data.split.val,
        )
        self._set_if_not_none(
            resolved,
            ["data", "split", "test"],
            self.cfg.data.split.test,
        )
        self._set_if_not_none(
            resolved,
            ["data", "split", "statistics", "train"],
            self.cfg.data.split.statistics.train,
        )
        self._set_if_not_none(
            resolved,
            ["data", "split", "statistics", "val"],
            self.cfg.data.split.statistics.val,
        )
        self._set_if_not_none(
            resolved,
            ["data", "split", "statistics", "test"],
            self.cfg.data.split.statistics.test,
        )

        # Keep a few generic convenience fields for downstream consumers that
        # may still read them directly.
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
        self._set_if_not_none(resolved, ["split_train"], self.cfg.data.split.train)
        self._set_if_not_none(resolved, ["split_val"], self.cfg.data.split.val)
        self._set_if_not_none(resolved, ["split_test"], self.cfg.data.split.test)
        self._set_if_not_none(
            resolved,
            ["statistics_split_map"],
            {
                "train": self.cfg.data.split.statistics.train,
                "val": self.cfg.data.split.statistics.val,
                "test": self.cfg.data.split.statistics.test,
            },
        )
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

        data_overrides = copy.deepcopy(self.cfg.data.overrides)
        if isinstance(data_overrides, dict) and len(data_overrides) > 0:
            resolved.setdefault("data", {})
            if (
                "data" in data_overrides
                and isinstance(data_overrides["data"], dict)
                and len(data_overrides) == 1
            ):
                # Backward-compatible legacy form:
                # data.overrides = {"data": {...}}
                self._deep_update(resolved["data"], data_overrides["data"])
            else:
                # Preferred form:
                # data.overrides = {...}  (interpreted relative to top-level data:)
                self._deep_update(resolved["data"], data_overrides)
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

        training_overrides = copy.deepcopy(self.cfg.training.overrides)

        # Legacy guard: model overrides belong in the top-level experiment
        # `model` section and are compiled into `model_resolved.yaml`. Do not
        # also inject them into the training run config, where they would be
        # misleading because Trainer builds from `configs.model_config`.
        training_overrides.pop("model", None)

        self._deep_update(config, training_overrides)

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

        default_generation_output_dir = self._nested_get(
            generation_run_cfg,
            ["generation_run", "outputs", "output_dir"],
            default=str(experiment_root / "generation"),
        )

        config: dict[str, Any] = {
            "evaluation_run": {
                "run_name": run_name,
                "paths": {
                    "generation_output_dir": default_generation_output_dir,
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

        if not run_cfg["paths"].get("generation_output_dir"):
            run_cfg["paths"]["generation_output_dir"] = str(
                Path(default_generation_output_dir).resolve()
            )
        if not run_cfg["paths"].get("dataset_config"):
            run_cfg["paths"]["dataset_config"] = (
                str(data_config_path.resolve()) if data_config_path is not None else None
            )
        if not run_cfg["paths"].get("evaluation_config"):
            run_cfg["paths"]["evaluation_config"] = (
                str(evaluation_base_config_path.resolve())
                if evaluation_base_config_path is not None
                else None
            )
        if not run_cfg["paths"].get("training_config"):
            run_cfg["paths"]["training_config"] = str(training_run_config_path.resolve())
        if not run_cfg["outputs"].get("output_dir"):
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
    def _compile_generation_base_config(
        self,
        *,
        base_generation: dict[str, Any] | None,
        base_sampler: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Merge generation config layers:

        1) sampler_base.yaml (algorithm defaults)
        2) generation_base.yaml (pipeline behavior)
        3) experiment overrides (highest priority)
        """

        resolved: dict[str, Any] = {}

        # 1) sampler defaults
        if base_sampler is not None:
            resolved = copy.deepcopy(base_sampler)

        # 2) generation base
        if base_generation is not None:
            self._deep_update(resolved, copy.deepcopy(base_generation))

        # 3) experiment overrides
        if isinstance(self.cfg.generation.overrides, dict):
            self._deep_update(resolved, copy.deepcopy(self.cfg.generation.overrides))

        # --- Validation ---
        sampler = self._nested_get(resolved, ["generation", "sampler"], default={})

        required = ["num_steps", "sigma_min", "sigma_max", "rho"]
        for key in required:
            if sampler.get(key) is None:
                raise ValueError(
                    f"Sampler parameter '{key}' is None after resolution. "
                    "Check sampler_base.yaml and generation_base.yaml."
                )

        self._log(
            "Resolved sampler parameters: "
            f"num_steps={sampler.get('num_steps')}, "
            f"sigma_min={sampler.get('sigma_min')}, "
            f"sigma_max={sampler.get('sigma_max')}, "
            f"rho={sampler.get('rho')}, "
            f"S_churn={sampler.get('S_churn')}, "
            f"S_min={sampler.get('S_min')}, "
            f"S_max={sampler.get('S_max')}, "
            f"S_noise={sampler.get('S_noise')}"
        )

        return resolved
