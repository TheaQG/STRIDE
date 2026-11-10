"""
Core generation orchestration for STRIDE.

This module is the generation-side analogue of `trainer.py`: it decides what to
load, which split to run on, which checkpoint weights to use, how many samples
to generate per case, and where outputs should be written.

Current scope
-------------
- load generation-run config from YAML
- infer checkpoint paths from a training config or use an explicit checkpoint
- resolve all referenced paths relative to the single generation-run config
- rebuild the model and load checkpoint weights
- build the requested dataset split and dataloader
- generate one or more samples per case
- save generated outputs as compressed `.npz` or NorCP temporal NetCDF stacks
- optionally save simple per-case metadata

Intentional non-goals for this first version
-------------------------------------------
- sophisticated distributed generation
- PMM computation implementation details
- rich plotting during full generation runs
- adapter-agnostic plugin discovery

The first milestone for this file is a clean, trustworthy path from:
    best checkpoint -> test split -> generated outputs on disk
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TYPE_CHECKING
import logging
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np

logger = logging.getLogger(__name__)

from stride_core.configs.adapter_config import AdapterConfig
from stride_core.generation.generation_utils import (
    build_preview_payload,
    generate_batch,
    inverse_transform_target_tensor,
    move_batch_to_device,
)
from stride_core.generation.output_saving import (
    case_output_dir as build_case_output_dir,
    save_aggregate,
    save_member,
    save_member_bundle,
    save_run_metadata,
)
from stride_core.generation.pmm import pmm_from_ensemble
from stride_core.models.build_model import build_model
from stride_core.training.checkpoints import _strip_parallel_prefix
from stride_core.training.data import build_dataset_adapter, stride_collate_fn

if TYPE_CHECKING:
    from stride_core.generation.netcdf_output import NetCDFWriter


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRunPaths:
    dataset_config_path: Path
    model_config_path: Path
    generation_config_path: Path
    training_config_path: Path | None


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    mode: str
    path: Path | None
    use_ema_weights: bool


@dataclass(frozen=True)
class GenerationDataConfig:
    split: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool


@dataclass(frozen=True)
class GenerationSamplingConfig:
    ensemble_size: int
    use_fixed_seed: bool
    base_seed: int


@dataclass(frozen=True)
class GenerationLimitsConfig:
    max_cases: int | None


@dataclass(frozen=True)
class GenerationOutputConfig:
    save_members: bool
    save_metadata: bool
    save_physical: bool
    save_ensemble_mean: bool
    save_pmm: bool
    save_plots: bool
    output_format: str
    storage_mode: str


@dataclass(frozen=True)
class GenerationRunConfig:
    run_name: str
    output_dir: Path
    device_accelerator: str
    paths: GenerationRunPaths
    checkpoint: CheckpointSelectionConfig
    data: GenerationDataConfig
    sampling: GenerationSamplingConfig
    limits: GenerationLimitsConfig
    outputs: GenerationOutputConfig
    config_path: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GenerationRunConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Generation run config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected generation run YAML to decode into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg, config_path=path)

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        *,
        config_path: str | Path,
    ) -> "GenerationRunConfig":
        run_cfg = cfg.get("generation_run", cfg)
        if not isinstance(run_cfg, dict):
            raise ValueError("Expected generation run config to be a dict")

        paths_cfg = run_cfg.get("paths", run_cfg.get("configs", {}))
        if not isinstance(paths_cfg, dict):
            raise ValueError(
                "Expected 'generation_run.paths' or 'generation_run.configs' to be a dict"
            )

        dataset_config_path = cls._resolve_path(
            paths_cfg["dataset_config"],
            config_path=config_path,
        )
        model_config_path = cls._resolve_path(
            paths_cfg["model_config"],
            config_path=config_path,
        )
        generation_config_path = cls._resolve_path(
            paths_cfg["generation_config"],
            config_path=config_path,
        )
        training_config_raw = paths_cfg.get("training_config", None)
        training_config_path = (
            None
            if training_config_raw is None
            else cls._resolve_path(training_config_raw, config_path=config_path)
        )

        with open(generation_config_path, "r", encoding="utf-8") as f:
            generation_base_cfg = yaml.safe_load(f)
        if not isinstance(generation_base_cfg, dict):
            raise ValueError(
                f"Expected generation base config YAML to decode into a dict, got {type(generation_base_cfg)}"
            )

        base_generation_cfg = generation_base_cfg.get("generation", generation_base_cfg)
        if not isinstance(base_generation_cfg, dict):
            raise ValueError(
                "Expected top-level 'generation' section in generation base config to be a dict"
            )

        checkpoint_cfg = run_cfg.get("checkpoint", base_generation_cfg.get("checkpoint", {}))
        data_cfg = base_generation_cfg.get("data", {}).copy()
        data_cfg.update(run_cfg.get("data", {}))
        sampling_cfg = base_generation_cfg.get("sampling", {}).copy()
        sampling_cfg.update(run_cfg.get("sampling", {}))
        limits_cfg = base_generation_cfg.get("limits", {}).copy()
        limits_cfg.update(run_cfg.get("limits", {}))
        outputs_cfg = base_generation_cfg.get("outputs", {}).copy()
        outputs_cfg.update(run_cfg.get("outputs", {}))
        device_cfg = base_generation_cfg.get("device", {}).copy()
        device_cfg.update(run_cfg.get("device", {}))

        if not isinstance(checkpoint_cfg, dict):
            raise ValueError("Expected generation checkpoint config to be a dict")
        if not isinstance(data_cfg, dict):
            raise ValueError("Expected generation data config to be a dict")
        if not isinstance(sampling_cfg, dict):
            raise ValueError("Expected generation sampling config to be a dict")
        if not isinstance(limits_cfg, dict):
            raise ValueError("Expected generation limits config to be a dict")
        if not isinstance(outputs_cfg, dict):
            raise ValueError("Expected generation outputs config to be a dict")
        if not isinstance(device_cfg, dict):
            raise ValueError("Expected generation device config to be a dict")

        checkpoint_mode = str(checkpoint_cfg.get("mode", "best"))
        checkpoint_path_raw = checkpoint_cfg.get("path", paths_cfg.get("checkpoint_path", None))
        checkpoint_path = (
            None
            if checkpoint_path_raw is None
            else cls._resolve_path(checkpoint_path_raw, config_path=config_path)
        )
        if checkpoint_mode not in {"best", "latest", "explicit"}:
            raise ValueError(
                "generation_run.checkpoint.mode must be one of: "
                f"best, latest, explicit. Got {checkpoint_mode!r}."
            )
        if checkpoint_mode == "explicit" and checkpoint_path is None:
            raise ValueError(
                "generation_run.checkpoint.path must be set when mode='explicit'"
            )

        batch_size = int(data_cfg.get("batch_size", 1))
        if batch_size <= 0:
            raise ValueError(f"generation_run.data.batch_size must be > 0, got {batch_size}")

        ensemble_size = int(sampling_cfg.get("ensemble_size", 1))
        if ensemble_size <= 0:
            raise ValueError(
                f"generation_run.sampling.ensemble_size must be > 0, got {ensemble_size}"
            )

        max_cases_raw = limits_cfg.get("max_cases", None)
        max_cases = None if max_cases_raw is None else int(max_cases_raw)
        if max_cases is not None and max_cases <= 0:
            raise ValueError(
                f"generation_run.limits.max_cases must be > 0 when set, got {max_cases}"
            )

        output_format = str(outputs_cfg.get("output_format", "npz"))
        if output_format not in {"npz", "netcdf"}:
            raise ValueError(
                f"output_format must be 'npz' or 'netcdf', got {output_format!r}"
            )

        storage_mode = str(outputs_cfg.get("storage_mode", "per_member"))
        if output_format == "netcdf":
            # This first NetCDF mode writes all three products in physical units.
            for key in ("save_physical", "save_members", "save_ensemble_mean", "save_pmm"):
                if not bool(outputs_cfg.get(key, True)):
                    raise ValueError(f"NetCDF output requires outputs.{key}=true")
            if bool(data_cfg.get("shuffle", False)):
                raise ValueError("NetCDF output requires data.shuffle=false (chronological order)")
            storage_mode = "temporal_stack"
        elif storage_mode not in {"per_member", "per_member_bundle"}:
            raise ValueError(
                "generation_run.outputs.storage_mode must be one of: "
                f"'per_member', 'per_member_bundle'. Got {storage_mode!r}."
            )

        return cls(
            run_name=str(run_cfg.get("run_name", run_cfg.get("name", "generation_run"))),
            output_dir=cls._resolve_path(
                run_cfg.get(
                    "output_dir",
                    outputs_cfg.get("output_dir", "runs/generation_run"),
                ),
                config_path=config_path,
            ),
            device_accelerator=str(device_cfg.get("accelerator", "auto")),
            paths=GenerationRunPaths(
                dataset_config_path=dataset_config_path,
                model_config_path=model_config_path,
                generation_config_path=generation_config_path,
                training_config_path=training_config_path,
            ),
            checkpoint=CheckpointSelectionConfig(
                mode=checkpoint_mode,
                path=checkpoint_path,
                use_ema_weights=bool(checkpoint_cfg.get("use_ema_weights", True)),
            ),
            data=GenerationDataConfig(
                split=str(data_cfg.get("split", "test")),
                batch_size=batch_size,
                num_workers=int(data_cfg.get("num_workers", 0)),
                pin_memory=bool(data_cfg.get("pin_memory", False)),
                shuffle=bool(data_cfg.get("shuffle", False)),
            ),
            sampling=GenerationSamplingConfig(
                ensemble_size=ensemble_size,
                use_fixed_seed=bool(sampling_cfg.get("use_fixed_seed", False)),
                base_seed=int(sampling_cfg.get("base_seed", 42)),
            ),
            limits=GenerationLimitsConfig(
                max_cases=max_cases,
            ),
            outputs=GenerationOutputConfig(
                save_members=bool(outputs_cfg.get("save_members", True)),
                save_metadata=bool(outputs_cfg.get("save_metadata", True)),
                save_physical=bool(outputs_cfg.get("save_physical", True)),
                save_ensemble_mean=bool(outputs_cfg.get("save_ensemble_mean", output_format == "netcdf")),
                save_pmm=bool(outputs_cfg.get("save_pmm", output_format == "netcdf")),
                save_plots=bool(outputs_cfg.get("save_plots", False)),
                output_format=output_format,
                storage_mode=storage_mode,
            ),
            config_path=Path(config_path),
        )
    @staticmethod
    def _resolve_path(raw_path: str | Path, *, config_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()

        config_path = Path(config_path).resolve()
        repo_root = Path(__file__).resolve().parents[2]

        config_relative = (config_path.parent / path).resolve()
        if config_relative.exists():
            return config_relative

        repo_relative = (repo_root / path).resolve()
        if repo_relative.exists():
            return repo_relative

        if len(path.parts) > 0 and path.parts[0] in {
            "runs",
            "configs",
            "data_adapters",
            "stride_core",
        }:
            return repo_relative

        return config_relative

# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------


class Generator:
    """
    Config-driven generation pipeline.
    """

    def __init__(self, config: GenerationRunConfig | str | Path) -> None:
        self.cfg = (
            config if isinstance(config, GenerationRunConfig) else GenerationRunConfig.from_yaml(config)
        )

        self.output_dir = self.cfg.output_dir
        self.samples_dir = self.output_dir / "samples"
        self.metadata_path = self.output_dir / "generation_metadata.json"

        self.device = self._resolve_device(self.cfg.device_accelerator)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.outputs.output_format == "npz":
            self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.adapter_cfg = AdapterConfig.from_yaml(self.cfg.paths.dataset_config_path)
        self.adapter = self._build_adapter()
        self.dataset = self._build_dataset()
        self.loader = self._build_dataloader()

        self.model = build_model(self.cfg.paths.model_config_path).to(self.device)
        self.model.eval()

        self.checkpoint_path = self._resolve_checkpoint_path()
        self.checkpoint_payload = self._load_checkpoint_into_model(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self) -> None:
        self._print_run_summary()
        output_context = (
            self._create_netcdf_writer()
            if self.cfg.outputs.output_format == "netcdf" else nullcontext()
        )
        # The NetCDF context publishes only after all requested cases succeed.
        with output_context as netcdf_writer:
            self._save_run_metadata()
            generated_case_count = 0
            for batch_idx, batch in enumerate(self.loader):
                if (
                    self.cfg.limits.max_cases is not None
                    and generated_case_count >= self.cfg.limits.max_cases
                ):
                    break

                batch = move_batch_to_device(batch, self.device)
                processed_cases = self._generate_for_batch(
                    batch,
                    batch_idx=batch_idx,
                    remaining_cases=None
                    if self.cfg.limits.max_cases is None
                    else self.cfg.limits.max_cases - generated_case_count,
                    netcdf_writer=netcdf_writer,
                    case_offset=generated_case_count,
                )
                generated_case_count += processed_cases

        if self.cfg.outputs.save_plots:
            logger.info(
                "Full-run generation plotting was requested, but plotting is not implemented yet in this first generator version."
            )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _create_netcdf_writer(self) -> NetCDFWriter:
        """Open one temporal stack for this NorCP run; never create NPZ output."""
        from data_adapters.norcp.adapter import NorCPDataset
        from stride_core.generation.netcdf_output import NetCDFWriter

        if not isinstance(self.dataset, NorCPDataset):
            raise ValueError("NetCDF generation currently supports the NorCP adapter only")
        if self.samples_dir.exists():
            raise FileExistsError("Use a new output directory for NetCDF; an existing samples/ directory was found")
        case_count = len(self.dataset)
        if self.cfg.limits.max_cases is not None:
            case_count = min(case_count, self.cfg.limits.max_cases)
        if case_count == 0:
            raise ValueError("Cannot generate a NetCDF stack from an empty dataset")
        metadata = self.dataset.get_output_metadata(0)
        self._netcdf_reference_metadata = metadata
        return NetCDFWriter(
            self.output_dir / "netcdf",
            variable_name=metadata.variable_name, units=metadata.units, grid=metadata.grid,
            variable_attributes=metadata.variable_attributes,
            calendar=metadata.calendar, with_time_bounds=metadata.time_bounds is not None,
            member_count=self.cfg.sampling.ensemble_size, expected_timesteps=case_count,
            global_attributes={
                "run_name": self.cfg.run_name,
                "checkpoint_path": str(self.checkpoint_path),
                "use_ema_weights": self.cfg.checkpoint.use_ema_weights,
                "weight_source": self.weight_source,
                "dataset_config": str(self.cfg.paths.dataset_config_path),
                "generation_run_config": str(self.cfg.config_path),
                "split": self.cfg.data.split,
                "stride_time_reference": "source target representative time",
                "stride_alignment_offset_hours": float(
                    self.dataset.target_time_offsets.get(metadata.variable_name, 0.0)
                ),
            },
        )

    def _resolve_device(self, accelerator: str) -> torch.device:
        if accelerator == "cpu":
            return torch.device("cpu")
        if accelerator == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            return torch.device("cuda")
        if accelerator == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raise ValueError(
            f"Unsupported generation_run.device.accelerator={accelerator!r}. "
            "Expected one of: auto, cpu, cuda."
        )

    def _build_adapter(self) -> Any:
        adapter_cfg = replace(self.adapter_cfg, split_name=self.cfg.data.split)
        return build_dataset_adapter(adapter_cfg)

    def _build_dataset(self) -> Any:
        adapter = self.adapter
        build_dataset = getattr(adapter, "build_dataset", None)
        if not callable(build_dataset):
            raise AttributeError(
                "Dataset adapter must provide a callable build_dataset() method"
            )
        return build_dataset()

    def _build_dataloader(self) -> DataLoader[dict[str, Any]]:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=self.cfg.data.shuffle,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=False,
            collate_fn=stride_collate_fn,
        )

    def _resolve_checkpoint_path(self) -> Path:
        if self.cfg.checkpoint.mode == "explicit":
            assert self.cfg.checkpoint.path is not None
            path = self.cfg.checkpoint.path
            if not path.exists():
                raise FileNotFoundError(f"Explicit checkpoint path does not exist: {path}")
            return path

        training_config_path = self.cfg.paths.training_config_path
        if training_config_path is None:
            raise ValueError(
                "generation_run.configs.training_config must be set when checkpoint.mode "
                "is not 'explicit'"
            )
        if not training_config_path.exists():
            raise FileNotFoundError(
                f"Training config for checkpoint inference does not exist: {training_config_path}"
            )

        with open(training_config_path, "r", encoding="utf-8") as f:
            training_cfg = yaml.safe_load(f)
        if not isinstance(training_cfg, dict):
            raise ValueError(
                f"Expected training config YAML to decode into a dict, got {type(training_cfg)}"
            )

        training_root = training_cfg.get("training", training_cfg)
        if not isinstance(training_root, dict):
            raise ValueError("Expected 'training' section in training config to be a dict")

        run_cfg = training_root.get("run", {})
        if not isinstance(run_cfg, dict):
            raise ValueError("Expected 'training.run' section to be a dict")

        training_output_dir_raw = run_cfg.get("output_dir", "runs/train_run")
        if not isinstance(training_output_dir_raw, str) or training_output_dir_raw.strip() == "":
            raise ValueError(
                "Expected 'training.run.output_dir' in training config to be a non-empty string"
            )
        training_output_dir = self.cfg._resolve_path(
            training_output_dir_raw,
            config_path=training_config_path,
        )
        checkpoint_dir = training_output_dir / "checkpoints"

        if self.cfg.checkpoint.mode == "best":
            path = checkpoint_dir / "checkpoint_best.pt"
        elif self.cfg.checkpoint.mode == "latest":
            path = checkpoint_dir / "checkpoint_latest.pt"
        else:
            raise ValueError(f"Unsupported checkpoint mode {self.cfg.checkpoint.mode!r}")

        if not path.exists():
            raise FileNotFoundError(
                f"Could not resolve requested checkpoint at: {path}"
            )
        return path

    @staticmethod
    def _ema_tensor_copies(
        shadows: Any, targets: dict[str, torch.Tensor], label: str, *, floating_only: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(shadows, dict) or not all(isinstance(name, str) for name in shadows):
            raise ValueError(f"Invalid {label}: expected a dictionary keyed by tensor name")

        copies = []
        for name, shadow in _strip_parallel_prefix(shadows).items():
            if name not in targets:
                raise ValueError(f"Invalid {label}: unknown tensor {name!r}")
            target = targets[name]
            if not isinstance(shadow, torch.Tensor):
                raise ValueError(f"Invalid {label}: {name!r} must be a tensor")
            if floating_only and not shadow.is_floating_point():
                raise ValueError(f"Invalid {label}: {name!r} has non-floating dtype {shadow.dtype}")
            if floating_only and (shadow.is_meta or shadow.layout != target.layout):
                raise ValueError(f"Invalid {label}: {name!r} must have storage and layout {target.layout}")
            if shadow.shape != target.shape:
                raise ValueError(
                    f"Invalid {label}: {name!r} has shape {tuple(shadow.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            copies.append((target, shadow))
        return copies

    @staticmethod
    def _raw_weight_state(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("model_state_dict", "model"):
            state = checkpoint.get(key)
            if (
                isinstance(state, dict)
                and all(isinstance(name, str) for name in state)
                and any(isinstance(value, torch.Tensor) for value in state.values())
            ):
                return _strip_parallel_prefix(state)
        return None

    def _load_flat_ema(self, state: dict[str, Any], label: str) -> None:
        if not all(isinstance(name, str) for name in state):
            raise ValueError(f"Invalid {label}: expected string state keys")
        state = _strip_parallel_prefix(state)
        expected = self.model.state_dict()
        missing = expected.keys() - state.keys()
        unexpected = state.keys() - expected.keys()
        if missing or unexpected:
            raise ValueError(
                f"Invalid {label}: missing keys {sorted(missing)}, "
                f"unexpected keys {sorted(unexpected)}"
            )
        tensors = {name: value for name, value in expected.items() if isinstance(value, torch.Tensor)}
        self._ema_tensor_copies({name: state[name] for name in tensors}, tensors, label)
        self.model.load_state_dict(state)

    def _load_ema_shadows(
        self, ema: dict[str, Any], label: str, raw_state: dict[str, Any] | None,
    ) -> None:
        params: dict[str, torch.Tensor] = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        shadows = ema.get("shadow_params")
        current_format = label == "ema_state_dict"
        if current_format:
            track_buffers = ema.get("track_buffers")
            if not isinstance(track_buffers, bool):
                raise ValueError(f"Invalid {label}.track_buffers: expected a boolean")
            params = {
                name: param for name, param in params.items()
                if param.requires_grad and param.is_floating_point()
            }
            buffers = {
                name: buffer for name, buffer in buffers.items()
                if track_buffers and buffer.is_floating_point()
            }
            shadow_buffers = ema.get("shadow_buffers")
        else:
            shadow_buffers = ema.get("shadow_buffers", {})
            if isinstance(shadows, dict) and all(isinstance(name, str) for name in shadows):
                # A complete legacy state takes precedence even when raw weights are present.
                state = _strip_parallel_prefix(shadows)
                if state.keys() == self.model.state_dict().keys():
                    copies = self._ema_tensor_copies(shadow_buffers, buffers, f"{label}.shadow_buffers")
                    self._load_flat_ema(shadows, f"{label}.shadow_params")
                    for target, shadow in copies:
                        target.copy_(shadow)
                    return
            params.update(buffers)

        # Validate both maps before loading the raw state or changing any tensors.
        copies = []
        for key, shadow_map, targets, names_key in (
            ("shadow_params", shadows, params, "tracked_param_names"),
            ("shadow_buffers", shadow_buffers, buffers, "tracked_buffer_names"),
        ):
            copies.extend(self._ema_tensor_copies(
                shadow_map, targets, f"{label}.{key}", floating_only=current_format,
            ))
            if current_format or names_key in ema:
                names = ema.get(names_key)
                if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
                    raise ValueError(f"Invalid {label}.{names_key}: expected a list of tensor names")
                normalized_names = _strip_parallel_prefix(dict.fromkeys(names))
                if (
                    len(names) != len(normalized_names)
                    or normalized_names.keys() != _strip_parallel_prefix(shadow_map).keys() # type: ignore
                ):
                    raise ValueError(f"Invalid {label}.{key}: tensor names do not match {names_key}")
        if not copies:
            raise ValueError(f"Invalid {label}: no EMA shadow tensors")
        if raw_state is None:
            raise KeyError(f"{label} shadows require 'model_state_dict' or legacy 'model'")
        self.model.load_state_dict(raw_state)
        # Copy into live tensors, including non-persistent buffers, like EMA.copy_to.
        for target, shadow in copies:
            target.copy_(shadow)

    @torch.no_grad()
    def _load_checkpoint_into_model(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Expected checkpoint to load into a dict, got {type(checkpoint)}"
            )

        ema_key = "ema_state_dict"
        ema_state = None
        if self.cfg.checkpoint.use_ema_weights:
            ema_state = checkpoint.get(ema_key)
            if ema_state is None:
                ema_key = "ema"
                ema_state = checkpoint.get(ema_key)

        raw_state = self._raw_weight_state(checkpoint)
        if ema_state is None:
            if raw_state is None:
                raise KeyError("Checkpoint has no loadable 'model_state_dict' or legacy 'model'")
            self.model.load_state_dict(raw_state)
            self.weight_source = "raw"
            if self.cfg.checkpoint.use_ema_weights:
                logger.warning(
                    "EMA weights requested but absent from %s. Using raw weights.", checkpoint_path,
                )
        else:
            if not isinstance(ema_state, dict) or not ema_state:
                raise ValueError(f"Invalid {ema_key}: expected a non-empty dictionary")
            if ema_key == "ema" or "shadow_params" in ema_state or "shadow_buffers" in ema_state:
                self._load_ema_shadows(ema_state, ema_key, raw_state)
            else:
                self._load_flat_ema(ema_state, ema_key)
            self.weight_source = "ema"

        self.model.eval()
        return checkpoint

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_for_batch(
        self,
        batch: dict[str, Any],
        *,
        batch_idx: int,
        remaining_cases: int | None,
        netcdf_writer: NetCDFWriter | None = None,
        case_offset: int = 0,
    ) -> int:
        batch_size = self._infer_batch_size(batch)
        if remaining_cases is not None:
            batch_size = min(batch_size, remaining_cases)
        dates = self._get_batch_meta_list(batch, "date", batch_size)
        timestamps = self._get_batch_meta_list(batch, "timestamp", batch_size)

        generated_members: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for member_idx in range(self.cfg.sampling.ensemble_size):
            self._set_member_seed(batch_idx=batch_idx, member_idx=member_idx)
            generated = generate_batch(
                self.model,
                batch,
                generation_config=self.cfg.paths.generation_config_path,
                device=self.device,
            )
            if not isinstance(generated, torch.Tensor):
                raise TypeError(
                    f"generate_batch returned type {type(generated)}, expected torch.Tensor"
                )
            if generated.ndim != 4:
                raise ValueError(
                    f"Expected generated tensor to have shape (B, C, H, W), got {tuple(generated.shape)}"
                )

            for case_idx in range(batch_size):
                generated_members[case_idx].append(
                    generated[case_idx : case_idx + 1].detach().cpu()
                )

        for case_idx in range(batch_size):
            case_batch = self._slice_batch(batch, case_idx)
            if netcdf_writer is not None:
                self._save_netcdf_case(
                    netcdf_writer, case_batch, generated_members[case_idx],
                    dataset_index=case_offset + case_idx,
                )
                continue  # NetCDF is an alternative to NPZ, not an additional copy.
            case_output_dir = build_case_output_dir(
                self.samples_dir,
                date=dates[case_idx],
                timestamp=timestamps[case_idx],
                fallback_name=f"batch_{batch_idx:04d}_case_{case_idx:02d}",
            )

            if self.cfg.outputs.save_members:
                if self.cfg.outputs.storage_mode == "per_member":
                    for member_idx, generated_member in enumerate(generated_members[case_idx]):
                        save_member(
                            case_batch,
                            generated_member,
                            case_output_dir=case_output_dir,
                            member_idx=member_idx,
                            checkpoint_path=self.checkpoint_path,
                            save_physical=self.cfg.outputs.save_physical,
                            save_metadata=self.cfg.outputs.save_metadata,
                        )
                elif self.cfg.outputs.storage_mode == "per_member_bundle":
                    save_member_bundle(
                        case_batch,
                        generated_members[case_idx],
                        case_output_dir=case_output_dir,
                        checkpoint_path=self.checkpoint_path,
                        save_physical=self.cfg.outputs.save_physical,
                        save_metadata=self.cfg.outputs.save_metadata,
                    )
                else:
                    raise ValueError(
                        f"Unsupported storage_mode {self.cfg.outputs.storage_mode!r}"
                    )

            # Physical aggregates must be computed AFTER converting each member.
            # Keep separate model-space aggregates for the existing `generated` key.
            physical_members = None
            if self.cfg.outputs.save_physical and (
                self.cfg.outputs.save_ensemble_mean or self.cfg.outputs.save_pmm
            ):
                physical_members = [
                    inverse_transform_target_tensor(case_batch, member)
                    for member in generated_members[case_idx]
                ]

            if self.cfg.outputs.save_ensemble_mean:
                ensemble_mean = self._compute_ensemble_mean(generated_members[case_idx])
                physical_mean = (
                    self._compute_ensemble_mean(physical_members)
                    if physical_members is not None else None
                )
                save_aggregate(
                    case_batch,
                    ensemble_mean,
                    case_output_dir=case_output_dir,
                    name="ensemble_mean",
                    checkpoint_path=self.checkpoint_path,
                    save_physical=self.cfg.outputs.save_physical,
                    save_metadata=self.cfg.outputs.save_metadata,
                    generated_physical=physical_mean,
                )

            if self.cfg.outputs.save_pmm:
                mask = self._build_pmm_mask(case_batch, generated_members[case_idx][0])
                pmm_field = self._compute_pmm(generated_members[case_idx], mask=mask)
                physical_pmm = (
                    self._compute_pmm(physical_members, mask=mask)
                    if physical_members is not None else None
                )
                save_aggregate(
                    case_batch,
                    pmm_field,
                    case_output_dir=case_output_dir,
                    name="pmm",
                    checkpoint_path=self.checkpoint_path,
                    save_physical=self.cfg.outputs.save_physical,
                    save_metadata=self.cfg.outputs.save_metadata,
                    generated_physical=physical_pmm,
                )

        return batch_size

    def _save_netcdf_case(
        self, writer: NetCDFWriter, batch: dict[str, Any],
        members: list[torch.Tensor], *, dataset_index: int,
    ) -> None:
        """Convert members first, then aggregate and append one physical record."""
        reference = self._netcdf_reference_metadata
        metadata = reference if dataset_index == 0 else self.dataset.get_output_metadata(dataset_index)
        timestamp = self._get_batch_meta_list(batch, "timestamp", 1)[0]
        if timestamp != metadata.aligned_timestamp.isoformat():
            raise ValueError("NetCDF metadata does not match the generated sample timestamp")
        for name in ("x", "y", "latitude", "longitude"):
            if not np.array_equal(getattr(metadata.grid, name), getattr(reference.grid, name)):
                raise ValueError(f"NorCP {name} coordinates changed within a fixed-grid NetCDF run")
        if metadata.calendar != reference.calendar or metadata.units != reference.units:
            raise ValueError("NorCP calendar or physical units changed within the NetCDF run")

        physical_members = [inverse_transform_target_tensor(batch, member) for member in members]
        mean = self._compute_ensemble_mean(physical_members)
        mask = self._build_pmm_mask(batch, members[0])
        pmm = self._compute_pmm(physical_members, mask=mask)
        writer.append(
            metadata.timestamp, time_bounds=metadata.time_bounds,
            members=torch.cat(physical_members, dim=0)[:, 0].numpy(),
            ensemble_mean=mean[0, 0].numpy(), pmm=pmm[0, 0].numpy(),
        )


    def _compute_ensemble_mean(self, generated_members: list[torch.Tensor]) -> torch.Tensor:
        if len(generated_members) == 0:
            raise ValueError("generated_members must contain at least one tensor")
        stacked = torch.stack(generated_members, dim=0)
        return torch.mean(stacked, dim=0)


    def _compute_pmm(
        self,
        generated_members: list[torch.Tensor],
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute PMM in the supplied members' space, using an already-built mask."""
        if len(generated_members) == 0:
            raise ValueError("generated_members must contain at least one tensor")

        ensemble = torch.cat(generated_members, dim=0)  # [M, 1, H, W]
        if ensemble.ndim != 4 or ensemble.shape[1] != 1:
            raise ValueError(
                "PMM currently expects univariate generated members with shape [M, 1, H, W], "
                f"got {tuple(ensemble.shape)}"
            )

        ensemble_univariate = ensemble[:, 0]  # [M, H, W]
        return pmm_from_ensemble(
            ensemble_univariate,
            mask=mask,
            exclude_zeros=False,
        )

    def _build_pmm_mask(
        self,
        batch: dict[str, Any],
        reference_generated: torch.Tensor,
    ) -> torch.Tensor | None:
        payload = build_preview_payload(batch, reference_generated)
        cond_static_physical = payload.get("cond_static_physical")
        if not isinstance(cond_static_physical, torch.Tensor):
            return None

        static_variables = tuple(self.adapter_cfg.static_variables)
        if "lsm" not in static_variables:
            return None

        lsm_idx = static_variables.index("lsm")
        if cond_static_physical.ndim != 4:
            return None
        if lsm_idx >= cond_static_physical.shape[1]:
            return None

        lsm = cond_static_physical[:, lsm_idx]  # [B, H, W]
        return lsm > 0.5


    def _slice_batch(self, batch: dict[str, Any], case_idx: int) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[case_idx : case_idx + 1]
            elif isinstance(value, dict):
                sliced_meta: dict[str, Any] = {}
                for meta_key, meta_value in value.items():
                    if isinstance(meta_value, list):
                        sliced_meta[meta_key] = [meta_value[case_idx]]
                    else:
                        sliced_meta[meta_key] = meta_value
                sliced[key] = sliced_meta
            else:
                sliced[key] = value
        return sliced


    # ------------------------------------------------------------------
    # Metadata / saving helpers
    # ------------------------------------------------------------------

    def _save_run_metadata(self) -> None:
        save_run_metadata(
            self.metadata_path,
            run_name=self.cfg.run_name,
            generation_run_config=self.cfg.config_path,
            dataset_config=self.cfg.paths.dataset_config_path,
            model_config=self.cfg.paths.model_config_path,
            generation_config=self.cfg.paths.generation_config_path,
            training_config=self.cfg.paths.training_config_path,
            checkpoint_mode=self.cfg.checkpoint.mode,
            checkpoint_path=self.checkpoint_path,
            use_ema_weights=self.cfg.checkpoint.use_ema_weights,
            weight_source=self.weight_source,
            split=self.cfg.data.split,
            batch_size=self.cfg.data.batch_size,
            ensemble_size=self.cfg.sampling.ensemble_size,
            max_cases=self.cfg.limits.max_cases,
            device=str(self.device),
            save_physical=self.cfg.outputs.save_physical,
            save_ensemble_mean=self.cfg.outputs.save_ensemble_mean,
            save_pmm=self.cfg.outputs.save_pmm,
            save_plots=self.cfg.outputs.save_plots,
            output_format=self.cfg.outputs.output_format,
            storage_mode=self.cfg.outputs.storage_mode,
        )


    def _infer_batch_size(self, batch: dict[str, Any]) -> int:
        target = batch.get("target")
        if isinstance(target, torch.Tensor):
            return int(target.shape[0])
        cond_dynamic = batch.get("cond_dynamic")
        if isinstance(cond_dynamic, torch.Tensor):
            return int(cond_dynamic.shape[0])
        raise ValueError("Could not infer batch size from batch tensors")

    def _get_batch_meta_list(
        self,
        batch: dict[str, Any],
        key: str,
        batch_size: int,
    ) -> list[Any]:
        meta = batch.get("meta", {})
        if not isinstance(meta, dict):
            return [None] * batch_size
        value = meta.get(key)
        if isinstance(value, list) and len(value) >= batch_size:
            # max_cases can select only a prefix of the collated batch.
            return value[:batch_size]
        return [None] * batch_size

    def _set_member_seed(self, *, batch_idx: int, member_idx: int) -> None:
        if not self.cfg.sampling.use_fixed_seed:
            return
        seed = int(self.cfg.sampling.base_seed + 10_000 * batch_idx + member_idx)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32 - 1))

    def _safe_len(self, obj: Any) -> int | None:
        try:
            return len(obj)
        except TypeError:
            return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _print_run_summary(self) -> None:
        logger.info("\n=======================")
        logger.info("STRIDE generation run")
        logger.info("=======================")
        logger.info(f"Run name:          {self.cfg.run_name}")
        logger.info(f"Device:            {self.device}")
        logger.info(f"Output dir:        {self.output_dir}")
        if self.cfg.outputs.output_format == "netcdf":
            logger.info(f"NetCDF dir:        {self.output_dir / 'netcdf'}")
        else:
            logger.info(f"Samples dir:       {self.samples_dir}")
        logger.info(f"Output format:     {self.cfg.outputs.output_format}")
        logger.info(f"Dataset config:    {self.cfg.paths.dataset_config_path}")
        logger.info(f"Model config:      {self.cfg.paths.model_config_path}")
        logger.info(f"Generation config: {self.cfg.paths.generation_config_path}")
        logger.info(f"Checkpoint path:   {self.checkpoint_path}")
        logger.info(f"Split:             {self.cfg.data.split}")
        logger.info(f"Shuffle:           {self.cfg.data.shuffle}")

        dataset_len = self._safe_len(self.dataset)
        loader_len = self._safe_len(self.loader)

        logger.info(
            f"Dataset length:    {'unknown' if dataset_len is None else dataset_len}"
        )
        logger.info(
            f"Num batches:       {'unknown' if loader_len is None else loader_len}"
        )
        logger.info(f"Ensemble size:     {self.cfg.sampling.ensemble_size}")
        logger.info(f"Max cases:         {self.cfg.limits.max_cases}")
        logger.info(f"Save members:      {self.cfg.outputs.save_members}")
        logger.info(f"Save ens. mean:    {self.cfg.outputs.save_ensemble_mean}")
        logger.info(f"Save PMM:          {self.cfg.outputs.save_pmm}")
        logger.info(f"Storage mode:      {self.cfg.outputs.storage_mode}")
        logger.info(f"Requested EMA:     {self.cfg.checkpoint.use_ema_weights}")
        logger.info(f"Weight source:     {self.weight_source}")
