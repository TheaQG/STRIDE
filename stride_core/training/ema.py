"""
Exponential Moving Average (EMA) utilities for STRIDE training.

This module keeps EMA handling separate from the trainer so the trainer can stay
focused on orchestration. The design keeps the useful functionality from the
legacy setup - shadow-parameter tracking, update scheduling, and checkpointable
state - but in a much smaller and cleaner form.

Current scope
-------------
- maintain a shadow copy of trainable model parameters
- optional shadow copy of model buffers
- delayed start via `update_after_step`
- configurable update frequency via `update_every`
- copy EMA weights into a model for validation / sampling
- save and load EMA state

Typical usage
-------------
    ema = EMA(model, decay=0.999, update_after_step=0, update_every=1)
    ...
    optimizer.step()
    ema.update(model)
    ...
    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)

Notes
-----
- EMA state is stored on CPU by default to reduce GPU memory pressure.
- Only floating-point trainable parameters are EMA-tracked.
- Buffers can optionally be tracked exactly rather than exponentially averaged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


@dataclass(frozen=True)
class EMAConfig:
    """
    Minimal EMA configuration extracted from the training YAML.
    """

    enabled: bool
    decay: float
    update_after_step: int
    update_every: int
    track_buffers: bool
    store_on_cpu: bool

    @classmethod
    def from_training_yaml(cls, training_config_path: str | Path) -> "EMAConfig":
        path = Path(training_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Training config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Expected training config YAML to load into a dict, got {type(cfg)}"
            )

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "EMAConfig":
        if "training" not in cfg:
            raise KeyError("Expected top-level key 'training' in training config")

        training_cfg = cfg["training"]
        if not isinstance(training_cfg, dict):
            raise ValueError(
                f"Expected 'training' section to be a dict, got {type(training_cfg)}"
            )

        ema_cfg = training_cfg.get("ema", {})
        if not isinstance(ema_cfg, dict):
            raise ValueError("Expected 'training.ema' to be a dict")

        enabled = bool(ema_cfg.get("enabled", True))
        decay = float(ema_cfg.get("decay", 0.999))
        update_after_step = int(ema_cfg.get("update_after_step", 0))
        update_every = int(ema_cfg.get("update_every", 1))
        track_buffers = bool(ema_cfg.get("track_buffers", True))
        store_on_cpu = bool(ema_cfg.get("store_on_cpu", True))

        if not (0.0 < decay < 1.0):
            raise ValueError(f"ema.decay must be in (0, 1), got {decay}")
        if update_after_step < 0:
            raise ValueError(
                f"ema.update_after_step must be >= 0, got {update_after_step}"
            )
        if update_every <= 0:
            raise ValueError(f"ema.update_every must be > 0, got {update_every}")

        return cls(
            enabled=enabled,
            decay=decay,
            update_after_step=update_after_step,
            update_every=update_every,
            track_buffers=track_buffers,
            store_on_cpu=store_on_cpu,
        )


class EMA:
    """
    Exponential moving average helper for model parameters.

    Parameters
    ----------
    model:
        Live training model whose parameters will be tracked.
    decay:
        EMA decay factor.
    update_after_step:
        Do not begin EMA updates before this optimization step.
    update_every:
        Only update EMA every `update_every` calls to `update()`.
    track_buffers:
        If True, copy model buffers exactly into EMA state during updates.
    store_on_cpu:
        If True, store EMA shadow state on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every: int = 1,
        track_buffers: bool = True,
        store_on_cpu: bool = True,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if update_after_step < 0:
            raise ValueError(
                f"update_after_step must be >= 0, got {update_after_step}"
            )
        if update_every <= 0:
            raise ValueError(f"update_every must be > 0, got {update_every}")

        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.track_buffers = bool(track_buffers)
        self.store_on_cpu = bool(store_on_cpu)

        self.num_updates = 0
        self.initialized = False

        self.shadow_params: dict[str, torch.Tensor] = {}
        self.shadow_buffers: dict[str, torch.Tensor] = {}

        self._tracked_param_names = self._get_tracked_param_names(model)
        self._tracked_buffer_names = self._get_tracked_buffer_names(model)

        self._initialize_from_model(model)

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        ema_config: EMAConfig | dict[str, Any] | str | Path,
    ) -> "EMA":
        cfg = _coerce_ema_config(ema_config)
        return cls(
            model=model,
            decay=cfg.decay,
            update_after_step=cfg.update_after_step,
            update_every=cfg.update_every,
            track_buffers=cfg.track_buffers,
            store_on_cpu=cfg.store_on_cpu,
        )

    def _target_device(self) -> torch.device | None:
        return torch.device("cpu") if self.store_on_cpu else None

    def _clone_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        cloned = tensor.detach().clone()
        target_device = self._target_device()
        if target_device is not None:
            cloned = cloned.to(target_device)
        return cloned

    def _get_tracked_param_names(self, model: nn.Module) -> list[str]:
        names: list[str] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.is_floating_point(param):
                continue
            names.append(name)
        return names

    def _get_tracked_buffer_names(self, model: nn.Module) -> list[str]:
        if not self.track_buffers:
            return []

        names: list[str] = []
        for name, buffer in model.named_buffers():
            if not torch.is_floating_point(buffer):
                continue
            names.append(name)
        return names

    def _initialize_from_model(self, model: nn.Module) -> None:
        named_params = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())

        self.shadow_params = {
            name: self._clone_tensor(named_params[name])
            for name in self._tracked_param_names
        }

        self.shadow_buffers = {
            name: self._clone_tensor(named_buffers[name])
            for name in self._tracked_buffer_names
        }

        self.initialized = True

    def should_update(self) -> bool:
        step = self.num_updates
        if step < self.update_after_step:
            return False
        if (step - self.update_after_step) % self.update_every != 0:
            return False
        return True

    @torch.no_grad()
    def update(self, model: nn.Module) -> bool:
        """
        Update EMA state from the live model.

        Returns
        -------
        bool
            True if an EMA update was performed, otherwise False.
        """
        if not self.initialized:
            self._initialize_from_model(model)

        do_update = self.should_update()

        if do_update:
            named_params = dict(model.named_parameters())
            for name in self._tracked_param_names:
                live = named_params[name].detach()
                shadow = self.shadow_params[name]
                live = live.to(device=shadow.device, dtype=shadow.dtype)
                shadow.mul_(self.decay).add_(live, alpha=1.0 - self.decay)

            if self.track_buffers:
                named_buffers = dict(model.named_buffers())
                for name in self._tracked_buffer_names:
                    live_buffer = named_buffers[name].detach()
                    self.shadow_buffers[name].copy_(
                        live_buffer.to(
                            device=self.shadow_buffers[name].device,
                            dtype=self.shadow_buffers[name].dtype,
                        )
                    )

        self.num_updates += 1
        return do_update

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA weights and tracked buffers into a model in-place.
        """
        named_params = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())

        for name, shadow in self.shadow_params.items():
            if name not in named_params:
                raise KeyError(f"Model is missing EMA-tracked parameter '{name}'")
            named_params[name].data.copy_(
                shadow.to(
                    device=named_params[name].device,
                    dtype=named_params[name].dtype,
                )
            )

        for name, shadow in self.shadow_buffers.items():
            if name not in named_buffers:
                raise KeyError(f"Model is missing EMA-tracked buffer '{name}'")
            named_buffers[name].data.copy_(
                shadow.to(
                    device=named_buffers[name].device,
                    dtype=named_buffers[name].dtype,
                )
            )

    def clone_model(self, model: nn.Module) -> nn.Module:
        """
        Create a deep-copied model with EMA weights loaded into it.
        """
        ema_model = deepcopy(model)
        self.copy_to(ema_model)
        return ema_model

    def state_dict(self) -> dict[str, Any]:
        """
        Return a checkpointable EMA state dictionary.
        """
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "track_buffers": self.track_buffers,
            "store_on_cpu": self.store_on_cpu,
            "num_updates": self.num_updates,
            "initialized": self.initialized,
            "tracked_param_names": list(self._tracked_param_names),
            "tracked_buffer_names": list(self._tracked_buffer_names),
            "shadow_params": {k: v.clone() for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.clone() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore EMA state from a checkpoint dictionary.
        """
        self.decay = float(state_dict["decay"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.update_every = int(state_dict["update_every"])
        self.track_buffers = bool(state_dict["track_buffers"])
        self.store_on_cpu = bool(state_dict["store_on_cpu"])
        self.num_updates = int(state_dict["num_updates"])
        self.initialized = bool(state_dict["initialized"])

        self._tracked_param_names = list(state_dict["tracked_param_names"])
        self._tracked_buffer_names = list(state_dict["tracked_buffer_names"])

        self.shadow_params = {
            k: v.detach().clone() for k, v in state_dict["shadow_params"].items()
        }
        self.shadow_buffers = {
            k: v.detach().clone() for k, v in state_dict["shadow_buffers"].items()
        }

    def to(self, device: torch.device | str) -> "EMA":
        """
        Move shadow tensors to a new device.

        This is mainly useful when `store_on_cpu=False` and you want explicit
        device control.
        """
        device = torch.device(device)
        self.shadow_params = {k: v.to(device) for k, v in self.shadow_params.items()}
        self.shadow_buffers = {k: v.to(device) for k, v in self.shadow_buffers.items()}
        self.store_on_cpu = device.type == "cpu"
        return self


def _coerce_ema_config(ema_config: EMAConfig | dict[str, Any] | str | Path) -> EMAConfig:
    if isinstance(ema_config, EMAConfig):
        return ema_config

    if isinstance(ema_config, (str, Path)):
        return EMAConfig.from_training_yaml(ema_config)

    if isinstance(ema_config, dict):
        return EMAConfig.from_dict(ema_config)

    raise TypeError(
        f"Unsupported ema_config type {type(ema_config)}. "
        "Expected EMAConfig, dict, str, or Path."
    )