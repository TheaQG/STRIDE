from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """
    Framework-level model specification for STRIDE.

    This is the validated, code-facing object that model constructors consume.
    """

    name: str = "edm_unet"

    # Spatial/data contract
    in_dynamic_channels: int = 2
    in_static_channels: int = 2
    out_channels: int = 1
    target_height: int = 128
    target_width: int = 128
    cond_height: int = 128
    cond_width: int = 128

    # Core UNet
    model_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0

    # Attention
    use_attention: bool = True
    attention_resolutions: tuple[int, ...] = (16,)
    num_heads: int = 4

    # Embeddings
    sigma_embed_dim: int = 128
    film_embed_dim: int = 128
    use_doy_film: bool = False
    use_variable_film: bool = False

    # Context branch
    use_context_encoder: bool = False
    context_in_channels: int = 0
    context_embed_dim: int = 128

    # Rain gate
    use_rain_gate: bool = False
    rain_gate_channels: int = 32

    # Future-facing switches
    use_temporal_stack: bool = False
    temporal_steps: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ModelSpec":
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Model YAML config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        if not isinstance(cfg_dict, dict):
            raise ValueError(
                f"Expected YAML config to load into a dict, got {type(cfg_dict)}"
            )

        return cls.from_dict(cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: dict[str, Any]) -> "ModelSpec":
        if "model" not in cfg_dict:
            raise KeyError("Expected top-level key 'model' in model config")

        model_cfg = cfg_dict["model"]
        if not isinstance(model_cfg, dict):
            raise ValueError(
                f"Expected 'model' section to be a dict, got {type(model_cfg)}"
            )

        spatial_cfg = model_cfg.get("spatial", {})
        unet_cfg = model_cfg.get("unet", {})
        attention_cfg = model_cfg.get("attention", {})
        embedding_cfg = model_cfg.get("embeddings", {})
        context_cfg = model_cfg.get("context", {})
        rain_gate_cfg = model_cfg.get("rain_gate", {})
        temporal_cfg = model_cfg.get("temporal", {})

        for name, section in [
            ("spatial", spatial_cfg),
            ("unet", unet_cfg),
            ("attention", attention_cfg),
            ("embeddings", embedding_cfg),
            ("context", context_cfg),
            ("rain_gate", rain_gate_cfg),
            ("temporal", temporal_cfg),
        ]:
            if not isinstance(section, dict):
                raise ValueError(f"Expected 'model.{name}' to be a dict")

        spec = cls(
            name=str(model_cfg.get("name", "edm_unet")),
            in_dynamic_channels=int(model_cfg.get("in_dynamic_channels", 2)),
            in_static_channels=int(model_cfg.get("in_static_channels", 2)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            target_height=int(spatial_cfg.get("target_height", 128)),
            target_width=int(spatial_cfg.get("target_width", 128)),
            cond_height=int(spatial_cfg.get("cond_height", 128)),
            cond_width=int(spatial_cfg.get("cond_width", 128)),
            model_channels=int(unet_cfg.get("model_channels", 64)),
            channel_mults=tuple(unet_cfg.get("channel_mults", [1, 2, 4, 4])),
            num_res_blocks=int(unet_cfg.get("num_res_blocks", 2)),
            dropout=float(unet_cfg.get("dropout", 0.0)),
            use_attention=bool(attention_cfg.get("use_attention", True)),
            attention_resolutions=tuple(
                attention_cfg.get("attention_resolutions", [16])
            ),
            num_heads=int(attention_cfg.get("num_heads", 4)),
            sigma_embed_dim=int(embedding_cfg.get("sigma_embed_dim", 128)),
            film_embed_dim=int(embedding_cfg.get("film_embed_dim", 128)),
            use_doy_film=bool(embedding_cfg.get("use_doy_film", False)),
            use_variable_film=bool(embedding_cfg.get("use_variable_film", False)),
            use_context_encoder=bool(context_cfg.get("use_context_encoder", False)),
            context_in_channels=int(context_cfg.get("context_in_channels", 0)),
            context_embed_dim=int(context_cfg.get("context_embed_dim", 128)),
            use_rain_gate=bool(rain_gate_cfg.get("use_rain_gate", False)),
            rain_gate_channels=int(rain_gate_cfg.get("rain_gate_channels", 32)),
            use_temporal_stack=bool(temporal_cfg.get("use_temporal_stack", False)),
            temporal_steps=int(temporal_cfg.get("temporal_steps", 1)),
        )

        spec._validate()
        return spec

    def _validate(self) -> None:
        if self.in_dynamic_channels < 0:
            raise ValueError("in_dynamic_channels must be >= 0")
        if self.in_static_channels < 0:
            raise ValueError("in_static_channels must be >= 0")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0")

        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError("target spatial dimensions must be positive")
        if self.cond_height <= 0 or self.cond_width <= 0:
            raise ValueError("conditioning spatial dimensions must be positive")

        if self.model_channels <= 0:
            raise ValueError("model_channels must be > 0")
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be > 0")
        if len(self.channel_mults) == 0:
            raise ValueError("channel_mults must not be empty")
        if any(mult <= 0 for mult in self.channel_mults):
            raise ValueError("all channel_mults must be > 0")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.sigma_embed_dim <= 0 or self.film_embed_dim <= 0:
            raise ValueError("embedding dimensions must be > 0")

        if self.use_context_encoder and self.context_in_channels <= 0:
            raise ValueError(
                "context_in_channels must be > 0 when use_context_encoder=True"
            )

        if self.use_temporal_stack and self.temporal_steps <= 1:
            raise ValueError(
                "temporal_steps must be > 1 when use_temporal_stack=True"
            )

    @property
    def total_cond_channels(self) -> int:
        return self.in_dynamic_channels + self.in_static_channels