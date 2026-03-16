"""
Shared adapter configuration schema for STRIDE.

This module defines the framework-level `AdapterConfig` used by dataset-specific
adapters. The goal is to keep experiment choices config-driven while keeping
adapter behavior in code.

Current supported schema (YAML / dict)
--------------------------------------

data:
  root_dir: /path/to/data_root
  size_tag: size_589x789

  split:
    manifest_path: data_adapters/danra_era5_small/saved/splits/random_seed42.json
    name: train
    stats_tag: random_seed42

  target:
    variable: prcp
    source: DANRA

  conditioning:
    dynamic:
      source: ERA5
      variables: [prcp, temp]
    static:
      source: STATIC
      variables: [lsm, topo]

  domain:
    tag: dk_128x128
    crop:
      anchor_y: 200
      anchor_x: 380
      height: 128
      width: 128

    spatial_shuffle:
      enabled: false
      train_only: true
      cutout_domain: [170, 350, 340, 520]   # [x1, x2, y1, y2]

  transforms:
    apply: true

Notes
-----
- This config class is framework-wide and should not contain dataset-specific
  loading logic.
- Relative paths inside YAML are resolved relative to the repository root when
  the config file lives under `configs/`, otherwise relative to the config file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT_DIR = Path("/Users/au728490/Data/Data_DiffMod_small")
DEFAULT_SIZE_TAG = "size_589x789"
DEFAULT_DYNAMIC_VARIABLES = ("prcp", "temp")
DEFAULT_STATIC_VARIABLES = ("lsm", "topo")
DEFAULT_TARGET_TIME_OFFSETS: dict[str, float] = {}
DEFAULT_DYNAMIC_TIME_OFFSETS: dict[str, float] = {}


@dataclass(frozen=True)
class AdapterConfig:
    """
    Shared STRIDE adapter configuration.
    """

    root_dir: str | Path = DEFAULT_ROOT_DIR
    size_tag: str = DEFAULT_SIZE_TAG
    scenario_name: str | None = None
    temporal_tag: str | None = None
    target_spatial_tag: str | None = None
    dynamic_spatial_tag: str | None = None
    split_manifest_path: str | Path = (
        Path("data_adapters/danra_era5_small/saved/splits/random_seed42.json")
    )
    split_name: str = "train"
    domain_tag: str = "dk_128x128"
    crop: tuple[int, int, int, int] | None = (200, 380, 128, 128)
    target_variable: str = "prcp"
    target_source: str = "DANRA"
    dynamic_variables: tuple[str, ...] = DEFAULT_DYNAMIC_VARIABLES
    dynamic_source: str = "ERA5"
    target_time_offsets: dict[str, float] = None  # type: ignore[assignment]
    dynamic_time_offsets: dict[str, float] = None  # type: ignore[assignment]
    static_variables: tuple[str, ...] = DEFAULT_STATIC_VARIABLES
    static_source: str = "STATIC"
    static_allow_missing: bool = True
    apply_transforms: bool = True
    split_stats_tag: str | None = None
    target_full_shape: tuple[int, int] | None = None
    dynamic_full_shape: tuple[int, int] | None = None
    upsample_to_target: bool = False
    upsample_mode: str = "bilinear"
    spatial_shuffle_enabled: bool = False
    spatial_shuffle_train_only: bool = True
    spatial_shuffle_cutout_domain: tuple[int, int, int, int] | None = None
    spatial_shuffle_seed: int | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "AdapterConfig":
        """
        Build an AdapterConfig from a YAML file following the STRIDE data schema.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter YAML config does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        if not isinstance(cfg_dict, dict):
            raise ValueError(
                f"Expected YAML config to load into a dict, got {type(cfg_dict)}"
            )

        return cls.from_dict(cfg_dict, config_path=path)

    @classmethod
    def from_dict(
        cls,
        cfg_dict: dict[str, Any],
        config_path: str | Path | None = None,
    ) -> "AdapterConfig":
        """
        Build an AdapterConfig from a dictionary following the STRIDE data schema.
        """
        if "data" not in cfg_dict:
            raise KeyError("Expected top-level key 'data' in adapter config")

        data_cfg = cfg_dict["data"]
        if not isinstance(data_cfg, dict):
            raise ValueError(
                f"Expected 'data' section to be a dict, got {type(data_cfg)}"
            )

        split_cfg = data_cfg.get("split", {})
        target_cfg = data_cfg.get("target", {})
        conditioning_cfg = data_cfg.get("conditioning", {})
        domain_cfg = data_cfg.get("domain", {})
        transforms_cfg = data_cfg.get("transforms", {})
        crop_cfg = domain_cfg.get("crop")
        spatial_shuffle_cfg = domain_cfg.get("spatial_shuffle", {})
        target_full_shape_cfg = domain_cfg.get("target_full_shape")
        dynamic_full_shape_cfg = domain_cfg.get("dynamic_full_shape")

        dynamic_cfg: dict[str, Any]
        static_cfg: dict[str, Any]

        # New nested schema:
        #   data.conditioning.dynamic.{source,variables}
        #   data.conditioning.static.{source,variables}
        # Backward-compatible fallback:
        #   data.conditioning.{source,variables}
        #   data.statics.{source,variables}
        if isinstance(conditioning_cfg, dict) and (
            "dynamic" in conditioning_cfg or "static" in conditioning_cfg
        ):
            dynamic_cfg = conditioning_cfg.get("dynamic", {})
            static_cfg = conditioning_cfg.get("static", {})
        else:
            dynamic_cfg = conditioning_cfg
            static_cfg = data_cfg.get("statics", {})

        if not isinstance(split_cfg, dict):
            raise ValueError("Expected 'data.split' to be a dict")
        if not isinstance(target_cfg, dict):
            raise ValueError("Expected 'data.target' to be a dict")
        if not isinstance(conditioning_cfg, dict):
            raise ValueError("Expected 'data.conditioning' to be a dict")
        if not isinstance(dynamic_cfg, dict):
            raise ValueError(
                "Expected dynamic conditioning config to be a dict "
                "(either 'data.conditioning.dynamic' or legacy 'data.conditioning')"
            )
        if not isinstance(static_cfg, dict):
            raise ValueError(
                "Expected static conditioning config to be a dict "
                "(either 'data.conditioning.static' or legacy 'data.statics')"
            )
        if not isinstance(domain_cfg, dict):
            raise ValueError("Expected 'data.domain' to be a dict")
        if not isinstance(transforms_cfg, dict):
            raise ValueError("Expected 'data.transforms' to be a dict")
        if crop_cfg is not None and not isinstance(crop_cfg, dict):
            raise ValueError("Expected 'data.domain.crop' to be a dict or null")
        if not isinstance(spatial_shuffle_cfg, dict):
            raise ValueError(
                "Expected 'data.domain.spatial_shuffle' to be a dict"
            )

        def _parse_shape(
            raw: Any,
            *,
            field_name: str,
        ) -> tuple[int, int] | None:
            if raw is None:
                return None
            if isinstance(raw, dict):
                if "height" not in raw or "width" not in raw:
                    raise ValueError(
                        f"Expected '{field_name}' dict to contain keys 'height' and 'width'"
                    )
                return (int(raw["height"]), int(raw["width"]))
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                return (int(raw[0]), int(raw[1]))
            raise ValueError(
                f"Expected '{field_name}' to be null, a dict with height/width, or a list/tuple of length 2"
            )

        manifest_path = split_cfg.get("manifest_path")
        if manifest_path is None:
            raise KeyError("Missing required key 'data.split.manifest_path'")

        manifest_path_resolved = cls._resolve_config_path(
            manifest_path,
            config_path=config_path,
        )

        crop_tuple: tuple[int, int, int, int] | None
        if crop_cfg is None:
            crop_tuple = None
        else:
            anchor_y = crop_cfg.get("anchor_y", crop_cfg.get("top"))
            anchor_x = crop_cfg.get("anchor_x", crop_cfg.get("left"))
            if anchor_y is None or anchor_x is None:
                raise ValueError(
                    "Expected 'data.domain.crop' to define either anchor_y/anchor_x or top/left"
                )
            crop_tuple = (
                int(anchor_y),
                int(anchor_x),
                int(crop_cfg["height"]),
                int(crop_cfg["width"]),
            )

        cutout_domain_raw = spatial_shuffle_cfg.get("cutout_domain")
        spatial_shuffle_cutout_domain: tuple[int, int, int, int] | None
        if cutout_domain_raw is None:
            spatial_shuffle_cutout_domain = None
        else:
            if not isinstance(cutout_domain_raw, (list, tuple)):
                raise ValueError(
                    "Expected 'data.domain.spatial_shuffle.cutout_domain' to be a "
                    "list or tuple of length 4"
                )
            if len(cutout_domain_raw) != 4:
                raise ValueError(
                    "Expected 'data.domain.spatial_shuffle.cutout_domain' to have "
                    f"length 4, got {len(cutout_domain_raw)}"
                )
            spatial_shuffle_cutout_domain = (
                int(cutout_domain_raw[0]),
                int(cutout_domain_raw[1]),
                int(cutout_domain_raw[2]),
                int(cutout_domain_raw[3]),
            )

        target_full_shape = _parse_shape(
            target_full_shape_cfg,
            field_name="data.domain.target_full_shape",
        )
        dynamic_full_shape = _parse_shape(
            dynamic_full_shape_cfg,
            field_name="data.domain.dynamic_full_shape",
        )

        return cls(
            root_dir=data_cfg.get("root_dir", DEFAULT_ROOT_DIR),
            size_tag=data_cfg.get("size_tag", DEFAULT_SIZE_TAG),
            scenario_name=(
                None
                if data_cfg.get("scenario_name") is None
                else str(data_cfg.get("scenario_name"))
            ),
            temporal_tag=(
                None
                if data_cfg.get("temporal_tag") is None
                else str(data_cfg.get("temporal_tag"))
            ),
            target_spatial_tag=(
                None
                if data_cfg.get("target_spatial_tag") is None
                else str(data_cfg.get("target_spatial_tag"))
            ),
            dynamic_spatial_tag=(
                None
                if data_cfg.get("dynamic_spatial_tag") is None
                else str(data_cfg.get("dynamic_spatial_tag"))
            ),
            split_manifest_path=manifest_path_resolved,
            split_name=str(split_cfg.get("name", "train")),
            domain_tag=str(domain_cfg.get("tag", "dk_128x128")),
            crop=crop_tuple,
            spatial_shuffle_enabled=bool(
                spatial_shuffle_cfg.get("enabled", False)
            ),
            spatial_shuffle_train_only=bool(
                spatial_shuffle_cfg.get("train_only", True)
            ),
            spatial_shuffle_cutout_domain=spatial_shuffle_cutout_domain,
            spatial_shuffle_seed=(
                None
                if spatial_shuffle_cfg.get("seed") is None
                else int(spatial_shuffle_cfg.get("seed")) # type: ignore
            ),
            target_variable=str(target_cfg.get("variable", "prcp")),
            target_source=str(target_cfg.get("source", "DANRA")),
            dynamic_variables=tuple(
                dynamic_cfg.get("variables", DEFAULT_DYNAMIC_VARIABLES)
            ),
            dynamic_source=str(dynamic_cfg.get("source", "ERA5")),
            target_time_offsets=dict(
                target_cfg.get("time_offsets", DEFAULT_TARGET_TIME_OFFSETS)
            ),
            dynamic_time_offsets=dict(
                dynamic_cfg.get("time_offsets", DEFAULT_DYNAMIC_TIME_OFFSETS)
            ),
            static_variables=tuple(
                static_cfg.get("variables", DEFAULT_STATIC_VARIABLES)
            ),
            static_source=str(static_cfg.get("source", "STATIC")),
            static_allow_missing=bool(static_cfg.get("allow_missing", True)),
            apply_transforms=bool(transforms_cfg.get("apply", True)),
            split_stats_tag=(
                str(split_cfg["stats_tag"])
                if split_cfg.get("stats_tag") is not None
                else None
            ),
            target_full_shape=target_full_shape,
            dynamic_full_shape=dynamic_full_shape,
            upsample_to_target=bool(dynamic_cfg.get("upsample_to_target", False)),
            upsample_mode=str(dynamic_cfg.get("upsample_mode", "bilinear")),
        )

    @staticmethod
    def _resolve_config_path(
        raw_path: str | Path,
        config_path: str | Path | None = None,
    ) -> Path:
        """
        Resolve a potentially relative path in a YAML config.

        Relative paths are interpreted relative to the repository root when the
        config file lives under `configs/`. Otherwise they are interpreted
        relative to the config file's parent.
        """
        path = Path(raw_path)
        if path.is_absolute():
            return path

        if config_path is None:
            return path

        config_path = Path(config_path).resolve()
        repo_root = Path(__file__).resolve().parents[2]

        if "configs" in config_path.parts:
            return repo_root / path

        return config_path.parent / path

    @property
    def split_tag(self) -> str:
        """
        Tag used to locate saved statistics.
        """
        if self.split_stats_tag is not None:
            return self.split_stats_tag
        return Path(self.split_manifest_path).stem

    @property
    def crop_dict(self) -> dict[str, int] | None:
        """
        Convenience representation of the crop as a dictionary.
        """
        if self.crop is None:
            return None
        anchor_y, anchor_x, height, width = self.crop
        return {
            "anchor_y": anchor_y,
            "anchor_x": anchor_x,
            "height": height,
            "width": width,
        }


    @property
    def spatial_shuffle_dict(self) -> dict[str, Any]:
        """
        Convenience representation of spatial shuffling settings.
        """
        return {
            "enabled": self.spatial_shuffle_enabled,
            "train_only": self.spatial_shuffle_train_only,
            "cutout_domain": (
                None
                if self.spatial_shuffle_cutout_domain is None
                else list(self.spatial_shuffle_cutout_domain)
            ),
        }

    @property
    def has_spatial_shuffle(self) -> bool:
        """
        Whether spatial shuffling is enabled in the adapter config.
        """
        return self.spatial_shuffle_enabled

    @property
    def spatial_shuffle(self) -> bool:
        """
        Backward-compatible alias used by some adapters.
        """
        return self.spatial_shuffle_enabled

    @property
    def shuffle_seed(self) -> int | None:
        """
        Backward-compatible alias used by some adapters.
        """
        return self.spatial_shuffle_seed