"""
Adapter skeleton for the small DANRA/ERA5 STRIDE setup.

This adapter ties together:
    - Canonical date indexing from `paths.py`
    - Split manifests from `splits.py`
    - Feature loading from `features.py`
    - Region selection from `regions.py`
    - Saved statistics + transforms from `statistics/` and `transforms.py`
    - Variable metadata from `variable_registry.py`

Current design notes
--------------------
- Target, dynamic conditioning, and static conditioning variables are all
  config-driven.
- Static file paths are resolved from the variable registry rather than being
  hard-coded to only `lsm` and `topo`.
- Default transform names are taken from the variable registry.
- Domain can be fixed-crop, full-domain, or train-time spatially shuffled inside a configured cutout region.
- No temporal stacking yet.
- No larger context branch yet.

This file intentionally remains fairly explicit so the data path is still easy
to inspect, while removing a few brittle assumptions from the first draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_adapters.danra_era5_small.features import (
    build_doy_sincos,
    build_feature_metadata,
    build_time_feature_metadata,
    load_dynamic_conditioning,
    load_static_features,
    load_target_field,
)
from data_adapters.danra_era5_small.paths import ExperimentSamplePaths, build_experiment_file_index
from data_adapters.danra_era5_small.regions import (
    CropSpec,
    crop_sample_fields,
    validate_crop_within_cutout,
    validate_cutout_domain,
)
from data_adapters.danra_era5_small.splits import get_dates_for_split, load_split_manifest
from data_adapters.danra_era5_small.statistics.io import build_stats_output_path, load_transform_stats
from data_adapters.danra_era5_small.transforms import BaseTransform, build_transform
from stride_core.configs.adapter_config import AdapterConfig
from data_adapters.danra_era5_small.variable_registry import VARIABLE_REGISTRY, canonicalize_variable_name




class DanraEra5SmallDataset(Dataset):
    """
    Dataset for the first small DANRA/ERA5 STRIDE experiment.

    Returned sample contract
    ------------------------
    {
        "target": torch.Tensor,         # [C_out, H, W]
        "cond_dynamic": torch.Tensor,   # [C_dyn, H, W]
        "cond_static": torch.Tensor | None,  # [C_static, H, W] or None
        "cond_coord": dict,
        "meta": dict,
        "time_features": torch.Tensor,  # [2] = [sin(DOY), cos(DOY)]
    }
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        self.cfg = cfg
        self.root_dir = Path(cfg.root_dir)

        self.base_crop_spec = self._build_crop_spec(cfg.crop)

        self.file_index = build_experiment_file_index(
            root_dir=self.root_dir,
            size_tag=cfg.size_tag,
            target_variable=cfg.target_variable,
            conditioning_variables=list(cfg.dynamic_variables),
            target_source=cfg.target_source,
            conditioning_source=cfg.dynamic_source,
        )

        self.split_manifest = load_split_manifest(cfg.split_manifest_path)
        self.dates = get_dates_for_split(self.split_manifest, cfg.split_name)
        self._validate_dates_exist()

        self.static_paths = self._build_static_paths()

        self.target_transform = self._build_target_transform()
        self.dynamic_transforms = self._build_dynamic_transforms()
        self.static_transforms = self._build_static_transforms()

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, Any]:
        date_str = self.dates[index]
        sample_paths: ExperimentSamplePaths = self.file_index[date_str]  # type: ignore

        target = load_target_field(
            target_path=sample_paths["target"],
            target_variable=self.cfg.target_variable,
            target_source=self.cfg.target_source,
        )
        cond_dynamic = load_dynamic_conditioning(
            dynamic_paths=sample_paths["cond_dynamic"],
            variable_order=list(self.cfg.dynamic_variables),
            source=self.cfg.dynamic_source,
        )
        cond_static = load_static_features(
            static_paths={k: self.static_paths[k] for k in self.cfg.static_variables},
            variable_order=list(self.cfg.static_variables),
            source=self.cfg.static_source,
        )

        crop_spec = self._resolve_crop_spec_for_sample(target.shape)
        cropped = crop_sample_fields(
            target=target,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            crop_spec=crop_spec,
        )

        target_arr = np.expand_dims(np.asarray(cropped["target"], dtype=np.float32), axis=0)
        cond_dynamic_arr = np.asarray(cropped["cond_dynamic"], dtype=np.float32)
        cond_static_arr = (
            np.asarray(cropped["cond_static"], dtype=np.float32)
            if cropped["cond_static"] is not None
            else None
        )

        if self.cfg.apply_transforms:
            target_arr = self.target_transform.forward(target_arr)
            cond_dynamic_arr = self._apply_channelwise_transforms(
                cond_dynamic_arr,
                variable_names=list(self.cfg.dynamic_variables),
                transforms=self.dynamic_transforms,
            )
            if cond_static_arr is not None:
                cond_static_arr = self._apply_channelwise_transforms(
                    cond_static_arr,
                    variable_names=list(self.cfg.static_variables),
                    transforms=self.static_transforms,
                )
        time_meta = build_time_feature_metadata(
            date_str,
            use_leap_years=False,
        )
        time_features_arr = build_doy_sincos(
            date_str,
            use_leap_years=False,
        )

        meta: dict[str, Any] = build_feature_metadata(
            target_variable=self.cfg.target_variable,
            dynamic_variables=list(self.cfg.dynamic_variables),
            static_variables=list(self.cfg.static_variables),
        )
        meta.update(
            {
                **time_meta,
                "split_name": self.cfg.split_name,
                "split_manifest_path": str(self.cfg.split_manifest_path),
                "domain_tag": self.cfg.domain_tag,
                "target_source": self.cfg.target_source,
                "dynamic_source": self.cfg.dynamic_source,
                "static_source": self.cfg.static_source,
                "apply_transforms": self.cfg.apply_transforms,
                "region_info": cropped["region_info"],
                "transform_metadata": {
                    "target": self.target_transform.get_metadata().__dict__,
                    "dynamic": {
                        name: transform.get_metadata().__dict__
                        for name, transform in self.dynamic_transforms.items()
                    },
                    "static": {
                        name: transform.get_metadata().__dict__
                        for name, transform in self.static_transforms.items()
                    },
                },
            }
        )

        target_tensor = torch.from_numpy(np.ascontiguousarray(target_arr)).to(torch.float32)
        cond_dynamic_tensor = torch.from_numpy(np.ascontiguousarray(cond_dynamic_arr)).to(torch.float32)
        cond_static_tensor = (
            torch.from_numpy(np.ascontiguousarray(cond_static_arr)).to(torch.float32)
            if cond_static_arr is not None
            else None
        )
        time_features_tensor = torch.from_numpy(
            np.ascontiguousarray(time_features_arr)
        ).to(torch.float32)

        return {
            "target": target_tensor,
            "cond_dynamic": cond_dynamic_tensor,
            "cond_static": cond_static_tensor,
            "cond_coord": cropped["region_info"],
            "meta": meta,
            "time_features": time_features_tensor,
        }
    def _should_use_spatial_shuffle(self) -> bool:
        if not self.cfg.spatial_shuffle_enabled:
            return False
        if not self.cfg.spatial_shuffle_train_only:
            return True
        return self.cfg.split_name == "train"

    def _resolve_crop_spec_for_sample(
        self,
        target_shape: tuple[int, ...],
    ) -> CropSpec | None:
        """
        Resolve the crop spec for a single sample.

        Behavior:
        - If no crop is configured, return None (full domain).
        - If spatial shuffling is disabled for this split, use the configured
        fixed crop.
        - If spatial shuffling is enabled, sample a random crop anchor inside
        the configured cutout domain.
        """
        if self.base_crop_spec is None:
            return None

        if not self._should_use_spatial_shuffle():
            return self.base_crop_spec

        if self.cfg.spatial_shuffle_cutout_domain is None:
            raise ValueError(
                "Spatial shuffling is enabled, but no cutout domain is configured."
            )

        if len(target_shape) < 2:
            raise ValueError(
                f"Expected target array to have at least 2 dimensions, got {target_shape}"
            )

        full_height = int(target_shape[-2])
        full_width = int(target_shape[-1])

        x1, x2, y1, y2 = self.cfg.spatial_shuffle_cutout_domain
        crop_h = self.base_crop_spec.height
        crop_w = self.base_crop_spec.width

        validate_cutout_domain(
            (x1, x2, y1, y2),
            full_shape=(full_height, full_width),
        )

        cutout_width = x2 - x1
        cutout_height = y2 - y1
        if crop_w > cutout_width or crop_h > cutout_height:
            raise ValueError(
                "Configured crop does not fit inside spatial shuffle cutout domain: "
                f"crop={(crop_h, crop_w)} cutout={(cutout_height, cutout_width)}"
            )

        max_anchor_x = x2 - crop_w
        max_anchor_y = y2 - crop_h

        anchor_x = int(np.random.randint(x1, max_anchor_x + 1))
        anchor_y = int(np.random.randint(y1, max_anchor_y + 1))

        crop_spec = CropSpec(
            anchor_y=anchor_y,
            anchor_x=anchor_x,
            height=crop_h,
            width=crop_w,
        )
        validate_crop_within_cutout(
            crop_spec,
            (x1, x2, y1, y2),
            full_shape=(full_height, full_width),
        )
        return crop_spec
    def _validate_dates_exist(self) -> None:
        missing_dates = [date for date in self.dates if date not in self.file_index]
        if missing_dates:
            raise ValueError(
                f"Some split dates are missing from the experiment file index: {missing_dates[:10]}"
            )

    def _build_target_transform(self) -> BaseTransform:
        canonical_variable = canonicalize_variable_name(self.cfg.target_variable)
        transform_name = self._get_default_transform_name(canonical_variable)

        if not self.cfg.apply_transforms:
            return build_transform("identity")

        stats_path = build_stats_output_path(
            split_name=self.cfg.split_tag,
            domain_tag=self.cfg.domain_tag,
            source=self.cfg.target_source,
            variable=canonical_variable,
            transform_name=transform_name,
        )
        stats = load_transform_stats(stats_path)
        clip_min = self._get_clip_min(transform_name)
        kwargs = {"transform_name": transform_name, "stats": stats}
        if clip_min is not None:
            kwargs["clip_min"] = clip_min
        return build_transform(**kwargs)

    def _build_dynamic_transforms(self) -> dict[str, BaseTransform]:
        transforms: dict[str, BaseTransform] = {}
        for variable in self.cfg.dynamic_variables:
            canonical_variable = canonicalize_variable_name(variable)
            transform_name = self._get_default_transform_name(canonical_variable)
            if not self.cfg.apply_transforms:
                transforms[variable] = build_transform("identity")
                continue

            stats_path = build_stats_output_path(
                split_name=self.cfg.split_tag,
                domain_tag=self.cfg.domain_tag,
                source=self.cfg.dynamic_source,
                variable=canonical_variable,
                transform_name=transform_name,
            )
            stats = load_transform_stats(stats_path)
            clip_min = self._get_clip_min(transform_name)
            kwargs = {"transform_name": transform_name, "stats": stats}
            if clip_min is not None:
                kwargs["clip_min"] = clip_min
            transforms[variable] = build_transform(**kwargs)
        return transforms

    def _build_static_transforms(self) -> dict[str, BaseTransform]:
        transforms: dict[str, BaseTransform] = {}
        for variable in self.cfg.static_variables:
            canonical_variable = canonicalize_variable_name(variable)
            transform_name = self._get_default_transform_name(canonical_variable)
            if not self.cfg.apply_transforms:
                transforms[variable] = build_transform("identity")
                continue

            stats_path = build_stats_output_path(
                split_name=self.cfg.split_tag,
                domain_tag=self.cfg.domain_tag,
                source=self.cfg.static_source,
                variable=canonical_variable,
                transform_name=transform_name,
            )
            stats = load_transform_stats(stats_path)
            clip_min = self._get_clip_min(transform_name)
            kwargs = {"transform_name": transform_name, "stats": stats}
            if clip_min is not None:
                kwargs["clip_min"] = clip_min
            transforms[variable] = build_transform(**kwargs)
        return transforms

    def _build_static_paths(self) -> dict[str, Path]:
        static_paths: dict[str, Path] = {}
        for variable in self.cfg.static_variables:
            canonical_variable = canonicalize_variable_name(variable)
            if canonical_variable not in VARIABLE_REGISTRY:
                raise KeyError(
                    f"Static variable '{variable}' is not registered in VARIABLE_REGISTRY"
                )

            spec = VARIABLE_REGISTRY[canonical_variable]
            if self.cfg.static_source not in spec.sources:
                raise KeyError(
                    f"Static source '{self.cfg.static_source}' is not registered for "
                    f"variable '{canonical_variable}'"
                )

            source_spec = spec.sources[self.cfg.static_source]
            static_paths[variable] = (
                self.root_dir
                / f"data_{canonical_variable}"
                / "truth_fullDomain"
                / f"{source_spec.file_prefix}.npz"
            )
        return static_paths

    @staticmethod
    def _get_default_transform_name(variable: str) -> str:
        if variable not in VARIABLE_REGISTRY:
            raise KeyError(f"Variable '{variable}' is not registered in VARIABLE_REGISTRY")
        return VARIABLE_REGISTRY[variable].default_transform

    @staticmethod
    def _get_clip_min(transform_name: str) -> float | None:
        return 0.0 if transform_name == "log_zscore" else None

    @staticmethod
    def _apply_channelwise_transforms(
        array: np.ndarray,
        variable_names: list[str],
        transforms: dict[str, BaseTransform],
    ) -> np.ndarray:
        if array.shape[0] != len(variable_names):
            raise ValueError(
                f"Channel count {array.shape[0]} does not match variable list {variable_names}"
            )

        transformed_channels: list[np.ndarray] = []
        for idx, variable_name in enumerate(variable_names):
            transformed = transforms[variable_name].forward(array[idx])
            transformed_channels.append(np.asarray(transformed, dtype=np.float32))
        return np.stack(transformed_channels, axis=0)

    @staticmethod
    def _build_crop_spec(
        crop: tuple[int, int, int, int] | None,
    ) -> CropSpec | None:
        if crop is None:
            return None

        anchor_y, anchor_x, height, width = crop
        return CropSpec(
            anchor_y=anchor_y,
            anchor_x=anchor_x,
            height=height,
            width=width,
        )


class DanraEra5SmallAdapter:
    """
    Thin adapter wrapper around the first dataset implementation.

    This wrapper exists so later extension points are obvious:
        - multiple split datasets at once
        - dataloader construction
        - temporal stacking
        - context branches
        - random spatial shuffle
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        self.cfg = cfg

    def build_dataset(self) -> DanraEra5SmallDataset:
        return DanraEra5SmallDataset(self.cfg)

    def build_datasets(self) -> dict[str, DanraEra5SmallDataset]:
        split_names = ("train", "val", "test")
        datasets: dict[str, DanraEra5SmallDataset] = {}
        for split_name in split_names:
            split_cfg = AdapterConfig(
                root_dir=self.cfg.root_dir,
                size_tag=self.cfg.size_tag,
                split_manifest_path=self.cfg.split_manifest_path,
                split_name=split_name,
                domain_tag=self.cfg.domain_tag,
                crop=self.cfg.crop,
                spatial_shuffle_enabled=self.cfg.spatial_shuffle_enabled,
                spatial_shuffle_train_only=self.cfg.spatial_shuffle_train_only,
                spatial_shuffle_cutout_domain=self.cfg.spatial_shuffle_cutout_domain,
                target_variable=self.cfg.target_variable,
                target_source=self.cfg.target_source,
                dynamic_variables=self.cfg.dynamic_variables,
                dynamic_source=self.cfg.dynamic_source,
                static_variables=self.cfg.static_variables,
                static_source=self.cfg.static_source,
                apply_transforms=self.cfg.apply_transforms,
                split_stats_tag=self.cfg.split_stats_tag,
            )
            datasets[split_name] = DanraEra5SmallDataset(split_cfg)
        return datasets
