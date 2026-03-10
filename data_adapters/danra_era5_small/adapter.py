"""
Adapter skeleton for the small DANRA/ERA5 STRIDE setup.

This adapter ties together:
    - Canonical date indexing from `paths.py`
    - Split manifests from `splits.py`
    - Feature loading from `features.py`
    - Region selection from `regions.py`
    - Saved statistics + transforms from `statistics/` and `transforms.py`

Current first-experiment setup
------------------------------
- Target: DANRA precipitation
- Conditioning: ERA5 precipitation + ERA5 temperature
- Statics: land-sea mask + topography
- Domain: fixed crop or full domain
- No temporal stacking yet
- No larger context branch yet

This file intentionally starts as a minimal, explicit skeleton rather than a
fully generalized adapter. The goal is to make the data path concrete and easy
to inspect before adding more abstraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data_adapters.danra_era5_small.features import (
    build_feature_metadata,
    load_dynamic_conditioning,
    load_static_features,
    load_target_field,
)
from data_adapters.danra_era5_small.paths import ExperimentSamplePaths, build_experiment_file_index
from data_adapters.danra_era5_small.regions import CropSpec, crop_sample_fields
from data_adapters.danra_era5_small.splits import get_dates_for_split, load_split_manifest
from data_adapters.danra_era5_small.statistics.io import build_stats_output_path, load_transform_stats
from data_adapters.danra_era5_small.transforms import BaseTransform, build_transform
from stride_core.configs.adapter_config import AdapterConfig




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
    }
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        self.cfg = cfg
        self.root_dir = Path(cfg.root_dir)

        self.crop_spec = self._build_crop_spec(cfg.crop)

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

        self.static_paths = {
            "lsm": self.root_dir / "data_lsm" / "truth_fullDomain" / "lsm_full.npz",
            "topo": self.root_dir / "data_topo" / "truth_fullDomain" / "topo_full.npz",
        }

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

        cropped = crop_sample_fields(
            target=target,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            crop_spec=self.crop_spec,
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

        meta: dict[str, Any] = build_feature_metadata(
            target_variable=self.cfg.target_variable,
            dynamic_variables=list(self.cfg.dynamic_variables),
            static_variables=list(self.cfg.static_variables),
        )
        meta.update(
            {
                "date": date_str,
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

        return {
            "target": target_tensor,
            "cond_dynamic": cond_dynamic_tensor,
            "cond_static": cond_static_tensor,
            "cond_coord": cropped["region_info"],
            "meta": meta,
        }

    def _validate_dates_exist(self) -> None:
        missing_dates = [date for date in self.dates if date not in self.file_index]
        if missing_dates:
            raise ValueError(
                f"Some split dates are missing from the experiment file index: {missing_dates[:10]}"
            )

    def _build_target_transform(self) -> BaseTransform:
        if not self.cfg.apply_transforms:
            return build_transform("identity")

        stats_path = build_stats_output_path(
            split_name=self.cfg.split_tag,
            domain_tag=self.cfg.domain_tag,
            source=self.cfg.target_source,
            variable=self.cfg.target_variable,
            transform_name="log_zscore",
        )
        stats = load_transform_stats(stats_path)
        return build_transform(
            transform_name="log_zscore",
            stats=stats,
            clip_min=0.0,
        )

    def _build_dynamic_transforms(self) -> dict[str, BaseTransform]:
        transforms: dict[str, BaseTransform] = {}
        for variable in self.cfg.dynamic_variables:
            transform_name = "log_zscore" if variable == "prcp" else "zscore"
            if not self.cfg.apply_transforms:
                transforms[variable] = build_transform("identity")
                continue

            stats_path = build_stats_output_path(
                split_name=self.cfg.split_tag,
                domain_tag=self.cfg.domain_tag,
                source=self.cfg.dynamic_source,
                variable=variable,
                transform_name=transform_name,
            )
            stats = load_transform_stats(stats_path)
            transforms[variable] = build_transform(
                transform_name=transform_name,
                stats=stats,
                clip_min=0.0,
            )
        return transforms

    def _build_static_transforms(self) -> dict[str, BaseTransform]:
        transforms: dict[str, BaseTransform] = {}
        for variable in self.cfg.static_variables:
            transform_name = "identity" if variable == "lsm" else "zscore"
            if not self.cfg.apply_transforms:
                transforms[variable] = build_transform("identity")
                continue

            stats_path = build_stats_output_path(
                split_name=self.cfg.split_tag,
                domain_tag=self.cfg.domain_tag,
                source=self.cfg.static_source,
                variable=variable,
                transform_name=transform_name,
            )
            stats = load_transform_stats(stats_path)
            transforms[variable] = build_transform(
                transform_name=transform_name,
                stats=stats,
                clip_min=0.0,
            )
        return transforms

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
        split_names = ("train", "valid", "test")
        datasets: dict[str, DanraEra5SmallDataset] = {}
        for split_name in split_names:
            split_cfg = AdapterConfig(
                root_dir=self.cfg.root_dir,
                size_tag=self.cfg.size_tag,
                split_manifest_path=self.cfg.split_manifest_path,
                split_name=split_name,
                domain_tag=self.cfg.domain_tag,
                crop=self.cfg.crop,
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
