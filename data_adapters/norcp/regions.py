

"""
Region and crop utilities for the NorCP STRIDE adapter.

This module provides spatial bookkeeping for NorCP fields, including:
- full-domain region descriptions
- optional crop specifications
- validation that crops stay within array bounds
- channel-first crop helpers for [C, H, W] tensors
- metadata builders for later adapter integration

Current NorCP conventions
-------------------------
- HR full-domain shape: [H, W] = [92, 68]
- LR full-domain shape: [H, W] = [23, 17]
- Arrays are treated consistently as [H, W]

This module is intentionally independent of file loading and transforms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


# -------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------

@dataclass(frozen=True)
class CropSpec:
    """
    Spatial crop specification in [H, W] convention.
    """

    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RegionInfo:
    """
    Lightweight description of the spatial region used for one sample.
    """

    name: str
    full_height: int
    full_width: int
    crop: dict[str, int] | None = None

    @property
    def is_cropped(self) -> bool:
        return self.crop is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "full_height": self.full_height,
            "full_width": self.full_width,
            "crop": self.crop,
            "is_cropped": self.is_cropped,
        }


# -------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------


def validate_positive_shape(height: int, width: int) -> None:
    """
    Ensure a spatial shape is strictly positive.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Spatial shape must be positive, got height={height}, width={width}"
        )



def validate_crop_spec(crop: CropSpec) -> None:
    """
    Validate that crop entries are non-negative and have positive extent.
    """
    if crop.top < 0 or crop.left < 0:
        raise ValueError(
            f"Crop top/left must be non-negative, got top={crop.top}, left={crop.left}"
        )
    if crop.height <= 0 or crop.width <= 0:
        raise ValueError(
            f"Crop height/width must be positive, got height={crop.height}, width={crop.width}"
        )



def validate_crop_within_shape(
    crop: CropSpec,
    *,
    full_height: int,
    full_width: int,
) -> None:
    """
    Validate that a crop lies within a full-domain shape.
    """
    validate_positive_shape(full_height, full_width)
    validate_crop_spec(crop)

    if crop.bottom > full_height or crop.right > full_width:
        raise ValueError(
            "Crop exceeds domain bounds: "
            f"crop=(top={crop.top}, left={crop.left}, height={crop.height}, width={crop.width}), "
            f"domain=(height={full_height}, width={full_width})"
        )



def validate_paired_hr_lr_crop(
    hr_crop: CropSpec,
    *,
    hr_height: int,
    hr_width: int,
    lr_height: int,
    lr_width: int,
    scale_factor: int = 4,
) -> CropSpec:
    """
    Validate an HR crop and derive the corresponding LR crop.

    The HR crop must align with the LR grid according to the integer scale
    factor. This is useful later when NorCP training uses HR crops together
    with corresponding LR conditioning patches.
    """
    validate_crop_within_shape(hr_crop, full_height=hr_height, full_width=hr_width)

    lr_crop = derive_scaled_crop(
        hr_crop,
        source_height=hr_height,
        source_width=hr_width,
        target_height=lr_height,
        target_width=lr_width,
        scale_factor=scale_factor,
    )
    return lr_crop

# -------------------------------------------------------------------
# Crop helpers for arrays
# -------------------------------------------------------------------


def derive_scaled_crop(
    source_crop: CropSpec,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    scale_factor: int,
) -> CropSpec:
    """
    Derive a crop on a target grid from a crop on a source grid using an
    integer scale factor.

    This is the general spatial helper that should be reused by adapters when a
    target-domain crop and a conditioning-domain crop are tied by a strict grid
    scaling relation.

    Notes
    -----
    - For the current NorCP setup, HR -> LR uses `scale_factor=4`.
    - This helper intentionally enforces exact divisibility to avoid silent
      shape mismatches.
    - Future larger-context conditioning setups may choose not to use this
      helper and instead specify LR crops explicitly.
    """
    validate_crop_within_shape(
        source_crop,
        full_height=source_height,
        full_width=source_width,
    )

    if source_crop.top % scale_factor != 0:
        raise ValueError(
            f"Crop top={source_crop.top} is not divisible by scale_factor={scale_factor}"
        )
    if source_crop.left % scale_factor != 0:
        raise ValueError(
            f"Crop left={source_crop.left} is not divisible by scale_factor={scale_factor}"
        )
    if source_crop.height % scale_factor != 0:
        raise ValueError(
            f"Crop height={source_crop.height} is not divisible by scale_factor={scale_factor}"
        )
    if source_crop.width % scale_factor != 0:
        raise ValueError(
            f"Crop width={source_crop.width} is not divisible by scale_factor={scale_factor}"
        )

    target_crop = CropSpec(
        top=source_crop.top // scale_factor,
        left=source_crop.left // scale_factor,
        height=source_crop.height // scale_factor,
        width=source_crop.width // scale_factor,
    )
    validate_crop_within_shape(
        target_crop,
        full_height=target_height,
        full_width=target_width,
    )
    return target_crop



def derive_lr_crop_from_hr_crop(
    hr_crop: CropSpec,
    *,
    hr_height: int = 92,
    hr_width: int = 68,
    lr_height: int = 23,
    lr_width: int = 17,
    scale_factor: int = 4,
) -> CropSpec:
    """
    Convenience wrapper for the current NorCP HR -> LR crop relation.

    This is appropriate for the present standard NorCP setup where LR and HR
    crops are tied by the native 4x grid ratio. For future larger-context
    conditioning, LR crop handling should be configurable independently.
    """
    return derive_scaled_crop(
        hr_crop,
        source_height=hr_height,
        source_width=hr_width,
        target_height=lr_height,
        target_width=lr_width,
        scale_factor=scale_factor,
    )


# -------------------------------------------------------------------
# Crop helpers for arrays
# -------------------------------------------------------------------


def crop_2d_field(array: np.ndarray, crop: CropSpec) -> np.ndarray:
    """
    Crop one 2D field in [H, W] format.
    """
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array for crop_2d_field, got shape {array.shape}")

    full_height, full_width = array.shape
    validate_crop_within_shape(crop, full_height=full_height, full_width=full_width)
    return np.asarray(array[crop.top:crop.bottom, crop.left:crop.right])



def crop_3d_channels(array: np.ndarray, crop: CropSpec) -> np.ndarray:
    """
    Crop one channel-first tensor in [C, H, W] format.
    """
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array for crop_3d_channels, got shape {array.shape}")

    _, full_height, full_width = array.shape
    validate_crop_within_shape(crop, full_height=full_height, full_width=full_width)
    return np.asarray(array[:, crop.top:crop.bottom, crop.left:crop.right])



def maybe_crop_2d(array: np.ndarray, crop: Optional[CropSpec]) -> np.ndarray:
    """
    Crop a 2D field if a crop spec is provided; otherwise return it unchanged.
    """
    if crop is None:
        return np.asarray(array)
    return crop_2d_field(array, crop)



def maybe_crop_3d(array: np.ndarray, crop: Optional[CropSpec]) -> np.ndarray:
    """
    Crop a [C, H, W] tensor if a crop spec is provided; otherwise return it unchanged.
    """
    if crop is None:
        return np.asarray(array)
    return crop_3d_channels(array, crop)


# -------------------------------------------------------------------
# Region metadata helpers
# -------------------------------------------------------------------


def build_region_info(
    *,
    name: str,
    full_height: int,
    full_width: int,
    crop: Optional[CropSpec] = None,
) -> RegionInfo:
    """
    Build a region description for one domain/crop configuration.
    """
    validate_positive_shape(full_height, full_width)
    if crop is not None:
        validate_crop_within_shape(crop, full_height=full_height, full_width=full_width)

    return RegionInfo(
        name=name,
        full_height=full_height,
        full_width=full_width,
        crop=None if crop is None else crop.to_dict(),
    )



def build_default_hr_region(crop: Optional[CropSpec] = None) -> RegionInfo:
    """
    Build the default NorCP HR region info.
    """
    return build_region_info(
        name="norcp_hr",
        full_height=92,
        full_width=68,
        crop=crop,
    )



def build_default_lr_region(crop: Optional[CropSpec] = None) -> RegionInfo:
    """
    Build the default NorCP LR region info.
    """
    return build_region_info(
        name="norcp_lr",
        full_height=23,
        full_width=17,
        crop=crop,
    )


# -------------------------------------------------------------------
# Sample-level crop application
# -------------------------------------------------------------------


def crop_sample_fields(
    *,
    target: np.ndarray,
    cond_dynamic: np.ndarray,
    cond_static: np.ndarray | None,
    hr_crop: Optional[CropSpec] = None,
    lr_crop: Optional[CropSpec] = None,
    scale_factor: int = 4,
) -> dict[str, np.ndarray | None]:
    """
    Apply region crops to one NorCP sample.

    Parameters
    ----------
    target:
        HR tensor [C, H, W].
    cond_dynamic:
        LR tensor [C, H, W].
    cond_static:
        Optional static tensor [C, H, W], typically on the HR grid.
    hr_crop:
        Optional HR crop.
    lr_crop:
        Optional LR crop. If omitted but `hr_crop` is provided, the LR crop is
        derived automatically from the HR crop using `scale_factor`.
    scale_factor:
        Integer HR/LR ratio. NorCP currently uses 4.
    """
    target = np.asarray(target)
    cond_dynamic = np.asarray(cond_dynamic)
    cond_static = None if cond_static is None else np.asarray(cond_static)

    if target.ndim != 3:
        raise ValueError(f"Expected target with shape [C,H,W], got {target.shape}")
    if cond_dynamic.ndim != 3:
        raise ValueError(
            f"Expected cond_dynamic with shape [C,H,W], got {cond_dynamic.shape}"
        )
    if cond_static is not None and cond_static.ndim != 3:
        raise ValueError(
            f"Expected cond_static with shape [C,H,W], got {cond_static.shape}"
        )

    _, hr_height, hr_width = target.shape
    _, lr_height, lr_width = cond_dynamic.shape

    derived_lr_crop = lr_crop
    if hr_crop is not None and lr_crop is None:
        derived_lr_crop = validate_paired_hr_lr_crop(
            hr_crop,
            hr_height=hr_height,
            hr_width=hr_width,
            lr_height=lr_height,
            lr_width=lr_width,
            scale_factor=scale_factor,
        )
    elif hr_crop is not None and lr_crop is not None:
        expected_lr_crop = validate_paired_hr_lr_crop(
            hr_crop,
            hr_height=hr_height,
            hr_width=hr_width,
            lr_height=lr_height,
            lr_width=lr_width,
            scale_factor=scale_factor,
        )
        if lr_crop != expected_lr_crop:
            raise ValueError(
                f"Provided lr_crop {lr_crop} does not match HR-derived crop {expected_lr_crop}"
            )
        derived_lr_crop = lr_crop
    elif hr_crop is None and lr_crop is not None:
        validate_crop_within_shape(lr_crop, full_height=lr_height, full_width=lr_width)
        derived_lr_crop = lr_crop

    cropped_target = maybe_crop_3d(target, hr_crop)
    cropped_dynamic = maybe_crop_3d(cond_dynamic, derived_lr_crop)
    cropped_static = None if cond_static is None else maybe_crop_3d(cond_static, hr_crop)

    return {
        "target": cropped_target,
        "cond_dynamic": cropped_dynamic,
        "cond_static": cropped_static,
    }