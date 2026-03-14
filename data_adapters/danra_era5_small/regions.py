"""
Region and cropping utilities for the small DANRA/ERA5 STRIDE adapter.

Current responsibilities:
    - Full-domain shape validation
    - Optional fixed crop
    - Train-time spatial shuffle validation
    - Coordinate bookkeeping

Potential later extensions:
    - Anchored input/output offsets
    - Larger context
    - Non-square regional specializations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CropSpec:
    """
    Fixed crop specification in pixel coordinates.

    Parameters
    ----------
    anchor_y
        Top-row index of the crop.
    anchor_x
        Left-column index of the crop.
    height
        Crop height in pixels.
    width
        Crop width in pixels.
    """

    anchor_y: int
    anchor_x: int
    height: int
    width: int

    @property
    def y_slice(self) -> slice:
        return slice(self.anchor_y, self.anchor_y + self.height)

    @property
    def x_slice(self) -> slice:
        return slice(self.anchor_x, self.anchor_x + self.width)


@dataclass(frozen=True)
class RegionInfo:
    """
    Lightweight metadata describing the selected spatial region.
    """

    full_height: int
    full_width: int
    crop_anchor_y: int
    crop_anchor_x: int
    crop_height: int
    crop_width: int

    def as_dict(self) -> dict[str, int]:
        return {
            "full_height": self.full_height,
            "full_width": self.full_width,
            "crop_anchor_y": self.crop_anchor_y,
            "crop_anchor_x": self.crop_anchor_x,
            "crop_height": self.crop_height,
            "crop_width": self.crop_width,
        }


def validate_cutout_domain(
    cutout_domain: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> None:
    """
    Validate that a cutout domain lies fully within a 2D field.

    Parameters
    ----------
    cutout_domain
        Tuple `(x1, x2, y1, y2)` describing the valid spatial-shuffle region.
    full_shape
        Full 2D shape `(H, W)` of the available field.
    """
    full_height, full_width = full_shape
    x1, x2, y1, y2 = cutout_domain

    if x1 < 0 or y1 < 0:
        raise ValueError(
            f"Cutout domain lower bounds must be non-negative, got {(x1, x2, y1, y2)}"
        )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Cutout domain must satisfy x2 > x1 and y2 > y1, got "
            f"{(x1, x2, y1, y2)}"
        )
    if x2 > full_width or y2 > full_height:
        raise ValueError(
            "Cutout domain exceeds available field shape: "
            f"cutout={(x1, x2, y1, y2)} full_shape={(full_height, full_width)}"
        )


def validate_crop_spec(crop_spec: CropSpec, full_shape: tuple[int, int]) -> None:
    """
    Validate that a crop lies fully within a 2D field.
    """
    full_height, full_width = full_shape

    if crop_spec.anchor_y < 0 or crop_spec.anchor_x < 0:
        raise ValueError(
            f"Crop anchors must be non-negative, got "
            f"anchor_y={crop_spec.anchor_y}, anchor_x={crop_spec.anchor_x}"
        )

    if crop_spec.height <= 0 or crop_spec.width <= 0:
        raise ValueError(
            f"Crop height/width must be positive, got "
            f"height={crop_spec.height}, width={crop_spec.width}"
        )

    if crop_spec.anchor_y + crop_spec.height > full_height:
        raise ValueError(
            f"Crop exceeds array height: anchor_y + height = "
            f"{crop_spec.anchor_y + crop_spec.height}, full_height = {full_height}"
        )

    if crop_spec.anchor_x + crop_spec.width > full_width:
        raise ValueError(
            f"Crop exceeds array width: anchor_x + width = "
            f"{crop_spec.anchor_x + crop_spec.width}, full_width = {full_width}"
        )


def validate_crop_within_cutout(
    crop_spec: CropSpec,
    cutout_domain: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> None:
    """
    Validate that a crop lies fully inside a validated cutout domain.

    Parameters
    ----------
    crop_spec
        Crop to validate.
    cutout_domain
        Tuple `(x1, x2, y1, y2)` describing the allowed spatial-shuffle region.
    full_shape
        Full 2D shape `(H, W)` of the available field.
    """
    validate_crop_spec(crop_spec, full_shape=full_shape)
    validate_cutout_domain(cutout_domain, full_shape=full_shape)

    x1, x2, y1, y2 = cutout_domain

    if crop_spec.anchor_x < x1 or crop_spec.anchor_y < y1:
        raise ValueError(
            "Crop anchor lies outside cutout domain lower bounds: "
            f"crop={(crop_spec.anchor_y, crop_spec.anchor_x, crop_spec.height, crop_spec.width)} "
            f"cutout={(x1, x2, y1, y2)}"
        )
    if crop_spec.anchor_x + crop_spec.width > x2:
        raise ValueError(
            "Crop exceeds cutout domain width bounds: "
            f"crop_right={crop_spec.anchor_x + crop_spec.width} x2={x2}"
        )
    if crop_spec.anchor_y + crop_spec.height > y2:
        raise ValueError(
            "Crop exceeds cutout domain height bounds: "
            f"crop_bottom={crop_spec.anchor_y + crop_spec.height} y2={y2}"
        )


def crop_2d_field(array: np.ndarray, crop_spec: CropSpec) -> np.ndarray:
    """
    Crop a 2D array of shape [H, W].
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array [H, W], got shape {array.shape}")
    height, width = array.shape
    validate_crop_spec(crop_spec, full_shape=(height, width))
    return array[crop_spec.y_slice, crop_spec.x_slice]


def crop_3d_channels(array: np.ndarray, crop_spec: CropSpec) -> np.ndarray:
    """
    Crop a channel-first 3D array of shape [C, H, W].
    """
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array [C, H, W], got shape {array.shape}")

    _, height, width = array.shape
    validate_crop_spec(crop_spec, full_shape=(height, width))
    return array[:, crop_spec.y_slice, crop_spec.x_slice]


def maybe_crop_field(array: np.ndarray, crop_spec: CropSpec | None) -> np.ndarray:
    """
    Apply a fixed crop to either a 2D [H, W] or 3D [C, H, W] array.

    If crop_spec is None, the original array is returned unchanged.
    """
    if crop_spec is None:
        return array

    if array.ndim == 2:
        return crop_2d_field(array, crop_spec)
    if array.ndim == 3:
        return crop_3d_channels(array, crop_spec)

    raise ValueError(
        f"Expected array with ndim 2 or 3 for cropping, got shape {array.shape}"
    )


def build_region_info(
    full_shape: tuple[int, int],
    crop_spec: CropSpec | None,
) -> RegionInfo:
    """
    Build region metadata for the current spatial selection.

    If no crop is provided, the full domain is treated as the selected region.
    """
    full_height, full_width = full_shape

    if crop_spec is None:
        crop_spec = CropSpec(anchor_y=0, anchor_x=0, height=full_height, width=full_width)
    else:
        validate_crop_spec(crop_spec, full_shape=full_shape)

    return RegionInfo(
        full_height=full_height,
        full_width=full_width,
        crop_anchor_y=crop_spec.anchor_y,
        crop_anchor_x=crop_spec.anchor_x,
        crop_height=crop_spec.height,
        crop_width=crop_spec.width,
    )


def crop_sample_fields(
    target: np.ndarray,
    cond_dynamic: np.ndarray,
    cond_static: np.ndarray | None,
    crop_spec: CropSpec | None,
) -> dict[str, Any]:
    """
    Apply the same fixed crop consistently across sample components.

    Parameters
    ----------
    target
        2D target field with shape [H, W].
    cond_dynamic
        Dynamic conditioning with shape [C_dyn, H, W].
    cond_static
        Static conditioning with shape [C_static, H, W], or None.
    crop_spec
        Fixed crop definition. If None, the full domain is returned.

    Returns
    -------
    dict
        Dictionary containing cropped arrays and region metadata:
            {
                "target": ...,
                "cond_dynamic": ...,
                "cond_static": ...,
                "region_info": ...,
            }
    """
    if target.ndim != 2:
        raise ValueError(f"Expected target with shape [H, W], got {target.shape}")
    if cond_dynamic.ndim != 3:
        raise ValueError(
            f"Expected cond_dynamic with shape [C_dyn, H, W], got {cond_dynamic.shape}"
        )
    if cond_static is not None and cond_static.ndim != 3:
        raise ValueError(
            f"Expected cond_static with shape [C_static, H, W], got {cond_static.shape}"
        )

    full_shape = (target.shape[0], target.shape[1])
    if cond_dynamic.shape[1:] != full_shape:
        raise ValueError(
            f"Target and cond_dynamic spatial shapes must match, got "
            f"target={target.shape}, cond_dynamic={cond_dynamic.shape}"
        )
    if cond_static is not None and cond_static.shape[1:] != full_shape:
        raise ValueError(
            f"Target and cond_static spatial shapes must match, got "
            f"target={target.shape}, cond_static={cond_static.shape}"
        )

    cropped_target = maybe_crop_field(target, crop_spec)
    cropped_cond_dynamic = maybe_crop_field(cond_dynamic, crop_spec)
    cropped_cond_static = maybe_crop_field(cond_static, crop_spec) if cond_static is not None else None
    region_info = build_region_info(full_shape=full_shape, crop_spec=crop_spec)

    return {
        "target": cropped_target,
        "cond_dynamic": cropped_cond_dynamic,
        "cond_static": cropped_cond_static,
        "region_info": region_info.as_dict(),
    }
  

