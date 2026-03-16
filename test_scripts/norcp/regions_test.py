import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.regions import (
    CropSpec,
    build_default_hr_region,
    build_default_lr_region,
    build_region_info,
    crop_2d_field,
    crop_3d_channels,
    crop_sample_fields,
    maybe_crop_2d,
    maybe_crop_3d,
    validate_crop_within_shape,
    validate_paired_hr_lr_crop,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def plot_region_crop_check(
    *,
    target: np.ndarray,
    cond_dynamic: np.ndarray,
    cond_static: np.ndarray | None,
    cropped_target: np.ndarray,
    cropped_dynamic: np.ndarray,
    cropped_static: np.ndarray | None,
    hr_crop: CropSpec,
    lr_crop: CropSpec,
) -> None:
    """
    Quick visual sanity check for region cropping.

    Shows full-domain and cropped examples for HR target, LR dynamic, and
    optional HR static fields in one compact figure.
    """
    print_section("Quick plot check")

    panels: list[tuple[str, np.ndarray]] = [
        ("Target full", target[0]),
        ("Target cropped", cropped_target[0]),
        ("Dynamic full", cond_dynamic[0]),
        ("Dynamic cropped", cropped_dynamic[0]),
    ]

    if cond_static is not None and cropped_static is not None:
        panels.extend(
            [
                ("Static full", cond_static[0]),
                ("Static cropped", cropped_static[0]),
            ]
        )

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, field) in zip(axes, panels):
        image = ax.imshow(field, origin="lower")
        ax.set_title(title)

        # Draw crop rectangles on full-domain panels
        if title == "Target full":
            rect = Rectangle(
                (hr_crop.left, hr_crop.top),
                hr_crop.width,
                hr_crop.height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        if title == "Dynamic full":
            rect = Rectangle(
                (lr_crop.left, lr_crop.top),
                lr_crop.width,
                lr_crop.height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        if title == "Static full" and cond_static is not None:
            rect = Rectangle(
                (hr_crop.left, hr_crop.top),
                hr_crop.width,
                hr_crop.height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("NorCP region crop check", fontsize=12)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Region metadata smoke test
# -----------------------------------------------------------------------------

print_section("Default region info")
hr_region = build_default_hr_region()
lr_region = build_default_lr_region()
print(hr_region.to_dict())
print(lr_region.to_dict())

custom_crop = CropSpec(top=8, left=4, height=20*3, width=16*3)
custom_region = build_region_info(
    name="norcp_hr_custom",
    full_height=92,
    full_width=68,
    crop=custom_crop,
)
print(custom_region.to_dict())


# -----------------------------------------------------------------------------
# Validation smoke test
# -----------------------------------------------------------------------------

print_section("Crop validation")
validate_crop_within_shape(custom_crop, full_height=92, full_width=68)
paired_lr_crop = validate_paired_hr_lr_crop(
    custom_crop,
    hr_height=92,
    hr_width=68,
    lr_height=23,
    lr_width=17,
    scale_factor=4,
)
print("HR crop:", custom_crop.to_dict())
print("Derived LR crop:", paired_lr_crop.to_dict())


# -----------------------------------------------------------------------------
# 2D crop smoke test
# -----------------------------------------------------------------------------

print_section("2D crop smoke test")
field_2d = np.arange(92 * 68, dtype=np.float32).reshape(92, 68)
cropped_2d = crop_2d_field(field_2d, custom_crop)
print(
    {
        "original_shape": field_2d.shape,
        "cropped_shape": cropped_2d.shape,
        "top_left_value": float(cropped_2d[0, 0]),
        "bottom_right_value": float(cropped_2d[-1, -1]),
    }
)

same_2d = maybe_crop_2d(field_2d, None)
print("maybe_crop_2d(None) shape:", same_2d.shape)


# -----------------------------------------------------------------------------
# 3D crop smoke test
# -----------------------------------------------------------------------------

print_section("3D crop smoke test")
field_3d = np.stack(
    [
        np.full((92, 68), fill_value=1.0, dtype=np.float32),
        np.full((92, 68), fill_value=2.0, dtype=np.float32),
        np.full((92, 68), fill_value=3.0, dtype=np.float32),
    ],
    axis=0,
)
cropped_3d = crop_3d_channels(field_3d, custom_crop)
print(
    {
        "original_shape": field_3d.shape,
        "cropped_shape": cropped_3d.shape,
        "channel_means": [float(cropped_3d[i].mean()) for i in range(cropped_3d.shape[0])],
    }
)

same_3d = maybe_crop_3d(field_3d, None)
print("maybe_crop_3d(None) shape:", same_3d.shape)


# -----------------------------------------------------------------------------
# Paired sample crop smoke test
# -----------------------------------------------------------------------------

print_section("Sample crop smoke test")

# HR sample [C, H, W] = [2, 92, 68]
target = np.stack(
    [
        np.arange(92 * 68, dtype=np.float32).reshape(92, 68),
        np.ones((92, 68), dtype=np.float32) * 10.0,
    ],
    axis=0,
)

# LR sample [C, H, W] = [3, 23, 17]
cond_dynamic = np.stack(
    [
        np.arange(23 * 17, dtype=np.float32).reshape(23, 17),
        np.ones((23, 17), dtype=np.float32) * 20.0,
        np.ones((23, 17), dtype=np.float32) * 30.0,
    ],
    axis=0,
)

# HR static [C, H, W] = [1, 92, 68]
cond_static = np.ones((1, 92, 68), dtype=np.float32) * 100.0

cropped = crop_sample_fields(
    target=target,
    cond_dynamic=cond_dynamic,
    cond_static=cond_static,
    hr_crop=custom_crop,
    lr_crop=None,
    scale_factor=4,
)

print(
    {
        "target_shape": None if cropped["target"] is None else tuple(int(v) for v in cropped["target"].shape),
        "cond_dynamic_shape": None if cropped["cond_dynamic"] is None else tuple(int(v) for v in cropped["cond_dynamic"].shape),
        "cond_static_shape": None if cropped["cond_static"] is None else tuple(int(v) for v in cropped["cond_static"].shape),
    }
)

plot_region_crop_check(
    target=target,
    cond_dynamic=cond_dynamic,
    cond_static=cond_static,
    cropped_target=cropped["target"], # type: ignore[union-attr]
    cropped_dynamic=cropped["cond_dynamic"], # type: ignore[union-attr]
    cropped_static=cropped["cond_static"],
    hr_crop=custom_crop,
    lr_crop=paired_lr_crop,
)


# -----------------------------------------------------------------------------
# Optional-static smoke test
# -----------------------------------------------------------------------------

print_section("Optional static smoke test")
cropped_no_static = crop_sample_fields(
    target=target,
    cond_dynamic=cond_dynamic,
    cond_static=None,
    hr_crop=custom_crop,
    lr_crop=None,
    scale_factor=4,
)
print(
    {
        "target_shape": None if cropped_no_static["target"] is None else tuple(int(v) for v in cropped_no_static["target"].shape),
        "cond_dynamic_shape": None if cropped_no_static["cond_dynamic"] is None else tuple(int(v) for v in cropped_no_static["cond_dynamic"].shape),
        "cond_static": cropped_no_static["cond_static"],
    }
)


# -----------------------------------------------------------------------------
# Final concise summary
# -----------------------------------------------------------------------------

print_section("Final quick sanity summary")
print(
    {
        "hr_region": hr_region.to_dict(),
        "lr_region": lr_region.to_dict(),
        "hr_crop": custom_crop.to_dict(),
        "lr_crop": paired_lr_crop.to_dict(),
        "cropped_target_shape": None if cropped["target"] is None else tuple(int(v) for v in cropped["target"].shape),
        "cropped_dynamic_shape": None if cropped["cond_dynamic"] is None else tuple(int(v) for v in cropped["cond_dynamic"].shape),
        "cropped_static_shape": None if cropped["cond_static"] is None else tuple(int(v) for v in cropped["cond_static"].shape),
    }
)