import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_adapters.norcp.features import (
    build_day_of_year_features,
    build_loaded_field_metadata,
    load_dynamic_stack,
    load_field_at_timestamp,
    load_static_stack,
    load_target_stack,
)
from data_adapters.norcp.indexing import build_norcp_sample_index, describe_sample_index
from stride_core.plotting.colormaps import get_variable_cmap


ROOT_DIR = Path("/Users/au728490/Data/NorCP/cropped")
SCENARIO_NAME = "ECMWF-ERAINT"
TEMPORAL_TAG = "6hr"

# Use canonical STRIDE variable names here.
TARGET_VARIABLES = ["prcp", "temp"]
DYNAMIC_VARIABLES = ["prcp", "temp", "hus1000", "ta1000", "ua1000", "va1000", "zg1000"]

# Precipitation is a 6-hour mean-rate field represented at 03/09/15/21 in the raw files.
# Shift it to interval start so it aligns with point variables at 00/06/12/18.
TARGET_TIME_OFFSETS = {"prcp": -3.0}
DYNAMIC_TIME_OFFSETS = {"prcp": -3.0}

# Optional static fields. Historical runs can use HR orography from the fixed-field directory,
# while future runs may omit topo entirely.
STATIC_FILES = {
    "topo": ROOT_DIR / SCENARIO_NAME / "3km" / "fx" / "orog" / "orog_3km_fx.nc",
}


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def plot_field_triplet(
    *,
    target_stack: np.ndarray,
    dynamic_stack: np.ndarray,
    static_stack: np.ndarray | None,
    target_variables: list[str],
    dynamic_variables: list[str],
    static_variables: list[str],
    timestamp_label: str,
) -> None:
    """
    Quick visual sanity check for NorCP fields.

    Creates one figure per field so it is easy to verify orientation,
    coastlines/orography structure, and rough physical plausibility.
    """
    print_section("Quick plot check")

    n_target = len(target_variables)
    n_dynamic = len(dynamic_variables)
    n_static = 0 if static_stack is None else len(static_variables)

    total_fields = n_target + n_dynamic + n_static

    cols = 4
    rows = int(np.ceil(total_fields / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()

    plot_index = 0

    # Target fields
    for idx, variable in enumerate(target_variables):
        ax = axes[plot_index]
        image = ax.imshow(target_stack[idx], origin="lower", cmap=get_variable_cmap(variable))
        ax.set_title(f"Target: {variable}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plot_index += 1

    # Dynamic fields
    for idx, variable in enumerate(dynamic_variables):
        ax = axes[plot_index]
        image = ax.imshow(dynamic_stack[idx], origin="lower", cmap=get_variable_cmap(variable))
        ax.set_title(f"Dynamic: {variable}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plot_index += 1

    # Static fields
    if static_stack is not None:
        for idx, variable in enumerate(static_variables):
            ax = axes[plot_index]
            image = ax.imshow(static_stack[idx], origin="lower", cmap=get_variable_cmap(variable))
            ax.set_title(f"Static: {variable}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            plot_index += 1

    # Turn off unused axes
    for ax in axes[plot_index:]:
        ax.axis("off")

    fig.suptitle(f"NorCP feature check | {timestamp_label}", fontsize=14)
    plt.tight_layout()
    plt.show()


sample_index = build_norcp_sample_index(
    root_dir=ROOT_DIR,
    scenario_name=SCENARIO_NAME,
    target_variables=TARGET_VARIABLES,
    dynamic_variables=DYNAMIC_VARIABLES,
    target_spatial_tag="3km",
    dynamic_spatial_tag="12km",
    temporal_tag=TEMPORAL_TAG,
    static_files=STATIC_FILES,  # type: ignore
    target_time_offsets=TARGET_TIME_OFFSETS,
    dynamic_time_offsets=DYNAMIC_TIME_OFFSETS,
)

print_section("Sample index summary")
print(f"n_samples = {len(sample_index)}")
print(describe_sample_index(sample_index, limit=2))

if not sample_index:
    raise RuntimeError("Sample index is empty. Cannot run feature-loading smoke test.")

sample = sample_index[0]
timestamp = sample.timestamp
print_section("Chosen timestamp")
print(timestamp.isoformat())


print_section("Single-field smoke test: HR precipitation")
hr_pr_field = load_field_at_timestamp(
    file_path=sample.target_files["prcp"],
    variable="prcp",
    source="NORCP_HR",
    timestamp=timestamp,
    file_time_offset_hours=TARGET_TIME_OFFSETS.get("prcp", 0.0),
)
print(
    {
        "shape": hr_pr_field.array.shape,
        "dtype": str(hr_pr_field.array.dtype),
        "min": float(hr_pr_field.array.min()),
        "max": float(hr_pr_field.array.max()),
        "path": str(hr_pr_field.path),
    }
)


print_section("Single-field smoke test: LR humidity")
lr_hus_field = load_field_at_timestamp(
    file_path=sample.dynamic_files["hus1000"],
    variable="hus1000",
    source="NORCP_LR",
    timestamp=timestamp,
    file_time_offset_hours=DYNAMIC_TIME_OFFSETS.get("hus1000", 0.0),
)
print(
    {
        "shape": lr_hus_field.array.shape,
        "dtype": str(lr_hus_field.array.dtype),
        "min": float(lr_hus_field.array.min()),
        "max": float(lr_hus_field.array.max()),
        "path": str(lr_hus_field.path),
    }
)


print_section("Target stack")
target_stack, target_fields = load_target_stack(
    target_files={k: str(v) for k, v in sample.target_files.items()},
    timestamp=timestamp,
    source="NORCP_HR",
    variable_order=TARGET_VARIABLES,
    variable_time_offsets=TARGET_TIME_OFFSETS,
)
print(f"target_stack.shape = {target_stack.shape}")
print(build_loaded_field_metadata(target_fields))


print_section("Dynamic stack")
dynamic_stack, dynamic_fields = load_dynamic_stack(
    dynamic_files={k: str(v) for k, v in sample.dynamic_files.items()},
    timestamp=timestamp,
    source="NORCP_LR",
    variable_order=DYNAMIC_VARIABLES,
    variable_time_offsets=DYNAMIC_TIME_OFFSETS,
)
print(f"dynamic_stack.shape = {dynamic_stack.shape}")
print(build_loaded_field_metadata(dynamic_fields))


print_section("Static stack")
static_stack, static_fields = load_static_stack(
    static_files={k: str(v) for k, v in sample.static_files.items()},
    source_by_variable={"topo": "NORCP_STATIC"},
    variable_order=["topo"],
)
if static_stack is None:
    print("No static fields loaded.")
else:
    print(f"static_stack.shape = {static_stack.shape}")
print(build_loaded_field_metadata(static_fields))


print_section("Time features")
time_features = build_day_of_year_features(timestamp)
print(time_features)
print(f"time_features.shape = {time_features.shape}")


plot_field_triplet(
    target_stack=target_stack,
    dynamic_stack=dynamic_stack,
    static_stack=static_stack,
    target_variables=TARGET_VARIABLES,
    dynamic_variables=DYNAMIC_VARIABLES,
    static_variables=["topo"] if static_stack is not None else [],
    timestamp_label=timestamp.isoformat(),
)


print_section("Final quick sanity summary")
print(
    {
        "timestamp": timestamp.isoformat(),
        "target_stack_shape": tuple(int(v) for v in target_stack.shape),
        "dynamic_stack_shape": tuple(int(v) for v in dynamic_stack.shape),
        "static_stack_shape": None if static_stack is None else tuple(int(v) for v in static_stack.shape),
        "time_features_shape": tuple(int(v) for v in time_features.shape),
    }
)