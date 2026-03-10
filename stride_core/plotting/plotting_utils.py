"""
Shared plotting helpers for STRIDE.

These utilities keep layout and small plotting mechanics out of training,
generation, and evaluation modules.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np



def get_channel_name(names: list[str], idx: int, prefix: str) -> str:
    if idx < len(names):
        return str(names[idx])
    return f"{prefix}[{idx}]"



def compute_shared_range(fields: list[np.ndarray]) -> tuple[float, float]:
    mins = [float(np.nanmin(np.asarray(field))) for field in fields]
    maxs = [float(np.nanmax(np.asarray(field))) for field in fields]
    return min(mins), max(maxs)



def find_matching_dynamic_index(
    target_name: str | None,
    dynamic_names: list[str],
) -> int | None:
    if target_name is None:
        return None
    target_name_norm = str(target_name).lower()
    for idx, name in enumerate(dynamic_names):
        if str(name).lower() == target_name_norm:
            return idx
    return None



def overlay_lsm_contour(ax: plt.Axes, lsm_case: np.ndarray | None) -> None:
    if lsm_case is None:
        return

    mask = np.asarray(lsm_case)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        return

    try:
        ax.contour(
            mask,
            levels=[0.5],
            colors="#2f2f2f",
            linewidths=0.45,
            origin="lower",
            alpha=0.8,
        )
    except ValueError:
        return



def add_side_boxplot(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
) -> None:
    clean = np.asarray(values, dtype=float).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        ax.axis("off")
        return

    ax.boxplot(
        clean,
        vert=True,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        boxprops={"facecolor": "#efefef", "edgecolor": "#555555", "linewidth": 0.8},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        medianprops={"color": "#222222", "linewidth": 1.0},
        meanprops={
            "marker": "x",
            "markeredgecolor": "#c44e52",
            "markerfacecolor": "#c44e52",
            "markersize": 4,
            "markeredgewidth": 0.8,
        },
        flierprops={
            "marker": ".",
            "markersize": 1.2,
            "markerfacecolor": "#2b8c67",
            "markeredgecolor": "#2b8c67",
            "alpha": 0.5,
        },
    )
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#666666")
    ax.tick_params(labelsize=11)



def plot_field_with_colorbar_and_optional_boxplot(
    ax: plt.Axes,
    field: np.ndarray,
    *,
    title: str,
    cmap: Any,
    vmin: float | None = None,
    vmax: float | None = None,
    lsm_overlay: np.ndarray | None = None,
    add_boxplot: bool = False,
) -> None:
    imshow_kwargs: dict[str, Any] = {"origin": "lower", "cmap": cmap}
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax

    im = ax.imshow(field, **imshow_kwargs)
    overlay_lsm_contour(ax, lsm_overlay)
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)

    if add_boxplot:
        bax = divider.append_axes("right", size="9%", pad=0.05)
        add_side_boxplot(
            bax,
            np.asarray(field),
            vmin=vmin,
            vmax=vmax,
        )
        cax = divider.append_axes("right", size="4%", pad=0.05)
    else:
        cax = divider.append_axes("right", size="4%", pad=0.04)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=12)
