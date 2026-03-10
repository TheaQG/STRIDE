

"""
EDM sampler for STRIDE.

This module implements a clean Karras-style EDM sampler with Heun updates,
adapted from the useful EDM-specific parts of the legacy `score_sampling.py`
while intentionally dropping legacy complexity that STRIDE v1 does not keep:

Current scope
-------------
- Karras sigma schedule
- optional churn / stochasticity injection
- Euler + Heun correction updates
- dynamic + optional static conditioning
- optional FiLM metadata passthrough (`doy`, `variable_labels`)

The sampler expects a model with a forward signature like:

    model(x, sigma, cond_dynamic, cond_static=None, doy=None, variable_labels=None)

which matches the current `EDMPrecondUNet` wrapper.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


@torch.no_grad()
def build_karras_sigma_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build the decreasing Karras sigma schedule and append a final zero.

    Returns
    -------
    torch.Tensor
        Shape `[num_steps + 1]`, where the last entry is zero.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError(
            f"sigma_min and sigma_max must be positive, got {(sigma_min, sigma_max)}"
        )
    if sigma_max < sigma_min:
        raise ValueError(
            f"sigma_max must be >= sigma_min, got {(sigma_min, sigma_max)}"
        )
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")

    step_idx = torch.arange(num_steps, device=device, dtype=dtype)
    ramp = step_idx / max(num_steps - 1, 1)

    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)
    return sigmas


@torch.no_grad()
def edm_sampler(
    model: nn.Module,
    cond_dynamic: torch.Tensor,
    cond_static: torch.Tensor | None = None,
    *,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
    doy: torch.Tensor | None = None,
    variable_labels: torch.Tensor | None = None,
    return_intermediates: bool = False,
) -> torch.Tensor | dict[str, Any]:
    """
    Sample from an EDM-preconditioned model using Heun updates.

    Parameters
    ----------
    model:
        EDM-preconditioned model.
    cond_dynamic:
        Dynamic conditioning tensor with shape `[B, C_dyn, H, W]`.
    cond_static:
        Optional static conditioning tensor with shape `[B, C_static, H, W]`.
    num_steps:
        Number of Karras steps before the final sigma=0 point.
    sigma_min, sigma_max, rho:
        Karras schedule parameters.
    S_churn, S_min, S_max, S_noise:
        EDM stochasticity / churn parameters.
    doy:
        Optional day-of-year tensor passed through to the model.
    variable_labels:
        Optional variable label tensor passed through to the model.
    return_intermediates:
        If True, return a dictionary containing the final sample and intermediate
        trajectory snapshots.

    Returns
    -------
    torch.Tensor or dict[str, Any]
        Final sample tensor by default. If `return_intermediates=True`, returns
        a dictionary with keys `sample`, `sigmas`, and `trajectory`.
    """
    if cond_dynamic.ndim != 4:
        raise ValueError(
            f"Expected cond_dynamic with shape [B, C, H, W], got {tuple(cond_dynamic.shape)}"
        )
    if cond_static is not None:
        if cond_static.ndim != 4:
            raise ValueError(
                f"Expected cond_static with shape [B, C, H, W], got {tuple(cond_static.shape)}"
            )
        if cond_static.shape[0] != cond_dynamic.shape[0]:
            raise ValueError(
                f"Batch mismatch between cond_dynamic and cond_static: {cond_dynamic.shape[0]} vs {cond_static.shape[0]}"
            )
        if cond_static.shape[2:] != cond_dynamic.shape[2:]:
            raise ValueError(
                "Spatial mismatch between cond_dynamic and cond_static: "
                f"{tuple(cond_dynamic.shape[2:])} vs {tuple(cond_static.shape[2:])}"
            )

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0.0 <= S_churn:
        raise ValueError(f"S_churn must be >= 0, got {S_churn}")
    if S_noise <= 0:
        raise ValueError(f"S_noise must be positive, got {S_noise}")

    device = cond_dynamic.device
    dtype = cond_dynamic.dtype
    batch_size = cond_dynamic.shape[0]
    height, width = cond_dynamic.shape[2:]

    out_channels = getattr(getattr(model, "spec", None), "out_channels", None)
    if out_channels is None:
        raise AttributeError(
            "The sampler could not infer `out_channels` from `model.spec.out_channels`."
        )

    sigmas = build_karras_sigma_schedule(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        dtype=dtype,
    )

    x = torch.randn(
        batch_size,
        int(out_channels),
        height,
        width,
        device=device,
        dtype=dtype,
    ) * sigmas[0]

    trajectory: list[torch.Tensor] = []
    if return_intermediates:
        trajectory.append(x.detach().cpu())

    for step_idx in range(num_steps):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        x_in = x

        if S_min <= float(sigma) <= S_max and S_churn > 0.0:
            gamma = min(S_churn / num_steps, math.sqrt(2.0) - 1.0)
            sigma_hat = sigma * (1.0 + gamma)
            eps = torch.randn_like(x) * S_noise
            x_in = x + eps * torch.sqrt(torch.clamp(sigma_hat**2 - sigma**2, min=0.0))
        else:
            sigma_hat = sigma

        sigma_hat_vec = torch.full(
            (batch_size,),
            float(sigma_hat),
            device=device,
            dtype=dtype,
        )

        denoised = model(
            x=x_in,
            sigma=sigma_hat_vec,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            doy=doy,
            variable_labels=variable_labels,
        )

        d_cur = (x_in - denoised) / sigma_hat
        x_euler = x_in + (sigma_next - sigma_hat) * d_cur

        if step_idx == num_steps - 1:
            x = x_euler
            if return_intermediates:
                trajectory.append(x.detach().cpu())
            break

        sigma_next_vec = torch.full(
            (batch_size,),
            float(sigma_next),
            device=device,
            dtype=dtype,
        )
        denoised_next = model(
            x=x_euler,
            sigma=sigma_next_vec,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            doy=doy,
            variable_labels=variable_labels,
        )
        d_next = (x_euler - denoised_next) / sigma_next

        x = x_in + (sigma_next - sigma_hat) * 0.5 * (d_cur + d_next)

        if return_intermediates:
            trajectory.append(x.detach().cpu())

    if not return_intermediates:
        return x

    return {
        "sample": x,
        "sigmas": sigmas,
        "trajectory": trajectory,
    }