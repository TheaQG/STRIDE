"""
Probability-Matched Mean (PMM) utilities for STRIDE generation.

Responsibilities
----------------
- compute ensemble mean for univariate fields
- compute Probability-Matched Mean (PMM) for univariate ensembles
- provide light validation helpers for ensemble tensor shape handling

Current assumptions
-------------------
- PMM is currently implemented for a single generated target field per case
- ensemble tensors are expected in shape [B, M, H, W] or [M, H, W]
- precipitation-like fields may optionally exclude exact zeros from the pooled
  distribution before quantile matching

Notes
-----
This follows the standard PMM recipe used for precipitation/QPF-style products:
1) compute the ensemble-mean spatial pattern
2) rank that mean field over valid pixels
3) pool all ensemble values over valid pixels and sort them
4) assign pooled quantiles back using the rank pattern of the mean field
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def ensure_bmhw(ens: torch.Tensor) -> torch.Tensor:
    """
    Normalize an ensemble tensor to shape [B, M, H, W].

    Accepted input shapes
    ---------------------
    - [B, M, H, W]
    - [M, H, W]   -> interpreted as a single case [1, M, H, W]

    Returns
    -------
    torch.Tensor
        Tensor with shape [B, M, H, W].
    """
    if ens.dim() == 4:
        return ens
    if ens.dim() == 3:
        return ens.unsqueeze(0)
    raise ValueError(
        f"Expected ensemble tensor with shape [B, M, H, W] or [M, H, W], got {tuple(ens.shape)}"
    )


@torch.no_grad()
def compute_ensemble_mean(ens: torch.Tensor) -> torch.Tensor:
    """
    Compute ensemble mean for a univariate ensemble.

    Parameters
    ----------
    ens:
        Ensemble tensor with shape [B, M, H, W] or [M, H, W].

    Returns
    -------
    torch.Tensor
        Mean field with shape [B, 1, H, W].
    """
    ens_bmhw = ensure_bmhw(ens)
    ens_bmhw = ens_bmhw if torch.is_floating_point(ens_bmhw) else ens_bmhw.float()
    mean_field = ens_bmhw.mean(dim=1, keepdim=True)
    return mean_field


@torch.no_grad()
def pmm_from_ensemble(
    ens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    exclude_zeros: bool = False,
) -> torch.Tensor:
    """
    Compute Probability-Matched Mean (PMM) for a univariate ensemble.

    For each sample b:
    1) Compute the ensemble-mean field μ(x) over members M.
    2) Take the ranks of μ(x) over valid pixels.
    3) Pool all ensemble values across members at valid pixels and sort them.
    4) Assign to each valid pixel the pooled value at the corresponding rank of μ(x).

    Invalid pixels (mask=False) fall back to the ensemble mean.

    Parameters
    ----------
    ens:
        Ensemble tensor with shape [B, M, H, W] or [M, H, W].
    mask:
        Optional boolean mask broadcastable to [B, H, W]. Pixels where mask=False
        are left as the ensemble mean.
    exclude_zeros:
        If True, exclude strictly 0.0 values from the pooled distribution before
        quantile matching. This is often useful for precipitation.

    Returns
    -------
    torch.Tensor
        PMM field with shape [B, 1, H, W].
    """
    ens_bmhw = ensure_bmhw(ens)
    ens_bmhw = ens_bmhw if torch.is_floating_point(ens_bmhw) else ens_bmhw.float()

    B, M, H, W = ens_bmhw.shape
    device = ens_bmhw.device

    if mask is None:
        mask_bhw = torch.ones((B, H, W), dtype=torch.bool, device=device)
    else:
        m = mask
        if m.dtype != torch.bool:
            m = m > 0.5
        while m.dim() < 3:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1)
        if tuple(m.shape[-2:]) != (H, W):
            raise ValueError(
                "Mask spatial shape must match ensemble spatial shape: "
                f"mask={tuple(m.shape)}, ensemble={(B, M, H, W)}"
            )
        mask_bhw = m.to(device)

    mean_field = ens_bmhw.mean(dim=1)  # [B, H, W]
    pmm_out = torch.empty((B, 1, H, W), dtype=ens_bmhw.dtype, device=device)

    for b in range(B):
        valid = mask_bhw[b].reshape(-1)
        if int(valid.sum().item()) == 0:
            pmm_out[b, 0] = mean_field[b]
            continue

        mean_vals = mean_field[b].reshape(-1)[valid]  # [Nv]
        ranks = torch.argsort(torch.argsort(mean_vals))

        pooled = ens_bmhw[b].reshape(M, -1)[:, valid]  # [M, Nv]
        pooled_flat = pooled.reshape(-1)
        if exclude_zeros:
            pooled_flat = pooled_flat[pooled_flat != 0.0]

        if pooled_flat.numel() == 0:
            out_vals = mean_vals
        else:
            pooled_sorted, _ = torch.sort(pooled_flat)
            num_valid = mean_vals.numel()
            q = (ranks.to(torch.float32) + 0.5) / float(num_valid)
            K = pooled_sorted.numel()
            idx = torch.clamp((q * (K - 1)).round().to(torch.long), 0, K - 1)
            out_vals = pooled_sorted[idx]

        flat = mean_field[b].reshape(-1).clone()
        flat[valid] = out_vals
        pmm_out[b, 0] = flat.view(H, W)

    return pmm_out