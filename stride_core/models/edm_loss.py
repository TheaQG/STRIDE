"""
EDM loss for STRIDE.

This module implements the Elucidated Diffusion Model (EDM) training loss from
Karras et al. (2022), adapted to the STRIDE batch contract and model wrapper.

Design goals
------------
- work with the new STRIDE adapter batch structure
- stay independent of dataset-specific assumptions
- keep the interface clean and easy to test
- avoid carrying over legacy SBGM / CFG / residual-era complexity

Current expected batch contract
-------------------------------
The loss expects at minimum:

    batch = {
        "target": torch.Tensor,         # [B, C_out, H, W]
        "cond_dynamic": torch.Tensor,   # [B, C_dyn, H, W]
        "cond_static": torch.Tensor | None,
        "meta": dict,
    }

Optional metadata currently supported:
- `meta["doy"]` or `meta["day_of_year"]` for future FiLM-style conditioning
- `meta["variable_labels"]` for future categorical FiLM conditioning

Notes
-----
The model passed to this loss is expected to be an `EDMPrecondUNet`-style model
with a forward signature like:

    model(x, sigma, cond_dynamic, cond_static=None, doy=None, variable_labels=None)

The loss computes:

    x_noisy = x_clean + sigma * noise
    pred = model(x_noisy, sigma, ...)
    loss = weight(sigma) * ||pred - x_clean||^2

with the EDM weighting:

    weight(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EDMLoss(nn.Module):
    """
    Karras EDM training loss adapted to STRIDE.

    Parameters
    ----------
    p_mean:
        Mean of the log-normal distribution used to sample sigma.
    p_std:
        Standard deviation of the log-normal sigma distribution.
    sigma_data:
        Data standard deviation used in the EDM weighting formula.
    reduction:
        Reduction applied to the final loss. Supported: `"mean"`, `"sum"`,
        `"none"`.
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if p_std <= 0:
            raise ValueError(f"p_std must be positive, got {p_std}")
        if sigma_data <= 0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be one of ['mean', 'sum', 'none'], got {reduction}"
            )

        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)
        self.reduction = reduction

    def sample_sigma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Sample sigma values from the EDM log-normal distribution.

        Returns
        -------
        torch.Tensor
            Shape `[B]`.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        rnd_normal = torch.randn(batch_size, device=device, dtype=dtype)
        return torch.exp(rnd_normal * self.p_std + self.p_mean)

    def edm_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the EDM loss weighting for sigma.

        Parameters
        ----------
        sigma:
            Tensor with shape `[B]` or `[B, 1, 1, 1]`.
        """
        sigma_data_sq = self.sigma_data ** 2
        return (sigma ** 2 + sigma_data_sq) / ((sigma * self.sigma_data) ** 2)

    @staticmethod
    def _get_optional_meta_tensor(
        meta: dict[str, Any],
        keys: tuple[str, ...],
        device: torch.device,
    ) -> torch.Tensor | None:
        for key in keys:
            if key in meta and meta[key] is not None:
                value = meta[key]
                if isinstance(value, torch.Tensor):
                    return value.to(device=device)
                return torch.as_tensor(value, device=device)
        return None

    def _extract_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        required_keys = ("target", "cond_dynamic", "meta")
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise KeyError(f"Batch is missing required keys: {missing}")

        target = batch["target"]
        cond_dynamic = batch["cond_dynamic"]
        cond_static = batch.get("cond_static")
        meta = batch["meta"]

        if not isinstance(target, torch.Tensor):
            raise TypeError(f"batch['target'] must be a torch.Tensor, got {type(target)}")
        if not isinstance(cond_dynamic, torch.Tensor):
            raise TypeError(
                f"batch['cond_dynamic'] must be a torch.Tensor, got {type(cond_dynamic)}"
            )
        if cond_static is not None and not isinstance(cond_static, torch.Tensor):
            raise TypeError(
                f"batch['cond_static'] must be a torch.Tensor or None, got {type(cond_static)}"
            )
        if not isinstance(meta, dict):
            raise TypeError(f"batch['meta'] must be a dict, got {type(meta)}")

        if target.ndim != 4:
            raise ValueError(
                f"batch['target'] must have shape [B, C, H, W], got {tuple(target.shape)}"
            )
        if cond_dynamic.ndim != 4:
            raise ValueError(
                "batch['cond_dynamic'] must have shape [B, C, H, W], got "
                f"{tuple(cond_dynamic.shape)}"
            )
        if cond_static is not None and cond_static.ndim != 4:
            raise ValueError(
                "batch['cond_static'] must have shape [B, C, H, W], got "
                f"{tuple(cond_static.shape)}"
            )

        return target, cond_dynamic, cond_static, meta

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        *,
        sigma: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute the EDM loss for one STRIDE batch.

        Parameters
        ----------
        model:
            EDM-preconditioned model.
        batch:
            STRIDE batch dict.
        sigma:
            Optional externally provided sigma tensor with shape `[B]`.
            If omitted, sigma is sampled internally.
        noise:
            Optional externally provided Gaussian noise with same shape as target.
        return_details:
            If True, return a dictionary containing intermediate tensors in
            addition to the reduced loss.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            Reduced loss by default, or a dictionary of details when
            `return_details=True`.
        """
        x_clean, cond_dynamic, cond_static, meta = self._extract_batch(batch)

        device = x_clean.device
        dtype = x_clean.dtype
        batch_size = x_clean.shape[0]

        if sigma is None:
            sigma = self.sample_sigma(batch_size, device=device, dtype=dtype)
        else:
            if sigma.ndim == 2 and sigma.shape[-1] == 1:
                sigma = sigma.squeeze(-1)
            if sigma.ndim != 1:
                raise ValueError(
                    f"Expected sigma with shape [B] or [B, 1], got {tuple(sigma.shape)}"
                )
            if sigma.shape[0] != batch_size:
                raise ValueError(
                    f"Sigma batch mismatch: expected {batch_size}, got {sigma.shape[0]}"
                )
            sigma = sigma.to(device=device, dtype=dtype)

        if noise is None:
            noise = torch.randn_like(x_clean)
        else:
            if not isinstance(noise, torch.Tensor):
                raise TypeError(f"noise must be a torch.Tensor, got {type(noise)}")
            if noise.shape != x_clean.shape:
                raise ValueError(
                    f"Noise shape mismatch: expected {tuple(x_clean.shape)}, got {tuple(noise.shape)}"
                )
            noise = noise.to(device=device, dtype=dtype)

        sigma_img = sigma[:, None, None, None]
        x_noisy = x_clean + sigma_img * noise

        doy = self._get_optional_meta_tensor(
            meta,
            keys=("doy", "day_of_year"),
            device=device,
        )
        variable_labels = self._get_optional_meta_tensor(
            meta,
            keys=("variable_labels",),
            device=device,
        )

        pred = model(
            x=x_noisy,
            sigma=sigma,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
            doy=doy,
            variable_labels=variable_labels,
        )

        if pred.shape != x_clean.shape:
            raise ValueError(
                f"Prediction shape mismatch: expected {tuple(x_clean.shape)}, got {tuple(pred.shape)}"
            )

        weight = self.edm_weight(sigma_img)
        per_pixel_loss = weight * (pred - x_clean) ** 2
        per_sample_loss = per_pixel_loss.reshape(batch_size, -1).mean(dim=1)

        if self.reduction == "mean":
            loss = per_sample_loss.mean()
        elif self.reduction == "sum":
            loss = per_sample_loss.sum()
        else:
            loss = per_sample_loss

        if not return_details:
            return loss

        return {
            "loss": loss,
            "per_sample_loss": per_sample_loss,
            "sigma": sigma,
            "weight": weight,
            "x_clean": x_clean,
            "x_noisy": x_noisy,
            "noise": noise,
            "pred": pred,
        }
