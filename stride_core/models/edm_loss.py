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
        "time_features": torch.Tensor | None,   # [B, 2] = [sin(DOY), cos(DOY)]
    }

Optional conditioning currently supported:
- `batch["time_features"]` for continuous FiLM-style DOY conditioning
- `meta["variable_labels"]` for future categorical FiLM conditioning

Notes
-----
The model passed to this loss is expected to be an `EDMPrecondUNet`-style model
with a forward signature like:

    model(
        x,
        sigma,
        cond_dynamic,
        cond_static=None,
        y=None,
        variable_labels=None,
        return_aux=False,
    )

The loss computes:

    x_noisy = x_clean + sigma * noise
    pred = model(x_noisy, sigma, ...)
    loss = weight(sigma) * ||pred - x_clean||^2

with the EDM weighting:

Optionally, the loss can also supervise an auxiliary RainGate head that predicts
pixel-wise wet/dry logits from conditioning inputs only.

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
    rain_gate_cfg:
        Optional nested RainGate loss configuration dictionary. Supported keys:
        `enabled`, `loss_weight`, `wet_threshold_mm`, `target_variable`,
        `use_loss_reweighting`, `reweight_detach`, and `reweight_power`.
        When provided, these values override the corresponding flat arguments.
    rain_gate_enabled:
        Whether auxiliary RainGate supervision is enabled.
    rain_gate_loss_weight:
        Weight applied to the auxiliary RainGate BCE loss.
    rain_gate_wet_threshold:
        Threshold in physical precipitation units used to construct the binary
        wet/dry target mask for RainGate supervision.
    rain_gate_target_variable:
        Target variable used for RainGate supervision. Currently only single-
        channel precipitation targets are supported.
    rain_gate_use_loss_reweighting:
        Placeholder flag for optional RainGate-based diffusion-loss reweighting.
        This is not yet implemented in the first STRIDE version.
    rain_gate_reweight_detach:
        Placeholder flag controlling whether RainGate reweighting would be
        detached from gradients. Not yet used.
    rain_gate_reweight_power:
        Placeholder exponent controlling how RainGate reweighting would be
        shaped. Not yet used.
    """

    def __init__(
        self,
        p_mean: float = -1.5,
        p_std: float = 1.2,
        sigma_data: float = 1.0,
        reduction: str = "mean",
        rain_gate_cfg: dict[str, Any] | None = None,
        rain_gate_enabled: bool | None = None,
        rain_gate_loss_weight: float = 0.0,
        rain_gate_wet_threshold: float = 0.1,
        rain_gate_target_variable: str = "prcp",
        rain_gate_use_loss_reweighting: bool = False,
        rain_gate_reweight_detach: bool = True,
        rain_gate_reweight_power: float = 1.0,
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
        if rain_gate_cfg is not None and not isinstance(rain_gate_cfg, dict):
            raise TypeError(
                f"rain_gate_cfg must be a dict or None, got {type(rain_gate_cfg)}"
            )

        if rain_gate_cfg is not None:
            rain_gate_enabled = bool(rain_gate_cfg.get("enabled", rain_gate_enabled))
            rain_gate_loss_weight = float(
                rain_gate_cfg.get("loss_weight", rain_gate_loss_weight)
            )
            rain_gate_wet_threshold = float(
                rain_gate_cfg.get("wet_threshold_mm", rain_gate_wet_threshold)
            )
            rain_gate_target_variable = str(
                rain_gate_cfg.get("target_variable", rain_gate_target_variable)
            )
            rain_gate_use_loss_reweighting = bool(
                rain_gate_cfg.get(
                    "use_loss_reweighting",
                    rain_gate_use_loss_reweighting,
                )
            )
            rain_gate_reweight_detach = bool(
                rain_gate_cfg.get("reweight_detach", rain_gate_reweight_detach)
            )
            rain_gate_reweight_power = float(
                rain_gate_cfg.get("reweight_power", rain_gate_reweight_power)
            )

        if rain_gate_enabled is None:
            rain_gate_enabled = rain_gate_loss_weight > 0.0
        if rain_gate_loss_weight < 0:
            raise ValueError(
                "rain_gate_loss_weight must be nonnegative, got "
                f"{rain_gate_loss_weight}"
            )
        if rain_gate_wet_threshold < 0:
            raise ValueError(
                "rain_gate_wet_threshold must be nonnegative, got "
                f"{rain_gate_wet_threshold}"
            )
        if not rain_gate_target_variable:
            raise ValueError("rain_gate_target_variable must be non-empty")
        if rain_gate_reweight_power < 0:
            raise ValueError(
                "rain_gate_reweight_power must be nonnegative, got "
                f"{rain_gate_reweight_power}"
            )

        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)
        self.reduction = reduction
        self.rain_gate_enabled = bool(rain_gate_enabled)
        self.rain_gate_loss_weight = float(rain_gate_loss_weight)
        self.rain_gate_wet_threshold = float(rain_gate_wet_threshold)
        self.rain_gate_target_variable = str(rain_gate_target_variable)
        self.rain_gate_use_loss_reweighting = bool(rain_gate_use_loss_reweighting)
        self.rain_gate_reweight_detach = bool(rain_gate_reweight_detach)
        self.rain_gate_reweight_power = float(rain_gate_reweight_power)
        self.rain_gate_bce = nn.BCEWithLogitsLoss(reduction="none")

    def _select_rain_gate_target_channel(self, x_clean: torch.Tensor) -> torch.Tensor:
        """
        Select the precipitation channel used for RainGate supervision.

        The current STRIDE implementation only supports single-channel targets.
        This keeps the loss explicit until a multi-target output ordering is
        formalized in the config.
        """
        if x_clean.ndim != 4:
            raise ValueError(
                f"Expected x_clean with shape [B, C, H, W], got {tuple(x_clean.shape)}"
            )
        if x_clean.shape[1] != 1:
            raise NotImplementedError(
                "RainGate target_variable selection currently supports only "
                "single-channel targets."
            )
        if self.rain_gate_target_variable != "prcp":
            raise NotImplementedError(
                "RainGate target_variable selection for multi-target outputs is not "
                f"implemented yet (got {self.rain_gate_target_variable!r})."
            )
        return x_clean[:, :1]

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

    def _extract_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        dict[str, Any],
        torch.Tensor | None,
    ]:
        required_keys = ("target", "cond_dynamic", "meta")
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise KeyError(f"Batch is missing required keys: {missing}")

        target = batch["target"]
        cond_dynamic = batch["cond_dynamic"]
        cond_static = batch.get("cond_static")
        meta = batch["meta"]
        time_features = batch.get("time_features")

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
        if time_features is not None and not isinstance(time_features, torch.Tensor):
            raise TypeError(
                f"batch['time_features'] must be a torch.Tensor or None, got {type(time_features)}"
            )

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
        if time_features is not None:
            if time_features.ndim != 2:
                raise ValueError(
                    "batch['time_features'] must have shape [B, 2], got "
                    f"{tuple(time_features.shape)}"
                )
            if time_features.shape[0] != target.shape[0]:
                raise ValueError(
                    "batch['time_features'] batch dimension mismatch: "
                    f"expected {target.shape[0]}, got {time_features.shape[0]}"
                )
            if time_features.shape[1] != 2:
                raise ValueError(
                    "batch['time_features'] must have final dimension 2 for "
                    "[sin(DOY), cos(DOY)], got "
                    f"{tuple(time_features.shape)}"
                )

        return target, cond_dynamic, cond_static, meta, time_features

    def _compute_rain_gate_loss(
        self,
        rain_gate_logits: torch.Tensor,
        x_clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute auxiliary RainGate BCE loss and the underlying wet mask target.

        Parameters
        ----------
        rain_gate_logits:
            Pixel-wise wet/dry logits with shape [B, 1, H, W].
        x_clean:
            Clean target tensor with shape [B, C_out, H, W]. The first output
            channel is interpreted as the precipitation field used for wet/dry
            supervision.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            `(rain_gate_loss, wet_mask)` where `rain_gate_loss` is a scalar and
            `wet_mask` has shape [B, 1, H, W].
        """
        if rain_gate_logits.ndim != 4:
            raise ValueError(
                "Expected rain_gate_logits with shape [B, 1, H, W], got "
                f"{tuple(rain_gate_logits.shape)}"
            )
        if rain_gate_logits.shape[1] != 1:
            raise ValueError(
                "Expected rain_gate_logits to have a single output channel, got "
                f"{rain_gate_logits.shape[1]}"
            )
        if x_clean.ndim != 4:
            raise ValueError(
                f"Expected x_clean with shape [B, C, H, W], got {tuple(x_clean.shape)}"
            )
        if x_clean.shape[0] != rain_gate_logits.shape[0]:
            raise ValueError(
                "Batch mismatch between x_clean and rain_gate_logits: "
                f"{x_clean.shape[0]} vs {rain_gate_logits.shape[0]}"
            )
        if x_clean.shape[2:] != rain_gate_logits.shape[2:]:
            raise ValueError(
                "Spatial mismatch between x_clean and rain_gate_logits: "
                f"{tuple(x_clean.shape[2:])} vs {tuple(rain_gate_logits.shape[2:])}"
            )

        target_channel = self._select_rain_gate_target_channel(x_clean)
        wet_mask = (target_channel >= self.rain_gate_wet_threshold).to(
            dtype=rain_gate_logits.dtype,
            device=rain_gate_logits.device,
        )
        rain_gate_loss_map = self.rain_gate_bce(rain_gate_logits, wet_mask)
        rain_gate_loss = rain_gate_loss_map.mean()
        return rain_gate_loss, wet_mask

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
        x_clean, cond_dynamic, cond_static, meta, time_features = self._extract_batch(batch)

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

        variable_labels = None
        if "variable_labels" in meta and meta["variable_labels"] is not None:
            variable_labels = meta["variable_labels"]
            if isinstance(variable_labels, torch.Tensor):
                variable_labels = variable_labels.to(device=device)
            else:
                variable_labels = torch.as_tensor(variable_labels, device=device)

        if time_features is not None:
            time_features = time_features.to(device=device, dtype=dtype)

        aux: dict[str, torch.Tensor] = {}
        if self.rain_gate_enabled and self.rain_gate_loss_weight > 0.0:
            pred, aux = model(
                x=x_noisy,
                sigma=sigma,
                cond_dynamic=cond_dynamic,
                cond_static=cond_static,
                y=time_features,
                variable_labels=variable_labels,
                return_aux=True,
            )
        else:
            pred = model(
                x=x_noisy,
                sigma=sigma,
                cond_dynamic=cond_dynamic,
                cond_static=cond_static,
                y=time_features,
                variable_labels=variable_labels,
                return_aux=False,
            )

        if pred.shape != x_clean.shape:
            raise ValueError(
                f"Prediction shape mismatch: expected {tuple(x_clean.shape)}, got {tuple(pred.shape)}"
            )

        rain_gate_loss = None
        wet_mask = None
        if self.rain_gate_enabled and self.rain_gate_loss_weight > 0.0:
            if "rain_gate_logits" not in aux:
                raise KeyError(
                    "RainGate auxiliary loss is enabled, but model aux outputs do not "
                    "contain 'rain_gate_logits'"
                )
            rain_gate_loss, wet_mask = self._compute_rain_gate_loss(
                rain_gate_logits=aux["rain_gate_logits"],
                x_clean=x_clean,
            )
        if self.rain_gate_use_loss_reweighting:
            raise NotImplementedError(
                "RainGate-based diffusion loss reweighting is not implemented yet in STRIDE."
            )

        weight = self.edm_weight(sigma_img)
        per_pixel_loss = weight * (pred - x_clean) ** 2
        per_sample_loss = per_pixel_loss.reshape(batch_size, -1).mean(dim=1)

        if self.reduction == "mean":
            diffusion_loss = per_sample_loss.mean()
        elif self.reduction == "sum":
            diffusion_loss = per_sample_loss.sum()
        else:
            diffusion_loss = per_sample_loss

        loss = diffusion_loss
        if rain_gate_loss is not None:
            if self.reduction == "none":
                loss = diffusion_loss + self.rain_gate_loss_weight * rain_gate_loss
            else:
                loss = diffusion_loss + self.rain_gate_loss_weight * rain_gate_loss

        if not return_details:
            return loss

        details: dict[str, torch.Tensor] = {
            "loss": loss,
            "diffusion_loss": diffusion_loss,
            "per_sample_loss": per_sample_loss,
            "sigma": sigma,
            "weight": weight,
            "x_clean": x_clean,
            "x_noisy": x_noisy,
            "noise": noise,
            "pred": pred,
        }
        if rain_gate_loss is not None:
            details["rain_gate_loss"] = rain_gate_loss
            details["rain_gate_logits"] = aux["rain_gate_logits"]
        if wet_mask is not None:
            details["rain_gate_wet_mask"] = wet_mask
        return details
