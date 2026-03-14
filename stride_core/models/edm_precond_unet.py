"""
EDM-preconditioned UNet wrapper for STRIDE.

This module wraps the plain conditional `EDMUNet` architecture with the
preconditioning formulation from Karras et al. (2022). The design goal is to
keep:

- architecture logic in `edm_unet.py`
- EDM scaling / preconditioning logic here

This separation makes the model easier to reason about, test, and extend.

Current scope
-------------
- wraps the plain conditional UNet
- supports dynamic + optional static conditioning
- supports optional FiLM conditioning tensors passed through to the plain UNet
- supports optional auxiliary outputs (e.g. RainGate logits)
- returns a denoised prediction in the original target space

Not included yet
----------------
- context encoder integration
- RainGate feature modulation / output gating
- temporal stacking-specific logic
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stride_core.configs.model_config import ModelSpec
from stride_core.models.edm_unet import EDMUNet


class EDMPrecondUNet(nn.Module):
    """
    EDM-preconditioned wrapper around `EDMUNet`.

    The wrapped model predicts the preconditioned residual branch `F(x)` and the
    final denoised estimate is assembled as:

        D(x, sigma) = c_skip * x + c_out * F(c_in * x, sigma, cond)

    where the coefficients follow the EDM parameterization.

    Parameters
    ----------
    spec:
        Validated `ModelSpec` used to build the plain UNet.
    sigma_data:
        Data standard deviation used in the EDM preconditioning coefficients.
        Default follows the common Karras EDM choice.
    """

    def __init__(self, spec: ModelSpec, sigma_data: float = 0.5) -> None:
        super().__init__()
        if sigma_data <= 0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}")

        self.spec = spec
        self.sigma_data = float(sigma_data)
        self.model = EDMUNet(spec)

    def _reshape_sigma(self, sigma: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize sigma into both vector form `[B]` and broadcastable image form
        `[B, 1, 1, 1]`.
        """
        if sigma.ndim == 0:
            sigma = sigma[None]
        if sigma.ndim == 2 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)
        if sigma.ndim != 1:
            raise ValueError(
                f"Expected sigma with shape [B] or [B, 1], got {tuple(sigma.shape)}"
            )
        if sigma.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch mismatch between x and sigma: {x.shape[0]} vs {sigma.shape[0]}"
            )

        sigma = sigma.to(device=x.device, dtype=x.dtype)
        sigma_img = sigma[:, None, None, None]
        return sigma, sigma_img

    def _preconditioning_coefficients(
        self,
        sigma_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute EDM preconditioning coefficients.

        Coefficients
        ------------
        c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
        c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
        c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
        c_noise = log(sigma) / 4

        Notes
        -----
        `c_noise` is returned in image-broadcast form here; the plain UNet uses
        the vector sigma input instead. We keep `c_noise` available because it is
        part of the canonical EDM formulation and may be useful later.
        """
        sigma_data_sq = self.sigma_data ** 2
        sigma_sq = sigma_img ** 2
        denom = torch.sqrt(sigma_sq + sigma_data_sq)

        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma_img * self.sigma_data / denom
        c_in = 1.0 / denom
        c_noise = torch.log(sigma_img) / 4.0
        return c_skip, c_out, c_in, c_noise

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        variable_labels: torch.Tensor | None = None,
        return_model_output: bool = False,
        return_aux: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """
        Forward pass of the EDM-preconditioned model.

        Parameters
        ----------
        x:
            Noisy target tensor with shape `[B, C_out, H, W]`.
        sigma:
            Noise levels with shape `[B]` or `[B, 1]`.
        cond_dynamic:
            Dynamic conditioning tensor with shape `[B, C_dyn, H, W]`.
        cond_static:
            Optional static conditioning tensor with shape `[B, C_static, H, W]`.
        y:
            Optional continuous temporal conditioning tensor for FiLM-style
            conditioning. Expected shape is `[B, 2]` containing
            `[sin(DOY), cos(DOY)]`.
        variable_labels:
            Optional categorical variable-label tensor for future FiLM-style
            conditioning.
        return_model_output:
            If True, also return the raw plain-UNet output before EDM output
            assembly. Useful for debugging.
        return_aux:
            If True, also return auxiliary outputs produced by the wrapped plain
            UNet, e.g. RainGate logits.

        Returns
        -------
        torch.Tensor or tuple
            By default returns the EDM denoised prediction.

            If `return_model_output=True`, returns:
                `(denoised, raw_model_output)`

            If `return_aux=True`, returns:
                `(denoised, aux_dict)`

            If both flags are True, returns:
                `(denoised, raw_model_output, aux_dict)`
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.spec.out_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.spec.out_channels}, got {x.shape[1]}"
            )

        sigma_vec, sigma_img = self._reshape_sigma(sigma, x)
        c_skip, c_out, c_in, _ = self._preconditioning_coefficients(sigma_img)

        model_input = c_in * x
        aux: dict[str, torch.Tensor] = {}

        if return_aux:
            model_output, aux = self.model(
                x=model_input,
                sigma=sigma_vec,
                cond_dynamic=cond_dynamic,
                cond_static=cond_static,
                y=y,
                variable_labels=variable_labels,
                return_aux=True,
            )
        else:
            model_output = self.model(
                x=model_input,
                sigma=sigma_vec,
                cond_dynamic=cond_dynamic,
                cond_static=cond_static,
                y=y,
                variable_labels=variable_labels,
                return_aux=False,
            )

        denoised = c_skip * x + c_out * model_output

        if return_model_output and return_aux:
            return denoised, model_output, aux
        if return_model_output:
            return denoised, model_output
        if return_aux:
            return denoised, aux
        return denoised