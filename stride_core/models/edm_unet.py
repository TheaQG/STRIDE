

"""
Plain conditional UNet for STRIDE EDM models.

This module assembles the reusable model building blocks into a first working
conditional UNet. It is intentionally kept separate from EDM preconditioning so
that:

- architecture logic lives here
- EDM scaling / wrapper logic lives in `edm_precond_unet.py`

Current v1 scope
----------------
- noisy HR input `x`
- sigma embedding
- dynamic conditioning
- optional static conditioning
- optional FiLM-style auxiliary embedding (e.g. day-of-year later)
- encoder + decoder backbone

Not included yet
----------------
- context encoder
- temporal stacking-specific logic
- RainGate integration
- EDM preconditioning math

The goal is to get a clean, testable forward pass alive before adding the more
specialized wrappers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from stride_core.configs.model_config import ModelSpec
from stride_core.models.decoder import Decoder
from stride_core.models.embeddings import (
    CategoricalEmbedding,
    DayOfYearEmbedding,
    SigmaEmbedding,
)
from stride_core.models.encoder import Encoder


class EDMUNet(nn.Module):
    """
    Plain conditional UNet used inside the EDM preconditioned wrapper.

    Parameters
    ----------
    spec:
        Validated `ModelSpec` describing channel counts, widths, attention
        layout, and optional conditioning features.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec

        self._validate_spec_for_v1(spec)

        self.in_dynamic_channels = spec.in_dynamic_channels
        self.in_static_channels = spec.in_static_channels
        self.out_channels = spec.out_channels
        self.input_channels = (
            spec.out_channels + spec.in_dynamic_channels + spec.in_static_channels
        )

        # Primary sigma embedding used throughout the UNet.
        self.sigma_embedding = SigmaEmbedding(
            embed_dim=spec.sigma_embed_dim,
            fourier_dim=spec.sigma_embed_dim,
            fourier_scale=16.0,
            hidden_dim=spec.sigma_embed_dim,
        )

        # Optional auxiliary FiLM embeddings.
        self.use_doy_film = spec.use_doy_film
        self.use_variable_film = spec.use_variable_film

        self.doy_embedding = (
            DayOfYearEmbedding(
                embed_dim=spec.film_embed_dim,
                hidden_dim=spec.film_embed_dim,
                period=365.0,
                project=True,
            )
            if spec.use_doy_film
            else None
        )

        self.variable_embedding = (
            CategoricalEmbedding(
                num_embeddings=max(spec.out_channels, 1),
                embed_dim=spec.film_embed_dim,
                out_dim=spec.film_embed_dim,
            )
            if spec.use_variable_film
            else None
        )

        use_any_film = spec.use_doy_film or spec.use_variable_film

        self.encoder = Encoder(
            in_channels=self.input_channels,
            base_channels=spec.model_channels,
            channel_mults=spec.channel_mults,
            num_res_blocks=spec.num_res_blocks,
            embed_dim=spec.sigma_embed_dim,
            attention_resolutions=spec.attention_resolutions,
            input_resolution=spec.target_height,
            num_heads=spec.num_heads,
            dropout=spec.dropout,
            use_film=use_any_film,
            film_embed_dim=spec.film_embed_dim if use_any_film else None,
        )

        skip_channels = tuple(stage.out_channels for stage in self.encoder.stage_specs)

        self.decoder = Decoder(
            bottleneck_channels=self.encoder.out_channels,
            skip_channels=skip_channels,
            out_channels=spec.out_channels,
            base_channels=spec.model_channels,
            channel_mults=spec.channel_mults,
            num_res_blocks=spec.num_res_blocks,
            embed_dim=spec.sigma_embed_dim,
            attention_resolutions=spec.attention_resolutions,
            bottleneck_resolution=self.encoder.bottleneck_resolution,
            num_heads=spec.num_heads,
            dropout=spec.dropout,
            use_film=use_any_film,
            film_embed_dim=spec.film_embed_dim if use_any_film else None,
        )

    @staticmethod
    def _validate_spec_for_v1(spec: ModelSpec) -> None:
        if spec.target_height != spec.target_width:
            raise ValueError(
                "The first EDMUNet implementation currently expects square HR "
                f"inputs, got {(spec.target_height, spec.target_width)}"
            )
        if spec.cond_height != spec.target_height or spec.cond_width != spec.target_width:
            raise ValueError(
                "The first EDMUNet implementation expects co-located conditioning "
                "with the same spatial size as the target. "
                f"Got target={(spec.target_height, spec.target_width)} and "
                f"cond={(spec.cond_height, spec.cond_width)}"
            )
        if spec.use_context_encoder:
            raise ValueError(
                "Context encoder support is not implemented in EDMUNet v1 yet. "
                "Set use_context_encoder=False."
            )
        if spec.use_temporal_stack:
            raise ValueError(
                "Temporal stacking support is not implemented in EDMUNet v1 yet. "
                "Set use_temporal_stack=False."
            )

    def _build_film_embedding(
        self,
        batch_size: int,
        x: torch.Tensor,
        doy: torch.Tensor | None = None,
        variable_labels: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        pieces: list[torch.Tensor] = []

        if self.doy_embedding is not None:
            if doy is None:
                raise ValueError("doy must be provided when use_doy_film=True")
            pieces.append(self.doy_embedding(doy.to(device=x.device, dtype=x.dtype)))

        if self.variable_embedding is not None:
            if variable_labels is None:
                raise ValueError(
                    "variable_labels must be provided when use_variable_film=True"
                )
            pieces.append(self.variable_embedding(variable_labels.to(device=x.device)))

        if not pieces:
            return None

        film_emb = pieces[0]
        for piece in pieces[1:]:
            if piece.shape != film_emb.shape:
                raise ValueError(
                    "All auxiliary FiLM embeddings must have matching shapes, got "
                    f"{film_emb.shape} and {piece.shape}"
                )
            film_emb = film_emb + piece

        if film_emb.shape[0] != batch_size:
            raise ValueError(
                f"FiLM embedding batch size mismatch: expected {batch_size}, got {film_emb.shape[0]}"
            )
        return film_emb

    def _assemble_input(
        self,
        x: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        if cond_dynamic.ndim != 4:
            raise ValueError(
                f"Expected cond_dynamic with shape [B, C, H, W], got {tuple(cond_dynamic.shape)}"
            )
        if x.shape[0] != cond_dynamic.shape[0]:
            raise ValueError(
                f"Batch mismatch between x and cond_dynamic: {x.shape[0]} vs {cond_dynamic.shape[0]}"
            )
        if x.shape[2:] != cond_dynamic.shape[2:]:
            raise ValueError(
                "Spatial mismatch between x and cond_dynamic: "
                f"{tuple(x.shape[2:])} vs {tuple(cond_dynamic.shape[2:])}"
            )
        if x.shape[1] != self.out_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.out_channels}, got {x.shape[1]}"
            )
        if cond_dynamic.shape[1] != self.in_dynamic_channels:
            raise ValueError(
                f"cond_dynamic channel mismatch: expected {self.in_dynamic_channels}, got {cond_dynamic.shape[1]}"
            )

        parts = [x, cond_dynamic]

        if self.in_static_channels > 0:
            if cond_static is None:
                raise ValueError(
                    "cond_static must be provided because spec.in_static_channels > 0"
                )
            if cond_static.ndim != 4:
                raise ValueError(
                    f"Expected cond_static with shape [B, C, H, W], got {tuple(cond_static.shape)}"
                )
            if cond_static.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Batch mismatch between x and cond_static: {x.shape[0]} vs {cond_static.shape[0]}"
                )
            if cond_static.shape[2:] != x.shape[2:]:
                raise ValueError(
                    "Spatial mismatch between x and cond_static: "
                    f"{tuple(x.shape[2:])} vs {tuple(cond_static.shape[2:])}"
                )
            if cond_static.shape[1] != self.in_static_channels:
                raise ValueError(
                    f"cond_static channel mismatch: expected {self.in_static_channels}, got {cond_static.shape[1]}"
                )
            parts.append(cond_static)
        else:
            if cond_static is not None:
                raise ValueError(
                    "cond_static was provided but spec.in_static_channels == 0"
                )

        return torch.cat(parts, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor | None = None,
        doy: torch.Tensor | None = None,
        variable_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the plain conditional UNet.

        Parameters
        ----------
        x:
            Noisy HR target tensor with shape `[B, C_out, H, W]`.
        sigma:
            Noise-level tensor with shape `[B]` or `[B, 1]`.
        cond_dynamic:
            Dynamic conditioning tensor with shape `[B, C_dyn, H, W]`.
        cond_static:
            Optional static conditioning tensor with shape `[B, C_static, H, W]`.
        doy:
            Optional day-of-year tensor used only when `use_doy_film=True`.
        variable_labels:
            Optional categorical variable-label tensor used only when
            `use_variable_film=True`.

        Returns
        -------
        torch.Tensor
            Predicted residual / denoised field with shape `[B, C_out, H, W]`.
        """
        model_input = self._assemble_input(
            x=x,
            cond_dynamic=cond_dynamic,
            cond_static=cond_static,
        )

        sigma_emb = self.sigma_embedding(sigma.to(device=x.device, dtype=x.dtype))
        film_emb = self._build_film_embedding(
            batch_size=x.shape[0],
            x=x,
            doy=doy,
            variable_labels=variable_labels,
        )

        bottleneck, skips = self.encoder(
            model_input,
            emb=sigma_emb,
            film_emb=film_emb,
        )
        out = self.decoder(
            bottleneck,
            skips=skips,
            emb=sigma_emb,
            film_emb=film_emb,
        )
        return out