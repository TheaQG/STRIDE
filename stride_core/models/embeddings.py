

"""
Embedding utilities for STRIDE models.

This module contains small, reusable embedding components that should stay
independent of any particular UNet implementation. The immediate goal is to
support the first EDM UNet port with:

- noise-level / sigma embeddings
- generic MLP projections for embeddings
- optional FiLM modulation helpers
- optional sinusoidal day-of-year embeddings

Design notes
------------
These implementations are based on the useful legacy pieces in `score_unet.py`,
but rewritten to be framework-level and data-agnostic:

- no hard-coded precipitation / topo / lsm assumptions
- no batch-contract assumptions
- no coupling to a specific UNet class

The first working model only strictly needs `GaussianFourierProjection` and
`SigmaEmbedding`, but the FiLM and day-of-year helpers are worth adding now
because they are small, self-contained, and likely to be used soon.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for continuous scalar inputs.

    This is the standard score-model / diffusion-style embedding used for noise
    levels or time variables. For an input scalar `x`, the module returns:

        [sin(2π W x), cos(2π W x)]

    where `W` is a fixed random vector sampled at initialization.

    Parameters
    ----------
    embed_dim:
        Output embedding dimension. Must be even.
    scale:
        Standard deviation of the Gaussian initialization for the fixed random
        projection matrix.
    """

    def __init__(self, embed_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be even for GaussianFourierProjection, got {embed_dim}"
            )

        weight = torch.randn(embed_dim // 2) * scale
        self.register_buffer("weight", weight)
        self.embed_dim = embed_dim
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor with shape `[B]` or `[B, 1]` containing continuous scalars.

        Returns
        -------
        torch.Tensor
            Tensor with shape `[B, embed_dim]`.
        """
        if x.ndim == 0:
            x = x[None]
        if x.ndim == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 1:
            raise ValueError(
                f"Expected input with shape [B] or [B, 1], got {tuple(x.shape)}"
            )

        x_proj = 2.0 * math.pi * x[:, None] * self.weight[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalEmbedding(nn.Module):
    """
    Transformer-style sinusoidal embedding for scalar inputs.

    This is deterministic (non-random) and can be useful for variables such as
    day-of-year, lead time, or generic scalar metadata.

    Parameters
    ----------
    embed_dim:
        Output embedding dimension. If odd, the final column is zero-padded.
    max_period:
        Controls the minimum frequency used in the sinusoidal basis.
    """

    def __init__(self, embed_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        self.embed_dim = embed_dim
        self.max_period = float(max_period)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            x = x[None]
        if x.ndim == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 1:
            raise ValueError(
                f"Expected input with shape [B] or [B, 1], got {tuple(x.shape)}"
            )

        half_dim = self.embed_dim // 2
        device = x.device
        dtype = x.dtype

        if half_dim == 0:
            return torch.zeros(x.shape[0], self.embed_dim, device=device, dtype=dtype)

        exponent = torch.arange(half_dim, device=device, dtype=dtype) / half_dim
        freqs = torch.exp(-math.log(self.max_period) * exponent)
        args = x[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLPEmbedding(nn.Module):
    """
    Small MLP used to project an embedding into a model-hidden representation.

    This is the generic projection block that replaces several legacy ad hoc
    embedding MLPs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError(
                f"in_dim, hidden_dim, and out_dim must be positive; "
                f"got {(in_dim, hidden_dim, out_dim)}"
            )

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SigmaEmbedding(nn.Module):
    """
    EDM-style sigma embedding.

    This wraps Gaussian Fourier features followed by a small MLP, which is the
    most directly reusable embedding pattern from the legacy diffusion model.

    Parameters
    ----------
    embed_dim:
        Output embedding dimension consumed by the UNet.
    fourier_dim:
        Internal Gaussian Fourier feature dimension.
    fourier_scale:
        Scale used for the Gaussian Fourier projection.
    hidden_dim:
        Hidden size of the embedding MLP. Defaults to `embed_dim` when omitted.
    """

    def __init__(
        self,
        embed_dim: int,
        fourier_dim: int = 128,
        fourier_scale: float = 16.0,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if fourier_dim <= 0:
            raise ValueError(f"fourier_dim must be positive, got {fourier_dim}")

        hidden_dim = embed_dim if hidden_dim is None else hidden_dim

        self.fourier = GaussianFourierProjection(
            embed_dim=fourier_dim,
            scale=fourier_scale,
        )
        self.mlp = MLPEmbedding(
            in_dim=fourier_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            activation=nn.SiLU,
            dropout=0.0,
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma_features = self.fourier(sigma)
        return self.mlp(sigma_features)


class DayOfYearEmbedding(nn.Module):
    """
    Cyclic day-of-year embedding using a deterministic sinusoidal encoding.

    Input is expected to be day-of-year values in `[1, 365]` or `[0, 365]`.
    The values are first mapped to the unit circle and then optionally projected
    through a small MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        period: float = 365.0,
        project: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        self.embed_dim = embed_dim
        self.period = float(period)
        self.project = bool(project)

        if self.project:
            hidden_dim = embed_dim if hidden_dim is None else hidden_dim
            self.proj = MLPEmbedding(
                in_dim=2,
                hidden_dim=hidden_dim,
                out_dim=embed_dim,
                activation=nn.SiLU,
                dropout=0.0,
            )
        else:
            self.proj = None
            if embed_dim != 2:
                raise ValueError(
                    "embed_dim must be 2 when project=False for DayOfYearEmbedding"
                )

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        if doy.ndim == 0:
            doy = doy[None]
        if doy.ndim == 2 and doy.shape[-1] == 1:
            doy = doy.squeeze(-1)
        if doy.ndim != 1:
            raise ValueError(
                f"Expected input with shape [B] or [B, 1], got {tuple(doy.shape)}"
            )

        angle = 2.0 * math.pi * doy / self.period
        cyc = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        if self.proj is None:
            return cyc
        return self.proj(cyc)


class CategoricalEmbedding(nn.Module):
    """
    Generic categorical embedding with optional projection.

    Useful later for variable-label FiLM conditioning or other discrete metadata.
    """

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0:
            raise ValueError(
                f"num_embeddings must be positive, got {num_embeddings}"
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.proj = (
            nn.Identity() if self.out_dim == embed_dim else nn.Linear(embed_dim, self.out_dim)
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 2 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if labels.ndim != 1:
            raise ValueError(
                f"Expected categorical labels with shape [B] or [B, 1], got {tuple(labels.shape)}"
            )
        emb = self.embedding(labels.long())
        return self.proj(emb)


class FiLMModulation(nn.Module):
    """
    Feature-wise linear modulation helper.

    Given an embedding vector `e`, this module predicts per-channel scale and
    shift coefficients `(gamma, beta)` and applies:

        y = x * (1 + gamma) + beta

    where `x` has shape `[B, C, H, W]`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_channels <= 0:
            raise ValueError(
                f"embed_dim and num_channels must be positive, got {(embed_dim, num_channels)}"
            )

        hidden_dim = embed_dim if hidden_dim is None else hidden_dim
        self.modulator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        if emb.ndim != 2:
            raise ValueError(f"Expected emb with shape [B, E], got {tuple(emb.shape)}")
        if x.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x and emb: {x.shape[0]} vs {emb.shape[0]}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.num_channels}, got {x.shape[1]}"
            )

        gamma_beta = self.modulator(emb)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta