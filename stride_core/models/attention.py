

"""
Attention modules for STRIDE models.

This file contains reusable attention components extracted from the useful parts
of the legacy `score_unet.py` implementation, but rewritten to be:

- framework-level
- model-agnostic
- free of dataset-specific assumptions
- easy to reuse in encoder / decoder blocks

For the first working STRIDE EDM UNet, the most important component is
`ImageSelfAttention`, which performs self-attention over flattened spatial
locations while keeping a channel-first image interface.

The implementation here follows the legacy idea closely:

    x -> LayerNorm -> MultiHeadAttention -> residual
      -> LayerNorm -> feed-forward -> residual

but is cleaned up and made more explicit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    Lightweight feed-forward block used inside attention residual blocks.

    Parameters
    ----------
    embed_dim:
        Token embedding dimension.
    hidden_dim:
        Hidden size of the feed-forward network. If omitted, defaults to
        `4 * embed_dim`, which is a common transformer-style choice.
    dropout:
        Dropout probability applied after each linear transformation.
    activation:
        Activation module class. GELU is used by default, matching the legacy
        implementation well.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive when provided, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        hidden_dim = 4 * embed_dim if hidden_dim is None else hidden_dim

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageSelfAttention(nn.Module):
    """
    Channel-first image self-attention over flattened spatial tokens.

    Input shape:
        [B, C, H, W]

    Internal token shape:
        [B, H*W, C]

    Output shape:
        [B, C, H, W]

    The block is a standard pre-norm attention residual block:

        h = x + MHA(LN(x))
        y = h + FF(LN(h))

    Parameters
    ----------
    input_channels:
        Number of channels in the image feature map.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout passed to the multi-head attention and feed-forward block.
    ff_hidden_dim:
        Optional hidden size of the feed-forward block. Defaults to
        `4 * input_channels` when omitted.
    """

    def __init__(
        self,
        input_channels: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if input_channels % num_heads != 0:
            raise ValueError(
                f"input_channels ({input_channels}) must be divisible by num_heads ({num_heads})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_channels = input_channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(input_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=input_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(input_channels)
        self.ff = FeedForwardBlock(
            embed_dim=input_channels,
            hidden_dim=ff_hidden_dim,
            dropout=dropout,
            activation=nn.GELU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input with shape [B, C, H, W], got {tuple(x.shape)}"
            )

        batch_size, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Channel mismatch: block expects {self.input_channels}, got {channels}"
            )

        # Convert channel-first image features to token representation.
        tokens = x.reshape(batch_size, channels, height * width).permute(0, 2, 1)

        # Pre-norm self-attention residual block.
        h = self.norm1(tokens)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = tokens + attn_out

        # Pre-norm feed-forward residual block.
        y = h + self.ff(self.norm2(h))

        # Convert back to channel-first image representation.
        return y.permute(0, 2, 1).reshape(batch_size, channels, height, width)


class AttentionBlock(nn.Module):
    """
    Thin convenience wrapper for image self-attention.

    This class exists mainly to give the rest of the UNet code a stable name to
    depend on, so later it can be extended to support other attention variants
    without changing encoder/decoder call sites.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.block = ImageSelfAttention(
            input_channels=channels,
            num_heads=num_heads,
            dropout=dropout,
            ff_hidden_dim=ff_hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IdentityAttention(nn.Module):
    """
    No-op attention block.

    Useful when attention is configured off for a given stage while keeping a
    consistent module interface in the encoder/decoder definitions.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_attention_block(
    channels: int,
    use_attention: bool,
    num_heads: int,
    dropout: float = 0.0,
    ff_hidden_dim: int | None = None,
) -> nn.Module:
    """
    Factory for creating either a real attention block or an identity block.

    This is convenient in encoder / decoder code where attention is only used at
    selected resolutions.
    """
    if not use_attention:
        return IdentityAttention()

    return AttentionBlock(
        channels=channels,
        num_heads=num_heads,
        dropout=dropout,
        ff_hidden_dim=ff_hidden_dim,
    )