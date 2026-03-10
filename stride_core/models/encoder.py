

"""
Encoder modules for STRIDE EDM-style UNets.

This file extracts the reusable encoder-side ideas from the legacy
`score_unet.py` implementation and rewrites them into a cleaner, framework-level
form.

Goals for this first STRIDE version
----------------------------------
- keep the useful residual-block structure from the legacy model
- support sigma/time embedding injection through additive conditioning
- support optional FiLM modulation later through an extra embedding vector
- support optional attention at selected stages
- produce skip connections for a decoder
- stay independent of any dataset-specific assumptions

The first working version intentionally does *not* include:
- context encoder logic
- temporal stacking-specific logic
- data-specific conditioning names
- residual prediction logic

Those belong one level above this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from stride_core.models.attention import build_attention_block
from stride_core.models.embeddings import FiLMModulation


@dataclass(frozen=True)
class EncoderStageSpec:
    """
    Lightweight description of one encoder stage.

    Parameters
    ----------
    in_channels:
        Number of incoming channels to the stage.
    out_channels:
        Number of outgoing channels from the stage.
    resolution:
        Spatial resolution at this stage, used to decide whether attention is
        active.
    downsample:
        Whether to downsample at the end of the stage.
    use_attention:
        Whether to apply attention inside this stage.
    """

    in_channels: int
    out_channels: int
    resolution: int
    downsample: bool
    use_attention: bool


# Helper function for GroupNorm group count
def choose_num_groups(num_channels: int, requested_groups: int) -> int:
    """
    Choose the largest valid GroupNorm group count not exceeding
    `requested_groups` and dividing `num_channels` exactly.
    """
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    if requested_groups <= 0:
        raise ValueError(
            f"requested_groups must be positive, got {requested_groups}"
        )

    max_groups = min(requested_groups, num_channels)
    for groups in range(max_groups, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """
    Residual convolution block with embedding injection.

    This is the most important reusable encoder/decoder building block. It keeps
    the useful functionality of the legacy model while cleaning up the interface.

    The block applies:

        GN -> SiLU -> Conv
        + projected embedding bias
        (+ optional FiLM modulation)
        GN -> SiLU -> Dropout -> Conv
        + skip connection

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    embed_dim:
        Dimension of the main embedding vector (typically sigma embedding).
    dropout:
        Dropout probability applied before the second convolution.
    use_film:
        Whether to additionally apply FiLM modulation from `film_emb`.
    film_embed_dim:
        Required when `use_film=True`.
    num_groups:
        Number of groups used in GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        dropout: float = 0.0,
        use_film: bool = False,
        film_embed_dim: int | None = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got {(in_channels, out_channels)}"
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if use_film and (film_embed_dim is None or film_embed_dim <= 0):
            raise ValueError(
                "film_embed_dim must be provided and positive when use_film=True"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_film = use_film

        norm1_groups = choose_num_groups(in_channels, num_groups)
        self.norm1 = nn.GroupNorm(num_groups=norm1_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

        self.film = (
            FiLMModulation(embed_dim=film_embed_dim, num_channels=out_channels) # type: ignore[call-arg]
            if use_film
            else None
        )

        norm2_groups = choose_num_groups(out_channels, num_groups)
        self.norm2 = nn.GroupNorm(num_groups=norm2_groups, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        film_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        if emb.ndim != 2:
            raise ValueError(f"Expected emb with shape [B, E], got {tuple(emb.shape)}")
        if x.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x and emb: {x.shape[0]} vs {emb.shape[0]}"
            )
        if self.use_film and film_emb is None:
            raise ValueError("film_emb must be provided when use_film=True")
        if film_emb is not None and film_emb.ndim != 2:
            raise ValueError(
                f"Expected film_emb with shape [B, E], got {tuple(film_emb.shape)}"
            )

        residual = self.skip(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        emb_bias = self.embed_proj(emb)[:, :, None, None]
        h = h + emb_bias

        if self.film is not None:
            h = self.film(h, film_emb)  # type: ignore[arg-type]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class Downsample2d(nn.Module):
    """
    2× spatial downsampling using a strided convolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class EncoderStage(nn.Module):
    """
    One encoder stage consisting of residual blocks, optional attention, and
    optional downsampling.

    Returns both the post-block feature map (for skip connections) and the stage
    output after optional downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        num_heads: int,
        dropout: float = 0.0,
        downsample: bool = True,
        use_film: bool = False,
        film_embed_dim: int | None = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if num_res_blocks <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {num_res_blocks}")

        blocks: list[nn.Module] = []
        current_in = in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=current_in,
                    out_channels=out_channels,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    use_film=use_film,
                    film_embed_dim=film_embed_dim,
                    num_groups=num_groups,
                )
            )
            current_in = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.attention = build_attention_block(
            channels=out_channels,
            use_attention=use_attention,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.downsample = Downsample2d(out_channels) if downsample else nn.Identity()
        self.has_downsample = downsample

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        film_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for block in self.blocks:
            h = block(h, emb, film_emb)

        h = self.attention(h)
        skip = h
        out = self.downsample(h)
        return out, skip


class Encoder(nn.Module):
    """
    Hierarchical UNet encoder.

    The encoder applies:
    - input projection
    - a sequence of encoder stages
    - a middle residual block
    - optional middle attention
    - a second middle residual block

    It returns:
    - bottleneck feature map
    - list of skip tensors, ordered from shallow to deep
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mults: tuple[int, ...],
        num_res_blocks: int,
        embed_dim: int,
        attention_resolutions: tuple[int, ...],
        input_resolution: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_film: bool = False,
        film_embed_dim: int | None = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if len(channel_mults) == 0:
            raise ValueError("channel_mults must not be empty")
        if input_resolution <= 0:
            raise ValueError(f"input_resolution must be positive, got {input_resolution}")

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        stages: list[nn.Module] = []
        stage_specs: list[EncoderStageSpec] = []

        current_channels = base_channels
        current_resolution = input_resolution

        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            use_attention = current_resolution in attention_resolutions
            downsample = level < len(channel_mults) - 1

            stage_specs.append(
                EncoderStageSpec(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    resolution=current_resolution,
                    downsample=downsample,
                    use_attention=use_attention,
                )
            )

            stages.append(
                EncoderStage(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    embed_dim=embed_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    dropout=dropout,
                    downsample=downsample,
                    use_film=use_film,
                    film_embed_dim=film_embed_dim,
                    num_groups=num_groups,
                )
            )

            current_channels = out_channels
            if downsample:
                current_resolution = current_resolution // 2

        self.stages = nn.ModuleList(stages)
        self.stage_specs = tuple(stage_specs)
        self.out_channels = current_channels
        self.bottleneck_resolution = current_resolution

        bottleneck_attention = current_resolution in attention_resolutions
        self.mid_block1 = ResidualBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            use_film=use_film,
            film_embed_dim=film_embed_dim,
            num_groups=num_groups,
        )
        self.mid_attn = build_attention_block(
            channels=current_channels,
            use_attention=bottleneck_attention,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mid_block2 = ResidualBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            use_film=use_film,
            film_embed_dim=film_embed_dim,
            num_groups=num_groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        film_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")

        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for stage in self.stages:
            h, skip = stage(h, emb, film_emb)
            skips.append(skip)

        h = self.mid_block1(h, emb, film_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb, film_emb)

        return h, skips