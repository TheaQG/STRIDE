

"""
Decoder modules for STRIDE EDM-style UNets.

This file mirrors the encoder structure and keeps the strongest reusable ideas
from the legacy `score_unet.py` implementation:

- residual blocks with embedding injection
- optional FiLM modulation
- optional attention at selected resolutions
- skip-connection fusion
- hierarchical upsampling

The goal is to keep the decoder generic and reusable. It should not know
anything about precipitation, temperature, statics, or specific data adapters.
Those decisions belong one level above this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from stride_core.models.attention import build_attention_block
from stride_core.models.encoder import ResidualBlock


@dataclass(frozen=True)
class DecoderStageSpec:
    """
    Lightweight description of one decoder stage.

    Parameters
    ----------
    in_channels:
        Number of incoming channels to the stage before skip fusion.
    skip_channels:
        Number of channels received from the corresponding encoder skip.
    out_channels:
        Number of output channels of the stage.
    resolution:
        Spatial resolution at this stage.
    upsample:
        Whether to upsample at the end of the stage.
    use_attention:
        Whether to apply attention inside the stage.
    """

    in_channels: int
    skip_channels: int
    out_channels: int
    resolution: int
    upsample: bool
    use_attention: bool


class Upsample2d(nn.Module):
    """
    2× spatial upsampling followed by a 3×3 convolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SkipFusion(nn.Module):
    """
    Fuse decoder features with an encoder skip tensor by channel concatenation.

    An optional 1x1 projection is applied after concatenation to stabilize the
    merged representation before residual processing.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or skip_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels, skip_channels, and out_channels must be positive, got "
                f"{(in_channels, skip_channels, out_channels)}"
            )
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or skip.ndim != 4:
            raise ValueError(
                f"Expected x and skip with shape [B, C, H, W], got {tuple(x.shape)} and {tuple(skip.shape)}"
            )
        if x.shape[0] != skip.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x and skip: {x.shape[0]} vs {skip.shape[0]}"
            )
        if x.shape[2:] != skip.shape[2:]:
            raise ValueError(
                f"Spatial mismatch between x and skip: {tuple(x.shape[2:])} vs {tuple(skip.shape[2:])}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )
        if skip.shape[1] != self.skip_channels:
            raise ValueError(
                f"skip channel mismatch: expected {self.skip_channels}, got {skip.shape[1]}"
            )

        fused = torch.cat([x, skip], dim=1)
        return self.proj(fused)


class DecoderStage(nn.Module):
    """
    One decoder stage consisting of:
    - skip fusion
    - residual blocks
    - optional attention
    - optional upsampling

    The stage expects a single encoder skip tensor aligned in spatial size with
    the current decoder feature map.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        embed_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        num_heads: int,
        dropout: float = 0.0,
        upsample: bool = True,
        use_film: bool = False,
        film_embed_dim: int | None = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if num_res_blocks <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {num_res_blocks}")

        self.skip_fusion = SkipFusion(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
        )

        blocks: list[nn.Module] = []
        current_in = out_channels
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
        self.upsample = Upsample2d(out_channels) if upsample else nn.Identity()
        self.has_upsample = upsample

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        emb: torch.Tensor,
        film_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.skip_fusion(x, skip)
        for block in self.blocks:
            h = block(h, emb, film_emb)
        h = self.attention(h)
        h = self.upsample(h)
        return h


class Decoder(nn.Module):
    """
    Hierarchical UNet decoder.

    This decoder mirrors the encoder and consumes skip tensors ordered from
    shallow to deep, i.e. the same order returned by `Encoder.forward(...)`.
    Internally it reverses them to decode from the bottleneck upward.

    It applies:
    - a sequence of decoder stages
    - final normalization + activation + output convolution
    """

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: tuple[int, ...],
        out_channels: int,
        base_channels: int,
        channel_mults: tuple[int, ...],
        num_res_blocks: int,
        embed_dim: int,
        attention_resolutions: tuple[int, ...],
        bottleneck_resolution: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_film: bool = False,
        film_embed_dim: int | None = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(channel_mults) == 0:
            raise ValueError("channel_mults must not be empty")
        if len(skip_channels) != len(channel_mults):
            raise ValueError(
                f"skip_channels length must match channel_mults length, got {len(skip_channels)} and {len(channel_mults)}"
            )
        if bottleneck_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError(
                f"bottleneck_channels, out_channels, and base_channels must be positive, got "
                f"{(bottleneck_channels, out_channels, base_channels)}"
            )
        if bottleneck_resolution <= 0:
            raise ValueError(
                f"bottleneck_resolution must be positive, got {bottleneck_resolution}"
            )

        reversed_mults = list(channel_mults[::-1])
        reversed_skips = list(skip_channels[::-1])

        stages: list[nn.Module] = []
        stage_specs: list[DecoderStageSpec] = []

        current_channels = bottleneck_channels
        current_resolution = bottleneck_resolution

        for level, (mult, skip_ch) in enumerate(zip(reversed_mults, reversed_skips)):
            stage_out_channels = base_channels * mult
            use_attention = current_resolution in attention_resolutions
            upsample = level < len(reversed_mults) - 1

            stage_specs.append(
                DecoderStageSpec(
                    in_channels=current_channels,
                    skip_channels=skip_ch,
                    out_channels=stage_out_channels,
                    resolution=current_resolution,
                    upsample=upsample,
                    use_attention=use_attention,
                )
            )

            stages.append(
                DecoderStage(
                    in_channels=current_channels,
                    skip_channels=skip_ch,
                    out_channels=stage_out_channels,
                    embed_dim=embed_dim,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    dropout=dropout,
                    upsample=upsample,
                    use_film=use_film,
                    film_embed_dim=film_embed_dim,
                    num_groups=num_groups,
                )
            )

            current_channels = stage_out_channels
            if upsample:
                current_resolution = current_resolution * 2

        self.stages = nn.ModuleList(stages)
        self.stage_specs = tuple(stage_specs)
        self.final_channels = current_channels

        final_num_groups = self._choose_num_groups(current_channels, num_groups)
        self.out_norm = nn.GroupNorm(num_groups=final_num_groups, num_channels=current_channels)
        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _choose_num_groups(num_channels: int, requested_groups: int) -> int:
        max_groups = min(requested_groups, num_channels)
        for groups in range(max_groups, 0, -1):
            if num_channels % groups == 0:
                return groups
        return 1

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        emb: torch.Tensor,
        film_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], got {tuple(x.shape)}")
        if len(skips) != len(self.stages):
            raise ValueError(
                f"Expected {len(self.stages)} skip tensors, got {len(skips)}"
            )

        h = x
        for stage, skip in zip(self.stages, reversed(skips)):
            h = stage(h, skip, emb, film_emb)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)