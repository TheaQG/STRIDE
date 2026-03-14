"""
RainGate module for auxiliary wet/dry prediction in STRIDE.

This module predicts pixel-wise wet/dry logits from conditioning inputs only.
It is intended to be used as an auxiliary head alongside the main diffusion
model, not as a hard output gate in the first implementation.

Input
-----
A conditioning tensor with shape:
    [B, C_in, H, W]

Typical input:
    concatenated dynamic and static conditioning channels

Output
------
Pixel-wise wet/dry logits with shape:
    [B, 1, H, W]

Design notes
------------
- Keep the module small and stable.
- Preserve spatial resolution throughout.
- Use simple convolution + normalization + SiLU blocks.
- Return logits rather than probabilities so BCEWithLogitsLoss can be used.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RainGateBlock(nn.Module):
    """
    Small convolutional block used inside RainGate.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"RainGateBlock channels must be positive, got {(in_channels, out_channels)}"
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RainGate(nn.Module):
    """
    Lightweight CNN for pixel-wise wet/dry prediction.

    Parameters
    ----------
    in_channels:
        Number of conditioning channels provided to RainGate.
    hidden_channels:
        Base hidden width used by the convolutional trunk.
    num_blocks:
        Number of intermediate convolutional blocks after the input block.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}"
            )
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        layers: list[nn.Module] = [RainGateBlock(in_channels, hidden_channels)]
        for _ in range(num_blocks - 1):
            layers.append(RainGateBlock(hidden_channels, hidden_channels))

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict wet/dry logits.

        Parameters
        ----------
        x:
            Conditioning tensor with shape [B, C_in, H, W].

        Returns
        -------
        torch.Tensor
            Wet/dry logits with shape [B, 1, H, W].
        """
        if x.ndim != 4:
            raise ValueError(
                f"RainGate expected input with shape [B, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                "RainGate input channel mismatch: expected "
                f"{self.in_channels}, got {x.shape[1]}"
            )

        features = self.trunk(x)
        logits = self.head(features)
        return logits