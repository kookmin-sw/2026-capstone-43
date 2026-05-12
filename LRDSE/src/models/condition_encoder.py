from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.left_pad = (self.kernel_size - 1) * self.dilation if self.causal else 0
        self.same_pad = ((self.kernel_size - 1) * self.dilation) // 2

        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(float(dropout))

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            return F.pad(x, (self.left_pad, 0))
        return F.pad(x, (self.same_pad, self.same_pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = F.silu(y)
        y = self.conv1(self._pad(y))
        y = self.dropout(y)
        y = self.norm2(y)
        y = F.silu(y)
        y = self.conv2(y)
        y = self.dropout(y)
        return x + y


class ConditionEncoder(nn.Module):
    """
    Predict log(1 + |STFT(noise)|) from frame-wise foot-force features.

    Input:
        force_feat: [B, 24, T]

    Output:
        mag_prior: [B, F, T]
    """

    def __init__(
        self,
        in_channels: int = 24,
        freq_bins: int = 256,
        hidden_channels: int = 256,
        num_layers: int = 8,
        kernel_size: int = 5,
        dropout: float = 0.05,
        causal: bool = True,
        max_dilation: int = 16,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if freq_bins <= 0:
            raise ValueError(f"freq_bins must be positive, got {freq_bins}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.in_channels = int(in_channels)
        self.freq_bins = int(freq_bins)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.max_dilation = int(max_dilation)

        self.input_proj = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        blocks = []
        dilation = 1
        for _ in range(self.num_layers):
            blocks.append(
                TemporalBlock(
                    channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                    causal=self.causal,
                )
            )
            dilation = min(dilation * 2, self.max_dilation)
        self.encoder = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.GroupNorm(_group_count(self.hidden_channels), self.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, self.freq_bins, 1),
        )

    def forward(
        self,
        force_feat: torch.Tensor,
        return_latent: bool = False,
    ):
        if force_feat.dim() != 3:
            raise ValueError(f"Expected force_feat [B,C,T], got {tuple(force_feat.shape)}")
        if force_feat.size(1) != self.in_channels:
            raise ValueError(
                f"Expected input channel={self.in_channels}, got {force_feat.size(1)}"
            )

        z = self.input_proj(force_feat)
        z = self.encoder(z)
        mag = F.softplus(self.decoder(z))

        if return_latent:
            return mag, z
        return mag
