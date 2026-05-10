"""SELDNet backbone modules for a spatial branch.

This is a cleaned subset of the DCASE SELD baseline model:
- Conv2D blocks
- BiGRU + gating
- MHSA + LayerNorm
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SeldNetSpatialEncoder(nn.Module):
    """SELD-style spatial encoder returning temporal embeddings [B, T, D]."""

    def __init__(
        self,
        in_channels: int = 7,
        nb_mel_bins: int = 64,
        nb_cnn2d_filt: int = 64,
        f_pool_size: Sequence[int] = (4, 4, 2),
        t_pool_size: Sequence[int] = (5, 1, 1),
        dropout_rate: float = 0.05,
        nb_rnn_layers: int = 2,
        rnn_size: int = 128,
        nb_heads: int = 8,
        nb_self_attn_layers: int = 2,
    ) -> None:
        super().__init__()
        if len(f_pool_size) != len(t_pool_size):
            raise ValueError("f_pool_size and t_pool_size must have the same length.")

        self.output_dim = rnn_size
        self.conv_block_list = nn.ModuleList()

        if len(f_pool_size):
            for idx in range(len(f_pool_size)):
                in_ch = nb_cnn2d_filt if idx else in_channels
                self.conv_block_list.append(
                    ConvBlock(in_channels=in_ch, out_channels=nb_cnn2d_filt)
                )
                self.conv_block_list.append(
                    nn.MaxPool2d((t_pool_size[idx], f_pool_size[idx]))
                )
                self.conv_block_list.append(nn.Dropout2d(p=dropout_rate))

        reduced_bins = int(np.floor(nb_mel_bins / np.prod(f_pool_size)))
        self.gru_input_dim = nb_cnn2d_filt * reduced_bins
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=rnn_size,
            num_layers=nb_rnn_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(nb_self_attn_layers):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=rnn_size,
                    num_heads=nb_heads,
                    dropout=dropout_rate,
                    batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(rnn_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: [batch, channels, frames, mel_bins]."""
        if x.ndim != 4:
            raise ValueError(
                f"SeldNetSpatialEncoder expects 4D input [B,C,T,F], got shape={tuple(x.shape)}"
            )

        for layer in self.conv_block_list:
            x = layer(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]

        for mhsa, ln in zip(self.mhsa_block_list, self.layer_norm_list):
            residual = x
            x, _ = mhsa(residual, residual, residual)
            x = ln(x + residual)

        return x

    def load_from_dcase_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str = "cpu",
    ) -> Dict[str, List[str]]:
        """Load matching backbone weights from a DCASE baseline checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError("Unsupported checkpoint format.")

        own_state = self.state_dict()
        accepted_prefixes = ("conv_block_list", "gru", "mhsa_block_list", "layer_norm_list")
        filtered = {}
        for key, value in state_dict.items():
            if not key.startswith(accepted_prefixes):
                continue
            if key not in own_state:
                continue
            if own_state[key].shape != value.shape:
                continue
            filtered[key] = value

        load_result = self.load_state_dict(filtered, strict=False)
        return {
            "loaded_keys": sorted(filtered.keys()),
            "missing_keys": load_result.missing_keys,
            "unexpected_keys": load_result.unexpected_keys,
        }

