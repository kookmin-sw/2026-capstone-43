from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class HashGrid4DConfig:
    """Configuration for a PyTorch 4D multi-scale hash grid.

    The input coordinate is (x, y, z, t). xyz is normalized by a scene bounding
    box, and t is normalized by a time range. The encoded feature can be used
    for open-vocabulary semantic logits, material embeddings, or uncertainty.
    """

    num_levels: int = 12
    features_per_level: int = 2
    log2_hashmap_size: int = 18
    base_resolution: int = 8
    finest_resolution: int = 256
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    output_dim: int = 16
    bbox_min: Tuple[float, float, float] = (-3.0, -3.0, -1.0)
    bbox_max: Tuple[float, float, float] = (3.0, 3.0, 3.0)
    time_min: float = 0.0
    time_max: float = 1.0


class MultiScaleHashGrid4D(nn.Module):
    """Small research-friendly 4D hash grid encoder with an MLP head.

    This intentionally avoids tiny-cuda-nn so it can run anywhere PyTorch runs.
    It is slower than a fused CUDA implementation, but much easier to inspect
    and modify while designing the semantic/material representation.
    """

    def __init__(self, config: HashGrid4DConfig):
        super().__init__()
        self.config = config
        self.hashmap_size = 2 ** config.log2_hashmap_size
        self.input_dim = config.num_levels * config.features_per_level

        if config.num_levels <= 1:
            growth = 1.0
        else:
            growth = (config.finest_resolution / config.base_resolution) ** (1.0 / (config.num_levels - 1))
        resolutions = [int(round(config.base_resolution * (growth**level))) for level in range(config.num_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long), persistent=False)
        self.register_buffer("bbox_min", torch.tensor(config.bbox_min, dtype=torch.float32), persistent=True)
        self.register_buffer("bbox_max", torch.tensor(config.bbox_max, dtype=torch.float32), persistent=True)
        self.register_buffer("time_range", torch.tensor([config.time_min, config.time_max], dtype=torch.float32), persistent=True)
        self.register_buffer("hash_primes", torch.tensor([1, 2654435761, 805459861, 3674653429], dtype=torch.long), persistent=False)

        tables = []
        for _ in range(config.num_levels):
            table = nn.Parameter(torch.empty((self.hashmap_size, config.features_per_level), dtype=torch.float32))
            nn.init.uniform_(table, -1e-4, 1e-4)
            tables.append(table)
        self.tables = nn.ParameterList(tables)

        layers = []
        in_dim = self.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.head = nn.Sequential(*layers)

    def normalize(self, xyz: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        xyz = xyz.to(dtype=torch.float32)
        bbox_extent = torch.clamp(self.bbox_max - self.bbox_min, min=1e-6)
        xyz01 = (xyz - self.bbox_min) / bbox_extent
        xyz01 = xyz01.clamp(0.0, 1.0)

        if time is None:
            t01 = torch.zeros((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
        else:
            time = time.reshape(-1, 1).to(dtype=torch.float32, device=xyz.device)
            t_extent = torch.clamp(self.time_range[1] - self.time_range[0], min=1e-6)
            t01 = ((time - self.time_range[0]) / t_extent).clamp(0.0, 1.0)
        return torch.cat([xyz01, t01], dim=-1)

    def _hash(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(torch.long)
        hashed = torch.zeros(indices.shape[:-1], dtype=torch.long, device=indices.device)
        primes = self.hash_primes.to(indices.device)
        for dim in range(4):
            hashed ^= indices[..., dim] * primes[dim]
        return torch.remainder(hashed, self.hashmap_size)

    def _encode_level(self, coords01: torch.Tensor, table: torch.Tensor, resolution: int) -> torch.Tensor:
        coords = coords01 * float(max(resolution - 1, 1))
        lower = torch.floor(coords).to(torch.long)
        frac = coords - lower.to(coords.dtype)

        out = torch.zeros((coords.shape[0], table.shape[1]), dtype=table.dtype, device=coords.device)
        # 4D multilinear interpolation: 2^4 corners.
        for mask in range(16):
            offset = torch.tensor(
                [(mask >> dim) & 1 for dim in range(4)],
                dtype=torch.long,
                device=coords.device,
            )
            corner = torch.clamp(lower + offset, min=0, max=resolution - 1)
            weight = torch.ones((coords.shape[0],), dtype=coords.dtype, device=coords.device)
            for dim in range(4):
                weight = weight * (frac[:, dim] if offset[dim] else (1.0 - frac[:, dim]))
            out = out + table[self._hash(corner)] * weight[:, None]
        return out

    def encode(self, xyz: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        coords01 = self.normalize(xyz, time)
        features = []
        for table, resolution in zip(self.tables, self.resolutions.tolist()):
            features.append(self._encode_level(coords01, table, int(resolution)))
        return torch.cat(features, dim=-1)

    def forward(self, xyz: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(xyz, time))

    @torch.no_grad()
    def predict_labels(self, xyz: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(xyz, time).argmax(dim=-1)
