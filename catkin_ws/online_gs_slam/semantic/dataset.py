from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class HashGridSupervision:
    xyz: torch.Tensor
    time: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    class_names: Optional[list[str]] = None

    @property
    def num_classes(self) -> int:
        return int(self.labels.max().item()) + 1 if self.labels.numel() else 0


def load_hash_grid_supervision(path: str | Path, device: str = "cpu") -> HashGridSupervision:
    """Load Gaussian/point supervision for the 4D semantic hash grid.

    Expected npz keys:
      - xyz or positions: float [N, 3]
      - labels: int [N]
    Optional keys:
      - time or timestamps: float [N]
      - weights: float [N]
      - class_names: string [C]
    """

    path = Path(path)
    data = np.load(path, allow_pickle=True)
    xyz_key = "xyz" if "xyz" in data else "positions"
    time_key = "time" if "time" in data else "timestamps" if "timestamps" in data else None
    if xyz_key not in data:
        raise KeyError(f"{path} must contain 'xyz' or 'positions'")
    if "labels" not in data:
        raise KeyError(f"{path} must contain integer 'labels'")

    xyz = torch.as_tensor(data[xyz_key], dtype=torch.float32, device=device)
    labels = torch.as_tensor(data["labels"], dtype=torch.long, device=device)
    if time_key is None:
        time = torch.zeros((xyz.shape[0],), dtype=torch.float32, device=device)
    else:
        time = torch.as_tensor(data[time_key], dtype=torch.float32, device=device)
    if "weights" in data:
        weights = torch.as_tensor(data["weights"], dtype=torch.float32, device=device)
    else:
        weights = torch.ones((xyz.shape[0],), dtype=torch.float32, device=device)
    class_names = None
    if "class_names" in data:
        class_names = [str(item) for item in data["class_names"].tolist()]
    return HashGridSupervision(xyz=xyz, time=time, labels=labels, weights=weights, class_names=class_names)
