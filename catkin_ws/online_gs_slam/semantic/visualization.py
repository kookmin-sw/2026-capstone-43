from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_PALETTE = np.array(
    [
        [166, 206, 227],
        [31, 120, 180],
        [178, 223, 138],
        [51, 160, 44],
        [251, 154, 153],
        [227, 26, 28],
        [253, 191, 111],
        [255, 127, 0],
        [202, 178, 214],
        [106, 61, 154],
        [255, 255, 153],
        [177, 89, 40],
    ],
    dtype=np.uint8,
)


def label_colors(labels: torch.Tensor | np.ndarray) -> np.ndarray:
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    return _PALETTE[np.mod(labels_np.astype(np.int64), len(_PALETTE))]


def write_labeled_ply(path: str | Path, xyz: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_np = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    colors = label_colors(labels)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz_np, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
