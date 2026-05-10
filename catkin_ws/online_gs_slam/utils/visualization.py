from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch


def save_uncertainty_bar(path: Union[str, Path], values: np.ndarray, width: int = 512, height: int = 64) -> None:
    path = Path(path)
    if values.size == 0:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        v = np.clip(values.astype(np.float32), 0.0, 1.0)
        hist, _ = np.histogram(v, bins=width, range=(0.0, 1.0))
        hist = hist.astype(np.float32) / max(float(hist.max()), 1.0)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for x, h in enumerate(hist):
            y0 = height - int(h * height)
            image[y0:, x] = (0, 128, 255)
    cv2.imwrite(str(path), image)


def save_rgb_debug(path: Union[str, Path], target_rgb: torch.Tensor, rendered_rgb: torch.Tensor) -> None:
    """Save target/render/error side-by-side.

    Tensors are expected as HWC RGB in [0, 1].
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target_rgb.detach().cpu().clamp(0.0, 1.0).numpy()
    render = rendered_rgb.detach().cpu().clamp(0.0, 1.0).numpy()
    error = np.abs(target - render)
    panel = np.concatenate([target, render, error], axis=1)
    panel_bgr = cv2.cvtColor((panel * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), panel_bgr)


def save_gaussian_ply(
    path: Union[str, Path],
    means: torch.Tensor,
    colors: torch.Tensor,
    uncertainty: torch.Tensor = None,
    max_points: int = 200000,
) -> None:
    """Export Gaussian centers as a colored PLY point cloud."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if means.numel() == 0:
        points = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
    else:
        points_t = means.detach().cpu()
        colors_t = colors.detach().cpu().clamp(0.0, 1.0)
        if points_t.shape[0] > max_points:
            idx = torch.linspace(0, points_t.shape[0] - 1, max_points).long()
            points_t = points_t[idx]
            colors_t = colors_t[idx]
        points = points_t.numpy().astype(np.float32)
        rgb = (colors_t.numpy() * 255.0).astype(np.uint8)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
