#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def read_ply_with_open3d(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
    if xyz.size == 0:
        raise RuntimeError(f"No points were loaded from {path}")
    return xyz, colors


def read_ply_with_plyfile(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    from plyfile import PlyData

    ply = PlyData.read(path)
    vertices = ply["vertex"]
    names = vertices.data.dtype.names or ()
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    colors = None
    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32) / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
        colors = np.clip(0.5 + 0.28209479177387814 * dc, 0.0, 1.0)
    return xyz, colors


def read_gaussian_ply(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        return read_ply_with_plyfile(path)
    except Exception as ply_error:
        try:
            return read_ply_with_open3d(path)
        except Exception as open3d_error:
            raise RuntimeError(
                f"Failed to read {path} with plyfile ({ply_error}) and Open3D ({open3d_error})"
            ) from open3d_error


def kmeans_numpy(values: np.ndarray, k: int, iterations: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(values) < k:
        return np.arange(len(values), dtype=np.int64)
    centers = values[rng.choice(len(values), size=k, replace=False)].copy()
    labels = np.zeros((len(values),), dtype=np.int64)
    for _ in range(iterations):
        dist = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1).astype(np.int64)
        for label in range(k):
            mask = labels == label
            if np.any(mask):
                centers[label] = values[mask].mean(axis=0)
    return labels


def make_labels(xyz: np.ndarray, colors: Optional[np.ndarray], mode: str, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    if num_classes < 2:
        raise ValueError("--num-classes must be >= 2")

    if mode == "height":
        values = xyz[:, 2]
        class_names = np.array([f"height_bin_{i}" for i in range(num_classes)])
        bins = np.quantile(values, np.linspace(0.0, 1.0, num_classes + 1)[1:-1])
        labels = np.digitize(values, bins).astype(np.int64)
        return labels, class_names

    if mode in {"axis-x", "axis-y", "axis-z"}:
        axis = {"axis-x": 0, "axis-y": 1, "axis-z": 2}[mode]
        values = xyz[:, axis]
        axis_name = ["x", "y", "z"][axis]
        class_names = np.array([f"{axis_name}_bin_{i}" for i in range(num_classes)])
        bins = np.quantile(values, np.linspace(0.0, 1.0, num_classes + 1)[1:-1])
        labels = np.digitize(values, bins).astype(np.int64)
        return labels, class_names

    if mode == "rgb-kmeans":
        if colors is None:
            raise ValueError("rgb-kmeans requires RGB/color properties in the PLY")
        labels = kmeans_numpy(colors.astype(np.float32), num_classes)
        class_names = np.array([f"rgb_cluster_{i}" for i in range(num_classes)])
        return labels, class_names

    if mode == "xyz-kmeans":
        normalized = (xyz - xyz.mean(axis=0, keepdims=True)) / (xyz.std(axis=0, keepdims=True) + 1e-6)
        labels = kmeans_numpy(normalized.astype(np.float32), num_classes)
        class_names = np.array([f"xyz_cluster_{i}" for i in range(num_classes)])
        return labels, class_names

    raise ValueError(f"Unknown label mode: {mode}")


def write_preview_ply(path: Path, xyz: np.ndarray, labels: np.ndarray) -> None:
    from online_gs_slam.semantic.visualization import write_labeled_ply

    write_labeled_ply(path, xyz, labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert exported Gaussian Splatting PLY centers into semantic_points.npz for 4D hash grid training."
    )
    parser.add_argument("--ply", required=True, type=Path, help="Gaussian Splatting PLY from ns-export gaussian-splat")
    parser.add_argument("--output", required=True, type=Path, help="Output semantic_points.npz")
    parser.add_argument("--preview-ply", type=Path, default=None, help="Optional colored label preview PLY")
    parser.add_argument("--label-mode", default="height", choices=["height", "axis-x", "axis-y", "axis-z", "rgb-kmeans", "xyz-kmeans"])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    xyz, colors = read_gaussian_ply(args.ply)
    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(xyz), size=args.max_points, replace=False)
        xyz = xyz[keep]
        if colors is not None:
            colors = colors[keep]

    labels, class_names = make_labels(xyz, colors, args.label_mode, args.num_classes)
    time = np.full((len(xyz),), args.time, dtype=np.float32)
    weights = np.ones((len(xyz),), dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xyz=xyz.astype(np.float32),
        time=time,
        labels=labels.astype(np.int64),
        weights=weights,
        class_names=class_names,
    )
    print(f"Wrote {args.output} with {len(xyz)} Gaussian centers")
    print(f"label_mode={args.label_mode} classes={class_names.tolist()}")

    if args.preview_ply is not None:
        write_preview_ply(args.preview_ply, xyz, labels)
        print(f"Wrote {args.preview_ply}")


if __name__ == "__main__":
    main()
