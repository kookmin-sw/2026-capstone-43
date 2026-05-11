#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


def quat_to_rot(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def world_from_camera(row):
    rot = quat_to_rot(float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
    return mat


def load_rows(poses_csv, start_index, keep_every, max_frames):
    rows = []
    with open(poses_csv, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_index:
                continue
            if keep_every > 1 and idx % keep_every != 0:
                continue
            rows.append(row)
            if max_frames is not None and len(rows) >= max_frames:
                break
    return rows


def depth_to_meters(depth, scale):
    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        return depth.astype(np.float32) * scale
    return depth.astype(np.float32)


def backproject(depth_m, intrinsics, stride, depth_min, depth_max):
    height, width = depth_m.shape[:2]
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    u = xs[valid].astype(np.float32)
    v = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points = np.stack([x, y, z], axis=1)
    pixels = np.stack([u, v], axis=1).astype(np.int32)
    return points, pixels


def voxel_downsample(points, colors, voxel_size):
    if voxel_size <= 0.0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return points[unique], colors[unique]


def write_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_colmap_points3d(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for idx, (point, color) in enumerate(zip(points, colors), start=1):
            f.write(
                f"{idx} {point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0.0\n"
            )


def build_points(args):
    dataset_dir = args.dataset_dir.expanduser().resolve()
    with open(dataset_dir / "camera_info.json") as f:
        intrinsics = json.load(f)

    rows = load_rows(dataset_dir / "poses.csv", args.start_index, args.keep_every, args.max_frames)
    rng = random.Random(args.random_seed)
    all_points = []
    all_colors = []

    for row in rows:
        depth_name = row.get("depth_filename", "")
        if not depth_name:
            continue

        depth_path = dataset_dir / args.depth_dir / depth_name
        rgb_path = dataset_dir / "rgb" / row["filename"]
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if depth is None or rgb_bgr is None:
            continue
        if depth.shape[:2] != rgb_bgr.shape[:2]:
            rgb_bgr = cv2.resize(rgb_bgr, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        cam_points, pixels = backproject(
            depth_to_meters(depth, args.depth_scale),
            intrinsics,
            args.point_stride,
            args.depth_min,
            args.depth_max,
        )
        if len(cam_points) == 0:
            continue

        if args.max_points_per_frame > 0 and len(cam_points) > args.max_points_per_frame:
            selected = np.array(rng.sample(range(len(cam_points)), args.max_points_per_frame), dtype=np.int64)
            cam_points = cam_points[selected]
            pixels = pixels[selected]

        cam_points_h = np.concatenate([cam_points.astype(np.float64), np.ones((len(cam_points), 1))], axis=1)
        world_points = (world_from_camera(row) @ cam_points_h.T).T[:, :3].astype(np.float32)
        rgb = rgb_bgr[pixels[:, 1], pixels[:, 0], ::-1].astype(np.uint8)

        all_points.append(world_points)
        all_colors.append(rgb)

    if not all_points:
        raise RuntimeError("No points generated. Check depth files, depth scale, and poses.csv depth_filename column.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if args.max_total_points > 0 and len(points) > args.max_total_points:
        selected = np.array(rng.sample(range(len(points)), args.max_total_points), dtype=np.int64)
        points = points[selected]
        colors = colors[selected]

    return voxel_downsample(points, colors, args.voxel_size)


def main():
    parser = argparse.ArgumentParser(
        description="Build COLMAP-free SfM-style initial points from aligned depth, RGB, and known camera poses."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--depth-dir", default="depth")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--colmap-points3d-output", type=Path, default=None)
    parser.add_argument("--keep-every", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=5.0)
    parser.add_argument("--point-stride", type=int, default=6)
    parser.add_argument("--max-points-per-frame", type=int, default=12000)
    parser.add_argument("--max-total-points", type=int, default=1500000)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output = args.output or (dataset_dir / "sparse_pc.ply")
    points, colors = build_points(args)
    write_ply(output, points, colors)
    print(f"Wrote {output} with {len(points)} points")

    if args.colmap_points3d_output:
        write_colmap_points3d(args.colmap_points3d_output, points, colors)
        print(f"Wrote {args.colmap_points3d_output} in COLMAP text points3D format")


if __name__ == "__main__":
    main()
