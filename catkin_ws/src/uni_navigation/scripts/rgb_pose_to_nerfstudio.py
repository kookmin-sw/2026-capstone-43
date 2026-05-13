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
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def matmul3(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def make_world_from_ros_optical(row):
    tx = float(row["tx"])
    ty = float(row["ty"])
    tz = float(row["tz"])
    if origin is not None:
        tx = origin[0] + translation_scale * (tx - origin[0])
        ty = origin[1] + translation_scale * (ty - origin[1])
        tz = origin[2] + translation_scale * (tz - origin[2])
    else:
        tx *= translation_scale
        ty *= translation_scale
        tz *= translation_scale
    qx = float(row["qx"])
    qy = float(row["qy"])
    qz = float(row["qz"])
    qw = float(row["qw"])
    rot = quat_to_rot(qx, qy, qz, qw)
    return np.array(
        [
            [rot[0][0], rot[0][1], rot[0][2], tx],
            [rot[1][0], rot[1][1], rot[1][2], ty],
            [rot[2][0], rot[2][1], rot[2][2], tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_transform(row, origin=None, translation_scale=1.0):
    tx = float(row["tx"])
    ty = float(row["ty"])
    tz = float(row["tz"])
    if origin is not None:
        tx = origin[0] + translation_scale * (tx - origin[0])
        ty = origin[1] + translation_scale * (ty - origin[1])
        tz = origin[2] + translation_scale * (tz - origin[2])
    else:
        tx *= translation_scale
        ty *= translation_scale
        tz *= translation_scale
    qx = float(row["qx"])
    qy = float(row["qy"])
    qz = float(row["qz"])
    qw = float(row["qw"])

    # ROS optical/OpenCV camera frame: +X right, +Y down, +Z forward.
    # Nerfstudio/OpenGL camera frame: +X right, +Y up, +Z backward.
    ros_optical_from_ns_camera = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    rot_world_from_ros_optical = quat_to_rot(qx, qy, qz, qw)
    rot_world_from_ns_camera = matmul3(rot_world_from_ros_optical, ros_optical_from_ns_camera)

    return [
        [rot_world_from_ns_camera[0][0], rot_world_from_ns_camera[0][1], rot_world_from_ns_camera[0][2], tx],
        [rot_world_from_ns_camera[1][0], rot_world_from_ns_camera[1][1], rot_world_from_ns_camera[1][2], ty],
        [rot_world_from_ns_camera[2][0], rot_world_from_ns_camera[2][1], rot_world_from_ns_camera[2][2], tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


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


def depth_to_meters(depth, depth_scale):
    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        return depth.astype(np.float32) * depth_scale
    return depth.astype(np.float32)


def backproject_depth(depth_m, intrinsics, sample_stride, depth_min, depth_max):
    h, w = depth_m.shape[:2]
    ys, xs = np.mgrid[0:h:sample_stride, 0:w:sample_stride]
    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    xs_valid = xs[valid].astype(np.float32)
    ys_valid = ys[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    x = (xs_valid - float(intrinsics["cx"])) / float(intrinsics["fx"]) * z_valid
    y = (ys_valid - float(intrinsics["cy"])) / float(intrinsics["fy"]) * z_valid
    points = np.stack([x, y, z_valid], axis=1)
    pixels = np.stack([xs_valid, ys_valid], axis=1).astype(np.int32)
    return points, pixels


def voxel_downsample(points, colors, voxel_size):
    if voxel_size <= 0.0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    return points[unique_indices], colors[unique_indices]


def write_ascii_ply(path, points, colors):
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


def generate_point_cloud(dataset_dir, rows, intrinsics, args):
    depth_dir = dataset_dir / args.depth_dir
    rgb_dir = dataset_dir / "rgb"
    all_points = []
    all_colors = []
    rng = random.Random(args.random_seed)
    origin = None
    if rows:
        origin = np.array([float(rows[0]["tx"]), float(rows[0]["ty"]), float(rows[0]["tz"])], dtype=np.float64)

    for row in rows:
        depth_filename = row.get("depth_filename", "")
        if not depth_filename:
            continue
        depth_path = depth_dir / depth_filename
        rgb_path = rgb_dir / row["filename"]
        if not depth_path.exists() or not rgb_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if depth is None or rgb_bgr is None:
            continue
        if depth.shape[:2] != rgb_bgr.shape[:2]:
            rgb_bgr = cv2.resize(rgb_bgr, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        depth_m = depth_to_meters(depth, args.depth_scale)
        cam_points, pixels = backproject_depth(depth_m, intrinsics, args.point_stride, args.depth_min, args.depth_max)
        if len(cam_points) == 0:
            continue

        if args.max_points_per_frame > 0 and len(cam_points) > args.max_points_per_frame:
            indices = np.array(rng.sample(range(len(cam_points)), args.max_points_per_frame), dtype=np.int64)
            cam_points = cam_points[indices]
            pixels = pixels[indices]

        world_from_camera = make_world_from_ros_optical(row)
        if origin is not None:
            world_from_camera[:3, 3] = origin + args.pose_translation_scale * (world_from_camera[:3, 3] - origin)
        else:
            world_from_camera[:3, 3] *= args.pose_translation_scale
        cam_h = np.concatenate([cam_points.astype(np.float64), np.ones((len(cam_points), 1))], axis=1)
        world_points = (world_from_camera @ cam_h.T).T[:, :3].astype(np.float32)
        rgb = rgb_bgr[pixels[:, 1], pixels[:, 0], ::-1]

        all_points.append(world_points)
        all_colors.append(rgb.astype(np.uint8))

    if not all_points:
        raise RuntimeError("No valid depth points were generated. Check aligned depth topic, depth filenames, and depth scale.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if args.max_total_points > 0 and len(points) > args.max_total_points:
        indices = np.array(rng.sample(range(len(points)), args.max_total_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]

    points, colors = voxel_downsample(points, colors, args.voxel_size)
    write_ascii_ply(args.pointcloud_output, points, colors)
    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Convert collected RGB+pose data to Nerfstudio transforms.json")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--keep-every", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--depth-dir", default="depth")
    parser.add_argument("--generate-point-cloud", action="store_true")
    parser.add_argument("--pointcloud-output", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=0.001, help="Scale uint depth values to meters. RealSense z16 default is 0.001.")
    parser.add_argument("--pose-translation-scale", type=float, default=1.0, help="Scale camera translations around the first exported pose.")
    parser.add_argument("--depth-min", type=float, default=0.15)
    parser.add_argument("--depth-max", type=float, default=5.0)
    parser.add_argument("--point-stride", type=int, default=6)
    parser.add_argument("--max-points-per-frame", type=int, default=12000)
    parser.add_argument("--max-total-points", type=int, default=1500000)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    poses_csv = dataset_dir / "poses.csv"
    camera_info_json = dataset_dir / "camera_info.json"
    output = args.output or (dataset_dir / "transforms.json")
    pointcloud_output = args.pointcloud_output or (output.parent / "sparse_pc.ply")
    args.pointcloud_output = pointcloud_output

    if not poses_csv.exists():
        raise FileNotFoundError(poses_csv)
    if not camera_info_json.exists():
        raise FileNotFoundError(f"{camera_info_json} is missing. Start the collector after /camera/color/camera_info is publishing.")

    with open(camera_info_json) as f:
        intrinsics = json.load(f)

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": intrinsics["fx"],
        "fl_y": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "w": intrinsics["width"],
        "h": intrinsics["height"],
        "frames": [],
    }
    if args.generate_point_cloud:
        transforms["ply_file_path"] = pointcloud_output.name if pointcloud_output.parent == output.parent else str(pointcloud_output)

    coeffs = intrinsics.get("distortion_coefficients", [])
    if coeffs:
        names = ["k1", "k2", "p1", "p2", "k3", "k4"]
        for name, value in zip(names, coeffs):
            transforms[name] = value

    rows = load_rows(poses_csv, args.start_index, args.keep_every, args.max_frames)
    origin = None
    if rows:
        origin = [float(rows[0]["tx"]), float(rows[0]["ty"]), float(rows[0]["tz"])]

    for row in rows:
        frame = {
            "file_path": f"rgb/{row['filename']}",
            "transform_matrix": make_transform(row, origin, args.pose_translation_scale),
        }
        depth_filename = row.get("depth_filename", "")
        if args.include_depth and depth_filename:
            frame["depth_file_path"] = f"{args.depth_dir}/{depth_filename}"
        transforms["frames"].append(frame)

    point_count = None
    if args.generate_point_cloud:
        point_count = generate_point_cloud(dataset_dir, rows, intrinsics, args)

    with open(output, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Wrote {output} with {len(transforms['frames'])} frames")
    if point_count is not None:
        print(f"Wrote {pointcloud_output} with {point_count} points")


if __name__ == "__main__":
    main()
