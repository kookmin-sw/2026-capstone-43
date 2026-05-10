#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


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


def make_transform(row):
    tx = float(row["tx"])
    ty = float(row["ty"])
    tz = float(row["tz"])
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


def main():
    parser = argparse.ArgumentParser(description="Convert collected RGB+pose data to Nerfstudio transforms.json")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--keep-every", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    poses_csv = dataset_dir / "poses.csv"
    camera_info_json = dataset_dir / "camera_info.json"
    output = args.output or (dataset_dir / "transforms.json")

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

    coeffs = intrinsics.get("distortion_coefficients", [])
    if coeffs:
        names = ["k1", "k2", "p1", "p2", "k3", "k4"]
        for name, value in zip(names, coeffs):
            transforms[name] = value

    with open(poses_csv, newline="") as f:
        reader = csv.DictReader(f)
        kept = 0
        for idx, row in enumerate(reader):
            if idx < args.start_index:
                continue
            if args.keep_every > 1 and idx % args.keep_every != 0:
                continue
            transforms["frames"].append(
                {
                    "file_path": f"rgb/{row['filename']}",
                    "transform_matrix": make_transform(row),
                }
            )
            kept += 1
            if args.max_frames is not None and kept >= args.max_frames:
                break

    with open(output, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Wrote {output} with {len(transforms['frames'])} frames")


if __name__ == "__main__":
    main()
