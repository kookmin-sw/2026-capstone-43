#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

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


def parse_camera_yaml(path):
    text = path.read_text()
    width = int(re.search(r"image_width:\s*(\d+)", text).group(1))
    height = int(re.search(r"image_height:\s*(\d+)", text).group(1))
    matrix_match = re.search(r"camera_matrix:\s*.*?data:\s*\[(.*?)\]", text, re.S)
    if matrix_match is None:
        raise RuntimeError(f"Could not parse camera_matrix from {path}")
    values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", matrix_match.group(1))]
    if len(values) != 9:
        raise RuntimeError(f"Expected 9 camera_matrix values in {path}, got {len(values)}")
    return {
        "w": width,
        "h": height,
        "fl_x": values[0],
        "fl_y": values[4],
        "cx": values[2],
        "cy": values[5],
    }


def load_poses(path):
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        stamp, x, y, z, qx, qy, qz, qw, node_id = parts[:9]
        poses.append(
            {
                "stamp": stamp,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
                "id": int(node_id),
            }
        )
    return poses


def nerfstudio_transform_from_ros_optical_pose(pose):
    world_from_ros_optical = np.eye(4, dtype=np.float64)
    world_from_ros_optical[:3, :3] = quat_to_rot(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    world_from_ros_optical[:3, 3] = [pose["x"], pose["y"], pose["z"]]

    # ROS/OpenCV optical camera: +X right, +Y down, +Z forward.
    # Nerfstudio/OpenGL camera: +X right, +Y up, +Z backward.
    ros_optical_from_nerfstudio_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    return world_from_ros_optical @ ros_optical_from_nerfstudio_camera


def run_rtabmap_export(db_path, raw_dir, prefix, ascii_cloud):
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "rtabmap-export",
        "--images",
        "--poses_camera",
        "--poses_format",
        "11",
        "--cloud",
        "--output",
        prefix,
        "--output_dir",
        str(raw_dir),
    ]
    if ascii_cloud:
        command.append("--ascii")
    command.append(str(db_path))
    print("[rtabmap-export]", " ".join(command))
    subprocess.run(command, check=True)


def link_or_copy(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Export an RTAB-Map database as a Nerfstudio/splatfacto dataset."
    )
    parser.add_argument("db", type=Path, nargs="?", default=Path.home() / ".ros" / "rtabmap.db")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "rtabmap_nerfstudio_dataset")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="rtabmap_export")
    parser.add_argument("--skip-rtdb-export", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--copy-cloud", action="store_true")
    parser.add_argument("--ascii-cloud", action="store_true")
    parser.add_argument("--keep-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = (args.raw_dir or output_dir / "_rtabmap_export").expanduser().resolve()

    if not args.skip_rtdb_export:
        run_rtabmap_export(db_path, raw_dir, args.prefix, args.ascii_cloud)

    rgb_dirs = sorted(raw_dir.glob("*_rgb"))
    calib_dirs = sorted(raw_dir.glob("*_calib"))
    pose_files = sorted(raw_dir.glob("*_camera_poses.txt"))
    cloud_files = sorted(raw_dir.glob("*_cloud.ply"))
    if not rgb_dirs or not calib_dirs or not pose_files:
        raise RuntimeError(f"Missing RTAB-Map export outputs under {raw_dir}")

    rgb_dir = rgb_dirs[0]
    calib_dir = calib_dirs[0]
    pose_file = pose_files[0]
    cloud_file = cloud_files[0] if cloud_files else None

    poses = load_poses(pose_file)
    frames = []
    intrinsics = None
    selected_count = 0
    for idx, pose in enumerate(poses):
        if args.keep_every > 1 and idx % args.keep_every != 0:
            continue
        if args.max_frames > 0 and selected_count >= args.max_frames:
            break

        stamp = pose["stamp"]
        image_candidates = sorted(rgb_dir.glob(f"{stamp}.*"))
        calib_path = calib_dir / f"{stamp}.yaml"
        if not image_candidates or not calib_path.exists():
            continue
        image_path = image_candidates[0]
        if intrinsics is None:
            intrinsics = parse_camera_yaml(calib_path)

        dst_name = f"{selected_count:06d}{image_path.suffix.lower()}"
        dst_image = output_dir / "images" / dst_name
        link_or_copy(image_path, dst_image, args.copy_images)

        transform = nerfstudio_transform_from_ros_optical_pose(pose)
        frames.append(
            {
                "file_path": f"images/{dst_name}",
                "transform_matrix": transform.tolist(),
                "timestamp": float(stamp),
                "rtabmap_id": pose["id"],
            }
        )
        selected_count += 1

    if intrinsics is None or not frames:
        raise RuntimeError("No usable image/pose pairs were exported.")

    sparse_pc_path = None
    if cloud_file is not None:
        sparse_pc_path = output_dir / "sparse_pc.ply"
        link_or_copy(cloud_file, sparse_pc_path, args.copy_cloud)

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": intrinsics["fl_x"],
        "fl_y": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "w": intrinsics["w"],
        "h": intrinsics["h"],
        "frames": frames,
    }
    if sparse_pc_path is not None:
        transforms["ply_file_path"] = "sparse_pc.ply"

    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_path = output_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2))

    print(f"Wrote {transforms_path}")
    print(f"Wrote {len(frames)} image/pose pairs to {output_dir / 'images'}")
    if sparse_pc_path is not None:
        print(f"Wrote initial point cloud link/copy to {sparse_pc_path}")


if __name__ == "__main__":
    main()
