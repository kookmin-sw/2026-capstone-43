#!/usr/bin/env python3

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="View a generated RGB-D point cloud PLY with Open3D.")
    parser.add_argument("ply", type=Path, nargs="?", default=Path.home() / "rgb_pose_dataset_01" / "sparse_pc.ply")
    parser.add_argument("--voxel-size", type=float, default=0.0)
    args = parser.parse_args()

    import open3d as o3d

    ply_path = args.ply.expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    cloud = o3d.io.read_point_cloud(str(ply_path))
    if args.voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(args.voxel_size)

    print(f"Loaded {ply_path}")
    print(f"points={len(cloud.points)}")
    if len(cloud.points) == 0:
        raise RuntimeError("Point cloud is empty.")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([cloud, frame], window_name="RGB-D pose point cloud")


if __name__ == "__main__":
    main()
