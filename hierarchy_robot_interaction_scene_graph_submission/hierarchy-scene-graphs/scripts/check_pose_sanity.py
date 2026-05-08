import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rgbd_data import ReplicaStyleRGBDDataset, resolve_replica_scene_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check posed RGB-D trajectory quality for hierarchy-scene-graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scene_path",
        help="Direct replica-style scene path containing results/ and traj.txt, or a dataset root.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Optional scene id used when scene_path is a dataset root rather than the scene folder itself.",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=40,
        help="How many neighboring frame pairs to evaluate.",
    )
    parser.add_argument(
        "--pair-step",
        type=int,
        default=20,
        help="Stride when choosing neighboring pairs to evaluate.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=1,
        help="Gap between source and target frame when forming a pair.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="Downsample size for per-frame local point clouds before alignment scoring.",
    )
    parser.add_argument(
        "--filter-distance",
        type=float,
        default=4.0,
        help="Max mean depth allowed when creating a frame point cloud. Use inf to disable.",
    )
    parser.add_argument(
        "--inlier-threshold",
        type=float,
        default=0.05,
        help="Distance threshold in meters used to report inlier overlap.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory where worst frame-pair overlays will be exported as PLY files.",
    )
    parser.add_argument(
        "--export-top-k",
        type=int,
        default=0,
        help="How many worst pairs to export. Set to 0 to disable export.",
    )
    parser.add_argument(
        "--export-mode",
        choices=["raw", "inverse", "both"],
        default="raw",
        help="Which pose interpretation(s) to export for the selected worst pairs.",
    )
    return parser.parse_args()


def resolve_scene_path(scene_path_arg: str, scene_id: str | None) -> Path:
    scene_path = Path(scene_path_arg)
    if (scene_path / "results").is_dir() and (scene_path / "traj.txt").is_file():
        return scene_path
    if scene_id is None:
        raise FileNotFoundError(
            "scene_path does not point to a replica-style scene directory. "
            "Pass --scene-id as well when giving a dataset root."
        )
    return resolve_replica_scene_path(str(scene_path), str(scene_id))


def load_poses(scene_path: Path) -> np.ndarray:
    poses = np.loadtxt(scene_path / "traj.txt", dtype=np.float64)
    return poses.reshape(-1, 4, 4)


def summarize_vectors(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    return (
        f"min={values.min():.6f} "
        f"mean={values.mean():.6f} "
        f"median={np.median(values):.6f} "
        f"max={values.max():.6f}"
    )


def relative_rotation_angles_deg(rotations: np.ndarray) -> np.ndarray:
    angles = []
    for idx in range(len(rotations) - 1):
        rel = rotations[idx].T @ rotations[idx + 1]
        cosine = float((np.trace(rel) - 1.0) / 2.0)
        cosine = np.clip(cosine, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    return np.asarray(angles, dtype=np.float64)


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1,
    )
    return (pose @ homogeneous.T).T[:, :3]


def symmetric_alignment_metrics(
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold: float,
) -> dict:
    tree_b = cKDTree(points_b)
    dist_ab, _ = tree_b.query(points_a, k=1, workers=-1)
    tree_a = cKDTree(points_a)
    dist_ba, _ = tree_a.query(points_b, k=1, workers=-1)

    dists = np.concatenate([dist_ab, dist_ba], axis=0)
    dists = dists[np.isfinite(dists)]
    if dists.size == 0:
        return {
            "mean_nn": np.inf,
            "median_nn": np.inf,
            "p90_nn": np.inf,
            "inlier_ratio": 0.0,
        }

    return {
        "mean_nn": float(dists.mean()),
        "median_nn": float(np.median(dists)),
        "p90_nn": float(np.percentile(dists, 90)),
        "inlier_ratio": float((dists <= threshold).mean()),
    }


def format_metric_summary(name: str, metrics: list[dict]) -> str:
    if not metrics:
        return f"{name}: no valid frame pairs"
    mean_nn = np.array([m["mean_nn"] for m in metrics], dtype=np.float64)
    median_nn = np.array([m["median_nn"] for m in metrics], dtype=np.float64)
    p90_nn = np.array([m["p90_nn"] for m in metrics], dtype=np.float64)
    inlier_ratio = np.array([m["inlier_ratio"] for m in metrics], dtype=np.float64)
    return (
        f"{name}: "
        f"mean_nn[{summarize_vectors(mean_nn)}] "
        f"median_nn[{summarize_vectors(median_nn)}] "
        f"p90_nn[{summarize_vectors(p90_nn)}] "
        f"inlier_ratio[{summarize_vectors(inlier_ratio)}]"
    )


def print_worst_pairs(name: str, metrics: list[dict], limit: int = 5):
    if not metrics:
        return
    ranked = sorted(metrics, key=lambda item: item["mean_nn"], reverse=True)[:limit]
    print(f"{name}_worst_pairs:")
    for item in ranked:
        frame_i, frame_j = item["pair"]
        print(
            f"  pair=({frame_i}, {frame_j}) "
            f"mean_nn={item['mean_nn']:.6f} "
            f"median_nn={item['median_nn']:.6f} "
            f"p90_nn={item['p90_nn']:.6f} "
            f"inlier_ratio={item['inlier_ratio']:.6f}"
        )


def get_local_points(
    dataset: ReplicaStyleRGBDDataset,
    frame_idx: int,
    voxel_size: float,
    filter_distance: float,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if frame_idx in cache:
        return cache[frame_idx]

    rgb_image, depth_image, _ = dataset[frame_idx]
    pcd = dataset.create_pcd(
        rgb_image,
        depth_image,
        camera_pose=None,
        filter_distance=filter_distance,
    )
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cache[frame_idx] = np.asarray(pcd.points).copy()
    return cache[frame_idx]


def get_local_pcd(
    dataset: ReplicaStyleRGBDDataset,
    frame_idx: int,
    voxel_size: float,
    filter_distance: float,
) -> o3d.geometry.PointCloud:
    rgb_image, depth_image, _ = dataset[frame_idx]
    pcd = dataset.create_pcd(
        rgb_image,
        depth_image,
        camera_pose=None,
        filter_distance=filter_distance,
    )
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def paint_and_transform(pcd: o3d.geometry.PointCloud, pose: np.ndarray, color: list[float]):
    transformed = copy.deepcopy(pcd)
    transformed.transform(pose)
    transformed.paint_uniform_color(color)
    return transformed


def export_worst_pairs(
    dataset: ReplicaStyleRGBDDataset,
    poses: np.ndarray,
    metrics: list[dict],
    export_dir: Path,
    export_top_k: int,
    export_mode: str,
    voxel_size: float,
    filter_distance: float,
):
    export_dir.mkdir(parents=True, exist_ok=True)
    selected_metrics = sorted(metrics, key=lambda item: item["mean_nn"], reverse=True)[:export_top_k]
    pose_modes = ["raw", "inverse"] if export_mode == "both" else [export_mode]
    for rank, metric in enumerate(selected_metrics, start=1):
        frame_i, frame_j = metric["pair"]
        local_i = get_local_pcd(dataset, frame_i, voxel_size, filter_distance)
        local_j = get_local_pcd(dataset, frame_j, voxel_size, filter_distance)
        pair_dir = export_dir / f"rank_{rank:02d}_pair_{frame_i:04d}_{frame_j:04d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        summary_path = pair_dir / "metrics.txt"
        with open(summary_path, "w") as outfile:
            outfile.write(
                "\n".join(
                    [
                        f"pair=({frame_i}, {frame_j})",
                        f"mean_nn={metric['mean_nn']:.6f}",
                        f"median_nn={metric['median_nn']:.6f}",
                        f"p90_nn={metric['p90_nn']:.6f}",
                        f"inlier_ratio={metric['inlier_ratio']:.6f}",
                    ]
                )
            )
        for mode in pose_modes:
            pose_i = poses[frame_i]
            pose_j = poses[frame_j]
            if mode == "inverse":
                pose_i = np.linalg.inv(pose_i)
                pose_j = np.linalg.inv(pose_j)
            world_i = paint_and_transform(local_i, pose_i, [1.0, 0.2, 0.2])
            world_j = paint_and_transform(local_j, pose_j, [0.2, 0.9, 1.0])
            overlay = world_i + world_j
            o3d.io.write_point_cloud(str(pair_dir / f"{mode}_frame_{frame_i:04d}.ply"), world_i)
            o3d.io.write_point_cloud(str(pair_dir / f"{mode}_frame_{frame_j:04d}.ply"), world_j)
            o3d.io.write_point_cloud(str(pair_dir / f"{mode}_overlay.ply"), overlay)


def evaluate_pose_mode(
    dataset: ReplicaStyleRGBDDataset,
    poses: np.ndarray,
    pairs: list[tuple[int, int]],
    voxel_size: float,
    filter_distance: float,
    inlier_threshold: float,
    mode: str,
) -> list[dict]:
    cache: dict[int, np.ndarray] = {}
    metrics = []
    for frame_i, frame_j in pairs:
        points_i = get_local_points(dataset, frame_i, voxel_size, filter_distance, cache)
        points_j = get_local_points(dataset, frame_j, voxel_size, filter_distance, cache)
        if points_i.shape[0] == 0 or points_j.shape[0] == 0:
            continue

        pose_i = poses[frame_i]
        pose_j = poses[frame_j]
        if mode == "inverse":
            pose_i = np.linalg.inv(pose_i)
            pose_j = np.linalg.inv(pose_j)
        elif mode != "raw":
            raise ValueError(f"Unsupported pose mode: {mode}")

        world_i = transform_points(points_i, pose_i)
        world_j = transform_points(points_j, pose_j)
        metric = symmetric_alignment_metrics(world_i, world_j, inlier_threshold)
        metric["pair"] = (frame_i, frame_j)
        metrics.append(metric)
    return metrics


def build_pairs(num_frames: int, pair_step: int, frame_gap: int, max_pairs: int) -> list[tuple[int, int]]:
    pairs = []
    upper = max(0, num_frames - frame_gap)
    for start_idx in range(0, upper, pair_step):
        end_idx = start_idx + frame_gap
        if end_idx >= num_frames:
            break
        pairs.append((start_idx, end_idx))
        if len(pairs) >= max_pairs:
            break
    return pairs


def print_pose_summary(poses: np.ndarray):
    rotations = poses[:, :3, :3]
    translations = poses[:, :3, 3]
    step_translation = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    step_rotation = relative_rotation_angles_deg(rotations)
    ortho_error = np.linalg.norm(
        np.matmul(rotations.transpose(0, 2, 1), rotations) - np.eye(3),
        axis=(1, 2),
    )
    determinants = np.linalg.det(rotations)

    print(f"pose_count: {poses.shape[0]}")
    print(f"translation_extent_min: {translations.min(axis=0)}")
    print(f"translation_extent_max: {translations.max(axis=0)}")
    print(f"camera_height_y: {summarize_vectors(translations[:, 1])}")
    print(f"step_translation_m: {summarize_vectors(step_translation)}")
    print(f"step_rotation_deg: {summarize_vectors(step_rotation)}")
    print(f"rotation_orthogonality_error: {summarize_vectors(ortho_error)}")
    print(f"rotation_determinant: {summarize_vectors(determinants)}")


def main():
    args = parse_args()
    scene_path = resolve_scene_path(args.scene_path, args.scene_id)
    dataset = ReplicaStyleRGBDDataset(str(scene_path))
    poses = load_poses(scene_path)

    print(f"scene_path: {scene_path}")
    print(f"rgb_depth_pairs: {len(dataset)}")
    print_pose_summary(poses)

    if len(dataset) != poses.shape[0]:
        print(
            "warning: number of RGB-D frames and poses differ "
            f"({len(dataset)} vs {poses.shape[0]})"
        )

    pairs = build_pairs(
        min(len(dataset), poses.shape[0]),
        pair_step=max(1, args.pair_step),
        frame_gap=max(1, args.frame_gap),
        max_pairs=max(1, args.pairs),
    )
    print(f"evaluated_pairs: {pairs[:5]}{' ...' if len(pairs) > 5 else ''}")
    print(f"pair_count: {len(pairs)}")

    raw_metrics = evaluate_pose_mode(
        dataset,
        poses,
        pairs,
        voxel_size=args.voxel_size,
        filter_distance=args.filter_distance,
        inlier_threshold=args.inlier_threshold,
        mode="raw",
    )
    inverse_metrics = evaluate_pose_mode(
        dataset,
        poses,
        pairs,
        voxel_size=args.voxel_size,
        filter_distance=args.filter_distance,
        inlier_threshold=args.inlier_threshold,
        mode="inverse",
    )

    print(format_metric_summary("raw_pose_alignment", raw_metrics))
    print(format_metric_summary("inverse_pose_alignment", inverse_metrics))
    print_worst_pairs("raw_pose_alignment", raw_metrics)
    print_worst_pairs("inverse_pose_alignment", inverse_metrics)

    if raw_metrics and inverse_metrics:
        raw_mean = float(np.mean([m["mean_nn"] for m in raw_metrics]))
        inv_mean = float(np.mean([m["mean_nn"] for m in inverse_metrics]))
        raw_inlier = float(np.mean([m["inlier_ratio"] for m in raw_metrics]))
        inv_inlier = float(np.mean([m["inlier_ratio"] for m in inverse_metrics]))
        print(
            "interpretation_hint: "
            f"raw mean_nn={raw_mean:.6f}, inverse mean_nn={inv_mean:.6f}, "
            f"raw inlier={raw_inlier:.6f}, inverse inlier={inv_inlier:.6f}"
        )
        if raw_mean < inv_mean and raw_inlier > inv_inlier:
            print("verdict: raw traj.txt poses are more self-consistent than inverse poses")
        elif inv_mean < raw_mean and inv_inlier > raw_inlier:
            print("verdict: inverse(traj.txt) looks more self-consistent than raw poses")
        else:
            print("verdict: raw and inverse are both plausible or both weak; inspect coordinate conventions next")

    if args.export_top_k > 0 and args.export_dir is not None:
        export_worst_pairs(
            dataset,
            poses,
            raw_metrics,
            Path(args.export_dir),
            export_top_k=max(1, args.export_top_k),
            export_mode=args.export_mode,
            voxel_size=args.voxel_size,
            filter_distance=args.filter_distance,
        )
        print(f"exported worst pairs to {args.export_dir}")


if __name__ == "__main__":
    main()
