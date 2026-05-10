#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from online_gs_slam.data.dataset import LiveRgbPoseDataset
from online_gs_slam.mapping.gaussian_insertion import GaussianInserter, GaussianInsertionConfig
from online_gs_slam.mapping.gaussian_map import GaussianMap, GaussianMapConfig
from online_gs_slam.mapping.keyframe_manager import KeyframeManager, KeyframeManagerConfig
from online_gs_slam.mapping.mapper import MapperConfig, OnlineMapper
from online_gs_slam.mapping.uncertainty import get_high_uncertainty_regions
from online_gs_slam.rendering.renderer import GsplatGaussianRenderer, GsplatRendererConfig, NullGaussianRenderer, SimpleSplatRendererConfig, SimpleTorchGaussianRenderer
from online_gs_slam.tracking.tracker import GaussianTracker, TrackingConfig
from online_gs_slam.utils.config import load_config
from online_gs_slam.utils.io import ensure_dir, save_trajectory
from online_gs_slam.utils.visualization import save_gaussian_ply, save_rgb_debug, save_uncertainty_bar


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online incremental Gaussian SLAM prototype")
    parser.add_argument("--data_dir", required=True, help="Collector output directory, e.g. /home/hd/rgb_pose_dataset_01")
    parser.add_argument("--config", default="configs/online_gs_slam.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--idle_timeout", type=float, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = cfg.get("system", {}).get("device", "cpu")
    save_every = int(cfg.get("system", {}).get("save_every_n_frames", 10))

    dataset = LiveRgbPoseDataset(
        args.data_dir,
        poll_interval=float(cfg.get("data", {}).get("poll_interval", 0.2)),
    )
    gmap = GaussianMap(GaussianMapConfig(device=device, **cfg.get("gaussian_map", {})))
    renderer_cfg = dict(cfg.get("renderer", {}))
    renderer_backend = renderer_cfg.pop("backend", "gsplat")
    if renderer_backend == "gsplat":
        renderer = GsplatGaussianRenderer(GsplatRendererConfig(**renderer_cfg), device=device)
    elif renderer_backend == "simple":
        renderer = SimpleTorchGaussianRenderer(SimpleSplatRendererConfig(**renderer_cfg), device=device)
    elif renderer_backend == "null":
        renderer = NullGaussianRenderer(device=device)
    else:
        raise ValueError(f"Unknown renderer backend: {renderer_backend}")
    inserter = GaussianInserter(GaussianInsertionConfig(**cfg.get("insertion", {})), device=device)
    keyframes = KeyframeManager(KeyframeManagerConfig(**cfg.get("keyframes", {})))
    tracker = GaussianTracker(TrackingConfig(**cfg.get("tracking", {})), renderer, gmap)
    mapper = OnlineMapper(MapperConfig(**cfg.get("mapping", {})), gmap, renderer, inserter, keyframes)

    trajectory: List[Dict] = []
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("data", {}).get("max_frames")
    idle_timeout = args.idle_timeout if args.idle_timeout is not None else cfg.get("data", {}).get("idle_timeout")

    print(f"[online-gs] watching {Path(args.data_dir).expanduser().resolve()}")
    print(f"[online-gs] writing outputs to {output_dir}")

    last_frame = None
    last_pose = None
    for frame in dataset.stream(max_frames=max_frames, idle_timeout=idle_timeout):
        pose = tracker.track(frame)
        stats = mapper.update(frame, pose)
        last_frame = frame
        last_pose = pose
        pose_np = pose.detach().cpu().numpy()
        trajectory.append(
            {
                "index": frame.index,
                "timestamp": frame.timestamp,
                "rgb_path": str(frame.rgb_path),
                "camera_to_world": pose_np.tolist(),
                "stats": stats,
            }
        )
        print(
            f"[online-gs] frame={frame.index:06d} "
            f"gaussians={gmap.num_gaussians} inserted={stats['inserted']} "
            f"visible={stats['visible']} train_frames={stats['train_frames']} "
            f"loss={stats['loss']} keyframe={stats['keyframe']}"
        )

        if (frame.index + 1) % save_every == 0:
            gmap.save_checkpoint(output_dir / "checkpoints" / f"gaussians_{frame.index:06d}.pt")
            save_trajectory(output_dir / "trajectory.json", trajectory)
            centers, values = get_high_uncertainty_regions(gmap, top_k=64)
            np.save(output_dir / "high_uncertainty_centers.npy", centers.detach().cpu().numpy())
            save_uncertainty_bar(output_dir / "uncertainty_debug.png", gmap.uncertainty.detach().cpu().numpy().reshape(-1))
            save_gaussian_ply(output_dir / "gaussians_latest.ply", gmap.means, gmap.colors, gmap.uncertainty)
            target_rgb, rendered_rgb = mapper.render_debug(frame, pose)
            save_rgb_debug(output_dir / "debug" / f"compare_{frame.index:06d}.png", target_rgb, rendered_rgb)

    gmap.save_checkpoint(output_dir / "checkpoints" / "gaussians_latest.pt")
    save_gaussian_ply(output_dir / "gaussians_latest.ply", gmap.means, gmap.colors, gmap.uncertainty)
    if last_frame is not None and last_pose is not None:
        target_rgb, rendered_rgb = mapper.render_debug(last_frame, last_pose)
        save_rgb_debug(output_dir / "debug" / "compare_latest.png", target_rgb, rendered_rgb)
    save_trajectory(output_dir / "trajectory.json", trajectory)
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"frames": len(trajectory), "gaussians": gmap.num_gaussians}, f, indent=2)
    print(f"[online-gs] done frames={len(trajectory)} gaussians={gmap.num_gaussians}")


if __name__ == "__main__":
    main()
