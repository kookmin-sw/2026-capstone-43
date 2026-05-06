#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online_gs_slam.data.frame import CameraIntrinsics
from online_gs_slam.mapping.gaussian_map import GaussianMap
from online_gs_slam.rendering.renderer import GsplatGaussianRenderer, GsplatRendererConfig


def load_intrinsics(data_dir: Path, scale: float) -> CameraIntrinsics:
    with open(data_dir / "camera_info.json") as f:
        info = json.load(f)
    return CameraIntrinsics(
        width=max(1, int(info["width"] * scale)),
        height=max(1, int(info["height"] * scale)),
        fx=float(info["fx"]) * scale,
        fy=float(info["fy"]) * scale,
        cx=float(info["cx"]) * scale,
        cy=float(info["cy"]) * scale,
        distortion_model=info.get("distortion_model", ""),
        distortion_coefficients=tuple(float(x) for x in info.get("distortion_coefficients", [])),
    )


def load_initial_pose(output_dir: Path, device: torch.device) -> torch.Tensor:
    trajectory_path = output_dir / "trajectory.json"
    if trajectory_path.exists():
        with open(trajectory_path) as f:
            trajectory = json.load(f)
        if trajectory:
            return torch.tensor(trajectory[-1]["camera_to_world"], dtype=torch.float32, device=device)
    return torch.eye(4, dtype=torch.float32, device=device)


def rotation_x(angle: float, device: torch.device) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=torch.float32,
        device=device,
    )


def rotation_y(angle: float, device: torch.device) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=torch.float32,
        device=device,
    )


def move_local(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    pose = pose.clone()
    pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ delta
    return pose


def rotate_local(pose: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    pose = pose.clone()
    pose[:3, :3] = pose[:3, :3] @ rot
    return pose


def render_image(renderer: GsplatGaussianRenderer, gmap: GaussianMap, pose: torch.Tensor, intrinsics: CameraIntrinsics) -> np.ndarray:
    with torch.no_grad():
        output = renderer.render(gmap, pose, intrinsics)
    rgb = output.rgb.detach().cpu().clamp(0.0, 1.0).numpy()
    return cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def draw_overlay(image: np.ndarray, pose: torch.Tensor, gmap: GaussianMap) -> np.ndarray:
    out = image.copy()
    text = [
        "gsplat viewer",
        "W/S forward/back, A/D left/right, Q/E down/up",
        "Arrows yaw/pitch, R reset, P save, ESC quit",
        f"gaussians: {gmap.num_gaussians}",
        f"pos: {pose[0,3].item():.2f}, {pose[1,3].item():.2f}, {pose[2,3].item():.2f}",
    ]
    for i, line in enumerate(text):
        cv2.putText(out, line, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive gsplat checkpoint viewer")
    parser.add_argument("--output_dir", default="outputs/scene01")
    parser.add_argument("--data_dir", default="/home/harudev/rgb_pose_dataset_01")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--rot_step_deg", type=float, default=3.0)
    parser.add_argument("--once", action="store_true", help="Render one image and exit")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else output_dir / "checkpoints" / "gaussians_latest.pt"
    device = torch.device(args.device)

    gmap = GaussianMap.load_checkpoint(checkpoint, device=args.device)
    intrinsics = load_intrinsics(data_dir, args.scale)
    renderer = GsplatGaussianRenderer(GsplatRendererConfig(downscale=1), device=args.device)
    initial_pose = load_initial_pose(output_dir, device)
    pose = initial_pose.clone()

    if args.once:
        image = render_image(renderer, gmap, pose, intrinsics)
        save_path = Path(args.save_path or (output_dir / "debug" / "viewer_once.png"))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)
        print(f"saved {save_path}")
        return

    cv2.namedWindow("online gsplat viewer", cv2.WINDOW_NORMAL)
    rot_step = math.radians(args.rot_step_deg)
    while True:
        image = render_image(renderer, gmap, pose, intrinsics)
        cv2.imshow("online gsplat viewer", draw_overlay(image, pose, gmap))
        key = cv2.waitKey(0)
        if key in (27, ord("x")):
            break
        if key == ord("r"):
            pose = initial_pose.clone()
        elif key == ord("p"):
            save_path = output_dir / "debug" / "viewer_snapshot.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            print(f"saved {save_path}")
        elif key == ord("w"):
            pose = move_local(pose, torch.tensor([0.0, 0.0, -args.step], dtype=torch.float32, device=device))
        elif key == ord("s"):
            pose = move_local(pose, torch.tensor([0.0, 0.0, args.step], dtype=torch.float32, device=device))
        elif key == ord("a"):
            pose = move_local(pose, torch.tensor([-args.step, 0.0, 0.0], dtype=torch.float32, device=device))
        elif key == ord("d"):
            pose = move_local(pose, torch.tensor([args.step, 0.0, 0.0], dtype=torch.float32, device=device))
        elif key == ord("q"):
            pose = move_local(pose, torch.tensor([0.0, -args.step, 0.0], dtype=torch.float32, device=device))
        elif key == ord("e"):
            pose = move_local(pose, torch.tensor([0.0, args.step, 0.0], dtype=torch.float32, device=device))
        elif key == 81:  # left arrow
            pose = rotate_local(pose, rotation_y(rot_step, device))
        elif key == 83:  # right arrow
            pose = rotate_local(pose, rotation_y(-rot_step, device))
        elif key == 82:  # up arrow
            pose = rotate_local(pose, rotation_x(rot_step, device))
        elif key == 84:  # down arrow
            pose = rotate_local(pose, rotation_x(-rot_step, device))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
