#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from hm3d_l3das23_single_mic.audio_renderer import (
    remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx,
)
from hm3d_l3das23_single_mic.config import load_config
from hm3d_l3das23_single_mic.geometry import local_to_world, yaw_rad_to_quaternion_wxyz
from hm3d_l3das23_single_mic.schemas import MicPose
from hm3d_l3das23_single_mic.scene_loader import HabitatSceneSession
from hm3d_l3das23_single_mic.split_builder import discover_hm3d_scenes


EXPECTED_GAINS = {
    "local_front": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "local_right": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "local_up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
}

LOCAL_VECTORS = {
    "local_front": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "local_right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "local_up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


def _make_fast_direct_config(config_path: Path):
    config = load_config(config_path)
    config.sensor_rig.enable_rgb = False
    config.sensor_rig.enable_depth = False
    config.sensor_rig.enable_semantic = False
    config.simulator.load_semantic_mesh = False
    config.speaker_proxy.enabled = False
    config.audio.enable_materials = False
    config.audio.max_ir_length_s = 0.12
    config.audio.direct = True
    config.audio.indirect = False
    config.audio.diffraction = False
    config.audio.transmission = False
    config.audio.direct_ray_count = 256
    config.audio.indirect_ray_count = 64
    config.audio.source_ray_count = 32
    config.audio.indirect_ray_depth = 1
    config.audio.source_ray_depth = 1
    return config


def _select_clear_mic_floor(session: HabitatSceneSession, *, seed: int) -> np.ndarray:
    _ = seed
    pathfinder = session.sim.pathfinder
    best_point = None
    best_clearance = -1.0
    for _ in range(2000):
        point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
        clearance = float(pathfinder.distance_to_closest_obstacle(point.astype(np.float32), 3.0))
        if np.isfinite(clearance) and clearance > best_clearance:
            best_point = point
            best_clearance = clearance
    if best_point is None:
        raise RuntimeError("Could not find a navigable microphone point for FOA sanity check.")
    return best_point


def _gains_vs_w(rir: np.ndarray) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32)
    w_channel = rir[0].astype(np.float64)
    peak_index = int(np.argmax(np.abs(w_channel)))
    lo = max(0, peak_index - 6)
    hi = min(rir.shape[1], peak_index + 7)
    w_window = rir[0, lo:hi].astype(np.float64)
    denom = float(np.dot(w_window, w_window)) + 1.0e-20
    return np.array(
        [
            float(np.dot(rir[channel, lo:hi].astype(np.float64), w_window) / denom)
            for channel in (1, 2, 3)
        ],
        dtype=np.float64,
    )


def run_check(
    *,
    config_path: Path,
    scene_index: int,
    trials: int,
    distance_m: float,
    tolerance: float,
) -> None:
    config = _make_fast_direct_config(config_path)
    scenes = discover_hm3d_scenes(config)
    if not scenes:
        raise RuntimeError(f"No HM3D scenes found from config: {config_path}")
    scene = scenes[int(scene_index) % len(scenes)]
    yaws_deg = (0.0, 90.0, 180.0, 270.0)

    failures: list[str] = []
    with HabitatSceneSession(config, scene) as session:
        for trial_index in range(int(trials)):
            floor_point = _select_clear_mic_floor(session, seed=20260506 + trial_index)
            mic_position = floor_point + np.array(
                [0.0, float(config.sensor_rig.mic_height_m), 0.0],
                dtype=np.float64,
            )
            for yaw_deg in yaws_deg:
                yaw_rad = math.radians(yaw_deg)
                mic = MicPose(
                    mic_index=trial_index,
                    floor_point_world=floor_point.astype(float).tolist(),
                    position_world=mic_position.astype(float).tolist(),
                    quaternion_wxyz=yaw_rad_to_quaternion_wxyz(yaw_rad),
                    yaw_rad=yaw_rad,
                    yaw_deg=yaw_deg,
                )
                for label, local_unit in LOCAL_VECTORS.items():
                    source = local_to_world(
                        mic.position_world,
                        mic.yaw_rad,
                        local_unit * float(distance_m),
                    )
                    raw_rir = session.render_rir(source.astype(float).tolist(), mic)
                    remapped = remap_rir_world_n3d_to_mic_ambix_sn3d_wyzx(
                        raw_rir,
                        mic.yaw_rad,
                    )
                    gains = _gains_vs_w(remapped)
                    expected = EXPECTED_GAINS[label]
                    error = float(np.max(np.abs(gains - expected)))
                    print(
                        f"trial={trial_index} yaw={yaw_deg:5.1f} {label:12s} "
                        f"gains(Y,Z,X)=({gains[0]:+.4f},{gains[1]:+.4f},{gains[2]:+.4f}) "
                        f"max_err={error:.4f}"
                    )
                    if error > float(tolerance):
                        failures.append(
                            f"trial={trial_index} yaw={yaw_deg} {label} "
                            f"expected={expected.tolist()} got={gains.tolist()}"
                        )

    if failures:
        joined = "\n".join(failures)
        raise SystemExit(f"FOA remap sanity check failed:\n{joined}")
    print("FOA remap sanity check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Habitat/RLR raw FOA remaps to mic-local AmbiX WYZX for multiple yaws."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--distance-m", type=float, default=0.75)
    parser.add_argument("--tolerance", type=float, default=0.08)
    args = parser.parse_args()
    run_check(
        config_path=args.config,
        scene_index=args.scene_index,
        trials=args.trials,
        distance_m=args.distance_m,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
