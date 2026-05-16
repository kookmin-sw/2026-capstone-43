from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.audio.windowed_beam import compute_windowed_beam_sequence
from heard_direction_overlay.utils.config import PipelineConfig
from heard_direction_overlay.utils.io import ensure_dir
from heard_direction_overlay.visualization.render_maps import render_planar_direction_map

from output_common import save_gif_from_frames, score_and_render_overlay_with_presence_floor


def generate(
    output_dir: Path,
    rgb: np.ndarray,
    pointcloud: dict[str, np.ndarray],
    gt_annotation: dict[str, Any],
    direction_grid: dict[str, np.ndarray],
    camera_fov: dict[str, np.ndarray],
    foa_stft,
    config: PipelineConfig,
    window_sec: float,
    hop_sec: float,
    gif_fps: float,
    min_presence_score: float,
    max_windows: int | None = None,
) -> dict[str, Any]:
    sequence_root = ensure_dir(output_dir / "15_windowed_beam")
    maps_dir = ensure_dir(sequence_root / "beam_power_maps")
    overlays_dir = ensure_dir(sequence_root / "beam_power_overlays")
    sequence = compute_windowed_beam_sequence(
        foa_stft=foa_stft,
        clip_start_sec=float(config.audio.start_sec),
        direction_grid=direction_grid,
        energy_percentile=config.audio.energy_percentile,
        smoothing_sigma=config.audio.beam_smoothing_sigma,
        window_sec=float(window_sec),
        hop_sec=float(hop_sec),
        max_windows=max_windows,
    )

    map_paths: list[Path] = []
    overlay_paths: list[Path] = []
    windows_summary: list[dict[str, Any]] = []
    for entry in sequence:
        index = int(entry["index"])
        start_sec = float(entry["window_start_sec"])
        end_sec = float(entry["window_end_sec"])
        prefix = f"{index:03d}_t{start_sec:06.2f}_{end_sec:06.2f}"

        map_path = maps_dir / f"{prefix}__beam_power_map.png"
        render_planar_direction_map(
            direction_map=entry["map"],
            azimuth_deg=direction_grid["azimuth_deg"],
            elevation_deg=direction_grid["elevation_deg"],
            output_path=map_path,
            title=f"15. Windowed Beam Power Map ({start_sec:.2f}s ~ {end_sec:.2f}s)",
            peak_direction=entry["peak_direction_camera"],
            gt_direction=gt_annotation["direction_camera"],
            camera_fov_boundary_directions=camera_fov["boundary_directions_camera"],
            camera_forward_direction=camera_fov["center_direction_camera"],
            cmap="viridis",
            note=f"window={window_sec:.2f}s, hop={hop_sec:.2f}s",
        )
        map_paths.append(map_path)

        overlay_path = overlays_dir / f"{prefix}__beam_filtered_overlay.png"
        overlay_stats = score_and_render_overlay_with_presence_floor(
            rgb=rgb,
            pointcloud=pointcloud,
            direction_grid=direction_grid,
            direction_map=entry["map"],
            output_path=overlay_path,
            title=f"15. Windowed Beam Overlay ({start_sec:.2f}s ~ {end_sec:.2f}s)",
            subtitle="Visible geometry consistent with windowed beam-power evidence",
            config=config,
            gt_annotation=gt_annotation,
            min_presence_score=float(min_presence_score),
        )
        overlay_paths.append(overlay_path)

        windows_summary.append(
            {
                "index": index,
                "window_start_sec": start_sec,
                "window_end_sec": end_sec,
                "num_frames": int(entry["num_frames"]),
                "selected_frames": int(entry["selected_frames"]),
                "peak_azimuth_deg": float(entry["peak_azimuth_deg"]),
                "peak_elevation_deg": float(entry["peak_elevation_deg"]),
                "map_path": str(map_path),
                "overlay_path": str(overlay_path),
                "overlay_stats": overlay_stats,
            }
        )

    maps_gif_path = save_gif_from_frames(
        frame_paths=map_paths,
        output_path=sequence_root / "15_beam_power_maps.gif",
        fps=float(gif_fps),
    )
    overlays_gif_path = save_gif_from_frames(
        frame_paths=overlay_paths,
        output_path=sequence_root / "15_beam_filtered_overlays.gif",
        fps=float(gif_fps),
    )
    return {
        "sequence_root": str(sequence_root),
        "beam_maps_dir": str(maps_dir),
        "beam_overlays_dir": str(overlays_dir),
        "num_windows": len(windows_summary),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "gif_fps": float(gif_fps),
        "min_presence_score": float(min_presence_score),
        "maps_gif_path": maps_gif_path,
        "overlays_gif_path": overlays_gif_path,
        "map_paths": [str(path) for path in map_paths],
        "overlay_paths": [str(path) for path in overlay_paths],
        "windows": windows_summary,
    }
