from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.audio.beamforming import compute_beam_power_direction_map
from heard_direction_overlay.audio.foa_stft import compute_foa_stft
from heard_direction_overlay.audio.intensity import compute_intensity_vector_direction_map
from heard_direction_overlay.geometry.depth_to_pointcloud import depth_to_pointcloud
from heard_direction_overlay.geometry.transforms import (
    build_camera_fov_boundary,
    build_direction_grid,
    build_intrinsics,
    resolve_mic_to_camera_rotation,
)
from heard_direction_overlay.utils.io import (
    ensure_dir,
    flatten_paths,
    load_audio_info,
    load_json,
    load_rgb_image,
    resolve_sample_paths,
    save_json,
    summarize_numpy,
)

from common import (
    build_pipeline_config,
    ensure_output_root,
    extract_gt_annotations,
    get_depth_for_sample,
    init_depth_adapter,
    resolve_sample_dirs,
    setup_logging,
)
from output01_rgb import generate as generate_output01_rgb
from output02_depth import generate as generate_output02_depth
from output03_raw_pointcloud import generate as generate_output03_raw_pointcloud
from output04_raw_overlay import generate as generate_output04_raw_overlay
from output05_intensity_map import generate as generate_output05_intensity_map
from output06_beam_map import generate as generate_output06_beam_map
from output12_beam_overlay import generate as generate_output12_beam_overlay
from output14_overview import generate as generate_output14_overview
from output15_windowed_beam import generate as generate_output15_windowed_beam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="02_pipeline: geometry + IV/beam visualization outputs")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/yu/Project_git/01_dataset/hm3d_losnlos_100x100"))
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--sample-id-contains", type=str, default=None)
    parser.add_argument("--selection", type=str, default="all", choices=["all", "los_fov"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spatialvla-root", type=Path, default=Path("/home/yu/Project_git/SpatialVLA"))
    parser.add_argument("--zoe-model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--rotation-json", type=Path, default=None)
    parser.add_argument("--foa-channel-order", type=str, default="WYZX")
    parser.add_argument("--flip-foa-x", action="store_true")
    parser.add_argument("--flip-foa-y", action="store_true")
    parser.add_argument("--flip-foa-z", action="store_true")
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--near-depth", type=float, default=0.05)
    parser.add_argument("--far-depth", type=float, default=20.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--energy-percentile", type=float, default=70.0)
    parser.add_argument("--azimuth-bins", type=int, default=181)
    parser.add_argument("--elevation-bins", type=int, default=91)
    parser.add_argument("--map-gaussian-sigma-deg", type=float, default=14.0)
    parser.add_argument("--beam-smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--raw-point-size", type=float, default=1.6)
    parser.add_argument("--raw-alpha", type=float, default=0.30)
    parser.add_argument("--filtered-point-size", type=float, default=4.8)
    parser.add_argument("--overlay-score-percentile", type=float, default=82.0)
    parser.add_argument("--overlay-gamma", type=float, default=0.85)
    parser.add_argument("--overlay-min-alpha", type=float, default=0.06)
    parser.add_argument("--overlay-max-alpha", type=float, default=0.95)
    parser.add_argument("--enable-output15", action="store_true")
    parser.add_argument("--output15-window-sec", type=float, default=0.50)
    parser.add_argument("--output15-hop-sec", type=float, default=0.25)
    parser.add_argument("--output15-max-windows", type=int, default=None)
    parser.add_argument("--output15-gif-fps", type=float, default=8.0)
    parser.add_argument("--output15-min-presence-score", type=float, default=0.10)
    parser.add_argument("--overview-columns", type=int, default=4)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def process_sample(sample, config, depth_adapter, depth_override: Path | None, rotation_json: Path | None, args) -> dict:
    assert config.runtime is not None
    start_time = time.time()
    output_dir = ensure_dir(config.runtime.output_root / sample.sample_id)

    metadata = load_json(sample.metadata_path)
    rgb = load_rgb_image(sample.rgb_path)
    height, width = rgb.shape[:2]

    depth, depth_source = get_depth_for_sample(sample, rgb, depth_override, depth_adapter)
    intrinsics = build_intrinsics(width, height, config.camera)
    gt_annotation = extract_gt_annotations(metadata, intrinsics, width, height)

    output01 = generate_output01_rgb(output_dir, rgb, gt_annotation)
    output02 = generate_output02_depth(output_dir, depth, rgb, gt_annotation)
    pointcloud = depth_to_pointcloud(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        point_stride=config.point_filter.point_stride,
        min_depth=config.camera.near_depth,
        max_depth=config.camera.far_depth,
    )
    output03 = generate_output03_raw_pointcloud(output_dir, pointcloud, gt_annotation)
    output04 = generate_output04_raw_overlay(output_dir, rgb, pointcloud, config, gt_annotation)

    mic_to_camera_rotation = resolve_mic_to_camera_rotation(config.foa, rotation_json)
    camera_fov = build_camera_fov_boundary(width, height, intrinsics)
    direction_grid = build_direction_grid(
        azimuth_bins=config.audio.azimuth_bins,
        elevation_bins=config.audio.elevation_bins,
        mic_to_camera_rotation=mic_to_camera_rotation,
    )
    foa_stft = compute_foa_stft(sample.audio_path, config.audio, config.foa)
    iv_result = compute_intensity_vector_direction_map(
        foa_stft=foa_stft,
        direction_grid=direction_grid,
        mic_to_camera_rotation=mic_to_camera_rotation,
        energy_percentile=config.audio.energy_percentile,
        sigma_deg=config.audio.map_gaussian_sigma_deg,
    )
    beam_result = compute_beam_power_direction_map(
        foa_stft=foa_stft,
        direction_grid=direction_grid,
        energy_percentile=config.audio.energy_percentile,
        smoothing_sigma=config.audio.beam_smoothing_sigma,
    )
    output05 = generate_output05_intensity_map(output_dir, direction_grid, iv_result, gt_annotation, camera_fov)
    output06 = generate_output06_beam_map(output_dir, direction_grid, beam_result, gt_annotation, camera_fov)
    output12 = generate_output12_beam_overlay(
        output_dir, rgb, pointcloud, direction_grid, beam_result, gt_annotation, config
    )
    output15_summary: dict | None = None
    if bool(args.enable_output15):
        output15_summary = generate_output15_windowed_beam(
            output_dir=output_dir,
            rgb=rgb,
            pointcloud=pointcloud,
            gt_annotation=gt_annotation,
            direction_grid=direction_grid,
            camera_fov=camera_fov,
            foa_stft=foa_stft,
            config=config,
            window_sec=float(args.output15_window_sec),
            hop_sec=float(args.output15_hop_sec),
            gif_fps=float(args.output15_gif_fps),
            min_presence_score=float(args.output15_min_presence_score),
            max_windows=args.output15_max_windows,
        )
    overview_inputs = generate_output14_overview(
        output_dir,
        sample.sample_dir,
        columns=config.runtime.overview_columns,
    )

    output_file_paths = [
        Path(output01),
        Path(output02["output_path"]),
        Path(output03["image_path"]),
        Path(output03["ply_path"]),
        Path(output03["npz_path"]),
        Path(output04),
        Path(output05),
        Path(output06),
        output_dir / "12_beam_filtered_overlay.png",
        output_dir / "14_overview.png",
    ]
    if output15_summary is not None:
        if output15_summary.get("maps_gif_path"):
            output_file_paths.append(Path(output15_summary["maps_gif_path"]))
        if output15_summary.get("overlays_gif_path"):
            output_file_paths.append(Path(output15_summary["overlays_gif_path"]))

    summary = {
        "sample_id": sample.sample_id,
        "sample_dir": str(sample.sample_dir),
        "metadata_path": str(sample.metadata_path),
        "rgb_path": str(sample.rgb_path),
        "audio_path": str(sample.audio_path),
        "audio_info": load_audio_info(sample.audio_path),
        "depth_source": depth_source,
        "depth_stats": output02["stats"],
        "intrinsics": intrinsics.tolist(),
        "mic_to_camera_rotation": mic_to_camera_rotation.tolist(),
        "metadata": {
            "geometry_los": metadata.get("geometry_los"),
            "in_fov": metadata.get("in_fov"),
            "azimuth_deg": metadata.get("azimuth_deg"),
            "elevation_deg": metadata.get("elevation_deg"),
            "source_distance": metadata.get("source_distance"),
            "projected_pixel_xy": metadata.get("projected_pixel_xy"),
        },
        "gt_annotation": gt_annotation,
        "pointcloud": {
            "num_points": int(len(pointcloud["points"])),
            "points_stats": summarize_numpy(pointcloud["points"]),
            "depth_stats": summarize_numpy(depth),
        },
        "directional_peaks": {
            "intensity": {
                "azimuth_deg": float(iv_result["peak_azimuth_deg"]),
                "elevation_deg": float(iv_result["peak_elevation_deg"]),
            },
            "beam": {
                "azimuth_deg": float(beam_result["peak_azimuth_deg"]),
                "elevation_deg": float(beam_result["peak_elevation_deg"]),
            },
        },
        "outputs": {
            "01_rgb": str(output01),
            "02_depth": output02["output_path"],
            "03_raw_pointcloud_image": output03["image_path"],
            "03_raw_pointcloud_ply": output03["ply_path"],
            "03_raw_pointcloud_npz": output03["npz_path"],
            "04_depth_colored_overlay": str(output04),
            "05_intensity_vector_direction_map": str(output05),
            "06_beam_power_direction_map": str(output06),
            "12_beam_filtered_overlay": str(output_dir / "12_beam_filtered_overlay.png"),
            "14_overview": str(output_dir / "14_overview.png"),
            "15_windowed_beam": None if output15_summary is None else output15_summary["sequence_root"],
        },
        "overlay_stats": {
            "12_beam_filtered_overlay": output12,
        },
        "output15": output15_summary,
        "runtime_sec": float(time.time() - start_time),
        "config": config.to_json_dict(),
        "output_files": flatten_paths(output_file_paths),
        "overview_inputs": flatten_paths(overview_inputs),
        "output15_config": {
            "enabled": bool(args.enable_output15),
            "window_sec": float(args.output15_window_sec),
            "hop_sec": float(args.output15_hop_sec),
            "max_windows": None if args.output15_max_windows is None else int(args.output15_max_windows),
            "gif_fps": float(args.output15_gif_fps),
            "min_presence_score": float(args.output15_min_presence_score),
        },
    }
    save_json(output_dir / "sample_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = build_pipeline_config(args)
    assert config.runtime is not None
    ensure_output_root(config)
    sample_dirs = resolve_sample_dirs(args)
    if not sample_dirs:
        raise RuntimeError("No samples found to process.")

    depth_override = args.depth_path.resolve() if args.depth_path is not None else None
    depth_adapter = init_depth_adapter(config, use_depth_override=depth_override is not None)

    run_summaries = []
    for sample_dir in sample_dirs:
        sample = resolve_sample_paths(sample_dir)
        single_depth_override = depth_override if len(sample_dirs) == 1 else None
        run_summaries.append(
            process_sample(sample, config, depth_adapter, single_depth_override, args.rotation_json, args)
        )

    batch_summary = {
        "num_samples": len(run_summaries),
        "sample_ids": [item["sample_id"] for item in run_summaries],
        "samples": run_summaries,
    }
    save_json(config.runtime.output_root / "batch_summary.json", batch_summary)


if __name__ == "__main__":
    main()
