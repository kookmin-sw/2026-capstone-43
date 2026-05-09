from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heard_direction_overlay.adapters.spatialvla_zoedepth import (
    SpatialVLAZoeDepthAdapter,
    resolve_default_zoe_model_path,
)
from heard_direction_overlay.geometry.transforms import normalize_vector
from heard_direction_overlay.utils.config import (
    AudioConfig,
    CameraConfig,
    FOAConventionConfig,
    PipelineConfig,
    PointFilterConfig,
    RuntimeConfig,
)
from heard_direction_overlay.utils.io import (
    SamplePaths,
    discover_sample_dirs_from_manifest,
    ensure_dir,
    load_depth_array,
    load_json,
)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    runtime = RuntimeConfig(
        spatialvla_root=args.spatialvla_root.resolve(),
        output_root=args.output_root.resolve(),
        zoe_model_path=args.zoe_model_path or resolve_default_zoe_model_path(args.spatialvla_root.resolve()),
        device=args.device,
        overview_columns=max(int(args.overview_columns), 1),
        log_level=args.log_level.upper(),
    )
    return PipelineConfig(
        camera=CameraConfig(
            hfov_deg=args.hfov_deg,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            near_depth=args.near_depth,
            far_depth=args.far_depth,
        ),
        foa=FOAConventionConfig(
            channel_order=args.foa_channel_order.upper(),
            flip_x=bool(args.flip_foa_x),
            flip_y=bool(args.flip_foa_y),
            flip_z=bool(args.flip_foa_z),
        ),
        audio=AudioConfig(
            target_sample_rate=args.target_sr,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            energy_percentile=args.energy_percentile,
            azimuth_bins=args.azimuth_bins,
            elevation_bins=args.elevation_bins,
            map_gaussian_sigma_deg=args.map_gaussian_sigma_deg,
            beam_smoothing_sigma=args.beam_smoothing_sigma,
        ),
        point_filter=PointFilterConfig(
            point_stride=args.point_stride,
            raw_point_size=args.raw_point_size,
            raw_alpha=args.raw_alpha,
            overlay_score_percentile=args.overlay_score_percentile,
            overlay_gamma=args.overlay_gamma,
            overlay_min_alpha=args.overlay_min_alpha,
            overlay_max_alpha=args.overlay_max_alpha,
            filtered_point_size=args.filtered_point_size,
        ),
        runtime=runtime,
    )


def _is_los_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if "nlos" in text:
            return False
        if "los" in text:
            return True
    return False


def resolve_sample_dirs(args: argparse.Namespace) -> list[Path]:
    if args.sample_dir is not None:
        return [args.sample_dir.resolve()]
    if args.manifest is None:
        raise ValueError("Provide either --sample-dir or --manifest.")
    discovered = discover_sample_dirs_from_manifest(
        dataset_root=args.dataset_root.resolve(),
        manifest_name=args.manifest,
        limit=None,
        sample_id_contains=args.sample_id_contains,
    )
    if args.selection == "all":
        if args.limit is not None:
            return discovered[: int(args.limit)]
        return discovered

    selected: list[Path] = []
    for sample_dir in discovered:
        metadata_path = sample_dir / "metadata" / "sample.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        if args.selection == "los_fov":
            if not bool(metadata.get("in_fov", False)):
                continue
            if not _is_los_flag(metadata.get("geometry_los")):
                continue
        selected.append(sample_dir)
        if args.limit is not None and len(selected) >= int(args.limit):
            break
    return selected


def extract_gt_annotations(
    metadata: dict[str, Any],
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    gt_camera_point = None
    local_xyz = metadata.get("source_pose_local_xyz")
    if isinstance(local_xyz, list) and len(local_xyz) == 3:
        x_right = float(local_xyz[0])
        z_forward = float(local_xyz[1])
        z_up = float(local_xyz[2])
        gt_camera_point = np.asarray([x_right, -z_up, z_forward], dtype=np.float32)

    gt_direction_camera = None
    if gt_camera_point is not None and float(np.linalg.norm(gt_camera_point)) > 1.0e-8:
        gt_direction_camera = normalize_vector(gt_camera_point)

    gt_pixel_xy = None
    gt_in_view = False
    out_of_view_reason = None
    projected_xy = metadata.get("projected_pixel_xy")
    if isinstance(projected_xy, list) and len(projected_xy) == 2:
        gt_pixel_xy = (float(projected_xy[0]), float(projected_xy[1]))
    elif gt_camera_point is not None and float(gt_camera_point[2]) > 1.0e-8:
        fx, fy, cx, cy = (
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        u = fx * float(gt_camera_point[0]) / float(gt_camera_point[2]) + cx
        v = fy * float(gt_camera_point[1]) / float(gt_camera_point[2]) + cy
        gt_pixel_xy = (u, v)

    if gt_pixel_xy is not None and gt_camera_point is not None and float(gt_camera_point[2]) > 1.0e-8:
        gt_in_view = (
            0.0 <= gt_pixel_xy[0] <= float(image_width - 1)
            and 0.0 <= gt_pixel_xy[1] <= float(image_height - 1)
        )
    else:
        gt_in_view = bool(metadata.get("in_fov", False))

    if not gt_in_view:
        out_of_view_reason = str(metadata.get("projection_reason") or "out of view")

    return {
        "camera_point": None if gt_camera_point is None else gt_camera_point.astype(np.float32),
        "direction_camera": None if gt_direction_camera is None else gt_direction_camera.astype(np.float32),
        "pixel_xy": gt_pixel_xy,
        "in_view": bool(gt_in_view),
        "out_of_view_reason": out_of_view_reason,
    }


def get_depth_for_sample(
    sample: SamplePaths,
    rgb: np.ndarray,
    depth_override: Path | None,
    depth_adapter: SpatialVLAZoeDepthAdapter | None,
) -> tuple[np.ndarray, str]:
    if depth_override is not None:
        return load_depth_array(depth_override), f"override:{depth_override}"
    if sample.depth_path is not None:
        return load_depth_array(sample.depth_path), f"sample:{sample.depth_path}"
    if depth_adapter is None:
        raise RuntimeError(
            "Depth is missing and ZoeDepth adapter is unavailable. "
            "Provide --spatialvla-root and a valid --zoe-model-path, or pass --depth-path."
        )
    return depth_adapter.predict_depth(rgb), f"zoedepth:{depth_adapter.model_path}"


def init_depth_adapter(config: PipelineConfig, use_depth_override: bool) -> SpatialVLAZoeDepthAdapter | None:
    assert config.runtime is not None
    if use_depth_override:
        return None
    return SpatialVLAZoeDepthAdapter(
        spatialvla_root=config.runtime.spatialvla_root,
        model_path=config.runtime.zoe_model_path,
        device=config.runtime.device,
    )


def ensure_output_root(config: PipelineConfig) -> Path:
    assert config.runtime is not None
    return ensure_dir(config.runtime.output_root)
