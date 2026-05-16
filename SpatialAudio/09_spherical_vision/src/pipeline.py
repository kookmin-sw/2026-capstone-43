from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .camera_utils import compute_intrinsics, describe_intrinsics
from .feature_export import save_feature_bundle
from .io_utils import (
    discover_images,
    ensure_dir,
    load_rgb_image,
    make_image_output_dir,
    save_json,
    save_rgb_image,
)
from .pointcloud_utils import depth_to_point_cloud, subsample_point_cloud, write_ply
from .pooling_utils import pool_azimuth_features_to_8way
from .spherical_projection import build_vision_feature_bundles, build_vision_sphere_meta
from .stats_utils import (
    aggregate_run_channel_stats,
    aggregate_run_summary,
    build_sample_stats,
    compute_channel_aggregate_payload,
    compute_depth_map_stats,
    resolve_depth_clip_bounds,
    summarize_pooled_tensor,
)
from .visualization import (
    save_8way_overlay_visualization,
    save_azimuth_multichannel_plot,
    save_channel_panel,
    save_depth_visualization,
    save_globe_visualization,
    save_metadata_text_box,
    save_point_cloud_views,
    save_spherical_channel_heatmap,
    save_summary_panel,
)
from .zoedepth_wrapper import ZoeDepthWrapper

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    project_root: Path
    zoe_model_path: str | Path | None = None
    device: str = "auto"
    hfov_deg: float = 69.0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    num_az_bins: int = 24
    num_el_bins: int = 8
    use_elevation: bool = True
    point_stride: int = 1
    max_points: int | None = 200_000
    plot_max_points: int = 40_000
    depth_clip_min: float | None = None
    depth_clip_max: float | None = None
    depth_clip_percentile_low: float | None = None
    depth_clip_percentile_high: float | None = None
    include_extra_depth_channels: bool = True
    export_pt: bool = False
    pooling_mode: str = "mean"
    pooling_mapping: str = "sector"
    save_globe: bool = False
    seed: int = 0
    log_level: str = "INFO"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["input_path"] = str(config.input_path)
    payload["output_dir"] = str(config.output_dir)
    payload["project_root"] = str(config.project_root)
    if config.zoe_model_path is not None:
        payload["zoe_model_path"] = str(config.zoe_model_path)
    return payload


def _build_sample_metadata_lines(sample_stats: dict[str, Any]) -> list[str]:
    return [
        f"image: {Path(sample_stats['input_image_path']).name}",
        f"size: {sample_stats['image_width']} x {sample_stats['image_height']}",
        (
            f"intrinsics: fx={sample_stats['intrinsics']['fx']:.2f}, "
            f"fy={sample_stats['intrinsics']['fy']:.2f}, "
            f"cx={sample_stats['intrinsics']['cx']:.2f}, "
            f"cy={sample_stats['intrinsics']['cy']:.2f}"
        ),
        f"hfov/vfov: {sample_stats['hfov_deg']:.2f} / {sample_stats['vfov_deg']:.2f} deg",
        f"bins: az={sample_stats['num_az_bins']}, el={sample_stats['num_el_bins']}",
        (
            f"observed/occupied/empty-observed: "
            f"{sample_stats['observed_bin_count']} / {sample_stats['occupied_bin_count']} / "
            f"{sample_stats['empty_but_observed_bin_count']}"
        ),
        (
            f"depth min/p10/p50/max: "
            f"{sample_stats['depth_stats']['min']:.3f} / "
            f"{sample_stats['depth_stats']['p10']:.3f} / "
            f"{sample_stats['depth_stats']['p50']:.3f} / "
            f"{sample_stats['depth_stats']['max']:.3f}"
        ),
        f"valid points: {sample_stats['valid_points']}",
        f"channels: {len(sample_stats['channel_names'])}",
    ]


def _log_feature_stats(label: str, summary: dict[str, Any]) -> None:
    LOGGER.info(
        (
            "%s | observed_bins=%s | occupied_bins=%s | empty_observed_bins=%s | "
            "most_occupied=%s | nearest_by_p10=%s"
        ),
        label,
        summary["observed_bin_count"],
        summary["occupied_bin_count"],
        summary["empty_but_observed_bin_count"],
        summary["most_occupied_bin"],
        summary["nearest_bin_by_p10_depth"],
    )


def process_single_image(
    image_path: Path,
    image_index: int,
    config: PipelineConfig,
    depth_model: ZoeDepthWrapper,
) -> dict[str, Any]:
    output_dir = make_image_output_dir(config.output_dir, image_path, image_index)
    rgb = load_rgb_image(image_path)
    image_height, image_width = rgb.shape[:2]

    LOGGER.info("Processing image %s", image_path)
    LOGGER.info(
        (
            "Image info | size=%dx%d | output_dir=%s | num_az_bins=%d | num_el_bins=%d | "
            "point_stride=%d"
        ),
        image_width,
        image_height,
        output_dir,
        config.num_az_bins,
        config.num_el_bins,
        config.point_stride,
    )

    intrinsics = compute_intrinsics(
        width=image_width,
        height=image_height,
        fx=config.fx,
        fy=config.fy,
        cx=config.cx,
        cy=config.cy,
        hfov_deg=config.hfov_deg,
    )
    LOGGER.info("Intrinsics | %s", describe_intrinsics(intrinsics))

    save_rgb_image(rgb, output_dir / "rgb.png")
    save_json(intrinsics.to_dict(), output_dir / "intrinsics.json")

    depth_prediction = depth_model.predict(rgb)
    depth_map = depth_prediction.depth
    depth_stats = compute_depth_map_stats(depth_map)
    LOGGER.info(
        "Depth stats | min=%.4f | max=%.4f | mean=%.4f | p10=%.4f | p50=%.4f | valid_ratio=%.4f",
        depth_stats["min"],
        depth_stats["max"],
        depth_stats["mean"],
        depth_stats["p10"],
        depth_stats["p50"],
        depth_stats["valid_ratio"],
    )
    save_json(
        {
            "stats": depth_stats,
            "metadata": depth_prediction.metadata,
        },
        output_dir / "depth_stats.json",
    )
    save_depth_visualization(depth_map, output_dir / "depth_vis.png")
    save_json({"depth_shape": list(depth_map.shape)}, output_dir / "depth_shape.json")
    from .io_utils import save_numpy  # local import to keep module import surface small

    save_numpy(depth_map, output_dir / "depth_raw.npy")

    clip_min, clip_max, clip_meta = resolve_depth_clip_bounds(
        depth_map=depth_map,
        depth_clip_min=config.depth_clip_min,
        depth_clip_max=config.depth_clip_max,
        depth_clip_percentile_low=config.depth_clip_percentile_low,
        depth_clip_percentile_high=config.depth_clip_percentile_high,
    )
    depth_processing_meta = {
        **clip_meta,
        "range_depth_reference": "camera-centered Euclidean norm of xyz",
        "include_extra_depth_channels": config.include_extra_depth_channels,
        "depth_transform_channels": [
            "inverse_mean_depth",
            "inverse_p10_depth",
            "log_mean_depth",
        ]
        if config.include_extra_depth_channels
        else [],
        "normalization_notes": {
            "raw_tensor_depth_channels": "stored as clipped raw range statistics",
            "inverse_and_log_channels": "provided as additional transform-friendly channels when enabled",
        },
    }
    save_json(depth_processing_meta, output_dir / "depth_processing.json")

    full_point_cloud = depth_to_point_cloud(
        depth_map=depth_map,
        rgb_image=rgb,
        intrinsics=intrinsics,
        point_stride=config.point_stride,
        depth_clip_min=clip_min,
        depth_clip_max=clip_max,
    )
    ply_point_cloud = subsample_point_cloud(
        full_point_cloud,
        max_points=config.max_points,
        seed=config.seed,
    )
    LOGGER.info(
        (
            "Point cloud stats | total_pixels=%d | sampled_pixels=%d | filtered_points=%d | "
            "valid_points=%d | kept_points=%d"
        ),
        full_point_cloud.stats.total_pixels,
        full_point_cloud.stats.sampled_pixels,
        full_point_cloud.stats.filtered_points,
        full_point_cloud.stats.valid_points,
        ply_point_cloud.stats.kept_points,
    )
    write_ply(output_dir / "pointcloud.ply", ply_point_cloud.points, ply_point_cloud.colors)
    save_json(
        {
            "full_point_cloud": full_point_cloud.stats.to_dict(),
            "saved_point_cloud": ply_point_cloud.stats.to_dict(),
        },
        output_dir / "pointcloud_stats.json",
    )

    full_bundle, azimuth_bundle = build_vision_feature_bundles(
        points=full_point_cloud.points,
        image_width=image_width,
        image_height=image_height,
        intrinsics=intrinsics,
        num_az_bins=config.num_az_bins,
        num_el_bins=config.num_el_bins,
        point_stride=config.point_stride,
        include_extra_depth_channels=config.include_extra_depth_channels,
        depth_processing=depth_processing_meta,
    )
    full_summary = full_bundle.summary_dict()
    azimuth_summary = azimuth_bundle.summary_dict()
    _log_feature_stats("Full spherical summary", full_summary)
    _log_feature_stats("Azimuth summary", azimuth_summary)

    vision_meta = build_vision_sphere_meta(full_bundle, azimuth_bundle, intrinsics)
    feature_export_paths = save_feature_bundle(
        output_dir=output_dir,
        full_bundle=full_bundle,
        azimuth_bundle=azimuth_bundle,
        meta=vision_meta,
        export_pt=config.export_pt,
    )

    pooled_tensor, pooled_max_tensor, pooled_meta = pool_azimuth_features_to_8way(
        azimuth_bundle=azimuth_bundle,
        pooling_mode=config.pooling_mode,
        mapping_mode=config.pooling_mapping,
    )
    save_numpy(pooled_tensor, output_dir / "vision_8way_pooled.npy")
    if pooled_max_tensor is not None:
        save_numpy(pooled_max_tensor, output_dir / "vision_8way_pooled_max.npy")
    pooled_meta = {
        **pooled_meta,
        "channel_names": list(full_bundle.channel_names),
        "primary_output_path": str((output_dir / "vision_8way_pooled.npy").resolve()),
        "max_output_path": ""
        if pooled_max_tensor is None
        else str((output_dir / "vision_8way_pooled_max.npy").resolve()),
        "primary_summary": summarize_pooled_tensor(pooled_tensor, full_bundle.channel_names, pooled_meta["labels"]),
        "max_summary": None
        if pooled_max_tensor is None
        else summarize_pooled_tensor(pooled_max_tensor, full_bundle.channel_names, pooled_meta["labels"]),
    }
    save_json(pooled_meta, output_dir / "vision_8way_meta.json")

    save_point_cloud_views(
        points=ply_point_cloud.points,
        colors=ply_point_cloud.colors,
        output_3d_path=output_dir / "pointcloud_3d.png",
        output_topdown_path=output_dir / "pointcloud_topdown.png",
        output_sideview_path=output_dir / "pointcloud_sideview.png",
        max_plot_points=config.plot_max_points,
        seed=config.seed,
    )
    save_spherical_channel_heatmap(
        bundle=full_bundle,
        channel_name="occupancy",
        output_path=output_dir / "vision_sphere_occupancy.png",
        title="Vision Sphere Occupancy",
        colorbar_label="Normalized occupancy",
        cmap_name="viridis",
    )
    save_spherical_channel_heatmap(
        bundle=full_bundle,
        channel_name="p10_depth",
        output_path=output_dir / "vision_sphere_p10_depth.png",
        title="Vision Sphere P10 Depth",
        colorbar_label="P10 range depth",
        cmap_name="magma",
    )
    save_spherical_channel_heatmap(
        bundle=full_bundle,
        channel_name="mean_depth",
        output_path=output_dir / "vision_sphere_mean_depth.png",
        title="Vision Sphere Mean Depth",
        colorbar_label="Mean range depth",
        cmap_name="plasma",
    )
    save_channel_panel(
        bundle=full_bundle,
        channel_names=["observed_mask", "has_points", "occupancy", "p10_depth", "mean_depth", "valid_ratio"],
        output_path=output_dir / "vision_sphere_channel_panel.png",
        title_prefix="Vision Sphere",
    )
    save_8way_overlay_visualization(
        azimuth_bundle=azimuth_bundle,
        output_path=output_dir / "vision_8way_overlay.png",
    )
    save_azimuth_multichannel_plot(
        azimuth_bundle=azimuth_bundle,
        output_path=output_dir / "vision_azimuth_multichannel.png",
    )

    sample_stats = build_sample_stats(
        image_path=image_path,
        output_dir=output_dir,
        image_width=image_width,
        image_height=image_height,
        intrinsics=intrinsics,
        depth_stats=depth_stats,
        depth_processing_meta=depth_processing_meta,
        point_cloud=full_point_cloud,
        full_bundle=full_bundle,
        azimuth_bundle=azimuth_bundle,
        pooled_tensor=pooled_tensor,
        pooled_meta=pooled_meta,
        channel_names=full_bundle.channel_names,
    )
    sample_stats["feature_exports"] = feature_export_paths
    metadata_lines = _build_sample_metadata_lines(sample_stats)
    save_metadata_text_box(metadata_lines, output_dir / "sample_metadata.png")

    if config.save_globe:
        save_globe_visualization(
            bundle=full_bundle,
            channel_name="occupancy",
            output_path=output_dir / "globe_3d_occupancy.png",
            title="3D Globe Occupancy",
            cmap_name="viridis",
        )
        save_globe_visualization(
            bundle=full_bundle,
            channel_name="p10_depth",
            output_path=output_dir / "globe_3d_p10_depth.png",
            title="3D Globe P10 Depth",
            cmap_name="magma",
        )

    save_summary_panel(
        [
            ("RGB", output_dir / "rgb.png"),
            ("Depth", output_dir / "depth_vis.png"),
            ("Top-Down Point Cloud", output_dir / "pointcloud_topdown.png"),
            ("Vision Sphere Occupancy", output_dir / "vision_sphere_occupancy.png"),
            ("Vision Sphere P10 Depth", output_dir / "vision_sphere_p10_depth.png"),
            ("8-Way Overlay", output_dir / "vision_8way_overlay.png"),
            ("Channel Panel", output_dir / "vision_sphere_channel_panel.png"),
            ("Sample Metadata", output_dir / "sample_metadata.png"),
        ],
        output_dir / "summary_panel.png",
    )

    save_json(sample_stats, output_dir / "sample_stats.json")
    save_json(sample_stats, output_dir / "summary.json")

    channel_payload = compute_channel_aggregate_payload(full_bundle)
    return {
        "sample_stats": sample_stats,
        "channel_payload": channel_payload,
    }


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    setup_logging(config.log_level)
    output_dir = ensure_dir(config.output_dir.resolve())
    images = discover_images(config.input_path)
    LOGGER.info("Discovered %d image(s) from %s", len(images), config.input_path)

    depth_model = ZoeDepthWrapper(
        model_path=config.zoe_model_path,
        project_root=config.project_root,
        device=config.device,
        allow_cpu_fallback=True,
    )

    processed_samples: list[dict[str, Any]] = []
    channel_payloads: list[dict[str, dict[str, float | int | str]]] = []
    failed_samples: list[dict[str, Any]] = []

    for index, image_path in enumerate(images):
        try:
            result = process_single_image(
                image_path=image_path,
                image_index=index,
                config=config,
                depth_model=depth_model,
            )
            processed_samples.append(result["sample_stats"])
            channel_payloads.append(result["channel_payload"])
        except Exception as exc:
            LOGGER.exception("Failed to process image %s", image_path)
            error_output_dir = make_image_output_dir(output_dir, image_path, index)
            error_payload = {
                "input_image_path": str(image_path),
                "output_dir": str(error_output_dir),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            save_json(error_payload, error_output_dir / "error.json")
            failed_samples.append(error_payload)

    config_dict = _config_to_dict(config)
    run_summary = aggregate_run_summary(
        config_dict=config_dict,
        processed_samples=processed_samples,
        failed_samples=failed_samples,
    )
    save_json(run_summary, output_dir / "run_summary.json")

    channel_names = processed_samples[0]["channel_names"] if processed_samples else []
    run_channel_stats = aggregate_run_channel_stats(channel_names, channel_payloads)
    save_json(run_channel_stats, output_dir / "run_channel_stats.json")

    LOGGER.info(
        "Run finished | processed=%d | failed=%d | output_dir=%s",
        len(processed_samples),
        len(failed_samples),
        output_dir,
    )
    return run_summary
