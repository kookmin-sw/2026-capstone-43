# 09_spherical_vision

Structured RGB -> Depth -> Point Cloud -> Vision Sphere export pipeline for building a learning-ready spherical vision representation before any audio fusion.

## Project Purpose

- This project upgrades the earlier visualization MVP into a standardized `V_sphere` export pipeline.
- The main goal is not pretty plots or maximum performance.
- The main goal is a reproducible, geometry-aware spherical tensor that can later align with an audio spherical tensor in a shared fusion space.
- The current scope is still vision only.

This project does **not** add:

- audio processing
- multimodal fusion
- training loops
- segmentation
- object detection
- hallucinated rear-side completion
- volumetric spherical voxel grids

## Current Pipeline

1. Load a single RGB image or a directory of images.
2. Run ZoeDepth monocular depth estimation.
3. Convert depth into a camera-centered point cloud.
4. Project visible geometry into a camera-centered spherical grid.
5. Export a structured spherical tensor plus rich metadata and debug visualizations.
6. Pool azimuth features into fixed 8-way directional sectors.

## Core Representation

The main learning export is:

- `vision_sphere.npy`
- shape: `[E, A, C]`

Where:

- `E`: number of elevation bins
- `A`: number of azimuth bins
- `C`: number of feature channels

The azimuth-collapsed version is also exported:

- `vision_sphere_azimuth.npy`
- shape: `[A, C]`

The tensor is accompanied by:

- `vision_sphere_channels.json`
- `vision_sphere_meta.json`
- optional `vision_sphere.pt`

## Channel Specification

Mandatory channels:

1. `observed_mask`
2. `fov_mask`
3. `has_points`
4. `occupancy`
5. `density`
6. `min_depth`
7. `p10_depth`
8. `mean_depth`
9. `median_depth`
10. `depth_std`
11. `valid_ratio`

Optional extra channels:

- `inverse_mean_depth`
- `inverse_p10_depth`
- `log_mean_depth`

### Channel Definitions

- `observed_mask`: this bin received at least one sampled camera ray from the image plane.
- `fov_mask`: the bin center lies inside the analytic camera FOV implied by the intrinsics.
- `has_points`: this bin contains at least one valid projected 3D point.
- `occupancy`: `valid_count / max(valid_count)` over bins.
- `density`: `valid_count / total_valid_points` for the sample.
- `min_depth`: smallest camera-centered range in the bin.
- `p10_depth`: 10th percentile camera-centered range in the bin. This is the main robust near-depth channel.
- `mean_depth`: mean camera-centered range in the bin.
- `median_depth`: median camera-centered range in the bin.
- `depth_std`: standard deviation of camera-centered range in the bin.
- `valid_ratio`: `valid_count / raw_projected_sample_count` in the bin.

## Unknown vs Empty vs Occupied

This distinction is explicit in both code and metadata.

- `unknown`: `observed_mask == 0`
- `observed but empty`: `observed_mask == 1` and `has_points == 0`
- `occupied`: `has_points == 1`

This is important because a front-view camera only observes a visible sector. The pipeline does not fill or guess unseen directions.

## Coordinate and Angular Convention

### Camera coordinates

- `x = right`
- `y = up`
- `z = forward`

Depth-to-point-cloud uses:

```text
x = (u - cx) * z / fx
y = -(v - cy) * z / fy
z = depth
```

### Azimuth

- `0 deg = forward`
- positive azimuth = right
- range = `[-180, 180]` or `[-pi, pi]`
- implemented as `atan2(x, z)`

### Elevation

- `0 deg = horizontal`
- positive elevation = up
- range = `[-90, 90]` or `[-pi/2, pi/2]`
- implemented as `atan2(y, sqrt(x^2 + z^2))`

### Bin edges and centers

The spherical grid is uniformly divided:

- azimuth edges span `[-pi, pi]`
- elevation edges span `[-pi/2, pi/2]`
- centers are midpoints of consecutive edges

Helpers are implemented in `src/spherical_projection.py`:

- `get_bin_centers()`
- `angle_to_bin()`
- `bin_to_angle()`

Metadata JSON stores the exact bin centers and edges in both radians and degrees.

## Robust Depth Handling

This project keeps the raw ZoeDepth output, but the spherical representation is made more robust by:

- invalid depth filtering
- optional absolute clipping: `--depth_clip_min`, `--depth_clip_max`
- optional percentile clipping:
  - `--depth_clip_percentile_low`
  - `--depth_clip_percentile_high`
- percentile-based near-depth summary via `p10_depth`
- median and standard deviation channels
- optional inverse/log depth transform channels

Important:

- raw depth is still monocular and may have scale ambiguity
- spherical depth channels represent **camera-centered Euclidean range**
- `p10_depth` is the recommended robust near-geometry channel for downstream learning

## 8-Way Pooling

An 8-way helper is provided for downstream direction classification.

Fixed label order:

1. `front-left`
2. `front`
3. `front-right`
4. `right`
5. `back-right`
6. `back`
7. `back-left`
8. `left`

Supported mapping modes:

- `sector`
- `nearest`

Supported pooling reductions:

- `mean`
- `max`
- `both`

Main pooled export:

- `vision_8way_pooled.npy`
- shape: `[8, C]`

Metadata:

- `vision_8way_meta.json`

## Main Output Files

Per sample, the pipeline saves at least:

- `rgb.png`
- `depth_raw.npy`
- `depth_vis.png`
- `pointcloud.ply`
- `pointcloud_topdown.png`
- `pointcloud_sideview.png`
- `vision_sphere.npy`
- `vision_sphere_azimuth.npy`
- `vision_sphere_channels.json`
- `vision_sphere_meta.json`
- `vision_sphere_occupancy.png`
- `vision_sphere_p10_depth.png`
- `vision_sphere_mean_depth.png`
- `vision_8way_pooled.npy`
- `vision_8way_meta.json`
- `summary_panel.png`
- `sample_stats.json`

Additional debug artifacts:

- `pointcloud_3d.png`
- `vision_sphere_channel_panel.png`
- `vision_8way_overlay.png`
- `vision_azimuth_multichannel.png`
- `sample_metadata.png`
- optional `vision_sphere.pt`
- optional `globe_3d_occupancy.png`
- optional `globe_3d_p10_depth.png`

Run-level outputs:

- `run_summary.json`
- `run_channel_stats.json`

## Visualization Notes

The plots are now debug tools for the tensor, not the primary deliverable.

Added analysis views include:

- spherical heatmaps with bin center overlays
- 8-way sector overlay figure
- multi-channel spherical panel
- azimuth multi-channel bar plots
- metadata text panel
- optional 3D globe view

The 3D globe view is only for interpretation and debugging. It is **not** a model input.

## Installation

ZoeDepth is assumed to be available locally. The wrapper checks:

1. `--zoe_model_path`
2. `ZOEDEPTH_MODEL_PATH`
3. `../pretrained/zoedepth-nyu-kitti`
4. local Hugging Face cache for `Intel/zoedepth-nyu-kitti`

Install dependencies:

```bash
cd 09_spherical_vision
pip install -r requirements.txt
```

Notes:

- The current implementation uses `transformers` ZoeDepth loading with a local model directory.
- If your environment has a `transformers` vs `huggingface-hub` version mismatch, the wrapper applies a small runtime compatibility patch automatically.
- `--device auto` intentionally uses CPU-first behavior for reproducibility. Use `--device cuda` only if your CUDA environment is confirmed to work with ZoeDepth.

## Run

Basic example:

```bash
cd 09_spherical_vision
python run_mvp.py \
  --input /path/to/image_or_dir \
  --output_dir outputs/demo \
  --hfov_deg 69 \
  --num_az_bins 24 \
  --num_el_bins 8 \
  --max_points 200000
```

Direct intrinsics:

```bash
python run_mvp.py \
  --input /path/to/image.png \
  --output_dir outputs/demo_intrinsics \
  --fx 500 --fy 500 --cx 320 --cy 240
```

Example with clipping and globe debug plots:

```bash
python run_mvp.py \
  --input /path/to/image_or_dir \
  --output_dir outputs/debug \
  --hfov_deg 69 \
  --num_az_bins 36 \
  --num_el_bins 12 \
  --depth_clip_percentile_low 1 \
  --depth_clip_percentile_high 99 \
  --pooling_mode both \
  --save_globe
```

Demo script:

```bash
bash scripts/run_demo.sh
```

## Sample Statistics

Each sample writes `sample_stats.json` with:

- input image path
- image width and height
- intrinsics
- hfov and vfov
- number of azimuth and elevation bins
- channel names
- total projected samples
- valid points
- observed bin count
- occupied bin count
- empty-but-observed bin count
- depth min / mean / max / p10 / p50
- 8-way pooled feature summary

## Run-Level Aggregates

For directory input, `run_summary.json` and `run_channel_stats.json` summarize:

- processed count
- failure count
- average observed bin count
- average occupied bin count
- average empty-but-observed bin count
- depth invalid ratio summary
- channel-wise min / max / mean

## Limitations

- Monocular depth remains scale-ambiguous.
- The representation only covers the camera-visible sector.
- The tensor is geometry-aware, but still derived from a single front-view image.
- `occupancy` is a projection-density statistic, not true volumetric occupancy.
- 8-way pooling is a coarse helper, not a learned alignment module.

## Recommended Downstream Use

For future fusion work, the suggested first-pass usage is:

- use `vision_sphere.npy` as the main vision tensor
- use `vision_sphere_channels.json` to keep channel order fixed
- align audio tensors to the same azimuth/elevation convention
- use `p10_depth`, `mean_depth`, `occupancy`, `observed_mask`, and `has_points` as the most informative early channels

## GitHub Cleanup Notes

Renumbered from `08_spherical_vision` to `09_spherical_vision` and placed before result-only experiment folders because it contains reusable representation/model-input code.

### Code Analysis
- Purpose: RGB image -> ZoeDepth -> point cloud -> spherical `V_sphere` tensor export for future audio/vision alignment.
- Core representation: `[num_el_bins, num_az_bins, channels]` vision sphere plus azimuth and 8-way pooled features.
- Main modules:
- `run_mvp.py`: functions `_load_yaml_defaults`, `build_parser`, `parse_args`, `main`
- `src/__init__.py`: module exports/package marker
- `src/camera_utils.py`: classes `CameraIntrinsics`; functions `coordinate_convention_dict`, `angular_convention_dict`, `compute_intrinsics`, `describe_intrinsics`, `compute_fov_angle_ranges`, `pixel_grid_to_camera_rays`
- `src/feature_export.py`: functions `validate_feature_bundle`, `build_channels_json`, `save_feature_bundle`
- `src/io_utils.py`: functions `ensure_dir`, `is_image_file`, `discover_images`, `load_rgb_image`, `save_rgb_image`, `save_numpy`
- `src/pipeline.py`: classes `PipelineConfig`; functions `setup_logging`, `_config_to_dict`, `_build_sample_metadata_lines`, `_log_feature_stats`, `process_single_image`, `run_pipeline`
- `src/pointcloud_utils.py`: classes `PointCloudStats`, `PointCloudData`; functions `depth_to_point_cloud`, `subsample_point_cloud`, `write_ply`
- `src/pooling_utils.py`: functions `_wrap_degrees`, `azimuth_to_8way_index`, `map_azimuth_bins_to_8way`, `_pool_channel_values`, `pool_azimuth_features_to_8way`
- `src/spherical_projection.py`: classes `AngularGrid`, `FeatureBundle`; functions `build_angular_grid`, `get_bin_centers`, `angle_to_bin`, `bin_to_angle`, `xyz_to_spherical`, `_compute_fov_mask`
- `src/stats_utils.py`: functions `compute_depth_map_stats`, `resolve_depth_clip_bounds`, `summarize_pooled_tensor`, `compute_channel_aggregate_payload`, `build_sample_stats`, `aggregate_run_summary`
- `src/visualization.py`: functions `_subsample_for_plot`, `colorize_depth`, `save_depth_visualization`, `_set_equal_3d_axes`, `save_point_cloud_views`, `_channel_display_map`
- `src/zoedepth_wrapper.py`: classes `DepthPrediction`, `ZoeDepthWrapper`; functions `_install_transformers_hub_version_patch`, `_load_transformers_zoe_classes`, `resolve_model_path`

### Result Summary
- Demo run processed 1 image(s) with 0 failure(s).
- Sample grid: 8 elevation bins x 24 azimuth bins, 14 channels, observed bins=20, occupied bins=20.
- Depth stats: min=0.988, mean=3.395, p10=1.569, p90=5.994, valid_ratio=1.000.
- Aggregate observed_bin_count mean=20.000, occupied_bin_count mean=20.000.
- Channel set retained in README summary: `observed_mask`, `fov_mask`, `has_points`, `occupancy`, `density`, `min_depth`, `p10_depth`, `mean_depth`, `median_depth`, `depth_std`, `valid_ratio`, `inverse_mean_depth`, `inverse_p10_depth`, `log_mean_depth`.

### Removed Artifacts
- Original source contained 31 files across 3 subdirectories (12.2 MB). Generated result files were removed or reduced to empty folder structure in this GitHub copy.
- `outputs/` directory structure was kept empty; generated `.json`, `.npy`, `.pt`, image, and point-cloud artifacts were removed.
- Python bytecode caches under `src/__pycache__/` were removed.
