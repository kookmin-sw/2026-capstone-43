# 10_spherical_audio

Audio-only spherical representation MVP for FOA/B-format recordings.

This project converts a 4-channel FOA wav into a learning-ready spherical audio tensor:

```text
FOA wav -> STFT -> directional audio features -> spherical projection -> A_sphere export
```

The goal is not source localization training. The goal is to build interpretable, analytic directional evidence on the same angular grid used by `09_spherical_vision`, so a later project can align `A_sphere` with `V_sphere` for shared spherical fusion.

## Input

The input must be a 4-channel FOA wav file or a directory of `.wav` files. Mono and stereo are intentionally unsupported.

FOA channel order varies across datasets, so the loader canonicalizes the first four channels into internal `WXYZ` order.

Examples:

```bash
--channel_order WXYZ
--channel_order WYZX
--channel_order W,Y,Z,X
```

The selected order must contain `W`, `X`, `Y`, and `Z` exactly once. The actual order and canonical remapping are saved in metadata.

## Angular Convention

This project intentionally matches `09_spherical_vision`.

Coordinate frame:

- `x = right`
- `y = up`
- `z = forward`
- reference frame is listener/camera centered

Angles:

- azimuth is `atan2(x, z)`
- `0 deg = forward`
- positive azimuth points right
- azimuth range is `[-180, 180]`
- elevation is `atan2(y, sqrt(x^2 + z^2))`
- `0 deg = horizontal`
- positive elevation points up
- elevation range is `[-90, 90]`

The default grid is `num_az_bins=24`, `num_el_bins=8`, giving `A_sphere.shape == [8, 24, C]`.

## Audio Sphere Channels

`audio_sphere.npy` stores `A_sphere` with shape `[E, A, C]`.

Default channel order:

1. `beam_power`: first-order FOA cardioid beamformer power per direction bin.
2. `aiv_score`: active-intensity-vector magnitude accumulated into each direction bin.
3. `diffuseness`: uncertainty proxy; high means diffuse, reverberant, or ambiguous evidence.
4. `dp_reliability`: direct-path reliability proxy combining directional energy, local contrast, AIV score, and low diffuseness.
5. `energy`: normalized directional energy proxy from beam and AIV evidence.
6. `stability`: temporal consistency proxy computed from window-wise directional energy.

These are analytic features, not a trained model prediction.

## Windowing And Aggregation

The audio is processed in analysis windows, default:

- `window_sec = 2.0`
- `hop_sec = 1.0`

Supported aggregation modes:

- `mean`: persistent directional evidence is emphasized.
- `max`: transient directional evidence is emphasized.
- `both`: saves mean as primary `audio_sphere.npy` and max as `audio_sphere_max.npy`.

## 8-Way Pooling

For downstream 8-way direction classification, azimuth bins are pooled into this fixed label order:

```text
front-left, front, front-right, right, back-right, back, back-left, left
```

The default mapping is sector-based with 45 degree sectors. `nearest` center mapping is also available.

Saved outputs:

- `audio_8way_pooled.npy`: primary pooled tensor, shape `[8, C]`
- `audio_8way_pooled_max.npy`: optional when max/both pooling is enabled
- `audio_8way_meta.json`: sector definitions and bin-to-sector mapping

## Install

Python 3.10+ is recommended.

```bash
cd 10_spherical_audio
pip install -r requirements.txt
```

PyTorch is optional. If installed, `--export_pt` also writes `audio_sphere.pt`.

## Run

Single wav:

```bash
python run_audio_mvp.py \
  --input path/to/foa.wav \
  --output_dir outputs/demo \
  --channel_order WXYZ \
  --num_az_bins 24 \
  --num_el_bins 8 \
  --window_sec 2.0 \
  --hop_sec 1.0 \
  --aggregation both \
  --pooling_mode both
```

Directory batch:

```bash
python run_audio_mvp.py \
  --input path/to/wav_dir \
  --output_dir outputs/batch \
  --channel_order WXYZ
```

Optional resampling:

```bash
python run_audio_mvp.py \
  --input path/to/foa.wav \
  --output_dir outputs/resampled \
  --target_sr 16000
```

Demo script:

```bash
bash scripts/run_demo.sh
```

## Outputs Per Wav

Required artifacts:

- `audio_waveform.png`: canonical WXYZ waveform sanity check.
- `stft_overview.png`: W-channel STFT overview.
- `beam_power_map.png`: spherical heatmap of beam power.
- `aiv_direction_map.png`: spherical heatmap of active-intensity direction evidence.
- `diffuseness_map.png`: spherical heatmap of uncertainty/diffuseness.
- `audio_sphere.npy`: primary `[E, A, C]` tensor.
- `audio_sphere_azimuth.npy`: azimuth aggregate `[A, C]`.
- `audio_sphere_channels.json`: channel names and definitions.
- `audio_sphere_meta.json`: angular convention, bin centers, feature settings, FOA/STFT metadata.
- `audio_8way_pooled.npy`: `[8, C]` pooled tensor.
- `audio_8way_meta.json`: 8-way sector metadata.
- `audio_sphere_channel_panel.png`: per-channel spherical visualization.
- `audio_azimuth_multichannel.png`: azimuth-only multi-channel bar plots.
- `summary_panel.png`: one-shot overview panel.
- `sample_stats.json`: sample-level sanity checks and feature summaries.

Additional artifacts:

- `audio_sphere_max.npy`: saved when `--aggregation both`.
- `audio_sphere_azimuth_max.npy`: saved when `--aggregation both`.
- `audio_8way_pooled_max.npy`: saved when `--pooling_mode both`.
- `audio_sphere.pt`: optional PyTorch export with `--export_pt`.
- `global_direction_hist.png`: accumulated AIV direction histogram.
- `windowwise_direction_track.png`: peak direction track across analysis windows.
- `audio_8way_pooled.png`: 8-way pooled feature visualization.

Batch-level artifacts:

- `run_summary.json`: processed/failure counts and aggregate label summary.
- `run_channel_stats.json`: channel-wise aggregate min/max/mean stats.

## Alignment With 09_spherical_vision

`A_sphere` and `V_sphere` are designed to share:

- identical coordinate axes: `x=right`, `y=up`, `z=forward`
- identical azimuth and elevation definitions
- identical azimuth/elevation ranges
- compatible bin center metadata
- compatible 8-way label order

If both projects use the same `num_az_bins` and `num_el_bins`, their tensors can be aligned by `[E, A]` grid position before future fusion. This project does not implement fusion.

## Limitations

- The features are analytic proxies, not physically exact direct-path estimators.
- FOA channel order and sign conventions are dataset-sensitive; use metadata and sanity plots to debug.
- `aiv_sign` may need flipping for datasets with opposite propagation/sign convention.
- Reverberation, multiple sources, and diffuse fields can make directional evidence ambiguous.
- This project is audio-only and does not include training, LLMs, semantics, or multimodal fusion.

## GitHub Cleanup Notes

Renumbered from `09_spherical_audio` to `10_spherical_audio` and placed next to `09_spherical_vision` because both define aligned spherical representations.

### Code Analysis
- Purpose: 4-channel FOA wav -> STFT -> analytic directional features -> spherical `A_sphere` tensor export.
- Core representation: `[num_el_bins, num_az_bins, channels]` audio sphere with beam power, active-intensity, diffuseness, reliability, energy, and stability channels.
- The angular convention intentionally matches `09_spherical_vision`: x=right, y=up, z=forward, azimuth 0 deg=forward, positive=right.
- Main modules:
- `run_audio_mvp.py`: functions `_config_value`, `build_arg_parser`, `main`
- `src/__init__.py`: module exports/package marker
- `src/directional_features.py`: classes `DirectionalFeatureResult`; functions `_safe_normalize_map`, `_beam_power_scan`, `_active_intensity_accumulate`, `_build_window_feature_map`, `compute_directional_features`
- `src/feature_export.py`: functions `validate_audio_tensor`, `validate_azimuth_tensor`, `save_audio_sphere`
- `src/foa_utils.py`: classes `FOAAudio`; functions `parse_channel_order`, `canonicalize_foa_channels`, `_read_wav`, `_resample_if_needed`, `load_foa_wav`
- `src/io_utils.py`: functions `ensure_dir`, `discover_wav_files`, `safe_stem`, `to_jsonable`, `write_json`, `read_yaml_config`
- `src/pipeline.py`: classes `AudioPipelineConfig`; functions `_channel_index`, `process_single_wav`, `run_pipeline`
- `src/pooling_utils.py`: functions `_wrap_degrees`, `azimuth_to_8way_index`, `map_azimuth_bins_to_8way`, `_pool`, `pool_audio_azimuth_to_8way`
- `src/spherical_projection.py`: classes `AngularGrid`, `AudioSphere`; functions `coordinate_convention_dict`, `build_angular_grid`, `get_bin_centers`, `_wrap_azimuth_rad`, `angle_to_bin`, `bin_to_angle`
- `src/stats_utils.py`: functions `_channel_stats`, `build_sample_stats`, `save_sample_stats`, `write_run_summaries`
- `src/stft_utils.py`: classes `WindowSTFT`; functions `_analysis_window`, `_frame_signal_1d`, `compute_multichannel_stft`, `compute_windowed_stfts`, `stft_metadata`
- `src/visualization.py`: functions `_finalize`, `_extent`, `plot_waveform`, `plot_stft_overview`, `plot_spherical_heatmap`, `plot_audio_sphere_channel_panel`

### Result Summary
- No completed demo result payload was present beyond the placeholder `outputs/.gitkeep`; that placeholder file was removed and `outputs/` was left as an empty result directory.

### Removed Artifacts
- Original source contained 1 files across 0 subdirectories (1 B). Generated result files were removed or reduced to empty folder structure in this GitHub copy.
- Python bytecode caches under `src/__pycache__/` were removed.
