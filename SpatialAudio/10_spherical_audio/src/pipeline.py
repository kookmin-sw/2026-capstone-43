from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .directional_features import compute_directional_features
from .feature_export import save_audio_sphere
from .foa_utils import load_foa_wav
from .io_utils import discover_wav_files, ensure_dir, safe_stem, write_json
from .pooling_utils import pool_audio_azimuth_to_8way
from .spherical_projection import AUDIO_CHANNEL_NAMES, aggregate_window_feature_maps, build_angular_grid
from .stats_utils import build_sample_stats, save_sample_stats, write_run_summaries
from .stft_utils import compute_windowed_stfts, stft_metadata
from .visualization import (
    plot_8way_pooled,
    plot_audio_azimuth_multichannel,
    plot_audio_sphere_channel_panel,
    plot_global_direction_hist,
    plot_spherical_heatmap,
    plot_stft_overview,
    plot_summary_panel,
    plot_waveform,
    plot_windowwise_direction_track,
)


@dataclass(frozen=True)
class AudioPipelineConfig:
    input_path: Path
    output_dir: Path
    channel_order: str = "WXYZ"
    target_sr: int | None = None
    normalize_audio: bool = False
    num_az_bins: int = 24
    num_el_bins: int = 8
    window_sec: float = 2.0
    hop_sec: float = 1.0
    aggregation: str = "both"
    pooling_mode: str = "both"
    pooling_mapping: str = "sector"
    n_fft: int = 1024
    stft_hop_length: int | None = None
    stft_window: str = "hann"
    aiv_sign: float = 1.0
    export_pt: bool = False


def _channel_index(name: str) -> int:
    return AUDIO_CHANNEL_NAMES.index(name)


def process_single_wav(path: Path, config: AudioPipelineConfig) -> dict[str, Any]:
    sample_dir = ensure_dir(config.output_dir / safe_stem(path))
    print(f"[PIPE] processing {path}")
    print(f"[PIPE] output_dir={sample_dir}")
    print(f"[PIPE] bins az/el={config.num_az_bins}/{config.num_el_bins}")

    foa = load_foa_wav(
        path,
        channel_order=config.channel_order,
        target_sample_rate=config.target_sr,
        normalize_audio=config.normalize_audio,
    )
    plot_waveform(foa.samples, foa.sample_rate, sample_dir / "audio_waveform.png")

    windows = compute_windowed_stfts(
        foa.samples,
        sample_rate=foa.sample_rate,
        window_sec=config.window_sec,
        hop_sec=config.hop_sec,
        n_fft=config.n_fft,
        stft_hop_length=config.stft_hop_length,
        stft_window=config.stft_window,
    )
    plot_stft_overview(windows, sample_dir / "stft_overview.png")

    grid = build_angular_grid(config.num_az_bins, config.num_el_bins)
    directional = compute_directional_features(windows, grid=grid, aiv_sign=config.aiv_sign)
    sphere = aggregate_window_feature_maps(
        directional.window_feature_maps,
        grid=grid,
        aggregation_mode=config.aggregation,
        extra_metadata={
            "input_wav_path": str(path),
            "foa": foa.metadata,
            "stft": stft_metadata(windows),
            "directional_features": directional.metadata,
            "window_peak_trace": directional.window_peak_trace,
            "pooling_mode": config.pooling_mode,
            "pooling_mapping": config.pooling_mapping,
        },
    )

    save_audio_sphere(
        sample_dir,
        tensor=sphere.tensor,
        azimuth_tensor=sphere.azimuth_tensor,
        channel_names=sphere.channel_names,
        meta=sphere.metadata,
        export_pt=config.export_pt,
        tensor_max=sphere.tensor_max,
        azimuth_tensor_max=sphere.azimuth_tensor_max,
    )

    pooled, pooled_max, pooling_meta = pool_audio_azimuth_to_8way(
        sphere.azimuth_tensor,
        azimuth_centers_rad=grid.azimuth_centers,
        channel_names=sphere.channel_names,
        pooling_mode=config.pooling_mode,
        mapping_mode=config.pooling_mapping,
    )
    np.save(sample_dir / "audio_8way_pooled.npy", pooled.astype(np.float32))
    if pooled_max is not None:
        np.save(sample_dir / "audio_8way_pooled_max.npy", pooled_max.astype(np.float32))
    write_json(sample_dir / "audio_8way_meta.json", pooling_meta)

    beam_idx = _channel_index("beam_power")
    aiv_idx = _channel_index("aiv_score")
    diff_idx = _channel_index("diffuseness")
    p10_like_title = "Beam power map"
    plot_spherical_heatmap(sphere.tensor[:, :, beam_idx], grid, p10_like_title, sample_dir / "beam_power_map.png", cmap="inferno")
    plot_spherical_heatmap(sphere.tensor[:, :, aiv_idx], grid, "AIV direction map", sample_dir / "aiv_direction_map.png", cmap="plasma")
    plot_spherical_heatmap(sphere.tensor[:, :, diff_idx], grid, "Diffuseness / uncertainty map", sample_dir / "diffuseness_map.png", cmap="magma")
    plot_audio_sphere_channel_panel(sphere.tensor, sphere.channel_names, grid, sample_dir / "audio_sphere_channel_panel.png")
    plot_audio_azimuth_multichannel(sphere.azimuth_tensor, sphere.channel_names, grid, sample_dir / "audio_azimuth_multichannel.png")
    plot_8way_pooled(pooled, sphere.channel_names, sample_dir / "audio_8way_pooled.png")
    plot_global_direction_hist(directional.aiv_histogram, grid, sample_dir / "global_direction_hist.png")
    plot_windowwise_direction_track(directional.window_peak_trace, sample_dir / "windowwise_direction_track.png")

    stats = build_sample_stats(
        input_path=path,
        sample_rate=foa.sample_rate,
        num_samples=foa.samples.shape[0],
        channel_order_input=foa.channel_order_input,
        channel_order_canonical=foa.channel_order_canonical,
        grid=grid,
        tensor=sphere.tensor,
        azimuth_tensor=sphere.azimuth_tensor,
        channel_names=sphere.channel_names,
        pooled_8way=pooled,
        pooling_meta=pooling_meta,
        window_sec=config.window_sec,
        hop_sec=config.hop_sec,
        aggregation_mode=config.aggregation,
    )
    save_sample_stats(sample_dir / "sample_stats.json", stats)
    plot_summary_panel(
        foa.samples,
        foa.sample_rate,
        windows,
        sphere.tensor,
        sphere.azimuth_tensor,
        pooled,
        sphere.channel_names,
        grid,
        stats,
        sample_dir / "summary_panel.png",
    )

    print(
        "[CHECK] nonzero_bins="
        f"{stats['nonzero_bins']} peak=({stats['peak_direction_azimuth_deg']:.1f} deg az, "
        f"{stats['peak_direction_elevation_deg']:.1f} deg el) "
        f"top8={stats['eight_way_top_label']}"
    )
    for channel_name, channel_stats in stats["per_channel_min_max_mean"].items():
        print(
            f"[CHECK] {channel_name}: min={channel_stats['min']:.4f} "
            f"max={channel_stats['max']:.4f} mean={channel_stats['mean']:.4f}"
        )

    return stats


def run_pipeline(config: AudioPipelineConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    wav_files = discover_wav_files(config.input_path)
    print(f"[RUN] found {len(wav_files)} wav file(s)")

    sample_stats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for wav_path in wav_files:
        try:
            sample_stats.append(process_single_wav(wav_path, config))
        except Exception as exc:
            failure = {"input_wav_path": str(wav_path), "error": repr(exc)}
            failures.append(failure)
            print(f"[ERROR] failed {wav_path}: {exc}")

    write_run_summaries(output_dir, sample_stats, failures, AUDIO_CHANNEL_NAMES)
    print(f"[RUN] completed processed={len(sample_stats)} failures={len(failures)} output={output_dir}")
    return {"sample_stats": sample_stats, "failures": failures, "output_dir": str(output_dir)}

