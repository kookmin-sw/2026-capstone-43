from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import write_json
from .pooling_utils import EIGHT_WAY_LABELS
from .spherical_projection import AngularGrid, bin_to_angle


def _channel_stats(tensor: np.ndarray, channel_names: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(channel_names):
        values = tensor[..., idx]
        stats[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return stats


def build_sample_stats(
    input_path: str | Path,
    sample_rate: int,
    num_samples: int,
    channel_order_input: str,
    channel_order_canonical: str,
    grid: AngularGrid,
    tensor: np.ndarray,
    azimuth_tensor: np.ndarray,
    channel_names: list[str],
    pooled_8way: np.ndarray,
    pooling_meta: dict[str, Any],
    window_sec: float,
    hop_sec: float,
    aggregation_mode: str,
) -> dict[str, Any]:
    score_channel = "dp_reliability" if "dp_reliability" in channel_names else channel_names[0]
    score_idx = channel_names.index(score_channel)
    score_map = tensor[:, :, score_idx]
    peak_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
    peak_angle = bin_to_angle(peak_idx[1], peak_idx[0], grid, output_in_degrees=True)
    nonzero_bins = int(np.count_nonzero(np.sum(np.abs(tensor), axis=-1) > 1.0e-8))

    top_label_idx = int(np.argmax(pooled_8way[:, score_idx]))
    pooled_summary = {
        EIGHT_WAY_LABELS[idx]: {
            channel_names[channel_idx]: float(pooled_8way[idx, channel_idx])
            for channel_idx in range(len(channel_names))
        }
        for idx in range(len(EIGHT_WAY_LABELS))
    }

    return {
        "input_wav_path": str(input_path),
        "sample_rate": int(sample_rate),
        "num_samples": int(num_samples),
        "duration_sec": float(num_samples / max(sample_rate, 1)),
        "channel_order_input": channel_order_input,
        "channel_order_canonical": channel_order_canonical,
        "num_az_bins": int(grid.num_az_bins),
        "num_el_bins": int(grid.num_el_bins),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "aggregation_mode": aggregation_mode,
        "nonzero_bins": nonzero_bins,
        "peak_direction_bin": {"elevation_idx": int(peak_idx[0]), "azimuth_idx": int(peak_idx[1])},
        "peak_direction_azimuth_deg": float(peak_angle["azimuth"]),
        "peak_direction_elevation_deg": float(peak_angle["elevation"]),
        "peak_direction_score_channel": score_channel,
        "peak_direction_score": float(score_map[peak_idx]),
        "per_channel_min_max_mean": _channel_stats(tensor, channel_names),
        "azimuth_per_channel_min_max_mean": _channel_stats(azimuth_tensor, channel_names),
        "eight_way_top_label": EIGHT_WAY_LABELS[top_label_idx],
        "eight_way_top_score": float(pooled_8way[top_label_idx, score_idx]),
        "eight_way_pooled_summary": pooled_summary,
        "eight_way_meta": pooling_meta,
    }


def save_sample_stats(output_path: str | Path, stats: dict[str, Any]) -> None:
    write_json(output_path, stats)


def write_run_summaries(
    output_dir: str | Path,
    sample_stats: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    channel_names: list[str],
) -> None:
    output_dir = Path(output_dir)
    processed = len(sample_stats)
    summary = {
        "processed_count": processed,
        "failure_count": len(failures),
        "failures": failures,
        "mean_nonzero_bins": float(np.mean([s["nonzero_bins"] for s in sample_stats])) if sample_stats else 0.0,
        "mean_duration_sec": float(np.mean([s["duration_sec"] for s in sample_stats])) if sample_stats else 0.0,
        "top_label_counts": {},
    }
    for stats in sample_stats:
        label = stats["eight_way_top_label"]
        summary["top_label_counts"][label] = summary["top_label_counts"].get(label, 0) + 1
    write_json(output_dir / "run_summary.json", summary)

    channel_stats: dict[str, dict[str, float]] = {}
    for name in channel_names:
        mins = [s["per_channel_min_max_mean"][name]["min"] for s in sample_stats]
        maxs = [s["per_channel_min_max_mean"][name]["max"] for s in sample_stats]
        means = [s["per_channel_min_max_mean"][name]["mean"] for s in sample_stats]
        channel_stats[name] = {
            "global_min": float(np.min(mins)) if mins else 0.0,
            "global_max": float(np.max(maxs)) if maxs else 0.0,
            "mean_of_sample_means": float(np.mean(means)) if means else 0.0,
        }
    write_json(output_dir / "run_channel_stats.json", {"channel_stats": channel_stats})

