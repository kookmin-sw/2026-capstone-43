from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spherical_projection import (
    EPSILON,
    WINDOW_CHANNEL_NAMES,
    AngularGrid,
    angle_to_bin,
    bin_to_angle,
    directions_to_angles,
)
from .stft_utils import WindowSTFT


@dataclass(frozen=True)
class DirectionalFeatureResult:
    window_feature_maps: np.ndarray
    window_peak_trace: list[dict[str, Any]]
    aiv_histogram: np.ndarray
    beam_power_mean: np.ndarray
    metadata: dict[str, Any]


def _safe_normalize_map(values: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    max_value = float(np.max(finite)) if finite.size else 0.0
    if max_value <= EPSILON:
        return np.zeros_like(finite, dtype=np.float32)
    return np.clip(finite / max_value, 0.0, 1.0).astype(np.float32)


def _beam_power_scan(stft: np.ndarray, directions: np.ndarray) -> np.ndarray:
    w = stft[0].reshape(-1)
    x = stft[1].reshape(-1)
    y = stft[2].reshape(-1)
    z = stft[3].reshape(-1)
    flat_dirs = directions.reshape(-1, 3)
    power = np.zeros(flat_dirs.shape[0], dtype=np.float32)
    for idx, (ux, uy, uz) in enumerate(flat_dirs):
        beam = 0.5 * (w + ux * x + uy * y + uz * z)
        power[idx] = float(np.mean(np.abs(beam) ** 2))
    return power.reshape(directions.shape[:2]).astype(np.float32)


def _active_intensity_accumulate(
    stft: np.ndarray,
    grid: AngularGrid,
    aiv_sign: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    w = stft[0]
    components = np.stack([stft[1], stft[2], stft[3]], axis=-1)
    intensity = aiv_sign * np.real(np.conj(w)[..., None] * components).astype(np.float32)
    vectors = intensity.reshape(-1, 3)
    magnitudes = np.linalg.norm(vectors, axis=1).astype(np.float32)
    valid = np.isfinite(magnitudes) & (magnitudes > EPSILON)
    flat_size = grid.num_el_bins * grid.num_az_bins

    if not np.any(valid):
        return np.zeros((grid.num_el_bins, grid.num_az_bins), dtype=np.float32), np.zeros(
            (grid.num_el_bins, grid.num_az_bins), dtype=np.float32
        ), 0.0

    valid_vectors = vectors[valid]
    valid_magnitudes = magnitudes[valid]
    unit_vectors = valid_vectors / np.maximum(valid_magnitudes[:, None], EPSILON)
    azimuth, elevation = directions_to_angles(unit_vectors)
    az_idx, el_idx = angle_to_bin(azimuth, elevation, grid)
    flat_idx = el_idx * grid.num_az_bins + az_idx

    sum_mag = np.bincount(flat_idx, weights=valid_magnitudes, minlength=flat_size).astype(np.float32)
    sum_x = np.bincount(flat_idx, weights=valid_vectors[:, 0], minlength=flat_size).astype(np.float32)
    sum_y = np.bincount(flat_idx, weights=valid_vectors[:, 1], minlength=flat_size).astype(np.float32)
    sum_z = np.bincount(flat_idx, weights=valid_vectors[:, 2], minlength=flat_size).astype(np.float32)
    sum_vec_norm = np.sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z)
    local_coherence = np.zeros(flat_size, dtype=np.float32)
    nonzero = sum_mag > EPSILON
    local_coherence[nonzero] = np.clip(sum_vec_norm[nonzero] / np.maximum(sum_mag[nonzero], EPSILON), 0.0, 1.0)

    global_vec = np.sum(valid_vectors, axis=0)
    global_coherence = float(np.linalg.norm(global_vec) / max(float(np.sum(valid_magnitudes)), EPSILON))
    return (
        sum_mag.reshape(grid.num_el_bins, grid.num_az_bins),
        local_coherence.reshape(grid.num_el_bins, grid.num_az_bins),
        float(np.clip(global_coherence, 0.0, 1.0)),
    )


def _build_window_feature_map(
    stft: np.ndarray,
    grid: AngularGrid,
    directions: np.ndarray,
    aiv_sign: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    if stft.shape[0] < 4:
        raise ValueError(f"Expected canonical WXYZ STFT with 4 channels, got {stft.shape}.")

    beam_power_raw = _beam_power_scan(stft[:4], directions)
    aiv_raw, local_coherence, global_coherence = _active_intensity_accumulate(stft[:4], grid, aiv_sign=aiv_sign)

    beam_power = _safe_normalize_map(beam_power_raw)
    aiv_score = _safe_normalize_map(aiv_raw)
    energy = _safe_normalize_map(0.5 * beam_power + 0.5 * aiv_score)

    median_energy = float(np.median(energy))
    local_contrast = _safe_normalize_map(np.maximum(energy - median_energy, 0.0))
    directional_coherence = np.clip(0.7 * local_coherence + 0.3 * global_coherence, 0.0, 1.0)
    diffuseness = np.clip(1.0 - directional_coherence * np.sqrt(np.clip(energy, 0.0, 1.0)), 0.0, 1.0)
    dp_reliability = np.clip((0.55 * energy + 0.30 * local_contrast + 0.15 * aiv_score) * (1.0 - diffuseness), 0.0, 1.0)

    feature_map = np.stack(
        [
            beam_power,
            aiv_score,
            diffuseness.astype(np.float32),
            dp_reliability.astype(np.float32),
            energy,
        ],
        axis=-1,
    ).astype(np.float32)

    peak_index = np.unravel_index(np.argmax(dp_reliability), dp_reliability.shape)
    peak_angles = bin_to_angle(peak_index[1], peak_index[0], grid, output_in_degrees=True)
    window_meta = {
        "global_aiv_coherence": global_coherence,
        "peak_el_idx": int(peak_index[0]),
        "peak_az_idx": int(peak_index[1]),
        "peak_azimuth_deg": float(peak_angles["azimuth"]),
        "peak_elevation_deg": float(peak_angles["elevation"]),
        "peak_dp_reliability": float(dp_reliability[peak_index]),
        "nonzero_aiv_bins": int(np.count_nonzero(aiv_raw > EPSILON)),
        "nonzero_beam_bins": int(np.count_nonzero(beam_power_raw > EPSILON)),
    }
    return feature_map, window_meta, aiv_raw.astype(np.float32), beam_power_raw.astype(np.float32)


def compute_directional_features(
    windows: list[WindowSTFT],
    grid: AngularGrid,
    aiv_sign: float = 1.0,
) -> DirectionalFeatureResult:
    if not windows:
        raise ValueError("No STFT windows were provided.")

    directions = grid.direction_vectors()
    feature_maps: list[np.ndarray] = []
    peak_trace: list[dict[str, Any]] = []
    aiv_hist = np.zeros((grid.num_el_bins, grid.num_az_bins), dtype=np.float32)
    beam_sum = np.zeros((grid.num_el_bins, grid.num_az_bins), dtype=np.float32)

    for window in windows:
        feature_map, window_meta, aiv_raw, beam_raw = _build_window_feature_map(
            window.stft,
            grid=grid,
            directions=directions,
            aiv_sign=aiv_sign,
        )
        window_meta.update({"window_index": window.index, "start_sec": window.start_sec, "end_sec": window.end_sec})
        feature_maps.append(feature_map)
        peak_trace.append(window_meta)
        aiv_hist += aiv_raw
        beam_sum += beam_raw

    stacked = np.stack(feature_maps, axis=0).astype(np.float32)
    beam_mean = beam_sum / max(len(windows), 1)
    metadata = {
        "window_channel_names": list(WINDOW_CHANNEL_NAMES),
        "num_windows": len(windows),
        "aiv_sign": float(aiv_sign),
        "feature_definitions": {
            "beam_power": "Normalized FOA cardioid beam power per direction and window.",
            "aiv_score": "Normalized active intensity magnitude accumulated by direction and window.",
            "diffuseness": "1 - coherence-weighted directional energy; high means ambiguous/diffuse.",
            "dp_reliability": "Analytic direct-path proxy from energy, contrast, AIV score, and low diffuseness.",
            "energy": "Normalized blend of beam power and AIV score.",
            "stability": "Added during aggregation from window-wise energy consistency.",
        },
    }
    print(
        "[DIR] window_maps="
        f"{stacked.shape} nonzero_aiv_bins={int(np.count_nonzero(aiv_hist > EPSILON))} "
        f"mean_global_aiv_coherence={np.mean([p['global_aiv_coherence'] for p in peak_trace]):.4f}"
    )
    return DirectionalFeatureResult(
        window_feature_maps=stacked,
        window_peak_trace=peak_trace,
        aiv_histogram=aiv_hist,
        beam_power_mean=beam_mean.astype(np.float32),
        metadata=metadata,
    )

