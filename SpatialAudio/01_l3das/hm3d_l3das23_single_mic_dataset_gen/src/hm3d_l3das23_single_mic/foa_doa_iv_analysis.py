from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, fftconvolve, stft

from .audio_renderer import _canonicalize_rir
from .config import DatasetGenerationConfig, load_config
from .manifest_io import iter_dataset_rows, row_in_fov, row_sample_id, row_scene_id, row_visibility_ratio
from .scene_loader import HabitatSceneSession
from .schemas import MicPose, SceneInfo
from .spatial_conventions import (
    FOA_CANONICAL_AXES,
    FOA_CANONICAL_CHANNEL_ORDER,
    FOA_RAW_CHANNEL_ORDER,
    local_angles_from_relative_xyz,
)
from .split_builder import discover_hm3d_scenes

LOGGER = logging.getLogger(__name__)

EPS = 1.0e-8
PROBE_DURATION_S = 0.5
SANITY_LOS_LIMIT = 8
SANITY_NLOS_LIMIT = 4
SANITY_DIRECT_WINDOW_S = 0.040
NLOS_CLUSTER_ANGLE_DEG = 20.0
EXPECTED_LOCAL_RFU_MAPPING_FROM_RAW_WYZX = "perm=1,3,2;signs=-++"


@dataclass(frozen=True)
class AnalysisOptions:
    dataset_root: Path
    config_path: Path
    mode: str
    split: str
    limit_los: int
    limit_nlos: int
    sample_ids: Optional[set[str]]
    out_dir: Path
    stft_win: int
    hop: int
    nfft: int
    energy_db_below_peak: float
    diffuseness_max: float
    beam_az_step: float
    beam_el_step: float
    probe_signals: tuple[str, ...]
    save_rendered_probes: bool


@dataclass(frozen=True)
class ChannelMapping:
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]

    @property
    def label(self) -> str:
        perm_text = ",".join(str(idx + 1) for idx in self.permutation)
        sign_text = "".join("+" if sign > 0 else "-" for sign in self.signs)
        return f"perm={perm_text};signs={sign_text}"


@dataclass
class AnalysisItem:
    item_id: str
    sample_id: str
    scene_id: str
    geometry_los: str
    source_kind: str
    signal_name: str
    gt_unit_vector: np.ndarray
    gt_azimuth_deg: float
    gt_elevation_deg: float
    waveform_path: Optional[Path] = None
    waveform: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    direct_window_s: Optional[tuple[float, float]] = None
    metadata: Optional[dict[str, Any]] = None
    notes: Optional[list[str]] = None


@dataclass
class PreparedAnalysisItem:
    item: AnalysisItem
    sample_rate: int
    waveform: np.ndarray
    freqs_hz: np.ndarray
    times_s: np.ndarray
    stft_matrix: np.ndarray
    forced_frame_mask: Optional[np.ndarray]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(payload), handle, indent=2, sort_keys=True)


def _hash_seed(*parts: str) -> int:
    digest = hashlib.blake2b("::".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _normalize_vector(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return np.full(3, np.nan, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= EPS:
        return np.full(3, np.nan, dtype=np.float64)
    return arr / norm


def local_vector_to_angles(local_xyz: Iterable[float]) -> tuple[float, float]:
    azimuth_deg, elevation_deg, _ = local_angles_from_relative_xyz(local_xyz)
    return float(azimuth_deg), float(elevation_deg)


def unit_vector_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    cos_el = math.cos(el)
    return np.array(
        [
            -cos_el * math.sin(az),
            cos_el * math.cos(az),
            math.sin(el),
        ],
        dtype=np.float64,
    )


def vector_to_angles(unit_vector: Iterable[float]) -> tuple[float, float]:
    return local_vector_to_angles(unit_vector)


def angular_error_deg(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    a = _normalize_vector(vector_a)
    b = _normalize_vector(vector_b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _circular_distance_deg(a: float, b: float) -> float:
    delta = ((float(a) - float(b) + 180.0) % 360.0) - 180.0
    return abs(delta)


def circular_median_deg(values_deg: Sequence[float]) -> float:
    if not values_deg:
        return float("nan")
    candidates = [float(value) for value in values_deg if np.isfinite(float(value))]
    if not candidates:
        return float("nan")
    return min(candidates, key=lambda candidate: sum(_circular_distance_deg(candidate, other) for other in candidates))


def _safe_percentile(values: Sequence[float], q: float) -> float:
    filtered = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if filtered.size == 0:
        return float("nan")
    return float(np.percentile(filtered, q))


def all_channel_mappings() -> list[ChannelMapping]:
    return [
        ChannelMapping(tuple(int(idx) for idx in perm), tuple(int(sign) for sign in signs))
        for perm in permutations(range(3))
        for signs in product((-1, 1), repeat=3)
    ]


def _read_multichannel_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = np.asarray(data, dtype=np.float32).T
        return waveform, int(sample_rate)
    except Exception:
        sample_rate, data = wavfile.read(path)
        samples = np.asarray(data)
        if np.issubdtype(samples.dtype, np.integer):
            max_abs = float(np.iinfo(samples.dtype).max)
            samples = samples.astype(np.float32) / max_abs
        else:
            samples = samples.astype(np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        return samples.T, int(sample_rate)


def _peak_normalize(samples: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= EPS:
        return samples.astype(np.float32)
    return (samples * (float(target_peak) / peak)).astype(np.float32)


def generate_probe_signal(signal_name: str, sample_rate: int, sample_id: str) -> np.ndarray:
    num_samples = int(round(PROBE_DURATION_S * float(sample_rate)))
    if num_samples <= 0:
        raise ValueError("probe duration produced zero samples")
    rng = np.random.default_rng(_hash_seed(sample_id, signal_name))
    if signal_name == "white":
        return _peak_normalize(rng.normal(0.0, 1.0, num_samples).astype(np.float32))
    if signal_name == "pink":
        white = rng.normal(0.0, 1.0, num_samples).astype(np.float32)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / float(sample_rate))
        scale = np.ones_like(freqs, dtype=np.float64)
        nonzero = freqs > 0.0
        scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
        pink = np.fft.irfft(spectrum * scale, n=num_samples).astype(np.float32)
        return _peak_normalize(pink)
    if signal_name == "chirp":
        time_axis = np.linspace(0.0, PROBE_DURATION_S, num_samples, endpoint=False, dtype=np.float64)
        max_freq = max(2000.0, 0.45 * float(sample_rate))
        sweep = chirp(
            time_axis,
            f0=80.0,
            f1=max_freq,
            t1=PROBE_DURATION_S,
            method="logarithmic",
        ).astype(np.float32)
        return _peak_normalize(sweep)
    raise ValueError(f"Unsupported probe signal: {signal_name}")


def _convolve_probe_with_rir(probe_signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    channels = [
        fftconvolve(np.asarray(probe_signal, dtype=np.float32), np.asarray(channel_ir, dtype=np.float32), mode="full").astype(np.float32)
        for channel_ir in np.asarray(rir, dtype=np.float32)
    ]
    return np.stack(channels, axis=0)


def estimate_direct_window_s(rir: np.ndarray, sample_rate: int) -> Optional[tuple[float, float]]:
    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim != 2 or rir.shape[0] < 1:
        return None
    w_channel = np.abs(rir[0])
    if w_channel.size == 0:
        return None
    peak = float(np.max(w_channel))
    if peak <= EPS:
        return None
    threshold = max(peak * 0.10, peak * 0.01)
    indices = np.flatnonzero(w_channel >= threshold)
    if indices.size == 0:
        return None
    start_index = int(indices[0])
    end_index = min(int(rir.shape[1]), start_index + int(round(SANITY_DIRECT_WINDOW_S * float(sample_rate))))
    return (float(start_index) / float(sample_rate), float(end_index) / float(sample_rate))


def _frame_mask_from_window(times_s: np.ndarray, window_s: Optional[tuple[float, float]]) -> Optional[np.ndarray]:
    if window_s is None:
        return None
    start_s, end_s = window_s
    return np.asarray((times_s >= float(start_s)) & (times_s <= float(end_s)), dtype=bool)


def _stft_matrix(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    win: int,
    hop: int,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = []
    freqs_hz: Optional[np.ndarray] = None
    times_s: Optional[np.ndarray] = None
    for channel in np.asarray(waveform, dtype=np.float32):
        freqs_hz, times_s, zxx = stft(
            channel,
            fs=float(sample_rate),
            nperseg=int(win),
            noverlap=max(0, int(win) - int(hop)),
            nfft=int(nfft),
            boundary=None,
            padded=False,
        )
        matrix.append(zxx)
    assert freqs_hz is not None
    assert times_s is not None
    return np.asarray(freqs_hz, dtype=np.float64), np.asarray(times_s, dtype=np.float64), np.stack(matrix, axis=0)


def _apply_mapping(stft_matrix: np.ndarray, mapping: ChannelMapping) -> tuple[np.ndarray, np.ndarray]:
    stft_matrix = np.asarray(stft_matrix)
    w_channel = stft_matrix[0]
    directional = stft_matrix[1:4]
    mapped = directional[list(mapping.permutation), ...].copy()
    mapped *= np.asarray(mapping.signs, dtype=np.float64)[:, None, None]
    return w_channel, mapped


def _compute_iv_features(stft_matrix: np.ndarray, mapping: ChannelMapping) -> dict[str, np.ndarray]:
    w_channel, directional = _apply_mapping(stft_matrix, mapping)
    x_channel, y_channel, z_channel = directional
    raw_iv = np.stack(
        [
            np.real(np.conj(w_channel) * x_channel),
            np.real(np.conj(w_channel) * y_channel),
            np.real(np.conj(w_channel) * z_channel),
        ],
        axis=0,
    )
    directional_energy_mean = (np.abs(x_channel) ** 2 + np.abs(y_channel) ** 2 + np.abs(z_channel) ** 2) / 3.0
    denom = np.abs(w_channel) ** 2 + directional_energy_mean + EPS
    normalized_iv = raw_iv / denom[None, ...]
    diffuseness = np.clip(1.0 - np.linalg.norm(normalized_iv, axis=0), 0.0, 1.0)
    total_energy = np.abs(w_channel) ** 2 + np.abs(x_channel) ** 2 + np.abs(y_channel) ** 2 + np.abs(z_channel) ** 2
    frame_energy = np.mean(np.abs(w_channel) ** 2, axis=0)
    return {
        "w_channel": w_channel,
        "directional": directional,
        "raw_iv": raw_iv,
        "normalized_iv": normalized_iv,
        "diffuseness": diffuseness,
        "total_energy": total_energy,
        "frame_energy": frame_energy,
    }


def build_tf_mask(
    stft_matrix: np.ndarray,
    mapping: ChannelMapping,
    *,
    energy_db_below_peak: float,
    diffuseness_max: float,
    forced_frame_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    features = _compute_iv_features(stft_matrix, mapping)
    frame_energy = features["frame_energy"]
    peak_energy = max(float(np.max(frame_energy)) if frame_energy.size else 0.0, EPS)
    frame_energy_db = 10.0 * np.log10((frame_energy + EPS) / peak_energy)
    selected_frames = np.asarray(frame_energy_db >= -float(energy_db_below_peak), dtype=bool)
    if forced_frame_mask is not None:
        forced_frame_mask = np.asarray(forced_frame_mask, dtype=bool)
        selected_frames &= forced_frame_mask

    total_energy = features["total_energy"]
    if np.any(selected_frames):
        selected_values = total_energy[:, selected_frames]
        energy_threshold = float(np.percentile(selected_values, 75.0))
    else:
        energy_threshold = float("inf")

    tf_mask = (total_energy >= energy_threshold) & (features["diffuseness"] <= float(diffuseness_max))
    tf_mask &= selected_frames[None, :]
    used_diffuseness_fallback = False
    if np.any(selected_frames) and not np.any(tf_mask):
        tf_mask = (total_energy >= energy_threshold) & selected_frames[None, :]
        used_diffuseness_fallback = True

    return {
        **features,
        "frame_energy_db": frame_energy_db,
        "selected_frames": selected_frames,
        "tf_mask": tf_mask,
        "energy_threshold": energy_threshold,
        "used_diffuseness_fallback": used_diffuseness_fallback,
    }


def _aggregate_frame_vectors(raw_iv: np.ndarray, tf_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(raw_iv.shape[-1])
    frame_vectors = np.full((num_frames, 3), np.nan, dtype=np.float64)
    valid_frames = np.zeros(num_frames, dtype=bool)
    for frame_index in range(num_frames):
        active = np.asarray(tf_mask[:, frame_index], dtype=bool)
        if not np.any(active):
            continue
        frame_vector = np.sum(raw_iv[:, active, frame_index], axis=1)
        frame_unit = _normalize_vector(frame_vector)
        if not np.isfinite(frame_unit).all():
            continue
        frame_vectors[frame_index] = frame_unit
        valid_frames[frame_index] = True
    return frame_vectors, valid_frames


def _beam_grid(azimuth_step_deg: float, elevation_step_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuths = np.arange(-180.0, 180.0 + 0.5 * float(azimuth_step_deg), float(azimuth_step_deg))
    elevations = np.arange(-90.0, 90.0 + 0.5 * float(elevation_step_deg), float(elevation_step_deg))
    grid_vectors = np.array(
        [
            unit_vector_from_angles(azimuth_deg, elevation_deg)
            for elevation_deg in elevations
            for azimuth_deg in azimuths
        ],
        dtype=np.float64,
    )
    return grid_vectors, azimuths, elevations


def _beam_scan_frame_vectors(
    stft_matrix: np.ndarray,
    mapping: ChannelMapping,
    tf_mask: np.ndarray,
    *,
    azimuth_step_deg: float,
    elevation_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = int(stft_matrix.shape[-1])
    beam_vectors = np.full((num_frames, 3), np.nan, dtype=np.float64)
    valid_frames = np.zeros(num_frames, dtype=bool)
    grid_vectors, _, _ = _beam_grid(azimuth_step_deg, elevation_step_deg)
    w_channel, directional = _apply_mapping(stft_matrix, mapping)
    for frame_index in range(num_frames):
        active = np.asarray(tf_mask[:, frame_index], dtype=bool)
        if not np.any(active):
            continue
        directional_frame = directional[:, active, frame_index]
        w_frame = w_channel[active, frame_index]
        projection = np.einsum("kd,df->kf", grid_vectors, directional_frame)
        beam = w_frame[None, :] + projection
        power = np.mean(np.abs(beam) ** 2, axis=1)
        if power.size == 0 or not np.isfinite(power).any():
            continue
        best_index = int(np.argmax(power))
        beam_vectors[frame_index] = grid_vectors[best_index]
        valid_frames[frame_index] = True
    return beam_vectors, valid_frames


def _vector_angles_array(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    azimuths = np.full(vectors.shape[0], np.nan, dtype=np.float64)
    elevations = np.full(vectors.shape[0], np.nan, dtype=np.float64)
    for index, vector in enumerate(vectors):
        if np.isfinite(vector).all():
            azimuths[index], elevations[index] = vector_to_angles(vector)
    return azimuths, elevations


def _frame_errors_to_gt(frame_vectors: np.ndarray, gt_unit_vector: np.ndarray) -> np.ndarray:
    errors = np.full(frame_vectors.shape[0], np.nan, dtype=np.float64)
    for index, vector in enumerate(frame_vectors):
        if np.isfinite(vector).all():
            errors[index] = angular_error_deg(vector, gt_unit_vector)
    return errors


def _frame_agreement(frame_vectors: np.ndarray, other_vectors: np.ndarray) -> np.ndarray:
    errors = np.full(frame_vectors.shape[0], np.nan, dtype=np.float64)
    for index, (vector_a, vector_b) in enumerate(zip(frame_vectors, other_vectors)):
        if np.isfinite(vector_a).all() and np.isfinite(vector_b).all():
            errors[index] = angular_error_deg(vector_a, vector_b)
    return errors


def _frame_steps(frame_vectors: np.ndarray, valid_frames: np.ndarray) -> np.ndarray:
    valid_indices = np.flatnonzero(valid_frames)
    if valid_indices.size < 2:
        return np.array([], dtype=np.float64)
    return np.asarray(
        [
            angular_error_deg(frame_vectors[left], frame_vectors[right])
            for left, right in zip(valid_indices[:-1], valid_indices[1:])
        ],
        dtype=np.float64,
    )


def _dominant_cluster_stats(frame_vectors: np.ndarray, valid_frames: np.ndarray) -> dict[str, float]:
    valid_vectors = np.asarray(frame_vectors[valid_frames], dtype=np.float64)
    if valid_vectors.size == 0:
        return {
            "cluster_center_azimuth_deg": float("nan"),
            "cluster_center_elevation_deg": float("nan"),
            "cluster_ratio_pct": float("nan"),
            "dispersion_p50_deg": float("nan"),
            "dispersion_p90_deg": float("nan"),
        }
    cluster_center = _normalize_vector(np.sum(valid_vectors, axis=0))
    if not np.isfinite(cluster_center).all():
        return {
            "cluster_center_azimuth_deg": float("nan"),
            "cluster_center_elevation_deg": float("nan"),
            "cluster_ratio_pct": float("nan"),
            "dispersion_p50_deg": float("nan"),
            "dispersion_p90_deg": float("nan"),
        }
    cluster_errors = np.asarray([angular_error_deg(vector, cluster_center) for vector in valid_vectors], dtype=np.float64)
    cluster_ratio_pct = 100.0 * float(np.mean(cluster_errors <= NLOS_CLUSTER_ANGLE_DEG))
    azimuth_deg, elevation_deg = vector_to_angles(cluster_center)
    return {
        "cluster_center_azimuth_deg": float(azimuth_deg),
        "cluster_center_elevation_deg": float(elevation_deg),
        "cluster_ratio_pct": float(cluster_ratio_pct),
        "dispersion_p50_deg": _safe_percentile(cluster_errors, 50.0),
        "dispersion_p90_deg": _safe_percentile(cluster_errors, 90.0),
    }


def _prepare_item(item: AnalysisItem, options: AnalysisOptions) -> PreparedAnalysisItem:
    if item.waveform is not None:
        waveform = np.asarray(item.waveform, dtype=np.float32)
        sample_rate = int(item.sample_rate or 48000)
    elif item.waveform_path is not None:
        waveform, sample_rate = _read_multichannel_wav(item.waveform_path)
    else:
        raise ValueError(f"Analysis item {item.item_id} does not have an audio source")

    if waveform.ndim != 2 or waveform.shape[0] != 4:
        raise ValueError(f"Expected FOA waveform with 4 channels for {item.item_id}, got shape {waveform.shape}")

    freqs_hz, times_s, stft_matrix = _stft_matrix(
        waveform,
        sample_rate,
        win=options.stft_win,
        hop=options.hop,
        nfft=options.nfft,
    )
    return PreparedAnalysisItem(
        item=item,
        sample_rate=int(sample_rate),
        waveform=waveform,
        freqs_hz=freqs_hz,
        times_s=times_s,
        stft_matrix=stft_matrix,
        forced_frame_mask=_frame_mask_from_window(times_s, item.direct_window_s),
    )


def _candidate_rows(
    dataset_root: Path,
    split: str,
    sample_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    rows = []
    for row in iter_dataset_rows(dataset_root):
        if str(row.get("split", "")).strip() != str(split).strip():
            continue
        if sample_ids is not None and row_sample_id(row) not in sample_ids:
            continue
        if str(row.get("rendering_status", "")).strip() != "success":
            continue
        if str(row.get("audio_format", "")).strip() != "foa_ambisonics_4ch":
            continue
        if str(row.get("audio_channel_layout", "")).strip() != "ambisonics":
            continue
        rows.append(row)
    rows.sort(key=lambda row: (row_scene_id(row), row_sample_id(row)))
    return rows


def _build_gt_from_row(row: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    local_xyz = row.get("source_mic_relative_position")
    if not isinstance(local_xyz, (list, tuple)) or len(local_xyz) != 3:
        raise ValueError("source_mic_relative_position must be a length-3 list")
    gt_unit_vector = _normalize_vector(local_xyz)
    if not np.isfinite(gt_unit_vector).all():
        raise ValueError("source_mic_relative_position is not a valid non-zero vector")
    azimuth_deg, elevation_deg = local_vector_to_angles(local_xyz)
    metadata_azimuth_deg = float(row.get("continuous_azimuth_deg", row.get("azimuth_deg", 0.0)))
    metadata_elevation_deg = float(row.get("continuous_elevation_deg", row.get("elevation_deg", 0.0)))
    if _circular_distance_deg(azimuth_deg, metadata_azimuth_deg) > 0.5 or abs(elevation_deg - metadata_elevation_deg) > 0.5:
        raise ValueError(
            "local-vector angles do not match metadata continuous angles "
            f"(computed=({azimuth_deg:.3f}, {elevation_deg:.3f}), "
            f"metadata=({metadata_azimuth_deg:.3f}, {metadata_elevation_deg:.3f}))"
        )
    return gt_unit_vector, float(azimuth_deg), float(elevation_deg)


def _select_diverse_rows(
    rows: Sequence[dict[str, Any]],
    limit: int,
    *,
    prefer_in_fov: bool = False,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    scored_rows = sorted(
        rows,
        key=lambda row: (
            -int(bool(row_in_fov(row))) if prefer_in_fov else 0,
            -(float(row_visibility_ratio(row) or 0.0)),
            row_scene_id(row),
            row_sample_id(row),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_scene_ids: set[str] = set()
    for row in scored_rows:
        sample_id = row_sample_id(row)
        scene_id = row_scene_id(row)
        if sample_id in seen_sample_ids or scene_id in seen_scene_ids:
            continue
        selected.append(row)
        seen_sample_ids.add(sample_id)
        seen_scene_ids.add(scene_id)
        if len(selected) >= limit:
            return selected
    for row in scored_rows:
        sample_id = row_sample_id(row)
        if sample_id in seen_sample_ids:
            continue
        selected.append(row)
        seen_sample_ids.add(sample_id)
        if len(selected) >= limit:
            break
    return selected


def _build_existing_items(
    rows: Sequence[dict[str, Any]],
    dataset_root: Path,
) -> tuple[list[AnalysisItem], list[dict[str, Any]]]:
    items: list[AnalysisItem] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        try:
            gt_unit_vector, gt_azimuth_deg, gt_elevation_deg = _build_gt_from_row(row)
            audio_relpath = str(row.get("foa_audio_path", "")).strip()
            if not audio_relpath:
                raise ValueError("missing foa_audio_path")
            audio_path = dataset_root / audio_relpath
            if not audio_path.exists():
                raise FileNotFoundError(f"foa wav not found: {audio_path}")
            sample_id = row_sample_id(row)
            items.append(
                AnalysisItem(
                    item_id=f"existing__{sample_id}",
                    sample_id=sample_id,
                    scene_id=row_scene_id(row),
                    geometry_los=str(row.get("geometry_los", "")).strip(),
                    source_kind="existing",
                    signal_name="speech",
                    gt_unit_vector=gt_unit_vector,
                    gt_azimuth_deg=gt_azimuth_deg,
                    gt_elevation_deg=gt_elevation_deg,
                    waveform_path=audio_path,
                    metadata=dict(row),
                    notes=[],
                )
            )
        except Exception as exc:
            skipped.append(
                {
                    "sample_id": row_sample_id(row),
                    "scene_id": row_scene_id(row),
                    "reason": str(exc),
                }
            )
    return items, skipped


def _build_scene_lookup(config: DatasetGenerationConfig) -> dict[str, SceneInfo]:
    return {scene.scene_id: scene for scene in discover_hm3d_scenes(config)}


def _build_mic_pose(row: dict[str, Any], config: DatasetGenerationConfig) -> MicPose:
    pose_world = row.get("mic_pose_world")
    if isinstance(pose_world, dict):
        position_world = list(pose_world.get("position_xyz", row.get("mic_world_position", [])))
        quaternion_wxyz = list(pose_world.get("quaternion_wxyz", row.get("mic_world_rotation", [])))
        yaw_rad = float(pose_world.get("yaw_rad", 0.0))
        yaw_deg = float(pose_world.get("yaw_deg", math.degrees(yaw_rad)))
    else:
        position_world = list(row.get("mic_world_position", []))
        quaternion_wxyz = list(row.get("mic_world_rotation", []))
        yaw_rad = 0.0
        yaw_deg = 0.0
    if len(position_world) != 3 or len(quaternion_wxyz) != 4:
        raise ValueError("mic pose metadata is incomplete")
    floor_point_world = list(position_world)
    floor_point_world[1] = float(position_world[1]) - float(config.sensor_rig.mic_height_m)
    return MicPose(
        mic_index=0,
        floor_point_world=[float(value) for value in floor_point_world],
        position_world=[float(value) for value in position_world],
        quaternion_wxyz=[float(value) for value in quaternion_wxyz],
        yaw_rad=float(yaw_rad),
        yaw_deg=float(yaw_deg),
    )


def build_sanity_items(
    config: DatasetGenerationConfig,
    rows: Sequence[dict[str, Any]],
    probe_signals: Sequence[str],
    *,
    save_rendered_probes: bool,
    out_dir: Path,
) -> tuple[list[AnalysisItem], list[dict[str, Any]]]:
    if not rows:
        return [], []
    scene_lookup = _build_scene_lookup(config)
    items: list[AnalysisItem] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row_sample_id(row)
        scene_id = row_scene_id(row)
        try:
            gt_unit_vector, gt_azimuth_deg, gt_elevation_deg = _build_gt_from_row(row)
            scene_info = scene_lookup.get(scene_id)
            if scene_info is None:
                raise KeyError(f"scene_id not found in config discovery: {scene_id}")
            mic_pose = _build_mic_pose(row, config)
            source_world_position = row.get("source_world_position")
            if not isinstance(source_world_position, (list, tuple)) or len(source_world_position) != 3:
                raise ValueError("missing source_world_position")
            with HabitatSceneSession(config, scene_info) as session:
                rir = _canonicalize_rir(
                    session.render_rir([float(value) for value in source_world_position], mic_pose),
                    expected_channels=int(config.audio.channel_count),
                )
            if rir.shape[0] != 4:
                raise ValueError(f"expected 4-channel FOA rir, got {rir.shape}")
            direct_window_s = estimate_direct_window_s(rir, int(config.audio.sample_rate))
            for signal_name in probe_signals:
                probe_signal = generate_probe_signal(signal_name, int(config.audio.sample_rate), sample_id)
                waveform = _peak_normalize(_convolve_probe_with_rir(probe_signal, rir))
                waveform_path: Optional[Path] = None
                if save_rendered_probes:
                    waveform_path = (
                        out_dir
                        / "samples"
                        / "sanity"
                        / f"{sample_id}__{signal_name}"
                        / f"{signal_name}.wav"
                    )
                    waveform_path.parent.mkdir(parents=True, exist_ok=True)
                    wavfile.write(
                        waveform_path,
                        int(config.audio.sample_rate),
                        waveform.T.astype(np.float32),
                    )
                items.append(
                    AnalysisItem(
                        item_id=f"sanity__{sample_id}__{signal_name}",
                        sample_id=sample_id,
                        scene_id=scene_id,
                        geometry_los=str(row.get("geometry_los", "")).strip(),
                        source_kind="sanity",
                        signal_name=signal_name,
                        gt_unit_vector=gt_unit_vector,
                        gt_azimuth_deg=gt_azimuth_deg,
                        gt_elevation_deg=gt_elevation_deg,
                        waveform_path=waveform_path,
                        waveform=waveform,
                        sample_rate=int(config.audio.sample_rate),
                        direct_window_s=direct_window_s,
                        metadata=dict(row),
                        notes=[],
                    )
                )
        except Exception as exc:
            skipped.append({"sample_id": sample_id, "scene_id": scene_id, "reason": str(exc)})
    return items, skipped


def _mapping_score_for_prepared_items(
    items: Sequence[PreparedAnalysisItem],
    mapping: ChannelMapping,
    options: AnalysisOptions,
) -> tuple[float, int]:
    pooled_errors: list[float] = []
    for prepared in items:
        if prepared.item.geometry_los != "gLOS":
            continue
        selection = build_tf_mask(
            prepared.stft_matrix,
            mapping,
            energy_db_below_peak=options.energy_db_below_peak,
            diffuseness_max=options.diffuseness_max,
            forced_frame_mask=prepared.forced_frame_mask,
        )
        frame_vectors, valid_frames = _aggregate_frame_vectors(selection["raw_iv"], selection["tf_mask"])
        errors = _frame_errors_to_gt(frame_vectors, prepared.item.gt_unit_vector)
        pooled_errors.extend(float(value) for value in errors[valid_frames] if np.isfinite(float(value)))
    if not pooled_errors:
        return float("inf"), 0
    return float(np.median(np.asarray(pooled_errors, dtype=np.float64))), len(pooled_errors)


def select_best_channel_mapping(
    sanity_items: Sequence[PreparedAnalysisItem],
    existing_los_items: Sequence[PreparedAnalysisItem],
    options: AnalysisOptions,
) -> dict[str, Any]:
    candidate_pool = [item for item in sanity_items if item.item.geometry_los == "gLOS"]
    source_label = "sanity"
    if not candidate_pool:
        candidate_pool = [item for item in existing_los_items if item.item.geometry_los == "gLOS"]
        source_label = "existing_fallback"
    if not candidate_pool:
        mapping = ChannelMapping((0, 1, 2), (1, 1, 1))
        return {
            "mapping": mapping,
            "mapping_label": mapping.label,
            "selection_source": "identity_default",
            "candidate_scores": [],
        }

    candidate_scores = []
    best_mapping = None
    best_score = float("inf")
    best_count = -1
    for mapping in all_channel_mappings():
        score, num_errors = _mapping_score_for_prepared_items(candidate_pool, mapping, options)
        candidate_scores.append(
            {
                "mapping_label": mapping.label,
                "score_median_error_deg": score,
                "num_frame_errors": int(num_errors),
            }
        )
        if num_errors > 0 and (score < best_score or (math.isclose(score, best_score) and num_errors > best_count)):
            best_mapping = mapping
            best_score = score
            best_count = num_errors
    if best_mapping is None:
        best_mapping = ChannelMapping((0, 1, 2), (1, 1, 1))
        source_label = "identity_default"
    return {
        "mapping": best_mapping,
        "mapping_label": best_mapping.label,
        "selection_source": source_label,
        "candidate_scores": candidate_scores,
    }


def _plot_selection_mask(
    prepared: PreparedAnalysisItem,
    selection: dict[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [
        float(prepared.times_s[0]) if prepared.times_s.size else 0.0,
        float(prepared.times_s[-1]) if prepared.times_s.size else 1.0,
        float(prepared.freqs_hz[0]) if prepared.freqs_hz.size else 0.0,
        float(prepared.freqs_hz[-1]) if prepared.freqs_hz.size else 1.0,
    ]
    ax.imshow(selection["tf_mask"].astype(np.float32), origin="lower", aspect="auto", extent=extent, interpolation="nearest")
    ax.set_title("Selected TF bins")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if prepared.item.direct_window_s is not None:
        ax.axvline(prepared.item.direct_window_s[0], color="white", linestyle="--", linewidth=1.0)
        ax.axvline(prepared.item.direct_window_s[1], color="white", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_direction_trace(
    prepared: PreparedAnalysisItem,
    iv_azimuth_deg: np.ndarray,
    iv_elevation_deg: np.ndarray,
    beam_azimuth_deg: np.ndarray,
    beam_elevation_deg: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(prepared.times_s, iv_azimuth_deg, label="IV azimuth", linewidth=1.2)
    axes[0].plot(prepared.times_s, beam_azimuth_deg, label="Beam azimuth", linewidth=1.0, alpha=0.8)
    axes[0].axhline(prepared.item.gt_azimuth_deg, color="black", linestyle="--", label="GT azimuth")
    axes[0].set_ylabel("Azimuth (deg)")
    axes[0].legend(loc="upper right")

    axes[1].plot(prepared.times_s, iv_elevation_deg, label="IV elevation", linewidth=1.2)
    axes[1].plot(prepared.times_s, beam_elevation_deg, label="Beam elevation", linewidth=1.0, alpha=0.8)
    axes[1].axhline(prepared.item.gt_elevation_deg, color="black", linestyle="--", label="GT elevation")
    axes[1].set_ylabel("Elevation (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_beam_vs_iv(
    prepared: PreparedAnalysisItem,
    iv_error_deg: np.ndarray,
    beam_error_deg: np.ndarray,
    iv_beam_agreement_deg: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(prepared.times_s, iv_error_deg, label="IV vs GT", linewidth=1.2)
    axes[0].plot(prepared.times_s, beam_error_deg, label="Beam vs GT", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Angular error (deg)")
    axes[0].legend(loc="upper right")
    axes[1].plot(prepared.times_s, iv_beam_agreement_deg, label="IV vs Beam", linewidth=1.2)
    axes[1].set_ylabel("Agreement (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_frame_metrics(
    output_path: Path,
    prepared: PreparedAnalysisItem,
    selection: dict[str, Any],
    iv_vectors: np.ndarray,
    beam_vectors: np.ndarray,
    iv_error_deg: np.ndarray,
    beam_error_deg: np.ndarray,
    iv_beam_agreement_deg: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iv_azimuth_deg, iv_elevation_deg = _vector_angles_array(iv_vectors)
    beam_azimuth_deg, beam_elevation_deg = _vector_angles_array(beam_vectors)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "time_s",
                "selected_frame",
                "frame_energy_db",
                "iv_x",
                "iv_y",
                "iv_z",
                "iv_azimuth_deg",
                "iv_elevation_deg",
                "beam_x",
                "beam_y",
                "beam_z",
                "beam_azimuth_deg",
                "beam_elevation_deg",
                "iv_error_deg",
                "beam_error_deg",
                "iv_beam_agreement_deg",
            ],
        )
        writer.writeheader()
        for index, time_s in enumerate(prepared.times_s.tolist()):
            writer.writerow(
                {
                    "frame_index": index,
                    "time_s": float(time_s),
                    "selected_frame": bool(selection["selected_frames"][index]),
                    "frame_energy_db": float(selection["frame_energy_db"][index]),
                    "iv_x": float(iv_vectors[index, 0]) if np.isfinite(iv_vectors[index, 0]) else "",
                    "iv_y": float(iv_vectors[index, 1]) if np.isfinite(iv_vectors[index, 1]) else "",
                    "iv_z": float(iv_vectors[index, 2]) if np.isfinite(iv_vectors[index, 2]) else "",
                    "iv_azimuth_deg": float(iv_azimuth_deg[index]) if np.isfinite(iv_azimuth_deg[index]) else "",
                    "iv_elevation_deg": float(iv_elevation_deg[index]) if np.isfinite(iv_elevation_deg[index]) else "",
                    "beam_x": float(beam_vectors[index, 0]) if np.isfinite(beam_vectors[index, 0]) else "",
                    "beam_y": float(beam_vectors[index, 1]) if np.isfinite(beam_vectors[index, 1]) else "",
                    "beam_z": float(beam_vectors[index, 2]) if np.isfinite(beam_vectors[index, 2]) else "",
                    "beam_azimuth_deg": float(beam_azimuth_deg[index]) if np.isfinite(beam_azimuth_deg[index]) else "",
                    "beam_elevation_deg": float(beam_elevation_deg[index]) if np.isfinite(beam_elevation_deg[index]) else "",
                    "iv_error_deg": float(iv_error_deg[index]) if np.isfinite(iv_error_deg[index]) else "",
                    "beam_error_deg": float(beam_error_deg[index]) if np.isfinite(beam_error_deg[index]) else "",
                    "iv_beam_agreement_deg": float(iv_beam_agreement_deg[index]) if np.isfinite(iv_beam_agreement_deg[index]) else "",
                }
            )


def analyze_prepared_item(
    prepared: PreparedAnalysisItem,
    mapping: ChannelMapping,
    options: AnalysisOptions,
    *,
    output_root: Path,
) -> dict[str, Any]:
    selection = build_tf_mask(
        prepared.stft_matrix,
        mapping,
        energy_db_below_peak=options.energy_db_below_peak,
        diffuseness_max=options.diffuseness_max,
        forced_frame_mask=prepared.forced_frame_mask,
    )
    iv_vectors, iv_valid_frames = _aggregate_frame_vectors(selection["raw_iv"], selection["tf_mask"])
    beam_vectors, beam_valid_frames = _beam_scan_frame_vectors(
        prepared.stft_matrix,
        mapping,
        selection["tf_mask"],
        azimuth_step_deg=options.beam_az_step,
        elevation_step_deg=options.beam_el_step,
    )
    iv_error_deg = _frame_errors_to_gt(iv_vectors, prepared.item.gt_unit_vector)
    beam_error_deg = _frame_errors_to_gt(beam_vectors, prepared.item.gt_unit_vector)
    iv_beam_agreement_deg = _frame_agreement(iv_vectors, beam_vectors)
    iv_steps_deg = _frame_steps(iv_vectors, iv_valid_frames)
    valid_iv_errors = np.asarray([value for value in iv_error_deg if np.isfinite(value)], dtype=np.float64)
    valid_beam_errors = np.asarray([value for value in beam_error_deg if np.isfinite(value)], dtype=np.float64)
    valid_agreements = np.asarray([value for value in iv_beam_agreement_deg if np.isfinite(value)], dtype=np.float64)
    iv_azimuth_deg, iv_elevation_deg = _vector_angles_array(iv_vectors)
    beam_azimuth_deg, beam_elevation_deg = _vector_angles_array(beam_vectors)

    summary: dict[str, Any] = {
        "item_id": prepared.item.item_id,
        "sample_id": prepared.item.sample_id,
        "scene_id": prepared.item.scene_id,
        "source_kind": prepared.item.source_kind,
        "signal_name": prepared.item.signal_name,
        "geometry_los": prepared.item.geometry_los,
        "mapping_label": mapping.label,
        "sample_rate": int(prepared.sample_rate),
        "num_frames": int(prepared.times_s.size),
        "selected_frame_count": int(np.count_nonzero(selection["selected_frames"])),
        "selected_tf_bin_count": int(np.count_nonzero(selection["tf_mask"])),
        "valid_iv_frame_count": int(np.count_nonzero(iv_valid_frames)),
        "valid_beam_frame_count": int(np.count_nonzero(beam_valid_frames)),
        "used_diffuseness_fallback": bool(selection["used_diffuseness_fallback"]),
        "gt_azimuth_deg": float(prepared.item.gt_azimuth_deg),
        "gt_elevation_deg": float(prepared.item.gt_elevation_deg),
        "median_angular_error_deg": _safe_percentile(valid_iv_errors, 50.0),
        "mean_angular_error_deg": float(np.mean(valid_iv_errors)) if valid_iv_errors.size else float("nan"),
        "p90_angular_error_deg": _safe_percentile(valid_iv_errors, 90.0),
        "median_beam_error_deg": _safe_percentile(valid_beam_errors, 50.0),
        "median_iv_beam_agreement_deg": _safe_percentile(valid_agreements, 50.0),
        "mean_iv_beam_agreement_deg": float(np.mean(valid_agreements)) if valid_agreements.size else float("nan"),
        "frame_to_frame_direction_variance_deg2": float(np.var(iv_steps_deg)) if iv_steps_deg.size else float("nan"),
        "threshold_acc_lt_5_deg_pct": 100.0 * float(np.mean(valid_iv_errors < 5.0)) if valid_iv_errors.size else float("nan"),
        "threshold_acc_lt_10_deg_pct": 100.0 * float(np.mean(valid_iv_errors < 10.0)) if valid_iv_errors.size else float("nan"),
        "threshold_acc_lt_20_deg_pct": 100.0 * float(np.mean(valid_iv_errors < 20.0)) if valid_iv_errors.size else float("nan"),
        "status": "ok" if np.count_nonzero(iv_valid_frames) > 0 else "no_valid_frames",
    }

    if prepared.item.geometry_los == "gNLOS":
        cluster_stats = _dominant_cluster_stats(iv_vectors, iv_valid_frames)
        summary.update(cluster_stats)
        summary["azimuth_circular_median_deg"] = circular_median_deg(
            [value for value in iv_azimuth_deg.tolist() if np.isfinite(value)]
        )
        summary["elevation_median_deg"] = _safe_percentile(
            [value for value in iv_elevation_deg.tolist() if np.isfinite(value)],
            50.0,
        )

    sample_dir = output_root / "samples" / prepared.item.source_kind / prepared.item.item_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_frame_metrics(
        sample_dir / "frame_metrics.csv",
        prepared,
        selection,
        iv_vectors,
        beam_vectors,
        iv_error_deg,
        beam_error_deg,
        iv_beam_agreement_deg,
    )
    _plot_selection_mask(prepared, selection, sample_dir / "selection_mask.png")
    _plot_direction_trace(
        prepared,
        iv_azimuth_deg,
        iv_elevation_deg,
        beam_azimuth_deg,
        beam_elevation_deg,
        sample_dir / "direction_trace.png",
    )
    _plot_beam_vs_iv(
        prepared,
        iv_error_deg,
        beam_error_deg,
        iv_beam_agreement_deg,
        sample_dir / "beam_vs_iv.png",
    )
    _write_json(sample_dir / "summary.json", summary)
    return summary


def _write_sample_metrics_csv(output_path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for summary in summaries for key in summary.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _to_builtin(summary.get(key, "")) for key in fieldnames})


def _summaries_for_group(
    summaries: Sequence[dict[str, Any]],
    *,
    source_kind: Optional[str] = None,
    geometry_los: Optional[str] = None,
) -> list[dict[str, Any]]:
    selected = []
    for summary in summaries:
        if source_kind is not None and str(summary.get("source_kind")) != str(source_kind):
            continue
        if geometry_los is not None and str(summary.get("geometry_los")) != str(geometry_los):
            continue
        selected.append(summary)
    return selected


def _group_stat(values: Sequence[float], reducer: str) -> float:
    filtered = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if filtered.size == 0:
        return float("nan")
    if reducer == "median":
        return float(np.median(filtered))
    if reducer == "mean":
        return float(np.mean(filtered))
    raise ValueError(f"Unsupported reducer: {reducer}")


def _build_aggregate_summary(
    summaries: Sequence[dict[str, Any]],
    mapping_info: dict[str, Any],
    skipped_existing: Sequence[dict[str, Any]],
    skipped_sanity: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    groups = {
        "existing_gLOS": _summaries_for_group(summaries, source_kind="existing", geometry_los="gLOS"),
        "existing_gNLOS": _summaries_for_group(summaries, source_kind="existing", geometry_los="gNLOS"),
        "sanity_gLOS": _summaries_for_group(summaries, source_kind="sanity", geometry_los="gLOS"),
        "sanity_gNLOS": _summaries_for_group(summaries, source_kind="sanity", geometry_los="gNLOS"),
    }
    summary = {
        "selected_mapping": mapping_info.get("mapping_label"),
        "mapping_selection_source": mapping_info.get("selection_source"),
        "mapping_candidate_scores": mapping_info.get("candidate_scores", []),
        "declared_foa_raw_channel_order": FOA_RAW_CHANNEL_ORDER,
        "declared_foa_canonical_channel_order": FOA_CANONICAL_CHANNEL_ORDER,
        "declared_foa_canonical_axes": FOA_CANONICAL_AXES,
        "expected_local_rfu_mapping_from_raw_wyzx": EXPECTED_LOCAL_RFU_MAPPING_FROM_RAW_WYZX,
        "selected_mapping_matches_declared_raw_wyzx": (
            str(mapping_info.get("mapping_label")) == EXPECTED_LOCAL_RFU_MAPPING_FROM_RAW_WYZX
        ),
        "num_items_analyzed": len(summaries),
        "num_items_skipped_existing": len(skipped_existing),
        "num_items_skipped_sanity": len(skipped_sanity),
        "skipped_existing": list(skipped_existing),
        "skipped_sanity": list(skipped_sanity),
        "group_counts": {key: len(value) for key, value in groups.items()},
        "los_existing_median_of_medians_deg": _group_stat(
            [summary["median_angular_error_deg"] for summary in groups["existing_gLOS"]],
            "median",
        ),
        "los_sanity_median_of_medians_deg": _group_stat(
            [summary["median_angular_error_deg"] for summary in groups["sanity_gLOS"]],
            "median",
        ),
        "nlos_existing_cluster_ratio_pct_median": _group_stat(
            [summary.get("cluster_ratio_pct", float("nan")) for summary in groups["existing_gNLOS"]],
            "median",
        ),
        "nlos_sanity_cluster_ratio_pct_median": _group_stat(
            [summary.get("cluster_ratio_pct", float("nan")) for summary in groups["sanity_gNLOS"]],
            "median",
        ),
    }
    summary["los_sanity_goal_pass"] = bool(
        np.isfinite(float(summary["los_sanity_median_of_medians_deg"]))
        and float(summary["los_sanity_median_of_medians_deg"]) <= 10.0
    )
    return summary


def _write_aggregate_markdown(output_path: Path, aggregate_summary: dict[str, Any], summaries: Sequence[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FOA DOA/IV Sanity Report",
        "",
        f"- Selected mapping: `{aggregate_summary.get('selected_mapping', '-')}`",
        f"- Mapping source: `{aggregate_summary.get('mapping_selection_source', '-')}`",
        f"- Analyzed items: `{aggregate_summary.get('num_items_analyzed', 0)}`",
        f"- Skipped existing items: `{aggregate_summary.get('num_items_skipped_existing', 0)}`",
        f"- Skipped sanity items: `{aggregate_summary.get('num_items_skipped_sanity', 0)}`",
        "",
        "## Aggregate Metrics",
        "",
        f"- Existing LOS median-of-medians: `{aggregate_summary.get('los_existing_median_of_medians_deg', float('nan')):.3f}` deg",
        f"- Sanity LOS median-of-medians: `{aggregate_summary.get('los_sanity_median_of_medians_deg', float('nan')):.3f}` deg",
        f"- Existing NLOS cluster ratio median: `{aggregate_summary.get('nlos_existing_cluster_ratio_pct_median', float('nan')):.3f}` %",
        f"- Sanity NLOS cluster ratio median: `{aggregate_summary.get('nlos_sanity_cluster_ratio_pct_median', float('nan')):.3f}` %",
        "",
        "## Item Summary",
        "",
        "| item_id | kind | signal | geometry | median error | p90 error | cluster ratio | status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary.get("item_id", "")),
                    str(summary.get("source_kind", "")),
                    str(summary.get("signal_name", "")),
                    str(summary.get("geometry_los", "")),
                    f"{float(summary.get('median_angular_error_deg', float('nan'))):.3f}",
                    f"{float(summary.get('p90_angular_error_deg', float('nan'))):.3f}",
                    f"{float(summary.get('cluster_ratio_pct', float('nan'))):.3f}",
                    str(summary.get("status", "")),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(options: AnalysisOptions) -> dict[str, Any]:
    config = load_config(options.config_path)
    rows = _candidate_rows(options.dataset_root, options.split, options.sample_ids)
    los_rows = [row for row in rows if str(row.get("geometry_los", "")).strip() == "gLOS"]
    nlos_rows = [row for row in rows if str(row.get("geometry_los", "")).strip() == "gNLOS"]

    existing_items: list[AnalysisItem] = []
    skipped_existing: list[dict[str, Any]] = []
    if options.mode in {"existing", "both"}:
        selected_existing_rows = _select_diverse_rows(los_rows, options.limit_los, prefer_in_fov=True) + _select_diverse_rows(
            nlos_rows,
            options.limit_nlos,
        )
        existing_items, skipped_existing = _build_existing_items(selected_existing_rows, options.dataset_root)

    sanity_items: list[AnalysisItem] = []
    skipped_sanity: list[dict[str, Any]] = []
    if options.mode in {"sanity", "both"}:
        selected_sanity_rows = _select_diverse_rows(los_rows, SANITY_LOS_LIMIT, prefer_in_fov=True) + _select_diverse_rows(
            nlos_rows,
            SANITY_NLOS_LIMIT,
        )
        sanity_items, skipped_sanity = build_sanity_items(
            config,
            selected_sanity_rows,
            options.probe_signals,
            save_rendered_probes=options.save_rendered_probes,
            out_dir=options.out_dir,
        )

    prepared_existing_items = [_prepare_item(item, options) for item in existing_items]
    prepared_sanity_items = [_prepare_item(item, options) for item in sanity_items]

    mapping_info = select_best_channel_mapping(prepared_sanity_items, prepared_existing_items, options)
    mapping = mapping_info["mapping"]
    summaries = []
    for prepared in [*prepared_existing_items, *prepared_sanity_items]:
        summaries.append(analyze_prepared_item(prepared, mapping, options, output_root=options.out_dir))

    aggregate_summary = _build_aggregate_summary(summaries, mapping_info, skipped_existing, skipped_sanity)
    aggregate_summary["selected_mapping"] = mapping.label
    aggregate_summary["analysis_mode"] = options.mode
    aggregate_summary["split"] = options.split
    aggregate_summary["probe_signals"] = list(options.probe_signals)

    _write_json(options.out_dir / "aggregate_summary.json", aggregate_summary)
    _write_aggregate_markdown(options.out_dir / "aggregate_summary.md", aggregate_summary, summaries)
    _write_sample_metrics_csv(options.out_dir / "sample_metrics.csv", summaries)
    return aggregate_summary
