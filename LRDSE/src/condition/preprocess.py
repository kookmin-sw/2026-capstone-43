# src/condition/preprocess.py

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import math

import numpy as np
import torch


@dataclass
class ConditionPreprocessConfig:
    """
    foot_force condition preprocessing 설정.

    condition 구성:
        raw 4ch:
            F_norm = F / raw_force_scale

        diff 4ch:
            dFdt = (F[t] - F[t-1]) / (t[t] - t[t-1])
            D_norm = clip(dFdt / d_force_scale, -1, 1)
            d_force_scale=255 means dFdt in [-255, 255] maps to [-1, 1]

    output:
        cond_8ch:
            [8, condition_num_frames]

        cond_mask:
            [condition_num_frames]
            True  = real lowstate token
            False = zero padding token
    """
    raw_force_scale: float = 220.0

    # dFdt normalization denominator. 255 maps [-255, 255] to [-1, 1].
    d_force_scale: float = 255.0
    # backward compatibility용 legacy field (현재 정규화에는 사용하지 않음)
    d_force_percentile: float = 80.0

    # foot_force smoothing window (1이면 smoothing 없음)
    smooth_win: int = 1

    # 약 2.04초 * 500Hz ≈ 1020 이므로 1024 고정 길이 사용
    condition_num_frames: int = 1024

    eps: float = 1e-8


def find_lowstate_file(run_dir: str) -> str:
    run_dir = Path(run_dir)

    candidates = [
        "lowstate_segment.jsonl",
        "lowstate.jsonl",
        "lowState.jsonl",
        "lowstate.json",
        "lowState.json",
        "low_level_state.jsonl",
        "low_level_state.json",
    ]

    for name in candidates:
        path = run_dir / name
        if path.is_file():
            return str(path)

    raise FileNotFoundError(f"lowstate file not found in: {run_dir}")


def find_anchor_file(run_dir: str) -> str:
    run_dir = Path(run_dir)

    candidates = [
        "anchor_segment.json",
        "anchors.json",
        "anchor.json",
    ]

    for name in candidates:
        path = run_dir / name
        if path.is_file():
            return str(path)

    raise FileNotFoundError(f"anchor file not found in: {run_dir}")


def get_time_sec(row: dict) -> Optional[float]:
    keys_ns = [
        "clock_monotonic_ns",
        "monotonic_ns",
        "timestamp_ns",
        "time_ns",
    ]

    for key in keys_ns:
        value = row.get(key, None)
        if isinstance(value, (int, float)):
            return float(value) / 1e9

    keys_sec = [
        "time_sec",
        "timestamp_sec",
        "t_sec",
    ]

    for key in keys_sec:
        value = row.get(key, None)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def get_foot_force(row: dict) -> Optional[List[float]]:
    msg = row.get("msg", None)

    if isinstance(msg, dict):
        ff = msg.get("foot_force", None)
        if isinstance(ff, list) and len(ff) >= 4:
            return [float(ff[0]), float(ff[1]), float(ff[2]), float(ff[3])]

    ff = row.get("foot_force", None)
    if isinstance(ff, list) and len(ff) >= 4:
        return [float(ff[0]), float(ff[1]), float(ff[2]), float(ff[3])]

    return None


def load_lowstate_time_and_force(lowstate_path: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(lowstate_path)
    suffix = path.suffix.lower()

    times = []
    forces = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                row = json.loads(line)
                t = get_time_sec(row)
                ff = get_foot_force(row)

                if t is None or ff is None:
                    continue

                if not math.isfinite(t):
                    continue

                times.append(t)
                forces.append(ff)

    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(raw, dict):
            if "data" in raw:
                raw = raw["data"]
            elif "records" in raw:
                raw = raw["records"]

        if not isinstance(raw, list):
            raise ValueError(f"Unsupported lowstate json format: {lowstate_path}")

        for row in raw:
            if not isinstance(row, dict):
                continue

            t = get_time_sec(row)
            ff = get_foot_force(row)

            if t is None or ff is None:
                continue

            if not math.isfinite(t):
                continue

            times.append(t)
            forces.append(ff)

    else:
        raise ValueError(f"Unsupported lowstate extension: {lowstate_path}")

    if len(times) < 2:
        raise ValueError(f"Not enough valid lowstate rows: {lowstate_path}")

    t = np.asarray(times, dtype=np.float64)
    f = np.asarray(forces, dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    f = f[order]

    valid_dt = np.concatenate([[True], np.diff(t) > 0])
    t = t[valid_dt]
    f = f[valid_dt]

    if len(t) < 2:
        raise ValueError(f"Not enough strictly increasing lowstate timestamps: {lowstate_path}")

    return t, f


def _append_segment_boundary_anchors(
    anchor_path: str,
    sample_idx: np.ndarray,
    mono_sec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    meta_path = Path(anchor_path).with_name("segment_meta.json")
    if not meta_path.is_file():
        return sample_idx, mono_sec

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return sample_idx, mono_sec

    if not isinstance(meta, dict):
        return sample_idx, mono_sec

    extra_samples = []
    extra_times = []

    start_ns = meta.get("noise_start_clock_monotonic_ns", None)
    if isinstance(start_ns, (int, float)):
        extra_samples.append(0.0)
        extra_times.append(float(start_ns) / 1e9)

    end_ns = meta.get("noise_end_clock_monotonic_ns", None)
    duration_sec = meta.get("duration_sec", None)
    sr = meta.get("source_sr", meta.get("noise_sr", None))
    if (
        isinstance(end_ns, (int, float))
        and isinstance(duration_sec, (int, float))
        and isinstance(sr, (int, float))
        and float(duration_sec) > 0.0
        and float(sr) > 0.0
    ):
        extra_samples.append(round(float(duration_sec) * float(sr)))
        extra_times.append(float(end_ns) / 1e9)

    if not extra_samples:
        return sample_idx, mono_sec

    sample_idx = np.concatenate(
        [sample_idx, np.asarray(extra_samples, dtype=np.float64)]
    )
    mono_sec = np.concatenate(
        [mono_sec, np.asarray(extra_times, dtype=np.float64)]
    )
    return sample_idx, mono_sec


def load_anchor_sample_to_time(anchor_path: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = json.loads(Path(anchor_path).read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "anchors" in raw:
        anchors = raw["anchors"]
    else:
        anchors = raw

    if not isinstance(anchors, list):
        raise ValueError(f"Unsupported anchors format: {anchor_path}")

    sample_idx = []
    mono_sec = []

    for a in anchors:
        if not isinstance(a, dict):
            continue

        s = None
        for key in ["sample_index_est", "sample_index"]:
            if key in a and isinstance(a[key], (int, float)):
                s = float(a[key])
                break

        t_ns = None
        for key in [
            "status_htstamp_clock_monotonic_ns",
            "trigger_htstamp_clock_monotonic_ns",
            "clock_monotonic_ns",
            "monotonic_ns",
        ]:
            if key in a and isinstance(a[key], (int, float)):
                t_ns = float(a[key])
                break

        if s is None or t_ns is None:
            continue

        sample_idx.append(s)
        mono_sec.append(t_ns / 1e9)

    if len(sample_idx) < 2:
        raise ValueError(f"Need at least 2 valid anchors: {anchor_path}")

    sample_idx = np.asarray(sample_idx, dtype=np.float64)
    mono_sec = np.asarray(mono_sec, dtype=np.float64)
    sample_idx, mono_sec = _append_segment_boundary_anchors(
        anchor_path=anchor_path,
        sample_idx=sample_idx,
        mono_sec=mono_sec,
    )

    order = np.argsort(sample_idx)
    sample_idx = sample_idx[order]
    mono_sec = mono_sec[order]

    sample_idx, unique_idx = np.unique(sample_idx, return_index=True)
    mono_sec = mono_sec[unique_idx]

    if len(sample_idx) < 2:
        raise ValueError(f"Need at least 2 unique anchors: {anchor_path}")

    return sample_idx, mono_sec


def sample_to_time_from_anchors(
    sample_index: float,
    anchor_sample_idx: np.ndarray,
    anchor_mono_sec: np.ndarray,
) -> float:
    return float(
        np.interp(
            float(sample_index),
            anchor_sample_idx,
            anchor_mono_sec,
        )
    )


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float64, copy=True)

    if win % 2 == 0:
        raise ValueError("smooth_win should be odd, e.g., 3 or 5")

    x = x.astype(np.float64, copy=False)
    y = np.empty_like(x)

    half = win // 2
    n = x.shape[0]

    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        y[i] = np.mean(x[s:e], axis=0)

    return y


def compute_force_and_derivative(
    t_low: np.ndarray,
    foot_force: np.ndarray,
    cfg: ConditionPreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        t_low:
            [N]

        force_norm:
            [N, 4]

        deriv_on_low:
            [N, 4]
            deriv_on_low[0] = 0
            deriv_on_low[i] = dFdt from i-1 -> i
    """
    if foot_force.ndim != 2 or foot_force.shape[1] != 4:
        raise ValueError(f"Expected foot_force shape [N, 4], got {foot_force.shape}")

    force_used = moving_average(foot_force, cfg.smooth_win)

    force_norm = force_used / max(cfg.raw_force_scale, cfg.eps)

    dt = np.diff(t_low)
    df = np.diff(force_used, axis=0)

    valid = np.isfinite(dt) & (dt > 0)

    d_fdt = np.zeros_like(df, dtype=np.float64)
    d_fdt[valid] = df[valid] / dt[valid, None]

    divisor = max(float(cfg.d_force_scale), cfg.eps)
    deriv_norm = d_fdt / divisor
    deriv_norm = np.clip(deriv_norm, -1.0, 1.0)

    deriv_on_low = np.zeros_like(force_norm, dtype=np.float64)
    deriv_on_low[1:] = deriv_norm

    return t_low, force_norm, deriv_on_low


@lru_cache(maxsize=2048)
def load_condition_source_cached(
    run_dir: str,
    raw_force_scale: float,
    d_force_scale: float,
    smooth_win: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = ConditionPreprocessConfig(
        raw_force_scale=float(raw_force_scale),
        d_force_scale=float(d_force_scale),
        smooth_win=int(smooth_win),
        eps=float(eps),
    )
    lowstate_path = find_lowstate_file(run_dir)
    anchor_path = find_anchor_file(run_dir)

    t_low, foot_force = load_lowstate_time_and_force(lowstate_path)
    anchor_sample_idx, anchor_mono_sec = load_anchor_sample_to_time(anchor_path)
    t_low, force_norm, deriv_on_low = compute_force_and_derivative(
        t_low=t_low,
        foot_force=foot_force,
        cfg=cfg,
    )
    return t_low, force_norm, deriv_on_low, anchor_sample_idx, anchor_mono_sec


@lru_cache(maxsize=2048)
def load_temp_contact_source_cached(
    run_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run_dir = str(Path(run_dir).expanduser().resolve())
    lowstate_path = find_lowstate_file(run_dir)
    anchor_path = find_anchor_file(run_dir)

    t_low, foot_force = load_lowstate_time_and_force(lowstate_path)
    anchor_sample_idx, anchor_mono_sec = load_anchor_sample_to_time(anchor_path)
    return t_low, foot_force, anchor_sample_idx, anchor_mono_sec


def build_frame_time_edges(frame_times: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if frame_times.ndim != 1:
        raise ValueError(f"Expected frame_times [T], got {frame_times.shape}")
    if frame_times.size <= 0:
        raise ValueError("frame_times must be non-empty")

    if frame_times.size == 1:
        half_step = 0.5 * eps
        return np.asarray(
            [frame_times[0] - half_step, frame_times[0] + half_step],
            dtype=np.float64,
        )

    mids = 0.5 * (frame_times[:-1] + frame_times[1:])
    first_step = max(float(frame_times[1] - frame_times[0]), eps)
    last_step = max(float(frame_times[-1] - frame_times[-2]), eps)

    edges = np.empty((frame_times.size + 1,), dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = frame_times[0] - 0.5 * first_step
    edges[-1] = frame_times[-1] + 0.5 * last_step
    return edges


def build_temp_contact_frames(
    run_dir: str,
    crop_start_sample: int,
    num_frames: int,
    hop_length: int,
    contact_threshold: float = 50.0,
    contact_lag_ms: float = 58.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build per-foot frame-wise contact-state channels aligned to audio STFT frames.

    foot_force timestamps are shifted later by contact_lag_ms before alignment.
    Output shape is [4, T]. A foot channel is 1 when any shifted lowstate sample
    inside that audio frame's monotonic-time window has force > contact_threshold.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    t_low, foot_force, anchor_sample_idx, anchor_mono_sec = (
        load_temp_contact_source_cached(str(Path(run_dir).expanduser().resolve()))
    )

    frame_samples = int(crop_start_sample) + (
        np.arange(int(num_frames), dtype=np.float64) * float(hop_length)
    )
    frame_times = np.interp(frame_samples, anchor_sample_idx, anchor_mono_sec)
    frame_edges = build_frame_time_edges(frame_times, eps=float(eps))

    shifted_times = t_low + (float(contact_lag_ms) / 1000.0)
    contact = foot_force > float(contact_threshold)
    out = np.zeros((4, int(num_frames)), dtype=np.float32)
    if shifted_times.size == 0:
        return torch.from_numpy(out)

    for frame_idx in range(int(num_frames)):
        if frame_idx == int(num_frames) - 1:
            in_frame = (
                (shifted_times >= frame_edges[frame_idx])
                & (shifted_times <= frame_edges[frame_idx + 1])
            )
        else:
            in_frame = (
                (shifted_times >= frame_edges[frame_idx])
                & (shifted_times < frame_edges[frame_idx + 1])
            )
        if np.any(in_frame):
            out[:, frame_idx] = np.any(contact[in_frame], axis=0).astype(np.float32)

    return torch.from_numpy(out)


def build_temp_contact_condition_for_train(
    run_dir: str,
    crop_start_sample: int,
    num_frames: int,
    hop_length: int,
    freq_bins: int,
    contact_threshold: float = 50.0,
    contact_lag_ms: float = 58.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    if freq_bins <= 0:
        raise ValueError(f"freq_bins must be positive, got {freq_bins}")

    frames = build_temp_contact_frames(
        run_dir=run_dir,
        crop_start_sample=crop_start_sample,
        num_frames=num_frames,
        hop_length=hop_length,
        contact_threshold=contact_threshold,
        contact_lag_ms=contact_lag_ms,
        eps=eps,
    ).float()
    return frames.view(4, 1, int(num_frames)).expand(4, int(freq_bins), int(num_frames)).contiguous()


def build_condition_tokens_from_crop_window(
    t_low: np.ndarray,
    force_norm: np.ndarray,
    deriv_on_low: np.ndarray,
    crop_start_time: float,
    crop_end_time: float,
    condition_num_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    audio crop 시간 구간 안에 들어온 lowstate sample을 condition token으로 사용.

    Returns:
        cond:
            [8, condition_num_frames]

        cond_mask:
            [condition_num_frames]
            True = real token
            False = zero padding

        cond_times:
            [condition_num_frames]
            absolute monotonic time (sec). padding 위치는 0

        real_token_count:
            crop 구간 안에서 실제로 발견된 lowstate sample 수
    """
    if condition_num_frames <= 0:
        raise ValueError(f"condition_num_frames must be positive, got {condition_num_frames}")

    valid = (
        np.isfinite(t_low)
        & (t_low >= crop_start_time)
        & (t_low <= crop_end_time)
    )

    t_win = t_low[valid]
    force_win = force_norm[valid]
    deriv_win = deriv_on_low[valid]

    real_token_count = int(len(t_win))

    cond = np.zeros((8, condition_num_frames), dtype=np.float32)
    cond_mask = np.zeros((condition_num_frames,), dtype=np.bool_)
    cond_times = np.zeros((condition_num_frames,), dtype=np.float64)

    if real_token_count == 0:
        return cond, cond_mask, cond_times, real_token_count

    token_values = np.concatenate([force_win, deriv_win], axis=1)  # [N, 8]

    if real_token_count <= condition_num_frames:
        n = real_token_count

        cond[:, :n] = token_values[:n].T.astype(np.float32)
        cond_mask[:n] = True
        cond_times[:n] = t_win[:n].astype(np.float64)

        return cond, cond_mask, cond_times, real_token_count

    # 혹시 lowstate sample이 1024개를 넘으면,
    # crop 구간을 1024개 시간 위치로 다시 보간해서 고정 길이에 맞춤.
    # 일반적으로는 2.04초 * 500Hz ≈ 1020이라 거의 안 넘을 가능성이 큼.
    query_times = np.linspace(
        crop_start_time,
        crop_end_time,
        condition_num_frames,
        endpoint=True,
        dtype=np.float64,
    )

    interp_values = np.zeros((condition_num_frames, 8), dtype=np.float64)

    for ch in range(8):
        interp_values[:, ch] = np.interp(
            query_times,
            t_win,
            token_values[:, ch],
            left=0.0,
            right=0.0,
        )

    cond[:, :] = interp_values.T.astype(np.float32)
    cond_mask[:] = True
    cond_times[:] = query_times.astype(np.float64)

    return cond, cond_mask, cond_times, real_token_count


def build_cond_10ch_with_time(
    cond_8ch: np.ndarray,
    cond_times: np.ndarray,
    cond_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build 10-channel condition:
        0..3: foot_force raw 4ch
        4..7: foot_force_diff 4ch
        8   : sin(2*pi*rel_time)
        9   : cos(2*pi*rel_time)
    Shape: [10, K]
    """
    if cond_8ch.ndim != 2 or cond_8ch.shape[0] != 8:
        raise ValueError(f"Expected cond_8ch shape [8, K], got {cond_8ch.shape}")
    if cond_times.ndim != 1 or cond_mask.ndim != 1:
        raise ValueError(
            f"Expected cond_times/cond_mask shape [K], got {cond_times.shape}/{cond_mask.shape}"
        )
    if cond_8ch.shape[1] != cond_times.shape[0] or cond_8ch.shape[1] != cond_mask.shape[0]:
        raise ValueError(
            f"Length mismatch: cond_8ch={cond_8ch.shape}, cond_times={cond_times.shape}, cond_mask={cond_mask.shape}"
        )

    k = cond_8ch.shape[1]
    raw_4ch = cond_8ch[0:4].astype(np.float32)
    diff_4ch = cond_8ch[4:8].astype(np.float32)

    sin_t = np.zeros((k,), dtype=np.float32)
    cos_t = np.zeros((k,), dtype=np.float32)

    valid_idx = np.where(cond_mask)[0]
    if valid_idx.size > 0:
        first = int(valid_idx[0])
        last = int(valid_idx[-1])
        t0 = float(cond_times[first])
        t1 = float(cond_times[last])
        denom = max(t1 - t0, eps)
        rel = (cond_times - t0) / denom
        rel = np.clip(rel, 0.0, 1.0)
        phase = (2.0 * math.pi) * rel
        sin_t = np.sin(phase).astype(np.float32)
        cos_t = np.cos(phase).astype(np.float32)
        sin_t[~cond_mask] = 0.0
        cos_t[~cond_mask] = 0.0

    cond_10ch = np.concatenate(
        [
            raw_4ch,
            diff_4ch,
            sin_t[None, :],
            cos_t[None, :],
        ],
        axis=0,
    )
    return cond_10ch


def preprocess_condition_for_train(
    run_dir: str,
    crop_start_sample: int,
    num_frames: int,
    hop_length: int,
    cfg: Optional[ConditionPreprocessConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    audio preprocess의 crop start에 맞춰 foot_force condition 생성.

    Args:
        run_dir:
            source.wav, anchor_segment.json 또는 anchors.json,
            lowstate_segment.jsonl이 들어있는 녹음 폴더

        crop_start_sample:
            audio preprocess에서 반환된 start

        num_frames:
            audio STFT frame 수. 현재 보통 256.
            condition token 수가 아님.

        hop_length:
            audio preprocess hop_length. 현재 128.

    Returns:
        cond_8ch:
            [8, 1024] 기본값

        cond_10ch:
            [10, 1024]
            channel order:
                [raw4, diff4, sin(rel_t), cos(rel_t)]

        cond_mask:
            [1024]

        cond_times:
            [1024]

        query_mono_times:
            [num_frames]
            STFT frame query용 crop-relative monotonic time (sec).
            absolute monotonic time은 audio/lowstate 정렬에만 사용하고,
            모델 time embedding에는 crop 시작 시간을 뺀 상대 시간을 사용한다.

        real_token_count:
            crop 구간 안에 실제로 들어온 lowstate sample 수
    """
    if cfg is None:
        cfg = ConditionPreprocessConfig()

    run_dir = str(Path(run_dir).expanduser().resolve())
    t_low, force_norm, deriv_on_low, anchor_sample_idx, anchor_mono_sec = (
        load_condition_source_cached(
            run_dir,
            float(cfg.raw_force_scale),
            float(cfg.d_force_scale),
            int(cfg.smooth_win),
            float(cfg.eps),
        )
    )

    crop_start_sample = int(crop_start_sample)

    # audio preprocess의 train_target_len은 (num_frames - 1) * hop_length
    # 기존 STFT frame center 기준 마지막 위치와 맞추기 위해 동일하게 사용.
    crop_end_sample = crop_start_sample + (int(num_frames) - 1) * int(hop_length)

    crop_start_time = sample_to_time_from_anchors(
        sample_index=crop_start_sample,
        anchor_sample_idx=anchor_sample_idx,
        anchor_mono_sec=anchor_mono_sec,
    )

    crop_end_time = sample_to_time_from_anchors(
        sample_index=crop_end_sample,
        anchor_sample_idx=anchor_sample_idx,
        anchor_mono_sec=anchor_mono_sec,
    )

    if crop_end_time < crop_start_time:
        crop_start_time, crop_end_time = crop_end_time, crop_start_time

    cond_np, mask_np, cond_abs_times_np, real_token_count = build_condition_tokens_from_crop_window(
        t_low=t_low,
        force_norm=force_norm,
        deriv_on_low=deriv_on_low,
        crop_start_time=crop_start_time,
        crop_end_time=crop_end_time,
        condition_num_frames=cfg.condition_num_frames,
    )

    cond_times_np = np.zeros_like(cond_abs_times_np, dtype=np.float64)
    cond_times_np[mask_np] = np.maximum(cond_abs_times_np[mask_np] - crop_start_time, 0.0)

    cond_8ch = torch.from_numpy(cond_np).float()
    cond_10ch = torch.from_numpy(
        build_cond_10ch_with_time(
            cond_8ch=cond_np,
            cond_times=cond_times_np,
            cond_mask=mask_np,
            eps=cfg.eps,
        )
    ).float()
    cond_mask = torch.from_numpy(mask_np).bool()
    cond_times = torch.from_numpy(cond_times_np).to(torch.float64)
    query_frame_sample_idx = crop_start_sample + (
        np.arange(int(num_frames), dtype=np.float64) * float(hop_length)
    )
    query_mono_times_np = np.interp(
        query_frame_sample_idx,
        anchor_sample_idx,
        anchor_mono_sec,
    ).astype(np.float64)
    query_abs_mono_times_np = query_mono_times_np
    query_mono_times_np = np.maximum(query_abs_mono_times_np - crop_start_time, 0.0)
    query_mono_times = torch.from_numpy(query_mono_times_np).to(torch.float64)

    return {
        "cond_8ch": cond_8ch,
        "cond_10ch": cond_10ch,
        "cond_mask": cond_mask,
        "cond_times": cond_times,
        "query_mono_times": query_mono_times,
        "real_token_count": torch.tensor(real_token_count, dtype=torch.long),
        "crop_start_time": torch.tensor(crop_start_time, dtype=torch.float64),
        "crop_end_time": torch.tensor(crop_end_time, dtype=torch.float64),
        "cond_abs_mono_times": torch.from_numpy(cond_abs_times_np).to(torch.float64),
        "query_abs_mono_times": torch.from_numpy(query_abs_mono_times_np).to(torch.float64),
    }
