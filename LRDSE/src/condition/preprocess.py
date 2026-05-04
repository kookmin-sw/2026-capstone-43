# src/condition/preprocess.py

from dataclasses import dataclass
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
            D_norm = tanh(dFdt / d_force_scale)

    output:
        cond_8ch:
            [8, condition_num_frames]

        cond_mask:
            [condition_num_frames]
            True  = real lowstate token
            False = zero padding token
    """
    raw_force_scale: float = 220.0

    # 전체 noisy 분석 결과 p99 기준
    d_force_scale: float = 9220.325595510363

    # p99 결과가 smooth_win=1 기준이면 1 유지
    # 3-sample moving average를 쓸 거면,
    # analyze_foot_force_scale.py --smooth-win 3으로 다시 s값을 잡는 게 좋음.
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

    deriv_norm = np.tanh(d_fdt / max(cfg.d_force_scale, cfg.eps))

    deriv_on_low = np.zeros_like(force_norm, dtype=np.float64)
    deriv_on_low[1:] = deriv_norm

    return t_low, force_norm, deriv_on_low


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
            padding 위치는 0

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
    cond_times = np.zeros((condition_num_frames,), dtype=np.float32)

    if real_token_count == 0:
        return cond, cond_mask, cond_times, real_token_count

    token_values = np.concatenate([force_win, deriv_win], axis=1)  # [N, 8]

    if real_token_count <= condition_num_frames:
        n = real_token_count

        cond[:, :n] = token_values[:n].T.astype(np.float32)
        cond_mask[:n] = True
        cond_times[:n] = t_win[:n].astype(np.float32)

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
    cond_times[:] = query_times.astype(np.float32)

    return cond, cond_mask, cond_times, real_token_count


def preprocess_condition_for_train(
    run_dir: str,
    crop_start_sample: int,
    num_frames: int,
    hop_length: int,
    freq_bins: Optional[int] = None,
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

        freq_bins:
            더 이상 사용하지 않음.
            이전 코드 호환을 위해 인자만 유지.

    Returns:
        cond_8ch:
            [8, 1024] 기본값

        cond_mask:
            [1024]

        cond_times:
            [1024]

        real_token_count:
            crop 구간 안에 실제로 들어온 lowstate sample 수
    """
    if cfg is None:
        cfg = ConditionPreprocessConfig()

    lowstate_path = find_lowstate_file(run_dir)
    anchor_path = find_anchor_file(run_dir)

    t_low, foot_force = load_lowstate_time_and_force(lowstate_path)
    anchor_sample_idx, anchor_mono_sec = load_anchor_sample_to_time(anchor_path)

    t_low, force_norm, deriv_on_low = compute_force_and_derivative(
        t_low=t_low,
        foot_force=foot_force,
        cfg=cfg,
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

    cond_np, mask_np, cond_times_np, real_token_count = build_condition_tokens_from_crop_window(
        t_low=t_low,
        force_norm=force_norm,
        deriv_on_low=deriv_on_low,
        crop_start_time=crop_start_time,
        crop_end_time=crop_end_time,
        condition_num_frames=cfg.condition_num_frames,
    )

    cond_8ch = torch.from_numpy(cond_np).float()
    cond_mask = torch.from_numpy(mask_np).bool()
    cond_times = torch.from_numpy(cond_times_np).float()

    return {
        "cond_8ch": cond_8ch,
        "cond_mask": cond_mask,
        "cond_times": cond_times,
        "real_token_count": torch.tensor(real_token_count, dtype=torch.long),
        "crop_start_time": torch.tensor(crop_start_time, dtype=torch.float32),
        "crop_end_time": torch.tensor(crop_end_time, dtype=torch.float32),
    }


# 아래 함수들은 예전 channel-concat 방식에서 쓰던 것들.
# cross-attention 방식에서는 사용하지 않는 것을 권장.
def expand_condition_to_freq(
    cond_8ch: torch.Tensor,
    freq_bins: int,
) -> torch.Tensor:
    """
    Deprecated.
    cross-attention 방식에서는 cond_8ch_freq를 만들지 않는 것을 권장.
    """
    if cond_8ch.dim() != 2:
        raise ValueError(f"Expected cond_8ch shape [8, K], got {cond_8ch.shape}")

    if cond_8ch.size(0) != 8:
        raise ValueError(f"Expected 8 condition channels, got {cond_8ch.size(0)}")

    return cond_8ch.unsqueeze(1).repeat(1, freq_bins, 1)


def make_conditioned_input(
    noisy_2ch: torch.Tensor,
    cond_8ch_freq: torch.Tensor,
) -> torch.Tensor:
    """
    Deprecated.
    cross-attention 방식에서는 noisy_stft와 condition을 channel concat하지 않음.
    """
    if noisy_2ch.dim() != 3 or noisy_2ch.size(0) != 2:
        raise ValueError(f"Expected noisy_2ch shape [2, F, K], got {noisy_2ch.shape}")

    if cond_8ch_freq.dim() != 3 or cond_8ch_freq.size(0) != 8:
        raise ValueError(f"Expected cond_8ch_freq shape [8, F, K], got {cond_8ch_freq.shape}")

    if noisy_2ch.shape[1:] != cond_8ch_freq.shape[1:]:
        raise ValueError(
            f"Shape mismatch: noisy_2ch={noisy_2ch.shape}, cond={cond_8ch_freq.shape}"
        )

    return torch.cat([noisy_2ch, cond_8ch_freq], dim=0)