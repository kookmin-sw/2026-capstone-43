from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FOAAudio:
    samples: np.ndarray
    sample_rate: int
    input_path: Path
    channel_order_input: str
    channel_order_canonical: str
    original_shape: tuple[int, ...]
    metadata: dict[str, Any]


def parse_channel_order(channel_order: str) -> list[str]:
    cleaned = channel_order.replace(",", "").replace(" ", "").upper()
    if len(cleaned) != 4 or set(cleaned) != {"W", "X", "Y", "Z"}:
        raise ValueError(
            "channel_order must contain exactly W, X, Y, Z once each. "
            f"Examples: WXYZ, WYZX, W,Y,Z,X. Got: {channel_order}"
        )
    return list(cleaned)


def canonicalize_foa_channels(audio: np.ndarray, channel_order: str) -> tuple[np.ndarray, dict[str, Any]]:
    order = parse_channel_order(channel_order)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape [samples, channels], got {audio.shape}.")
    if audio.shape[1] < 4:
        raise ValueError(
            "FOA input requires at least 4 channels. Mono/stereo are intentionally unsupported. "
            f"Got shape {audio.shape}."
        )

    selected = audio[:, :4]
    channel_index_by_name = {name: idx for idx, name in enumerate(order)}
    canonical_indices = [channel_index_by_name[name] for name in ["W", "X", "Y", "Z"]]
    canonical = selected[:, canonical_indices].astype(np.float32, copy=False)
    meta = {
        "channel_order_input": "".join(order),
        "channel_order_canonical": "WXYZ",
        "canonical_indices_from_input_first4": canonical_indices,
        "input_channels": int(audio.shape[1]),
        "used_first_four_channels": True,
        "ignored_extra_channels": max(0, int(audio.shape[1]) - 4),
    }
    return canonical, meta


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, always_2d=True)
        return data.astype(np.float32), int(sample_rate)
    except Exception as soundfile_error:
        try:
            from scipy.io import wavfile
        except ImportError as exc:
            raise ImportError("Install soundfile or scipy to read wav files.") from exc

        sample_rate, data = wavfile.read(path)
        if data.ndim == 1:
            data = data[:, None]
        if np.issubdtype(data.dtype, np.integer):
            max_abs = float(np.iinfo(data.dtype).max)
            data = data.astype(np.float32) / max_abs
        else:
            data = data.astype(np.float32)
        print(f"[WARN] soundfile failed for {path}: {soundfile_error}. Used scipy.io.wavfile fallback.")
        return data, int(sample_rate)


def _resample_if_needed(audio: np.ndarray, sample_rate: int, target_sample_rate: int | None) -> tuple[np.ndarray, int]:
    if target_sample_rate is None or int(target_sample_rate) == int(sample_rate):
        return audio, int(sample_rate)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError("scipy is required for --target_sr resampling.") from exc

    gcd = math.gcd(int(sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // gcd
    down = int(sample_rate) // gcd
    resampled = resample_poly(audio, up=up, down=down, axis=0).astype(np.float32)
    return resampled, int(target_sample_rate)


def load_foa_wav(
    path: str | Path,
    channel_order: str = "WXYZ",
    target_sample_rate: int | None = None,
    normalize_audio: bool = False,
) -> FOAAudio:
    path = Path(path)
    raw_audio, sample_rate = _read_wav(path)
    original_shape = tuple(raw_audio.shape)
    audio, sample_rate = _resample_if_needed(raw_audio, sample_rate, target_sample_rate)
    canonical, channel_meta = canonicalize_foa_channels(audio, channel_order=channel_order)

    peak_abs = float(np.max(np.abs(canonical))) if canonical.size else 0.0
    if normalize_audio and peak_abs > 0.0:
        canonical = canonical / peak_abs

    metadata: dict[str, Any] = {
        **channel_meta,
        "input_path": str(path),
        "sample_rate": int(sample_rate),
        "num_samples": int(canonical.shape[0]),
        "duration_sec": float(canonical.shape[0] / max(sample_rate, 1)),
        "original_shape": list(original_shape),
        "canonical_shape": list(canonical.shape),
        "peak_abs_before_optional_normalize": peak_abs,
        "normalize_audio": bool(normalize_audio),
    }
    print(
        "[FOA] loaded "
        f"path={path} shape={original_shape} sr={sample_rate} "
        f"canonical=WXYZ from order={channel_meta['channel_order_input']}"
    )
    if channel_meta["ignored_extra_channels"] > 0:
        print(f"[FOA] warning: ignored {channel_meta['ignored_extra_channels']} extra channel(s).")

    return FOAAudio(
        samples=canonical.astype(np.float32),
        sample_rate=int(sample_rate),
        input_path=path,
        channel_order_input=channel_meta["channel_order_input"],
        channel_order_canonical="WXYZ",
        original_shape=original_shape,
        metadata=metadata,
    )

