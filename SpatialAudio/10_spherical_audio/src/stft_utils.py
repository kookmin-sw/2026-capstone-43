from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WindowSTFT:
    index: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    stft: np.ndarray
    n_fft: int
    hop_length: int
    window_name: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.stft.shape)


def _analysis_window(window_name: str, n_fft: int) -> np.ndarray:
    name = window_name.lower()
    if name in {"hann", "hanning"}:
        return np.hanning(n_fft).astype(np.float32)
    if name == "hamming":
        return np.hamming(n_fft).astype(np.float32)
    if name in {"boxcar", "rect", "rectangular"}:
        return np.ones(n_fft, dtype=np.float32)
    raise ValueError(f"Unsupported STFT window: {window_name}")


def _frame_signal_1d(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}.")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError(f"n_fft and hop_length must be positive, got {n_fft}, {hop_length}.")

    if x.shape[0] < n_fft:
        pad_width = n_fft - x.shape[0]
        x = np.pad(x, (0, pad_width), mode="constant")

    n_frames = int(np.ceil((x.shape[0] - n_fft) / hop_length)) + 1
    padded_len = (n_frames - 1) * hop_length + n_fft
    if padded_len > x.shape[0]:
        x = np.pad(x, (0, padded_len - x.shape[0]), mode="constant")

    stride = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, n_fft),
        strides=(hop_length * stride, stride),
        writeable=False,
    )
    return np.asarray(frames)


def compute_multichannel_stft(
    audio: np.ndarray,
    n_fft: int = 1024,
    hop_length: int | None = None,
    window_name: str = "hann",
) -> np.ndarray:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [samples, channels], got {audio.shape}.")
    hop_length = int(hop_length or n_fft // 2)
    window = _analysis_window(window_name, n_fft)
    channel_specs = []
    for channel_idx in range(audio.shape[1]):
        frames = _frame_signal_1d(audio[:, channel_idx].astype(np.float32), n_fft, hop_length)
        framed = frames * window[None, :]
        spec = np.fft.rfft(framed, n=n_fft, axis=1)
        channel_specs.append(spec.T.astype(np.complex64))
    return np.stack(channel_specs, axis=0)


def compute_windowed_stfts(
    audio: np.ndarray,
    sample_rate: int,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    n_fft: int = 1024,
    stft_hop_length: int | None = None,
    stft_window: str = "hann",
) -> list[WindowSTFT]:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [samples, channels], got {audio.shape}.")
    if window_sec <= 0.0 or hop_sec <= 0.0:
        raise ValueError(f"window_sec and hop_sec must be positive, got {window_sec}, {hop_sec}.")

    num_samples = int(audio.shape[0])
    window_len = max(1, int(round(window_sec * sample_rate)))
    hop_len = max(1, int(round(hop_sec * sample_rate)))

    if num_samples <= window_len:
        starts = [0]
    else:
        starts = list(range(0, num_samples - window_len + 1, hop_len))
        last_start = num_samples - window_len
        if starts[-1] != last_start:
            starts.append(last_start)

    windows: list[WindowSTFT] = []
    for index, start in enumerate(starts):
        end = min(start + window_len, num_samples)
        chunk = audio[start:end]
        stft = compute_multichannel_stft(
            chunk,
            n_fft=n_fft,
            hop_length=stft_hop_length,
            window_name=stft_window,
        )
        windows.append(
            WindowSTFT(
                index=index,
                start_sample=int(start),
                end_sample=int(end),
                start_sec=float(start / sample_rate),
                end_sec=float(end / sample_rate),
                stft=stft,
                n_fft=int(n_fft),
                hop_length=int(stft_hop_length or n_fft // 2),
                window_name=stft_window,
            )
        )

    print(
        "[STFT] windows="
        f"{len(windows)} first_shape={windows[0].shape if windows else None} "
        f"n_fft={n_fft} hop={stft_hop_length or n_fft // 2}"
    )
    return windows


def stft_metadata(windows: list[WindowSTFT]) -> dict[str, Any]:
    if not windows:
        return {}
    return {
        "num_analysis_windows": len(windows),
        "first_stft_shape_channels_freq_frames": list(windows[0].shape),
        "n_fft": int(windows[0].n_fft),
        "stft_hop_length": int(windows[0].hop_length),
        "stft_window": windows[0].window_name,
        "analysis_windows": [
            {
                "index": w.index,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "stft_shape": list(w.shape),
            }
            for w in windows
        ],
    }

