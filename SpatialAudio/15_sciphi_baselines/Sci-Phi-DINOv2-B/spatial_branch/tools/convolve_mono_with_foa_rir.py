#!/usr/bin/env python3
"""Convolve mono dry audio with FOA RIR and export a 4-channel FOA wav."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning


def _read_wav_float(path: Path) -> tuple[int, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=WavFileWarning)
        sr, data = wavfile.read(str(path))

    if data.dtype.kind in ("i", "u"):
        max_int = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(max_int)
    else:
        data = data.astype(np.float32)
    return sr, data


def _resample_if_needed(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    gcd = np.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return scipy.signal.resample_poly(x, up, down, axis=0).astype(np.float32)


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1).astype(np.float32)


def convolve_mono_with_foa_rir(
    dry: np.ndarray,
    rir_foa: np.ndarray,
) -> np.ndarray:
    if rir_foa.ndim != 2 or rir_foa.shape[1] < 4:
        raise ValueError(f"RIR must be at least 4-channel, got shape={rir_foa.shape}")

    rir_foa = rir_foa[:, :4]
    out = []
    for ch_idx in range(4):
        y = scipy.signal.fftconvolve(dry, rir_foa[:, ch_idx], mode="full")
        out.append(y.astype(np.float32))
    return np.stack(out, axis=1)


def _normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = np.max(np.abs(x))
    if m < 1e-8:
        return x
    return (x / m * peak).astype(np.float32)


def _write_int16(path: Path, sr: int, data: np.ndarray) -> None:
    clipped = np.clip(data, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_wav", required=True, help="Mono/stereo dry source wav path.")
    parser.add_argument("--rir_wav", required=True, help="FOA RIR wav path (4ch).")
    parser.add_argument("--out_foa_wav", required=True, help="Output convolved FOA wav (4ch).")
    parser.add_argument(
        "--out_mono_wav",
        default="",
        help="Optional output mono downmix wav for quick debugging.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=24000,
        help="Target sample-rate for output wavs.",
    )
    args = parser.parse_args()

    dry_path = Path(args.dry_wav)
    rir_path = Path(args.rir_wav)
    out_foa_path = Path(args.out_foa_wav)
    out_mono_path = Path(args.out_mono_wav) if args.out_mono_wav else None

    dry_sr, dry = _read_wav_float(dry_path)
    rir_sr, rir = _read_wav_float(rir_path)

    dry = _to_mono(dry)
    dry = _resample_if_needed(dry, dry_sr, args.target_sr)
    rir = _resample_if_needed(rir, rir_sr, args.target_sr)

    out_foa = convolve_mono_with_foa_rir(dry, rir)
    out_foa = _normalize_peak(out_foa)

    out_foa_path.parent.mkdir(parents=True, exist_ok=True)
    _write_int16(out_foa_path, args.target_sr, out_foa)

    if out_mono_path is not None:
        out_mono = out_foa.mean(axis=1)
        _write_int16(out_mono_path, args.target_sr, out_mono)

    print("dry:", dry_path, "sr=", dry_sr, "->", args.target_sr)
    print("rir:", rir_path, "sr=", rir_sr, "->", args.target_sr, "shape=", rir.shape)
    print("out_foa:", out_foa_path, "shape=", out_foa.shape)
    if out_mono_path is not None:
        print("out_mono:", out_mono_path, "shape=", out_mono.shape)


if __name__ == "__main__":
    main()

