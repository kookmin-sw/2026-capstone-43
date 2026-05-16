#!/usr/bin/env python3
"""
Visualize values before and after the SGMSE-style STFT preprocessing path.

This script follows the same paired preprocessing used by SpeechEnhancementDataset:

    load clean/noisy wav
    -> same crop/pad
    -> normalize_pair
    -> raw complex STFT
    -> spec_fwd
    -> real/imag 2-channel tensor

Example:
    python3 -m src.plot.plot_stft_before_after \
        --manifest data/manifest_val.csv \
        --index 0 \
        --out-dir outputs/plots/stft_before_after
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import resolve_manifest_path  # noqa: E402
from src.audio.preprocess import (  # noqa: E402
    AudioPreprocessConfig,
    complex_to_channels,
    crop_or_pad_for_train,
    load_wav,
    normalize_pair,
    spec_fwd,
    stft,
)


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            return candidate
    return path


def safe_name(value: str) -> str:
    value = str(value).strip() or "sample"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def amp_to_db(x: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    if torch.is_complex(x):
        x = x.abs()
    return to_numpy(20.0 * torch.log10(x.clamp_min(eps)))


def robust_limits(values, lower_q: float = 0.01, upper_q: float = 0.99):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    lo = float(np.quantile(values, lower_q))
    hi = float(np.quantile(values, upper_q))
    if hi <= lo:
        pad = max(abs(hi), 1.0) * 0.05
        return lo - pad, hi + pad
    return lo, hi


def symmetric_limit(values, q: float = 0.995):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    limit = float(np.quantile(np.abs(values), q))
    return max(limit, 1e-8)


def spec_extent(spec: torch.Tensor, cfg: AudioPreprocessConfig):
    frames = int(spec.shape[-1])
    max_time = max(frames - 1, 1) * cfg.hop_length / cfg.sample_rate
    return [0.0, max_time, 0.0, cfg.sample_rate / 2.0]


def add_spec_image(fig, ax, image, title, cfg, vmin=None, vmax=None, cmap="magma"):
    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=spec_extent_from_image(image, cfg),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def spec_extent_from_image(image: np.ndarray, cfg: AudioPreprocessConfig):
    frames = int(image.shape[-1])
    max_time = max(frames - 1, 1) * cfg.hop_length / cfg.sample_rate
    return [0.0, max_time, 0.0, cfg.sample_rate / 2.0]


def read_manifest(manifest_path: Path, valid_only: bool):
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if valid_only and str(row.get("valid", "1")) != "1":
                continue
            if not row.get("clean_wav") or not row.get("noisy_wav"):
                continue
            rows.append(dict(row))
    if not rows:
        raise RuntimeError(f"No usable rows found in manifest: {manifest_path}")
    return rows


def select_row(rows, index: int, sample_id: str):
    if sample_id:
        for row in rows:
            candidates = {
                str(row.get("id", "")),
                str(row.get("source_id", "")),
                str(row.get("noisy_wav", "")),
                str(row.get("clean_wav", "")),
            }
            if sample_id in candidates:
                return row
        raise RuntimeError(f"sample id/source_id/path not found: {sample_id}")

    if index < 0 or index >= len(rows):
        raise IndexError(f"index={index} out of range for {len(rows)} rows")
    return rows[index]


def load_pair(row, cfg: AudioPreprocessConfig):
    clean_path = resolve_manifest_path(row["clean_wav"])
    noisy_path = resolve_manifest_path(row["noisy_wav"])
    clean, clean_sr = load_wav(clean_path, cfg)
    noisy, noisy_sr = load_wav(noisy_path, cfg)
    if clean_sr != noisy_sr:
        raise RuntimeError(f"sample rate mismatch: clean={clean_sr}, noisy={noisy_sr}")

    # Keep paired crop alignment stable if a manifest row has a tiny length mismatch.
    min_len = min(clean.size(-1), noisy.size(-1))
    clean = clean[..., :min_len]
    noisy = noisy[..., :min_len]
    return clean, noisy, clean_path, noisy_path


def preprocess_pair_for_visuals(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    cfg: AudioPreprocessConfig,
    crop_start: int,
    random_crop: bool,
):
    if clean.size(-1) != noisy.size(-1):
        raise ValueError("clean/noisy must be length-aligned before preprocessing")

    if crop_start >= 0:
        max_start = max(0, clean.size(-1) - cfg.train_target_len)
        if crop_start > max_start:
            raise ValueError(
                f"crop_start={crop_start} exceeds max_start={max_start} "
                f"for waveform length={clean.size(-1)}"
            )
        clean_crop, start = crop_or_pad_for_train(
            clean,
            cfg,
            start=crop_start,
            random_crop=False,
        )
    else:
        clean_crop, start = crop_or_pad_for_train(
            clean,
            cfg,
            start=None,
            random_crop=random_crop,
        )

    noisy_crop, _ = crop_or_pad_for_train(
        noisy,
        cfg,
        start=int(start),
        random_crop=False,
    )

    clean_norm, noisy_norm, normfac = normalize_pair(clean_crop, noisy_crop, cfg)

    clean_raw_spec = stft(clean_norm, cfg)
    noisy_raw_spec = stft(noisy_norm, cfg)
    clean_model_spec = spec_fwd(clean_raw_spec, cfg)
    noisy_model_spec = spec_fwd(noisy_raw_spec, cfg)
    clean_2ch = complex_to_channels(clean_model_spec)
    noisy_2ch = complex_to_channels(noisy_model_spec)

    return {
        "clean_crop": clean_crop,
        "noisy_crop": noisy_crop,
        "clean_norm": clean_norm,
        "noisy_norm": noisy_norm,
        "normfac": normfac,
        "start": torch.tensor(int(start), dtype=torch.long),
        "clean_raw_spec": clean_raw_spec,
        "noisy_raw_spec": noisy_raw_spec,
        "clean_model_spec": clean_model_spec,
        "noisy_model_spec": noisy_model_spec,
        "clean_2ch": clean_2ch,
        "noisy_2ch": noisy_2ch,
    }


def plot_waveforms(clean, noisy, processed, cfg, title, out_path: Path):
    sr = cfg.sample_rate
    start = int(processed["start"].item())
    target_len = cfg.train_target_len

    clean_full = to_numpy(clean.squeeze(0))
    noisy_full = to_numpy(noisy.squeeze(0))
    clean_norm = to_numpy(processed["clean_norm"].squeeze(0))
    noisy_norm = to_numpy(processed["noisy_norm"].squeeze(0))
    residual = noisy_norm - clean_norm

    t_full = np.arange(clean_full.size) / sr
    t_crop = np.arange(clean_norm.size) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(t_full, clean_full, linewidth=0.7, alpha=0.75, label="clean full")
    axes[0].plot(t_full, noisy_full, linewidth=0.7, alpha=0.75, label="noisy full")
    axes[0].axvspan(start / sr, (start + target_len) / sr, color="#FFB703", alpha=0.22)
    axes[0].set_title(f"{title} | full waveform before STFT")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_crop, clean_norm, linewidth=0.8, label="clean cropped+normalized")
    axes[1].plot(t_crop, noisy_norm, linewidth=0.8, label="noisy cropped+normalized")
    axes[1].set_title(
        f"Model waveform before STFT | start={start}, normfac={processed['normfac'].item():.6g}"
    )
    axes[1].set_xlabel("Crop-relative time (s)")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_crop, residual, linewidth=0.8, color="#D62828", label="noisy - clean")
    axes[2].set_title("Waveform residual before STFT")
    axes[2].set_xlabel("Crop-relative time (s)")
    axes[2].set_ylabel("Residual")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_stft_magnitude(processed, cfg, title, out_path: Path):
    clean_raw_db = amp_to_db(processed["clean_raw_spec"].squeeze(0))
    noisy_raw_db = amp_to_db(processed["noisy_raw_spec"].squeeze(0))
    clean_model_db = amp_to_db(processed["clean_model_spec"].squeeze(0))
    noisy_model_db = amp_to_db(processed["noisy_model_spec"].squeeze(0))

    raw_vmin, raw_vmax = robust_limits(np.concatenate([clean_raw_db.reshape(-1), noisy_raw_db.reshape(-1)]))
    model_vmin, model_vmax = robust_limits(
        np.concatenate([clean_model_db.reshape(-1), noisy_model_db.reshape(-1)])
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    add_spec_image(
        fig,
        axes[0, 0],
        clean_raw_db,
        "clean raw STFT magnitude (dB)",
        cfg,
        vmin=raw_vmin,
        vmax=raw_vmax,
    )
    add_spec_image(
        fig,
        axes[0, 1],
        noisy_raw_db,
        "noisy raw STFT magnitude (dB)",
        cfg,
        vmin=raw_vmin,
        vmax=raw_vmax,
    )
    add_spec_image(
        fig,
        axes[1, 0],
        clean_model_db,
        "clean spec_fwd magnitude (dB)",
        cfg,
        vmin=model_vmin,
        vmax=model_vmax,
    )
    add_spec_image(
        fig,
        axes[1, 1],
        noisy_model_db,
        "noisy spec_fwd magnitude (dB)",
        cfg,
        vmin=model_vmin,
        vmax=model_vmax,
    )
    fig.suptitle(f"{title} | STFT after transform", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_real_imag_channels(processed, cfg, title, out_path: Path):
    clean_2ch = to_numpy(processed["clean_2ch"])
    noisy_2ch = to_numpy(processed["noisy_2ch"])
    limit = symmetric_limit(np.concatenate([clean_2ch.reshape(-1), noisy_2ch.reshape(-1)]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], clean_2ch[0], "clean spec_fwd real"),
        (axes[0, 1], clean_2ch[1], "clean spec_fwd imag"),
        (axes[1, 0], noisy_2ch[0], "noisy spec_fwd real"),
        (axes[1, 1], noisy_2ch[1], "noisy spec_fwd imag"),
    ]

    for ax, image, panel_title in panels:
        add_spec_image(
            fig,
            ax,
            image,
            panel_title,
            cfg,
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
        )

    fig.suptitle(f"{title} | real/imag 2-channel tensor after spec_fwd", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_value_distributions(processed, title, out_path: Path):
    clean_norm = to_numpy(processed["clean_norm"]).reshape(-1)
    noisy_norm = to_numpy(processed["noisy_norm"]).reshape(-1)
    raw_mag = torch.cat(
        [
            processed["clean_raw_spec"].abs().reshape(-1),
            processed["noisy_raw_spec"].abs().reshape(-1),
        ]
    )
    model_mag = torch.cat(
        [
            processed["clean_model_spec"].abs().reshape(-1),
            processed["noisy_model_spec"].abs().reshape(-1),
        ]
    )
    model_ri = np.concatenate(
        [
            to_numpy(processed["clean_2ch"]).reshape(-1),
            to_numpy(processed["noisy_2ch"]).reshape(-1),
        ]
    )

    raw_mag_np = to_numpy(raw_mag)
    model_mag_np = to_numpy(model_mag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].hist(clean_norm, bins=160, alpha=0.65, density=True, label="clean")
    axes[0, 0].hist(noisy_norm, bins=160, alpha=0.65, density=True, label="noisy")
    axes[0, 0].set_title("Waveform values before STFT")
    axes[0, 0].set_xlabel("normalized amplitude")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.25)

    raw_hi = max(float(np.quantile(raw_mag_np, 0.995)), 1e-8)
    axes[0, 1].hist(raw_mag_np, bins=180, range=(0.0, raw_hi), alpha=0.78, density=True)
    axes[0, 1].set_title("Raw STFT magnitude values")
    axes[0, 1].set_xlabel("|STFT|")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].grid(True, alpha=0.25)

    model_hi = max(float(np.quantile(model_mag_np, 0.995)), 1e-8)
    axes[1, 0].hist(model_mag_np, bins=180, range=(0.0, model_hi), alpha=0.78, density=True)
    axes[1, 0].set_title("spec_fwd magnitude values")
    axes[1, 0].set_xlabel("|spec_fwd(STFT)|")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].grid(True, alpha=0.25)

    ri_lim = symmetric_limit(model_ri)
    axes[1, 1].hist(model_ri, bins=180, range=(-ri_lim, ri_lim), alpha=0.78, density=True)
    axes[1, 1].set_title("spec_fwd real/imag channel values")
    axes[1, 1].set_xlabel("channel value")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"{title} | value distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def stats_for_tensor(x: torch.Tensor):
    dtype = str(x.dtype)
    shape = list(x.shape)
    values = x.abs() if torch.is_complex(x) else x
    values_np = to_numpy(values).reshape(-1)
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return {"shape": shape, "dtype": dtype, "count": 0}
    return {
        "shape": shape,
        "dtype": dtype,
        "count": int(values_np.size),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
        "mean": float(np.mean(values_np)),
        "std": float(np.std(values_np)),
        "p50": float(np.quantile(values_np, 0.50)),
        "p95": float(np.quantile(values_np, 0.95)),
        "p99": float(np.quantile(values_np, 0.99)),
    }


def write_summary(row, clean_path, noisy_path, processed, cfg, out_path: Path):
    summary = {
        "id": row.get("id", ""),
        "source_id": row.get("source_id", ""),
        "clean_wav": clean_path,
        "noisy_wav": noisy_path,
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "num_frames": cfg.num_frames,
        "target_length": cfg.train_target_len,
        "crop_start": int(processed["start"].item()),
        "normfac": float(processed["normfac"].item()),
        "spec_factor": cfg.spec_factor,
        "spec_abs_exponent": cfg.spec_abs_exponent,
        "normalize": cfg.normalize,
        "tensors": {
            "clean_wave_before_stft": stats_for_tensor(processed["clean_norm"]),
            "noisy_wave_before_stft": stats_for_tensor(processed["noisy_norm"]),
            "clean_raw_complex_stft": stats_for_tensor(processed["clean_raw_spec"]),
            "noisy_raw_complex_stft": stats_for_tensor(processed["noisy_raw_spec"]),
            "clean_spec_fwd_complex": stats_for_tensor(processed["clean_model_spec"]),
            "noisy_spec_fwd_complex": stats_for_tensor(processed["noisy_model_spec"]),
            "clean_spec_fwd_2ch_real_imag": stats_for_tensor(processed["clean_2ch"]),
            "noisy_spec_fwd_2ch_real_imag": stats_for_tensor(processed["noisy_2ch"]),
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest_val.csv")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--out-dir", default="outputs/plots/stft_before_after")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument(
        "--crop-start",
        type=int,
        default=-1,
        help="Fixed crop start sample. Negative means center crop unless --random-crop is set.",
    )

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=32640)
    parser.add_argument("--n-fft", type=int, default=510)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=510)
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--spec-factor", type=float, default=0.15)
    parser.add_argument("--spec-abs-exponent", type=float, default=0.5)
    parser.add_argument("--normalize", default="noisy", choices=["noisy", "clean", "not"])
    return parser


def main():
    args = build_argparser().parse_args()

    if args.win_length != args.n_fft:
        raise ValueError(
            f"Current preprocess.py assumes win_length == n_fft. "
            f"Got win_length={args.win_length}, n_fft={args.n_fft}"
        )

    cfg = AudioPreprocessConfig(
        sample_rate=args.target_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_frames=args.num_frames,
        center=True,
        spec_factor=args.spec_factor,
        spec_abs_exponent=args.spec_abs_exponent,
        normalize=args.normalize,
    )
    if args.target_length != cfg.train_target_len:
        raise ValueError(
            f"target_length mismatch: got {args.target_length}, "
            f"expected {cfg.train_target_len} from (num_frames - 1) * hop_length"
        )

    manifest_path = resolve_project_path(args.manifest)
    rows = read_manifest(manifest_path, valid_only=(not args.include_invalid))
    row = select_row(rows, index=args.index, sample_id=args.sample_id)

    clean, noisy, clean_path, noisy_path = load_pair(row, cfg)
    processed = preprocess_pair_for_visuals(
        clean=clean,
        noisy=noisy,
        cfg=cfg,
        crop_start=args.crop_start,
        random_crop=args.random_crop,
    )

    sample_label = row.get("id") or row.get("source_id") or f"index_{args.index:04d}"
    prefix = safe_name(sample_label)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title = str(sample_label)
    paths = {
        "waveform": out_dir / f"{prefix}_00_waveform_before_stft.png",
        "magnitude": out_dir / f"{prefix}_01_stft_magnitude_before_after_spec_fwd.png",
        "real_imag": out_dir / f"{prefix}_02_spec_fwd_real_imag_channels.png",
        "distribution": out_dir / f"{prefix}_03_value_distributions.png",
        "summary": out_dir / f"{prefix}_summary.json",
    }

    plot_waveforms(clean, noisy, processed, cfg, title, paths["waveform"])
    plot_stft_magnitude(processed, cfg, title, paths["magnitude"])
    plot_real_imag_channels(processed, cfg, title, paths["real_imag"])
    plot_value_distributions(processed, title, paths["distribution"])
    write_summary(row, clean_path, noisy_path, processed, cfg, paths["summary"])

    print("sample:", sample_label)
    print("clean:", clean_path)
    print("noisy:", noisy_path)
    print("crop_start:", int(processed["start"].item()))
    print("normfac:", float(processed["normfac"].item()))
    print("wave before STFT:", tuple(processed["noisy_norm"].shape), processed["noisy_norm"].dtype)
    print("raw complex STFT:", tuple(processed["noisy_raw_spec"].shape), processed["noisy_raw_spec"].dtype)
    print("spec_fwd complex:", tuple(processed["noisy_model_spec"].shape), processed["noisy_model_spec"].dtype)
    print("spec_fwd 2ch real/imag:", tuple(processed["noisy_2ch"].shape), processed["noisy_2ch"].dtype)
    for key, path in paths.items():
        print(f"saved {key}: {path}")


if __name__ == "__main__":
    main()
