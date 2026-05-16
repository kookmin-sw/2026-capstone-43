#!/usr/bin/env python3
"""
Compare clean, noisy, and enhanced waveforms before/after STFT preprocessing.

Typical usage with a generated sample:
    python3 -m src.plot.plot_clean_noisy_enhanced \
        --manifest data/manifest_val.csv \
        --index 0 \
        --enhanced-wav checkpoints/.../samples/step_XXXX/00_xxx_enhanced_full.wav \
        --out-dir outputs/plots/clean_noisy_enhanced

You can also provide all three wav paths directly:
    python3 -m src.plot.plot_clean_noisy_enhanced \
        --clean-wav clean.flac \
        --noisy-wav noisy.wav \
        --enhanced-wav enhanced.wav
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
    crop_or_pad_for_train,
    load_wav,
    normalize_noisy,
    spec_fwd,
    stft,
)


COLORS = {
    "clean": "#2A9D8F",
    "noisy": "#E76F51",
    "enhanced": "#4361EE",
    "noisy_error": "#F4A261",
    "enhanced_error": "#3A0CA3",
}


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


def read_manifest(manifest_path: Path, valid_only: bool):
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if valid_only and str(row.get("valid", "1")) != "1":
                continue
            if row.get("clean_wav") and row.get("noisy_wav"):
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


def load_three_wavs(args, cfg: AudioPreprocessConfig):
    row = None
    clean_path = args.clean_wav
    noisy_path = args.noisy_wav

    if not clean_path or not noisy_path:
        manifest_path = resolve_project_path(args.manifest)
        rows = read_manifest(manifest_path, valid_only=(not args.include_invalid))
        row = select_row(rows, index=args.index, sample_id=args.sample_id)
        clean_path = resolve_manifest_path(row["clean_wav"])
        noisy_path = resolve_manifest_path(row["noisy_wav"])
    else:
        clean_path = str(resolve_project_path(clean_path))
        noisy_path = str(resolve_project_path(noisy_path))

    enhanced_path = str(resolve_project_path(args.enhanced_wav))
    if not Path(enhanced_path).exists():
        raise FileNotFoundError(f"enhanced_wav not found: {enhanced_path}")

    clean, clean_sr = load_wav(clean_path, cfg)
    noisy, noisy_sr = load_wav(noisy_path, cfg)
    enhanced, enhanced_sr = load_wav(enhanced_path, cfg)
    if len({clean_sr, noisy_sr, enhanced_sr}) != 1:
        raise RuntimeError(
            f"sample rate mismatch: clean={clean_sr}, noisy={noisy_sr}, enhanced={enhanced_sr}"
        )

    return {
        "row": row,
        "clean_path": clean_path,
        "noisy_path": noisy_path,
        "enhanced_path": enhanced_path,
        "clean": clean,
        "noisy": noisy,
        "enhanced": enhanced,
    }


def align_and_prepare(wavs, cfg: AudioPreprocessConfig, crop_start: int, random_crop: bool, full: bool):
    clean = wavs["clean"]
    noisy = wavs["noisy"]
    enhanced = wavs["enhanced"]

    min_len = min(clean.size(-1), noisy.size(-1), enhanced.size(-1))
    clean = clean[..., :min_len]
    noisy = noisy[..., :min_len]
    enhanced = enhanced[..., :min_len]

    if full:
        clean_used = clean
        noisy_used = noisy
        enhanced_used = enhanced
        start = 0
    else:
        if crop_start >= 0:
            max_start = max(0, min_len - cfg.train_target_len)
            if crop_start > max_start:
                raise ValueError(
                    f"crop_start={crop_start} exceeds max_start={max_start} "
                    f"for aligned length={min_len}"
                )
            clean_used, start = crop_or_pad_for_train(
                clean,
                cfg,
                start=crop_start,
                random_crop=False,
            )
        else:
            clean_used, start = crop_or_pad_for_train(
                clean,
                cfg,
                start=None,
                random_crop=random_crop,
            )
        noisy_used, _ = crop_or_pad_for_train(noisy, cfg, start=int(start), random_crop=False)
        enhanced_used, _ = crop_or_pad_for_train(enhanced, cfg, start=int(start), random_crop=False)

    noisy_norm, normfac = normalize_noisy(noisy_used, cfg)
    clean_norm = clean_used / normfac
    enhanced_norm = enhanced_used / normfac

    specs = {}
    for name, wav in {
        "clean": clean_norm,
        "noisy": noisy_norm,
        "enhanced": enhanced_norm,
    }.items():
        raw_spec = stft(wav, cfg)
        model_spec = spec_fwd(raw_spec, cfg)
        specs[f"{name}_raw_spec"] = raw_spec
        specs[f"{name}_model_spec"] = model_spec

    return {
        "aligned_len": int(min_len),
        "start": int(start),
        "normfac": normfac,
        "clean": clean,
        "noisy": noisy,
        "enhanced": enhanced,
        "clean_used": clean_used,
        "noisy_used": noisy_used,
        "enhanced_used": enhanced_used,
        "clean_norm": clean_norm,
        "noisy_norm": noisy_norm,
        "enhanced_norm": enhanced_norm,
        **specs,
    }


def db_image(spec: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    return to_numpy(20.0 * torch.log10(spec.abs().squeeze(0).clamp_min(eps)))


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


def spec_extent(image: np.ndarray, cfg: AudioPreprocessConfig):
    frames = int(image.shape[-1])
    max_time = max(frames - 1, 1) * cfg.hop_length / cfg.sample_rate
    return [0.0, max_time, 0.0, cfg.sample_rate / 2.0]


def add_spec_image(fig, ax, image, title, cfg, vmin=None, vmax=None, cmap="magma"):
    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=spec_extent(image, cfg),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    estimate = estimate.reshape(-1).float()
    reference = reference.reshape(-1).float()
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    ref_energy = torch.sum(reference * reference).clamp_min(eps)
    projection = torch.sum(estimate * reference) * reference / ref_energy
    noise = estimate - projection
    ratio = torch.sum(projection * projection) / torch.sum(noise * noise).clamp_min(eps)
    return float(10.0 * torch.log10(ratio.clamp_min(eps)).item())


def signal_metrics(signal: torch.Tensor, clean: torch.Tensor):
    signal = signal.reshape(-1).float()
    clean = clean.reshape(-1).float()
    err = signal - clean
    mse = torch.mean(err.pow(2)).item()
    mae = torch.mean(err.abs()).item()
    clean_power = torch.mean(clean.pow(2)).clamp_min(1e-8)
    err_power = torch.mean(err.pow(2)).clamp_min(1e-8)
    snr = 10.0 * torch.log10(clean_power / err_power).item()
    return {
        "mse": float(mse),
        "mae": float(mae),
        "snr_db": float(snr),
        "si_sdr_db": si_sdr(signal, clean),
    }


def plot_waveform_comparison(processed, cfg, title, out_path: Path):
    clean = to_numpy(processed["clean_norm"].squeeze(0))
    noisy = to_numpy(processed["noisy_norm"].squeeze(0))
    enhanced = to_numpy(processed["enhanced_norm"].squeeze(0))
    noisy_err = noisy - clean
    enhanced_err = enhanced - clean
    t = np.arange(clean.size) / cfg.sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

    axes[0].plot(t, clean, linewidth=0.75, color=COLORS["clean"], label="clean")
    axes[0].plot(t, noisy, linewidth=0.65, color=COLORS["noisy"], alpha=0.72, label="noisy")
    axes[0].plot(t, enhanced, linewidth=0.75, color=COLORS["enhanced"], alpha=0.82, label="enhanced")
    axes[0].set_title(
        f"{title} | waveform before STFT | start={processed['start']} "
        f"| normfac={processed['normfac'].item():.6g}"
    )
    axes[0].set_ylabel("normalized amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, noisy_err, linewidth=0.75, color=COLORS["noisy_error"], label="noisy - clean")
    axes[1].plot(t, enhanced_err, linewidth=0.75, color=COLORS["enhanced_error"], label="enhanced - clean")
    axes[1].set_title("Time-domain error against clean")
    axes[1].set_ylabel("error")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, np.abs(noisy_err), linewidth=0.75, color=COLORS["noisy_error"], label="|noisy - clean|")
    axes[2].plot(
        t,
        np.abs(enhanced_err),
        linewidth=0.75,
        color=COLORS["enhanced_error"],
        label="|enhanced - clean|",
    )
    axes[2].set_title("Absolute error against clean")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("absolute error")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_magnitude_triplet(processed, cfg, title, out_path: Path, spec_kind: str):
    images = {
        "clean": db_image(processed[f"clean_{spec_kind}_spec"]),
        "noisy": db_image(processed[f"noisy_{spec_kind}_spec"]),
        "enhanced": db_image(processed[f"enhanced_{spec_kind}_spec"]),
    }
    vmin, vmax = robust_limits(np.concatenate([v.reshape(-1) for v in images.values()]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, sharey=True)
    for ax, name in zip(axes, ["clean", "noisy", "enhanced"]):
        add_spec_image(
            fig,
            ax,
            images[name],
            f"{name} {spec_kind} STFT magnitude (dB)",
            cfg,
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle(f"{title} | {spec_kind} STFT magnitude comparison", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_error_spectrograms(processed, cfg, title, out_path: Path):
    noisy_err = (processed["noisy_model_spec"] - processed["clean_model_spec"]).abs().squeeze(0)
    enhanced_err = (processed["enhanced_model_spec"] - processed["clean_model_spec"]).abs().squeeze(0)
    improvement = noisy_err - enhanced_err

    noisy_db = to_numpy(20.0 * torch.log10(noisy_err.clamp_min(1e-8)))
    enhanced_db = to_numpy(20.0 * torch.log10(enhanced_err.clamp_min(1e-8)))
    improvement_np = to_numpy(improvement)

    err_vmin, err_vmax = robust_limits(np.concatenate([noisy_db.reshape(-1), enhanced_db.reshape(-1)]))
    improve_lim = max(float(np.quantile(np.abs(improvement_np.reshape(-1)), 0.99)), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, sharey=True)
    add_spec_image(
        fig,
        axes[0],
        noisy_db,
        "|noisy - clean| in spec_fwd space (dB)",
        cfg,
        vmin=err_vmin,
        vmax=err_vmax,
    )
    add_spec_image(
        fig,
        axes[1],
        enhanced_db,
        "|enhanced - clean| in spec_fwd space (dB)",
        cfg,
        vmin=err_vmin,
        vmax=err_vmax,
    )
    add_spec_image(
        fig,
        axes[2],
        improvement_np,
        "improvement: |noisy-clean| - |enhanced-clean|",
        cfg,
        vmin=-improve_lim,
        vmax=improve_lim,
        cmap="coolwarm",
    )
    fig.suptitle(f"{title} | error comparison after spec_fwd", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_value_distributions(processed, title, out_path: Path):
    waves = {
        "clean": to_numpy(processed["clean_norm"]).reshape(-1),
        "noisy": to_numpy(processed["noisy_norm"]).reshape(-1),
        "enhanced": to_numpy(processed["enhanced_norm"]).reshape(-1),
    }
    model_mag = {
        name: to_numpy(processed[f"{name}_model_spec"].abs()).reshape(-1)
        for name in ["clean", "noisy", "enhanced"]
    }
    model_real = {
        name: to_numpy(processed[f"{name}_model_spec"].real).reshape(-1)
        for name in ["clean", "noisy", "enhanced"]
    }
    model_imag = {
        name: to_numpy(processed[f"{name}_model_spec"].imag).reshape(-1)
        for name in ["clean", "noisy", "enhanced"]
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for name in ["clean", "noisy", "enhanced"]:
        axes[0, 0].hist(
            waves[name],
            bins=150,
            alpha=0.43,
            density=True,
            label=name,
            color=COLORS[name],
        )
    axes[0, 0].set_title("Waveform values before STFT")
    axes[0, 0].set_xlabel("normalized amplitude")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.25)

    mag_hi = max(
        float(np.quantile(np.concatenate([v for v in model_mag.values()]), 0.995)),
        1e-8,
    )
    for name in ["clean", "noisy", "enhanced"]:
        axes[0, 1].hist(
            model_mag[name],
            bins=150,
            range=(0.0, mag_hi),
            alpha=0.43,
            density=True,
            label=name,
            color=COLORS[name],
        )
    axes[0, 1].set_title("spec_fwd magnitude values")
    axes[0, 1].set_xlabel("|spec_fwd(STFT)|")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.25)

    ri_all = np.concatenate([*model_real.values(), *model_imag.values()])
    ri_lim = max(float(np.quantile(np.abs(ri_all), 0.995)), 1e-8)
    for name in ["clean", "noisy", "enhanced"]:
        axes[1, 0].hist(
            model_real[name],
            bins=150,
            range=(-ri_lim, ri_lim),
            alpha=0.43,
            density=True,
            label=name,
            color=COLORS[name],
        )
    axes[1, 0].set_title("spec_fwd real values")
    axes[1, 0].set_xlabel("real")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.25)

    for name in ["clean", "noisy", "enhanced"]:
        axes[1, 1].hist(
            model_imag[name],
            bins=150,
            range=(-ri_lim, ri_lim),
            alpha=0.43,
            density=True,
            label=name,
            color=COLORS[name],
        )
    axes[1, 1].set_title("spec_fwd imag values")
    axes[1, 1].set_xlabel("imag")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"{title} | clean/noisy/enhanced value distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def stats_for_tensor(x: torch.Tensor):
    values = x.abs() if torch.is_complex(x) else x
    arr = to_numpy(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    out = {"shape": list(x.shape), "dtype": str(x.dtype), "count": int(arr.size)}
    if arr.size == 0:
        return out
    out.update(
        {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
        }
    )
    return out


def write_summary(wavs, processed, cfg, out_path: Path):
    clean_norm = processed["clean_norm"]
    noisy_norm = processed["noisy_norm"]
    enhanced_norm = processed["enhanced_norm"]
    row = wavs.get("row") or {}

    summary = {
        "id": row.get("id", ""),
        "source_id": row.get("source_id", ""),
        "clean_wav": wavs["clean_path"],
        "noisy_wav": wavs["noisy_path"],
        "enhanced_wav": wavs["enhanced_path"],
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "num_frames": cfg.num_frames,
        "aligned_len": processed["aligned_len"],
        "visualized_len": int(clean_norm.size(-1)),
        "crop_start": processed["start"],
        "normfac_from_noisy": float(processed["normfac"].item()),
        "metrics_against_clean_normalized": {
            "noisy": signal_metrics(noisy_norm, clean_norm),
            "enhanced": signal_metrics(enhanced_norm, clean_norm),
        },
        "tensors": {},
    }

    for name in ["clean", "noisy", "enhanced"]:
        summary["tensors"][f"{name}_wave_before_stft"] = stats_for_tensor(processed[f"{name}_norm"])
        summary["tensors"][f"{name}_raw_complex_stft"] = stats_for_tensor(processed[f"{name}_raw_spec"])
        summary["tensors"][f"{name}_spec_fwd_complex"] = stats_for_tensor(processed[f"{name}_model_spec"])

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest_val.csv")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--clean-wav", default="")
    parser.add_argument("--noisy-wav", default="")
    parser.add_argument("--enhanced-wav", required=True)
    parser.add_argument("--out-dir", default="outputs/plots/clean_noisy_enhanced")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--full", action="store_true", help="Compare full aligned wavs instead of train crop.")
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument(
        "--crop-start",
        type=int,
        default=-1,
        help="Fixed crop start sample for non-full mode. Negative means center crop unless --random-crop is set.",
    )

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=32640)
    parser.add_argument("--n-fft", type=int, default=510)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=510)
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--spec-factor", type=float, default=0.15)
    parser.add_argument("--spec-abs-exponent", type=float, default=0.5)
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
        normalize="noisy",
    )
    if args.target_length != cfg.train_target_len:
        raise ValueError(
            f"target_length mismatch: got {args.target_length}, "
            f"expected {cfg.train_target_len} from (num_frames - 1) * hop_length"
        )

    wavs = load_three_wavs(args, cfg)
    processed = align_and_prepare(
        wavs=wavs,
        cfg=cfg,
        crop_start=args.crop_start,
        random_crop=args.random_crop,
        full=args.full,
    )

    row = wavs.get("row") or {}
    sample_label = row.get("id") or row.get("source_id") or Path(wavs["enhanced_path"]).stem
    if args.full:
        sample_label = f"{sample_label}_full"
    prefix = safe_name(sample_label)

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "waveform": out_dir / f"{prefix}_00_waveform_clean_noisy_enhanced.png",
        "raw_stft": out_dir / f"{prefix}_01_raw_stft_magnitude.png",
        "spec_fwd": out_dir / f"{prefix}_02_spec_fwd_magnitude.png",
        "error_spec": out_dir / f"{prefix}_03_spec_fwd_error_vs_clean.png",
        "distribution": out_dir / f"{prefix}_04_value_distributions.png",
        "summary": out_dir / f"{prefix}_summary.json",
    }

    plot_waveform_comparison(processed, cfg, sample_label, paths["waveform"])
    plot_magnitude_triplet(processed, cfg, sample_label, paths["raw_stft"], spec_kind="raw")
    plot_magnitude_triplet(processed, cfg, sample_label, paths["spec_fwd"], spec_kind="model")
    plot_error_spectrograms(processed, cfg, sample_label, paths["error_spec"])
    plot_value_distributions(processed, sample_label, paths["distribution"])
    write_summary(wavs, processed, cfg, paths["summary"])

    print("sample:", sample_label)
    print("clean:", wavs["clean_path"])
    print("noisy:", wavs["noisy_path"])
    print("enhanced:", wavs["enhanced_path"])
    print("aligned_len:", processed["aligned_len"])
    print("visualized_len:", int(processed["clean_norm"].size(-1)))
    print("crop_start:", processed["start"])
    print("normfac_from_noisy:", float(processed["normfac"].item()))
    print("clean/noisy/enhanced wave shape:", tuple(processed["clean_norm"].shape))
    print("raw STFT shape:", tuple(processed["clean_raw_spec"].shape), processed["clean_raw_spec"].dtype)
    print("spec_fwd shape:", tuple(processed["clean_model_spec"].shape), processed["clean_model_spec"].dtype)
    for key, path in paths.items():
        print(f"saved {key}: {path}")


if __name__ == "__main__":
    main()
