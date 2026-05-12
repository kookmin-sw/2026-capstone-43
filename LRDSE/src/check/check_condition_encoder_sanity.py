#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.condition.condition_encoder import (
    ConditionEncoderDataset,
    align_by_delay,
    compute_condition_encoder_losses,
    estimate_delay_from_loader,
)
from src.models.condition_encoder import ConditionEncoder


def namespace_from_args_dict(args_dict):
    defaults = {
        "manifest": "",
        "val_manifest": "",
        "target_sr": 16000,
        "target_length": 32640,
        "n_fft": 510,
        "hop_length": 128,
        "num_frames": 256,
        "normalize_audio": "not",
        "apply_mix_gain": False,
        "raw_force_scale": 220.0,
        "d_force_scale": 255.0,
        "condition_smooth_win": 1,
        "hidden_channels": 256,
        "num_layers": 8,
        "kernel_size": 5,
        "dropout": 0.05,
        "causal": True,
        "max_dilation": 16,
        "encoder_conv_type": "standard",
        "delay_frames": 0,
    }
    defaults.update(args_dict or {})
    return SimpleNamespace(**defaults)


def build_dataset(args, manifest, limit):
    return ConditionEncoderDataset(
        manifest_path=manifest,
        target_sr=args.target_sr,
        target_length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_frames=args.num_frames,
        center=True,
        normalize_audio=args.normalize_audio,
        random_crop=False,
        valid_only=True,
        limit=limit,
        apply_mix_gain=args.apply_mix_gain,
        raw_force_scale=args.raw_force_scale,
        d_force_scale=args.d_force_scale,
        condition_smooth_win=args.condition_smooth_win,
    )


def build_model(args):
    return ConditionEncoder(
        in_channels=24,
        freq_bins=args.n_fft // 2 + 1,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        causal=args.causal,
        max_dilation=args.max_dilation,
        encoder_conv_type=args.encoder_conv_type,
    )


@torch.no_grad()
def l1_at_delay(pred, target, delay_frames):
    pa, ta = align_by_delay(pred, target, delay_frames)
    return F.l1_loss(pa, ta)


@torch.no_grad()
def run_matched_shuffled(model, loader, device, delay_frames, num_batches):
    totals = {
        "matched": 0.0,
        "shuffled_target": 0.0,
        "shuffled_force": 0.0,
    }
    count = 0
    usable_shuffle_batches = 0

    for bi, batch in enumerate(loader):
        force = batch["force_feat"].to(device)
        target = batch["target_mag"].to(device)

        pred = model(force)
        matched = l1_at_delay(pred, target, delay_frames)
        totals["matched"] += float(matched.item()) * force.size(0)
        count += force.size(0)

        if force.size(0) >= 2:
            target_shuf = torch.roll(target, shifts=1, dims=0)
            force_shuf = torch.roll(force, shifts=1, dims=0)
            pred_force_shuf = model(force_shuf)
            totals["shuffled_target"] += float(l1_at_delay(pred, target_shuf, delay_frames).item()) * force.size(0)
            totals["shuffled_force"] += float(l1_at_delay(pred_force_shuf, target, delay_frames).item()) * force.size(0)
            usable_shuffle_batches += force.size(0)

        if bi + 1 >= num_batches:
            break

    denom = max(1, count)
    shuffle_denom = max(1, usable_shuffle_batches)
    out = {
        "matched_l1": totals["matched"] / denom,
        "shuffled_target_l1": totals["shuffled_target"] / shuffle_denom,
        "shuffled_force_l1": totals["shuffled_force"] / shuffle_denom,
        "num_samples": int(count),
        "num_shuffle_samples": int(usable_shuffle_batches),
    }
    if out["matched_l1"] > 0:
        out["shuffled_target_ratio"] = out["shuffled_target_l1"] / out["matched_l1"]
        out["shuffled_force_ratio"] = out["shuffled_force_l1"] / out["matched_l1"]
    else:
        out["shuffled_target_ratio"] = float("nan")
        out["shuffled_force_ratio"] = float("nan")
    out["passes_ratio_1p2"] = bool(
        math.isfinite(out["shuffled_target_ratio"])
        and out["shuffled_target_ratio"] > 1.2
    )
    return out


@torch.no_grad()
def run_time_shift_curve(model, loader, device, min_shift, max_shift, num_batches):
    shifts = list(range(int(min_shift), int(max_shift) + 1))
    sums = {s: 0.0 for s in shifts}
    count = 0

    for bi, batch in enumerate(loader):
        force = batch["force_feat"].to(device)
        target = batch["target_mag"].to(device)
        pred = model(force)
        for shift in shifts:
            loss = compute_condition_encoder_losses(
                pred_mag=pred,
                target_mag=target,
                delay_frames=shift,
                band_weight=0.0,
                event_weight=0.0,
            )["loss"]
            sums[shift] += float(loss.item()) * force.size(0)
        count += force.size(0)
        if bi + 1 >= num_batches:
            break

    denom = max(1, count)
    curve = [{"delay_frames": s, "l_mag": sums[s] / denom} for s in shifts]
    best = min(curve, key=lambda item: item["l_mag"]) if curve else {"delay_frames": 0}
    return {
        "best_delay_frames": int(best["delay_frames"]),
        "curve": curve,
        "num_samples": int(count),
    }


def plot_delay_curve(curve, out_path, y_key, title, vertical_delay=None):
    xs = [item["delay_frames"] for item in curve]
    ys = [item[y_key] for item in curve]
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    if vertical_delay is not None:
        plt.axvline(int(vertical_delay), color="tab:red", linestyle="--", linewidth=1.0)
    plt.xlabel("delay frames")
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


@torch.no_grad()
def save_magnitude_visualizations(model, loader, device, out_dir, delay_frames, num_viz):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for batch in loader:
        force = batch["force_feat"].to(device)
        target = batch["target_mag"].to(device)
        pred = model(force)
        force_shuf = torch.roll(force, shifts=1, dims=0) if force.size(0) >= 2 else force.flip(dims=[-1])
        pred_shuf = model(force_shuf)

        pred_a, target_a = align_by_delay(pred, target, delay_frames)
        pred_shuf_a, _ = align_by_delay(pred_shuf, target, delay_frames)

        metas = batch.get("meta", {})
        ids = metas.get("id", None) if isinstance(metas, dict) else None

        for i in range(force.size(0)):
            if saved >= num_viz:
                return
            target_i = target_a[i].detach().cpu()
            pred_i = pred_a[i].detach().cpu()
            pred_shuf_i = pred_shuf_a[i].detach().cpu()
            err_i = (target_i - pred_i).abs()

            vmax = float(torch.quantile(target_i, 0.995).item())
            vmax = max(vmax, 1e-6)
            sample_id = ids[i] if ids is not None else f"sample_{saved:03d}"
            safe_id = str(sample_id).replace("/", "_").replace(" ", "_")

            fig, axes = plt.subplots(4, 1, figsize=(9.0, 9.5), sharex=True)
            panels = [
                ("target M", target_i, 0.0, vmax),
                ("predicted M_hat", pred_i, 0.0, vmax),
                ("abs error", err_i, 0.0, float(torch.quantile(err_i, 0.995).item()) + 1e-6),
                ("predicted with shuffled force", pred_shuf_i, 0.0, vmax),
            ]
            for ax, (name, mat, vmin, vmax_panel) in zip(axes, panels):
                im = ax.imshow(
                    mat.numpy(),
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax_panel,
                )
                ax.set_ylabel("freq bin")
                ax.set_title(name)
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            axes[-1].set_xlabel("frame")
            fig.suptitle(str(sample_id))
            fig.tight_layout()
            fig.savefig(out_dir / f"{saved:03d}_{safe_id}_magnitude.png", dpi=160)
            plt.close(fig)
            saved += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--out-dir", default="./outputs/condition_encoder_sanity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--delay-min-frames", type=int, default=-24)
    parser.add_argument("--delay-max-frames", type=int, default=24)
    parser.add_argument("--num-viz", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = namespace_from_args_dict(ckpt.get("args", {}))
    manifest = args.manifest or train_args.val_manifest or train_args.manifest
    if not manifest:
        raise ValueError("No manifest provided and checkpoint args do not contain one.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        train_args,
        manifest=manifest,
        limit=args.limit_samples if args.limit_samples > 0 else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    delay_corr = estimate_delay_from_loader(
        loader,
        min_delay=args.delay_min_frames,
        max_delay=args.delay_max_frames,
        num_batches=args.num_batches,
        device=device,
    )
    matched = run_matched_shuffled(
        model,
        loader,
        device=device,
        delay_frames=train_args.delay_frames,
        num_batches=args.num_batches,
    )
    shift_curve = run_time_shift_curve(
        model,
        loader,
        device=device,
        min_shift=args.delay_min_frames,
        max_shift=args.delay_max_frames,
        num_batches=args.num_batches,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(manifest),
        "trained_delay_frames": int(train_args.delay_frames),
        "cross_correlation_delay": delay_corr,
        "matched_vs_shuffled": matched,
        "time_shift_curve": shift_curve,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plots:
        plot_delay_curve(
            delay_corr["curve"],
            out_dir / "force_noise_cross_corr_delay.png",
            y_key="corr",
            title="force derivative vs noise energy cross-correlation",
            vertical_delay=delay_corr["best_delay_frames"],
        )
        plot_delay_curve(
            shift_curve["curve"],
            out_dir / "time_shift_lmag_curve.png",
            y_key="l_mag",
            title="M_hat vs target magnitude L1 over target delay",
            vertical_delay=shift_curve["best_delay_frames"],
        )
        save_magnitude_visualizations(
            model,
            loader,
            device=device,
            out_dir=out_dir / "magnitude_maps",
            delay_frames=train_args.delay_frames,
            num_viz=args.num_viz,
        )

    print("--------------------------------------------------")
    print("[condition encoder sanity]")
    print(f"checkpoint              : {args.checkpoint}")
    print(f"manifest                : {manifest}")
    print(f"trained_delay_frames    : {train_args.delay_frames}")
    print(f"xcor best delay         : {delay_corr['best_delay_frames']}")
    print(f"shift-curve best delay  : {shift_curve['best_delay_frames']}")
    print(f"matched L1              : {matched['matched_l1']:.6f}")
    print(f"shuffled target L1      : {matched['shuffled_target_l1']:.6f}")
    print(f"shuffled target ratio   : {matched['shuffled_target_ratio']:.3f}")
    print(f"shuffled force ratio    : {matched['shuffled_force_ratio']:.3f}")
    print(f"ratio > 1.2             : {matched['passes_ratio_1p2']}")
    print(f"saved                   : {out_dir}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
