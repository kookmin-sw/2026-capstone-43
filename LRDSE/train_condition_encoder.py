#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.condition.condition_encoder import (
    ConditionEncoderDataset,
    build_band_slices,
    compute_condition_encoder_losses,
    estimate_delay_from_loader,
    parse_band_edges_hz,
)
from src.models.condition_encoder import ConditionEncoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=True)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def temporal_receptive_field_frames(args) -> int:
    receptive_field = 1
    dilation = 1
    for _ in range(args.num_layers):
        receptive_field += (args.kernel_size - 1) * dilation
        dilation = min(dilation * 2, args.max_dilation)
    return receptive_field


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


def build_dataset(args, manifest, random_crop, limit):
    return ConditionEncoderDataset(
        manifest_path=manifest,
        target_sr=args.target_sr,
        target_length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_frames=args.num_frames,
        center=True,
        normalize_audio=args.normalize_audio,
        random_crop=random_crop,
        valid_only=True,
        limit=limit,
        apply_mix_gain=args.apply_mix_gain,
        raw_force_scale=args.raw_force_scale,
        d_force_scale=args.d_force_scale,
        condition_smooth_win=args.condition_smooth_win,
    )


def batch_to_device(batch, device):
    force = batch["force_feat"].to(device, non_blocking=True)
    target = batch["target_mag"].to(device, non_blocking=True)
    return force, target


@torch.no_grad()
def evaluate(model, loader, args, device, band_slices):
    model.eval()
    totals = {"loss": 0.0, "l_mag": 0.0, "l_band": 0.0, "l_event": 0.0}
    count = 0

    for batch in loader:
        force, target = batch_to_device(batch, device)
        pred = model(force)
        losses = compute_condition_encoder_losses(
            pred_mag=pred,
            target_mag=target,
            delay_frames=args.delay_frames,
            band_slices=band_slices,
            band_weight=args.band_weight,
            event_weight=args.event_weight,
            event_percentile=args.event_percentile,
        )
        batch_size = force.size(0)
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        count += batch_size

    model.train()
    denom = max(1, count)
    return {key: value / denom for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, scaler, args, step, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": dict(metrics),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), ckpt.get("metrics", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", default="")
    parser.add_argument("--save-dir", default="./checkpoints/condition_encoder")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=32640)
    parser.add_argument("--n-fft", type=int, default=510)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=256)
    parser.add_argument("--normalize-audio", default="not", choices=["not", "noise"])
    parser.add_argument(
        "--apply-mix-gain",
        action="store_true",
        default=False,
        help="Use only for compatibility with synthetic noisy metadata. "
             "Step 1 noise-only training should leave this disabled.",
    )
    parser.add_argument("--no-apply-mix-gain", dest="apply_mix_gain", action="store_false")

    parser.add_argument("--raw-force-scale", type=float, default=220.0)
    parser.add_argument("--d-force-scale", type=float, default=255.0)
    parser.add_argument("--condition-smooth-win", type=int, default=1)

    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-dilation", type=int, default=8)
    parser.add_argument(
        "--encoder-conv-type",
        default="separable",
        choices=["separable", "standard"],
        help="separable is the lightweight real-time default; standard keeps the old dense Conv1d blocks.",
    )
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--non-causal", dest="causal", action="store_false")

    parser.add_argument("--delay-frames", type=int, default=0)
    parser.add_argument("--auto-delay", action="store_true")
    parser.add_argument("--delay-min-frames", type=int, default=-12)
    parser.add_argument("--delay-max-frames", type=int, default=12)
    parser.add_argument("--delay-batches", type=int, default=32)

    parser.add_argument("--band-weight", type=float, default=0.0)
    parser.add_argument("--event-weight", type=float, default=0.0)
    parser.add_argument("--event-percentile", type=float, default=85.0)
    parser.add_argument("--band-edges-hz", default="0,250,500,1000,2000,4000,8000")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--max-epochs", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--eval-every-epochs",
        type=float,
        default=0.0,
        help="0보다 크면 epoch 기준 validation/best checkpoint 주기. "
             "예: 1이면 매 epoch 끝에서 validation 후 best.pt 갱신.",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-every-epochs", type=float, default=0.0)
    parser.add_argument("--epoch-loss-csv", default="epoch_losses.csv")
    parser.add_argument("--disable-checkpoint-save", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.target_length != (args.num_frames - 1) * args.hop_length:
        raise ValueError(
            f"target_length mismatch: got {args.target_length}, "
            f"expected {(args.num_frames - 1) * args.hop_length}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.overfit_samples > 0:
        dataset_limit = args.overfit_samples
        train_shuffle = False
        train_random_crop = False
    elif args.limit_samples > 0:
        dataset_limit = args.limit_samples
        train_shuffle = True
        train_random_crop = True
    else:
        dataset_limit = None
        train_shuffle = True
        train_random_crop = True

    train_dataset = build_dataset(
        args,
        manifest=args.manifest,
        random_crop=train_random_crop,
        limit=dataset_limit,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise RuntimeError("train DataLoader is empty")

    val_loader = None
    if args.val_manifest:
        val_batch_size = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
        val_num_workers = args.val_num_workers if args.val_num_workers >= 0 else args.num_workers
        val_dataset = build_dataset(
            args,
            manifest=args.val_manifest,
            random_crop=False,
            limit=None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(val_num_workers > 0),
            drop_last=False,
        )
        if len(val_loader) == 0:
            raise RuntimeError("validation DataLoader is empty")

    if args.auto_delay:
        delay_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
            drop_last=False,
        )
        delay_info = estimate_delay_from_loader(
            delay_loader,
            min_delay=args.delay_min_frames,
            max_delay=args.delay_max_frames,
            num_batches=args.delay_batches,
            device=device,
        )
        args.delay_frames = int(delay_info["best_delay_frames"])
        with (save_dir / "delay_estimate.json").open("w", encoding="utf-8") as f:
            json.dump(delay_info, f, indent=2)
        print(f"[delay] selected delay_frames={args.delay_frames}")

    band_edges = parse_band_edges_hz(args.band_edges_hz)
    band_slices = build_band_slices(
        freq_bins=args.n_fft // 2 + 1,
        sample_rate=args.target_sr,
        n_fft=args.n_fft,
        band_edges_hz=band_edges,
    )

    with (save_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = (
        max(1, int(math.ceil(updates_per_epoch * args.max_epochs)))
        if args.max_epochs > 0
        else int(args.max_steps)
    )
    save_every_steps = (
        max(1, int(math.ceil(updates_per_epoch * args.save_every_epochs)))
        if args.save_every_epochs > 0
        else int(args.save_every)
    )
    eval_every_steps = (
        max(1, int(math.ceil(updates_per_epoch * args.eval_every_epochs)))
        if args.eval_every_epochs > 0
        else int(args.eval_every)
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(args.amp and device.type == "cuda"),
    )

    start_step = 0
    if args.resume:
        start_step, metrics = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        print(f"[resume] path={args.resume} step={start_step} metrics={metrics}")

    csv_path = save_dir / args.epoch_loss_csv
    write_header = (start_step == 0) or (not csv_path.exists())
    with csv_path.open("w" if write_header else "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "step",
                "epoch",
                "train_loss",
                "train_l_mag",
                "train_l_band",
                "train_l_event",
                "val_loss",
                "val_l_mag",
                "val_l_band",
                "val_l_event",
                "timestamp",
            ])

    print("--------------------------------------------------")
    print("[condition encoder train config]")
    print(f"device             : {device}")
    print(f"train_samples      : {len(train_dataset)}")
    print(f"train_batches/ep   : {len(train_loader)}")
    print(f"val_manifest       : {args.val_manifest if args.val_manifest else '(none)'}")
    print(f"updates_per_epoch  : {updates_per_epoch}")
    print(f"max_steps          : {total_steps}")
    print(f"max_epochs         : {args.max_epochs}")
    print(f"batch_size         : {args.batch_size}")
    print(f"eval_every_steps   : {eval_every_steps}")
    print(f"eval_every_epochs  : {args.eval_every_epochs}")
    print(f"save_every_steps   : {save_every_steps}")
    print(f"save_every_epochs  : {args.save_every_epochs}")
    print(f"lr                 : {args.lr}")
    print(f"amp                : {args.amp}")
    print(f"save_dir           : {save_dir}")
    print("--------------------------------------------------")
    print("[preprocess]")
    print(f"target_sr          : {args.target_sr}")
    print(f"n_fft/hop/T        : {args.n_fft}/{args.hop_length}/{args.num_frames}")
    print(f"normalize_audio    : {args.normalize_audio}")
    print(f"apply_mix_gain     : {args.apply_mix_gain}")
    print(f"raw_force_scale    : {args.raw_force_scale}")
    print(f"d_force_scale      : {args.d_force_scale}")
    print("--------------------------------------------------")
    print("[model/loss]")
    print(f"params             : {count_params(model):,}")
    print(f"hidden/layers      : {args.hidden_channels}/{args.num_layers}")
    print(f"encoder_conv_type  : {args.encoder_conv_type}")
    rf_frames = temporal_receptive_field_frames(args)
    print(f"encoder_rf         : {rf_frames} frames ({rf_frames * args.hop_length / args.target_sr * 1000:.1f} ms)")
    print(f"causal             : {args.causal}")
    print(f"delay_frames       : {args.delay_frames}")
    print(f"band_weight        : {args.band_weight}")
    print(f"event_weight       : {args.event_weight}")
    print(f"band_slices        : {band_slices}")
    print("--------------------------------------------------")

    train_iter = cycle(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    best_metric = float("inf")
    running = {"loss": 0.0, "l_mag": 0.0, "l_band": 0.0, "l_event": 0.0}
    running_count = 0
    last_time = time.time()

    for step in range(start_step + 1, total_steps + 1):
        step_sums = {"loss": 0.0, "l_mag": 0.0, "l_band": 0.0, "l_event": 0.0}

        for _ in range(args.grad_accum):
            batch = next(train_iter)
            force, target = batch_to_device(batch, device)

            with autocast_context(device, enabled=(args.amp and device.type == "cuda")):
                pred = model(force)
                losses = compute_condition_encoder_losses(
                    pred_mag=pred,
                    target_mag=target,
                    delay_frames=args.delay_frames,
                    band_slices=band_slices,
                    band_weight=args.band_weight,
                    event_weight=args.event_weight,
                    event_percentile=args.event_percentile,
                )
                loss = losses["loss"] / max(1, args.grad_accum)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"loss is NaN or Inf at step {step}")

            scaler.scale(loss).backward()
            for key in step_sums:
                step_sums[key] += float(losses[key].item())

        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for key in step_sums:
            step_sums[key] /= max(1, args.grad_accum)
            running[key] += step_sums[key]
        running_count += 1

        val_metrics = None
        should_eval = (
            val_loader is not None
            and eval_every_steps > 0
            and (step % eval_every_steps == 0 or step == total_steps)
        )
        if should_eval:
            val_metrics = evaluate(model, val_loader, args, device, band_slices)

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            avg = {key: running[key] / max(1, running_count) for key in running}
            speed = args.log_every / max(now - last_time, 1e-8)
            msg = (
                f"[step {step:07d}/{total_steps:07d}] "
                f"loss={avg['loss']:.6f} mag={avg['l_mag']:.6f} "
                f"band={avg['l_band']:.6f} event={avg['l_event']:.6f} "
                f"speed={speed:.2f} step/s"
            )
            if val_metrics is not None:
                msg += f" val={val_metrics['loss']:.6f}"
            print(msg)
            running = {key: 0.0 for key in running}
            running_count = 0
            last_time = now

        if val_metrics is not None:
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    step / updates_per_epoch,
                    step_sums["loss"],
                    step_sums["l_mag"],
                    step_sums["l_band"],
                    step_sums["l_event"],
                    val_metrics["loss"],
                    val_metrics["l_mag"],
                    val_metrics["l_band"],
                    val_metrics["l_event"],
                    time.time(),
                ])

            metric = val_metrics["loss"]
            if metric < best_metric and not args.disable_checkpoint_save:
                best_metric = metric
                save_checkpoint(save_dir / "best.pt", model, optimizer, scaler, args, step, val_metrics)

        if (
            not args.disable_checkpoint_save
            and save_every_steps > 0
            and (step % save_every_steps == 0 or step == total_steps)
        ):
            metrics = val_metrics if val_metrics is not None else step_sums
            save_checkpoint(save_dir / "latest.pt", model, optimizer, scaler, args, step, metrics)

    if not args.disable_checkpoint_save:
        final_metrics = evaluate(model, val_loader, args, device, band_slices) if val_loader else step_sums
        save_checkpoint(save_dir / "latest.pt", model, optimizer, scaler, args, total_steps, final_metrics)

    print("[done] condition encoder training complete")


if __name__ == "__main__":
    main()
