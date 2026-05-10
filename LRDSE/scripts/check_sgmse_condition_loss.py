#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import SpeechEnhancementDataset
from denoise_sgmse import (
    load_ema_if_available,
    load_model_weights,
    load_training_args,
    resolve_device,
    torch_load,
    validate_args,
)
from train_sgmse import build_model, set_seed, twoch_to_complex_batch


MODE_CHOICES = {
    "real",
    "zero",
    "zero_padded",
    "random",
    "shuffle",
    "no_condition",
}


def parse_modes(value: str):
    modes = [m.strip() for m in value.split(",") if m.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one mode")

    invalid = [m for m in modes if m not in MODE_CHOICES]
    if invalid:
        raise ValueError(
            f"invalid mode(s): {invalid}. choices={sorted(MODE_CHOICES)}"
        )
    return modes


def has_nonfinite(x: torch.Tensor) -> bool:
    return bool((~torch.isfinite(x)).any().item())


def condition_stats(cond: torch.Tensor, cond_mask: torch.Tensor):
    valid_mask = cond_mask.bool().unsqueeze(1).expand_as(cond)
    if valid_mask.any():
        values = cond[valid_mask]
    else:
        values = cond.reshape(-1)

    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "valid_tokens": int(cond_mask.sum().item()),
        "total_tokens": int(cond_mask.numel()),
    }


def make_random_condition(
    cond: torch.Tensor,
    cond_mask: torch.Tensor,
    generator: torch.Generator,
):
    valid_mask = cond_mask.bool().unsqueeze(1).expand_as(cond)
    if valid_mask.any():
        values = cond[valid_mask]
    else:
        values = cond.reshape(-1)

    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1e-6)
    random_cond = torch.randn(
        cond.shape,
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    random_cond = random_cond * std + mean

    return torch.where(valid_mask, random_cond, torch.zeros_like(random_cond))


def apply_condition_mode(
    mode: str,
    cond: torch.Tensor,
    cond_times: torch.Tensor,
    cond_mask: torch.Tensor,
    query_mono_times: torch.Tensor,
    generator: torch.Generator,
):
    if mode == "real":
        return cond, cond_times, cond_mask, query_mono_times

    if mode == "zero":
        return torch.zeros_like(cond), cond_times, cond_mask, query_mono_times

    if mode == "zero_padded":
        return (
            torch.zeros_like(cond),
            cond_times,
            torch.zeros_like(cond_mask, dtype=torch.bool),
            query_mono_times,
        )

    if mode == "random":
        return (
            make_random_condition(cond, cond_mask, generator),
            cond_times,
            cond_mask,
            query_mono_times,
        )

    if mode == "shuffle":
        if cond.size(0) < 2:
            shuffled_cond = make_random_condition(cond, cond_mask, generator)
            return shuffled_cond, cond_times, cond_mask, query_mono_times

        # Roll the whole condition packet so each sample sees another sample's
        # condition with matching condition timestamps/mask.
        return (
            torch.roll(cond, shifts=1, dims=0),
            torch.roll(cond_times, shifts=1, dims=0),
            torch.roll(cond_mask, shifts=1, dims=0),
            query_mono_times,
        )

    if mode == "no_condition":
        return None, None, None, None

    raise ValueError(f"unsupported mode: {mode}")


def rng_devices(device: torch.device):
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [device.index]
    return [torch.cuda.current_device()]


@torch.no_grad()
def loss_with_fixed_seed(model, payload, seed: int, device: torch.device):
    with torch.random.fork_rng(devices=rng_devices(device), enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return model._step(payload, batch_idx=0)


def build_payload(batch, args, device):
    clean = twoch_to_complex_batch(batch["clean_stft"].to(device, non_blocking=True))
    noisy = twoch_to_complex_batch(batch["noisy_stft"].to(device, non_blocking=True))

    if "cond" not in batch:
        raise RuntimeError(
            "Batch does not contain condition tensors. "
            "Use a manifest whose noisy samples have lowstate/anchor files."
        )

    cond = batch["cond"].to(device, non_blocking=True)
    cond_times = batch["cond_times"].to(device, non_blocking=True)
    cond_mask = batch["cond_mask"].to(device, non_blocking=True).bool()
    query_mono_times = batch["query_mono_times"].to(device, non_blocking=True)

    if cond.dim() != 3:
        raise RuntimeError(f"Expected cond shape [B, C, K], got {tuple(cond.shape)}")
    if cond.size(1) != args.aux_cond_dim:
        raise RuntimeError(
            f"Expected cond channel={args.aux_cond_dim}, got {cond.size(1)}"
        )

    return clean, noisy, cond, cond_times, cond_mask, query_mono_times


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    finite = torch.isfinite(tensor)
    if not finite.any():
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "nonfinite": int((~finite).sum().item()),
            "count": int(tensor.numel()),
        }

    finite_values = tensor[finite]
    return {
        "mean": float(finite_values.mean().item()),
        "std": float(finite_values.std(unbiased=False).item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "nonfinite": int((~finite).sum().item()),
        "count": int(tensor.numel()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SGMSE aux-condition loss under real/zero/random/shuffled "
            "condition inputs."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--args-json", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", default="real,zero,zero_padded,random,shuffle,no_condition")
    parser.add_argument("--valid-only", action="store_true", default=True)
    parser.add_argument("--include-invalid", dest="valid_only", action="store_false")
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--out-csv", default="")
    parser.add_argument(
        "--fail-ratio",
        type=float,
        default=0.0,
        help="0보다 크면 mean_loss / real_mean_loss가 이 값 이상인 mode가 있을 때 실패 처리.",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    modes = parse_modes(cli.modes)
    set_seed(cli.seed)

    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    args = load_training_args(checkpoint_path, checkpoint, cli.args_json)
    args.device = cli.device
    args.seed = cli.seed
    args.use_ema = not cli.no_ema
    validate_args(args)

    if not args.use_aux_cond:
        raise RuntimeError(
            "Checkpoint args say use_aux_cond=False, so condition ablation would not affect the model. "
            "Use an aux-condition SGMSE checkpoint."
        )

    device = resolve_device(args.device)
    dataset = SpeechEnhancementDataset(
        manifest_path=cli.manifest,
        target_sr=args.target_sr,
        target_length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        num_frames=args.num_frames,
        spec_factor=args.spec_factor,
        spec_abs_exponent=args.spec_abs_exponent,
        normalize=args.normalize,
        random_crop=cli.random_crop,
        valid_only=cli.valid_only,
        limit=max(1, cli.batch_size * cli.num_batches),
        use_condition=True,
        condition_repr="8ch",
        raw_force_scale=args.raw_force_scale,
        d_force_scale=args.d_force_scale,
        condition_smooth_win=args.condition_smooth_win,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(args)
    weights_source, incompatible = load_model_weights(
        model=model,
        checkpoint=checkpoint,
        strict=(not cli.non_strict),
    )
    ema_loaded = load_ema_if_available(model, checkpoint, args)
    model = model.to(device)
    model.eval(no_ema=(not args.use_ema))

    step = checkpoint.get("step", "?") if isinstance(checkpoint, dict) else "?"
    print("--------------------------------------------------")
    print("[condition loss check]")
    print(f"checkpoint       : {checkpoint_path}")
    print(f"weights          : {weights_source}")
    print(f"step             : {step}")
    print(f"device           : {device}")
    print(f"use_ema          : {args.use_ema} (loaded={ema_loaded})")
    print(f"manifest         : {cli.manifest}")
    print(f"samples          : {len(dataset)}")
    print(f"batch_size       : {cli.batch_size}")
    print(f"num_batches      : {cli.num_batches}")
    print(f"repeats          : {cli.repeats}")
    print(f"modes            : {', '.join(modes)}")
    print(f"random_crop      : {cli.random_crop}")
    print(f"aux_cond_dim     : {args.aux_cond_dim}")
    if cli.non_strict:
        print(f"missing_keys     : {len(incompatible.missing_keys)}")
        print(f"unexpected_keys  : {len(incompatible.unexpected_keys)}")
    print("--------------------------------------------------")

    losses = defaultdict(list)
    rows = []
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(cli.seed + 12345)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= cli.num_batches:
            break

        clean, noisy, cond, cond_times, cond_mask, query_mono_times = build_payload(
            batch=batch,
            args=args,
            device=device,
        )
        stats = condition_stats(cond, cond_mask)
        print(
            f"[batch {batch_idx + 1:03d}] "
            f"clean={tuple(clean.shape)} noisy={tuple(noisy.shape)} "
            f"cond={tuple(cond.shape)} "
            f"valid_tokens={stats['valid_tokens']}/{stats['total_tokens']} "
            f"cond_mean={stats['mean']:.6f} cond_std={stats['std']:.6f}"
        )

        for repeat_idx in range(cli.repeats):
            loss_seed = cli.seed + batch_idx * 1009 + repeat_idx

            for mode in modes:
                (
                    mode_cond,
                    mode_cond_times,
                    mode_cond_mask,
                    mode_query_mono_times,
                ) = apply_condition_mode(
                    mode=mode,
                    cond=cond,
                    cond_times=cond_times,
                    cond_mask=cond_mask,
                    query_mono_times=query_mono_times,
                    generator=random_generator,
                )

                payload = {
                    "x": clean,
                    "y": noisy,
                    "aux_cond": mode_cond,
                    "aux_cond_times": mode_cond_times,
                    "aux_cond_mask": mode_cond_mask,
                    "aux_query_times": mode_query_mono_times,
                }
                loss = loss_with_fixed_seed(
                    model=model,
                    payload=payload,
                    seed=loss_seed,
                    device=device,
                )
                loss_value = float(loss.detach().cpu().item())
                losses[mode].append(loss_value)
                rows.append(
                    {
                        "batch_idx": batch_idx,
                        "repeat_idx": repeat_idx,
                        "mode": mode,
                        "loss": loss_value,
                        "nonfinite": int(has_nonfinite(loss)),
                    }
                )

                if has_nonfinite(loss):
                    print(
                        f"[nonfinite] batch={batch_idx} repeat={repeat_idx} "
                        f"mode={mode} loss={loss_value}"
                    )

    print("--------------------------------------------------")
    print("[summary]")
    real_mean = summarize(losses["real"])["mean"] if "real" in losses else None
    failed = False
    summary_rows = []

    for mode in modes:
        s = summarize(losses[mode])
        ratio = float("nan")
        if real_mean is not None and real_mean == real_mean and abs(real_mean) > 1e-12:
            ratio = s["mean"] / real_mean

        if s["nonfinite"] > 0:
            failed = True
        if cli.fail_ratio > 0 and ratio == ratio and ratio >= cli.fail_ratio:
            failed = True

        summary_rows.append({
            "mode": mode,
            **s,
            "ratio_vs_real": ratio,
        })

        print(
            f"{mode:14s} "
            f"mean={s['mean']:.8f} "
            f"std={s['std']:.8f} "
            f"min={s['min']:.8f} "
            f"max={s['max']:.8f} "
            f"ratio_vs_real={ratio:.4f} "
            f"nonfinite={s['nonfinite']}/{s['count']}"
        )

    if cli.out_csv:
        out_csv = Path(cli.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "batch_idx",
                    "repeat_idx",
                    "mode",
                    "loss",
                    "nonfinite",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "mode",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "nonfinite",
                    "count",
                    "ratio_vs_real",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"[saved] {out_csv}")
        print(f"[saved] {summary_csv}")

    if failed:
        raise RuntimeError("Condition loss check failed. See nonfinite/ratio summary above.")

    print("[OK] condition loss check completed")


if __name__ == "__main__":
    main()
