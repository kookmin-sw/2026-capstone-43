#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import SpeechEnhancementDataset, infer_run_dir
from denoise_sgmse import (
    load_noisy_audio,
    load_ema_if_available,
    load_model_weights,
    load_training_args,
    resolve_device,
    stft_data_loss,
    torch_load,
    validate_args,
    waveform_metrics,
)
from train_sgmse import build_model, enhance_full_wav, set_seed, twoch_to_complex_batch


MODE_CHOICES = {
    "real",
    "zero",
    "zero_padded",
    "random",
    "shuffle",
    "no_condition",
}

DENOISE_MODE_CHOICES = {
    "real",
    "zero",
    "zero_padded",
    "random",
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


def parse_denoise_modes(value: str):
    modes = [m.strip() for m in value.split(",") if m.strip()]
    if not modes:
        raise ValueError("--sample-modes must contain at least one mode")

    invalid = [m for m in modes if m not in DENOISE_MODE_CHOICES]
    if invalid:
        raise ValueError(
            f"invalid sample mode(s): {invalid}. "
            f"choices={sorted(DENOISE_MODE_CHOICES)}"
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


def apply_sampling_overrides(args, cli):
    if cli.sampling_N is not None:
        args.sampling_N = int(cli.sampling_N)
    if cli.sampler_type is not None:
        args.sampler_type = cli.sampler_type
    if cli.predictor:
        args.predictor = cli.predictor
    if cli.corrector:
        args.corrector = cli.corrector
    if cli.corrector_steps is not None:
        args.corrector_steps = int(cli.corrector_steps)
    if cli.snr is not None:
        args.snr = float(cli.snr)
    if cli.sample_chunk_hop is not None:
        args.sample_chunk_hop = int(cli.sample_chunk_hop)
    if cli.sample_max_rms_ratio is not None:
        args.sample_max_rms_ratio = float(cli.sample_max_rms_ratio)


def build_dataset(args, cli, limit, use_condition: bool):
    return SpeechEnhancementDataset(
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
        limit=limit,
        use_condition=use_condition,
        condition_repr="8ch",
        raw_force_scale=args.raw_force_scale,
        d_force_scale=args.d_force_scale,
        condition_smooth_win=args.condition_smooth_win,
    )


def sanitize_name(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "sample"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def align_waveform_length(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.numel() == target_len:
        return wav
    if wav.numel() > target_len:
        return wav[:target_len]
    return torch.nn.functional.pad(wav, (0, target_len - wav.numel()), mode="constant")


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@torch.no_grad()
def enhance_with_fixed_seed(
    model,
    noisy_wav: torch.Tensor,
    args,
    device: torch.device,
    run_dir: str,
    seed: int,
):
    with torch.random.fork_rng(devices=rng_devices(device), enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return enhance_full_wav(
            model=model,
            noisy_wav=noisy_wav,
            args=args,
            device=device,
            run_dir=run_dir,
        )


def run_loss_check(
    cli,
    args,
    checkpoint_path: Path,
    checkpoint,
    model,
    device: torch.device,
    weights_source: str,
    incompatible,
    ema_loaded: bool,
):
    modes = parse_modes(cli.modes)
    dataset = build_dataset(
        args=args,
        cli=cli,
        limit=max(1, cli.batch_size * cli.num_batches),
        use_condition=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

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

        summary_rows.append(
            {
                "mode": mode,
                **s,
                "ratio_vs_real": ratio,
            }
        )

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


def run_sample_denoise_comparison(
    cli,
    args,
    checkpoint_path: Path,
    model,
    device: torch.device,
):
    sample_modes = parse_denoise_modes(cli.sample_modes)
    rows_dataset = build_dataset(
        args=args,
        cli=cli,
        limit=None,
        use_condition=False,
    )

    if cli.sample_index < 0 or cli.sample_index >= len(rows_dataset):
        raise IndexError(
            f"--sample-index out of range: {cli.sample_index} "
            f"(available 0..{len(rows_dataset) - 1})"
        )

    row = rows_dataset.rows[cli.sample_index]
    noisy_path = Path(row["noisy_wav"]).expanduser().resolve()
    clean_path = Path(row["clean_wav"]).expanduser().resolve()
    run_dir = infer_run_dir(row) if args.use_aux_cond else None

    noisy_wav, noisy_original_sr = load_noisy_audio(noisy_path, target_sr=args.target_sr)
    clean_wav, clean_original_sr = load_noisy_audio(clean_path, target_sr=args.target_sr)

    if cli.sample_max_sec > 0:
        max_len = int(cli.sample_max_sec * args.target_sr)
        noisy_wav = noisy_wav[:max_len]
        clean_wav = clean_wav[:max_len]

    if noisy_wav.numel() == 0:
        raise ValueError(f"No audio samples left after trimming: {noisy_path}")

    clean_wav = align_waveform_length(clean_wav, noisy_wav.numel())

    sample_id = row.get("id", "") or row.get("source_id", "") or noisy_path.stem
    safe_sample_id = sanitize_name(sample_id)
    out_dir = (
        Path(cli.sample_out_dir).expanduser().resolve()
        if cli.sample_out_dir
        else (
            checkpoint_path.parent
            / "condition_denoise"
            / f"{cli.sample_index:04d}_{safe_sample_id}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    noisy_out = out_dir / f"{safe_sample_id}_noisy.wav"
    clean_out = out_dir / f"{safe_sample_id}_clean.wav"
    sf.write(str(noisy_out), noisy_wav.detach().cpu().numpy(), args.target_sr)
    sf.write(str(clean_out), clean_wav.detach().cpu().numpy(), args.target_sr)

    sampling_seed = cli.seed + cli.sample_index * 1009
    original_aux_mode = getattr(args, "aux_cond_mode", "real")
    original_aux_seed = getattr(args, "aux_cond_seed", cli.sample_aux_cond_seed)

    print("--------------------------------------------------")
    print("[condition denoise compare]")
    print(f"sample_index      : {cli.sample_index}")
    print(f"sample_id         : {sample_id}")
    print(f"checkpoint        : {checkpoint_path}")
    print(f"noisy_wav         : {noisy_path}")
    print(f"clean_wav         : {clean_path}")
    print(f"run_dir           : {run_dir}")
    print(f"original_sr       : noisy={noisy_original_sr}, clean={clean_original_sr}")
    print(f"duration          : {noisy_wav.numel() / args.target_sr:.2f}s")
    print(f"sample_modes      : {', '.join(sample_modes)}")
    print(f"sampling_seed     : {sampling_seed}")
    print(f"aux_cond_seed     : {cli.sample_aux_cond_seed}")
    print(f"sampler_type      : {args.sampler_type}")
    print(f"sampling_N        : {args.sampling_N}")
    print(f"saved_input_noisy : {noisy_out}")
    print(f"saved_input_clean : {clean_out}")
    print("--------------------------------------------------")

    baseline_metrics = {
        "waveform_error_vs_clean": waveform_metrics(clean_wav, noisy_wav),
        "stft_data_loss_vs_clean": stft_data_loss(
            reference=clean_wav,
            estimate=noisy_wav,
            norm_source=noisy_wav,
            args=args,
        ),
    }

    results = {
        "sample_index": int(cli.sample_index),
        "sample_id": str(sample_id),
        "checkpoint": str(checkpoint_path),
        "manifest": str(Path(cli.manifest).expanduser().resolve()),
        "noisy_wav": str(noisy_path),
        "clean_wav": str(clean_path),
        "run_dir": None if run_dir is None else str(run_dir),
        "sample_rate": int(args.target_sr),
        "sampling_seed": int(sampling_seed),
        "aux_cond_seed": int(cli.sample_aux_cond_seed),
        "sample_modes": sample_modes,
        "saved_noisy_wav": str(noisy_out),
        "saved_clean_wav": str(clean_out),
        "noisy_vs_clean": baseline_metrics,
        "modes": {},
    }

    enhanced_by_mode = {}

    try:
        for mode in sample_modes:
            args.aux_cond_mode = mode
            args.aux_cond_seed = cli.sample_aux_cond_seed

            enhanced_wav = enhance_with_fixed_seed(
                model=model,
                noisy_wav=noisy_wav,
                args=args,
                device=device,
                run_dir=run_dir,
                seed=sampling_seed,
            )
            if not cli.sample_no_clamp:
                enhanced_wav = torch.clamp(enhanced_wav, min=-1.0, max=1.0)

            out_path = out_dir / f"{safe_sample_id}_{mode}.wav"
            sf.write(str(out_path), enhanced_wav.detach().cpu().numpy(), args.target_sr)
            enhanced_by_mode[mode] = enhanced_wav.detach().cpu()

            mode_metrics = {
                "output_wav": str(out_path),
                "waveform_error_vs_clean": waveform_metrics(clean_wav, enhanced_wav),
                "stft_data_loss_vs_clean": stft_data_loss(
                    reference=clean_wav,
                    estimate=enhanced_wav,
                    norm_source=noisy_wav,
                    args=args,
                ),
            }
            results["modes"][mode] = mode_metrics

            print(
                f"[mode {mode:11s}] "
                f"snr_db={mode_metrics['waveform_error_vs_clean']['snr_db']:.4f} "
                f"rmse={mode_metrics['waveform_error_vs_clean']['rmse']:.8f} "
                f"complex_mse={mode_metrics['stft_data_loss_vs_clean']['complex_mse']:.8f} "
                f"saved={out_path}"
            )
    finally:
        args.aux_cond_mode = original_aux_mode
        args.aux_cond_seed = original_aux_seed

    real_wav = enhanced_by_mode.get("real", None)
    if real_wav is not None:
        for mode, enhanced_wav in enhanced_by_mode.items():
            if mode == "real":
                continue
            results["modes"][mode]["waveform_delta_vs_real"] = waveform_metrics(
                real_wav,
                enhanced_wav,
            )
            results["modes"][mode]["stft_data_loss_vs_real"] = stft_data_loss(
                reference=real_wav,
                estimate=enhanced_wav,
                norm_source=noisy_wav,
                args=args,
            )

    summary_rows = []
    for mode in sample_modes:
        entry = results["modes"][mode]
        delta_wave = entry.get("waveform_delta_vs_real", {})
        delta_stft = entry.get("stft_data_loss_vs_real", {})
        summary_rows.append(
            {
                "mode": mode,
                "output_wav": entry["output_wav"],
                "clean_snr_db": entry["waveform_error_vs_clean"]["snr_db"],
                "clean_rmse": entry["waveform_error_vs_clean"]["rmse"],
                "clean_complex_mse": entry["stft_data_loss_vs_clean"]["complex_mse"],
                "delta_vs_real_rmse": delta_wave.get("rmse", float("nan")),
                "delta_vs_real_complex_mse": delta_stft.get("complex_mse", float("nan")),
            }
        )

    summary_json = out_dir / "metrics.json"
    summary_csv = out_dir / "metrics_summary.csv"
    write_json(summary_json, results)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "output_wav",
                "clean_snr_db",
                "clean_rmse",
                "clean_complex_mse",
                "delta_vs_real_rmse",
                "delta_vs_real_complex_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("[baseline noisy vs clean]")
    print(
        f"snr_db={baseline_metrics['waveform_error_vs_clean']['snr_db']:.4f} "
        f"rmse={baseline_metrics['waveform_error_vs_clean']['rmse']:.8f} "
        f"complex_mse={baseline_metrics['stft_data_loss_vs_clean']['complex_mse']:.8f}"
    )
    print(f"[saved] {summary_json}")
    print(f"[saved] {summary_csv}")
    print("[OK] condition denoise comparison completed")


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
        "--skip-loss",
        action="store_true",
        help="Loss 비교 루프를 건너뛰고 sample denoise 비교만 실행.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="0 이상이면 filtered manifest 기준 이 index의 full sample을 denoise 비교.",
    )
    parser.add_argument(
        "--sample-modes",
        default="real,zero,random",
        help="Sample denoise 비교 mode 목록. choices=real,zero,zero_padded,random",
    )
    parser.add_argument(
        "--sample-out-dir",
        default="",
        help="Sample denoise 결과 저장 디렉터리. 비우면 checkpoint 옆에 자동 생성.",
    )
    parser.add_argument(
        "--sample-max-sec",
        type=float,
        default=0.0,
        help="0보다 크면 sample denoise 비교 시 앞부분만 사용.",
    )
    parser.add_argument(
        "--sample-aux-cond-seed",
        type=int,
        default=1234,
        help="Sample denoise 비교에서 random cond 생성 seed.",
    )
    parser.add_argument(
        "--sample-no-clamp",
        action="store_true",
        help="Sample denoise 결과 wav를 [-1, 1]로 clamp하지 않음.",
    )
    parser.add_argument("--sampling-N", type=int, default=None)
    parser.add_argument("--sampler-type", choices=["auto", "pc", "ode", "sde"], default=None)
    parser.add_argument("--predictor", default="")
    parser.add_argument("--corrector", default="")
    parser.add_argument("--corrector-steps", type=int, default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--sample-chunk-hop", type=int, default=None)
    parser.add_argument("--sample-max-rms-ratio", type=float, default=None)
    parser.add_argument(
        "--fail-ratio",
        type=float,
        default=0.0,
        help="0보다 크면 mean_loss / real_mean_loss가 이 값 이상인 mode가 있을 때 실패 처리.",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    set_seed(cli.seed)

    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    args = load_training_args(checkpoint_path, checkpoint, cli.args_json)
    args.device = cli.device
    args.seed = cli.seed
    args.use_ema = not cli.no_ema
    apply_sampling_overrides(args, cli)
    validate_args(args)

    if not args.use_aux_cond:
        raise RuntimeError(
            "Checkpoint args say use_aux_cond=False, so condition ablation would not affect the model. "
            "Use an aux-condition SGMSE checkpoint."
        )

    device = resolve_device(args.device)
    model = build_model(args)
    weights_source, incompatible = load_model_weights(
        model=model,
        checkpoint=checkpoint,
        strict=(not cli.non_strict),
    )
    ema_loaded = load_ema_if_available(model, checkpoint, args)
    model = model.to(device)
    model.eval(no_ema=(not args.use_ema))

    run_loss = (not cli.skip_loss) and cli.num_batches > 0
    run_sample = cli.sample_index >= 0
    if not run_loss and not run_sample:
        raise RuntimeError(
            "Nothing to do. Use loss mode (default) or pass --sample-index >= 0 "
            "for sample denoise comparison."
        )

    if run_loss:
        run_loss_check(
            cli=cli,
            args=args,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model=model,
            device=device,
            weights_source=weights_source,
            incompatible=incompatible,
            ema_loaded=ema_loaded,
        )

    if run_sample:
        run_sample_denoise_comparison(
            cli=cli,
            args=args,
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
        )


if __name__ == "__main__":
    main()
