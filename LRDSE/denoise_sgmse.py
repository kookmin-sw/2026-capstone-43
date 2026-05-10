#!/usr/bin/env python3
import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.audio.preprocess import AudioPreprocessConfig, ensure_mono_2d, spec_fwd, stft
from train_sgmse import build_model, enhance_full_wav, set_seed


DEFAULT_TRAIN_ARGS: Dict[str, Any] = {
    "manifest": "",
    "val_manifest": "",
    "save_dir": "./checkpoints/sgmse_se",
    "device": "cuda",
    "seed": 10,
    "target_sr": 16000,
    "target_length": 32640,
    "n_fft": 510,
    "hop_length": 128,
    "win_length": 510,
    "num_frames": 256,
    "spec_factor": 0.15,
    "spec_abs_exponent": 0.5,
    "normalize": "noisy",
    "backbone": "ncsnpp_v2",
    "sde": "ouve",
    "nf": 128,
    "ch_mult": "(1, 1, 2, 2, 2, 2, 2)",
    "num_res_blocks": 2,
    "attn_resolutions": "(16,)",
    "theta": 1.5,
    "sigma_min": 0.05,
    "sigma_max": 0.5,
    "sde_N": 30,
    "k": 2.6,
    "c": 0.4,
    "sb_eps": 1e-8,
    "sampler_type": "auto",
    "t_eps": 0.03,
    "loss_type": "score_matching",
    "loss_weighting": "sigma^2",
    "network_scaling": "auto",
    "c_in": "1",
    "c_out": "1",
    "c_skip": "0",
    "sigma_data": 0.1,
    "l1_weight": 0.001,
    "pesq_weight": 0.0,
    "use_aux_cond": False,
    "aux_encoder": "identity",
    "aux_cond_dim": 8,
    "aux_hidden_dim": 128,
    "aux_scale_init": 0.1,
    "aux_time_scale": 0.05,
    "aux_time_embed_dim": 128,
    "aux_time_max_period": 10000.0,
    "raw_force_scale": 220.0,
    "d_force_scale": 255.0,
    "condition_smooth_win": 1,
    "batch_size": 4,
    "val_batch_size": 0,
    "num_workers": 2,
    "val_num_workers": -1,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "max_steps": 10000,
    "max_epochs": 0.0,
    "grad_accum": 1,
    "max_grad_norm": 1.0,
    "amp": False,
    "use_ema": True,
    "ema_decay": 0.999,
    "log_every": 10,
    "save_every": 1000,
    "save_every_epochs": 0.0,
    "disable_checkpoint_save": False,
    "save_step_checkpoints": False,
    "epoch_loss_csv": "epoch_losses.csv",
    "sample_every": 0,
    "sample_every_epochs": 0.0,
    "num_sample_wavs": 2,
    "sample_max_sec": 0.0,
    "sample_chunk_hop": 0,
    "sample_max_rms_ratio": 3.0,
    "predictor": "reverse_diffusion",
    "corrector": "ald",
    "corrector_steps": 1,
    "snr": 0.5,
    "sampling_N": 0,
    "resume": "",
    "overfit_samples": 0,
    "limit_samples": 0,
}


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def infer_aux_encoder_from_checkpoint(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        return ""

    state_dict = None
    if isinstance(checkpoint.get("model"), dict):
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint

    if not state_dict:
        return ""

    if any(str(k).startswith("aux_context_encoder.value_proj.") for k in state_dict):
        return "mlp"

    for key, value in state_dict.items():
        if "ContextK.weight" in str(key) and torch.is_tensor(value) and value.dim() == 3:
            in_channels = int(value.shape[1])
            if in_channels == 8:
                return "identity"
            if in_channels == 128:
                return "mlp"

    return ""


def load_training_args(
    checkpoint_path: Path,
    checkpoint: Any,
    explicit_args_json: str = "",
) -> Namespace:
    args_dict = dict(DEFAULT_TRAIN_ARGS)

    adjacent_args_json = checkpoint_path.parent / "args.json"
    if adjacent_args_json.exists():
        args_dict.update(read_json(adjacent_args_json))

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        checkpoint_args = checkpoint["args"]
        args_dict.update(checkpoint["args"])
    else:
        checkpoint_args = {}

    if explicit_args_json:
        args_dict.update(read_json(Path(explicit_args_json)))
    elif "aux_encoder" not in checkpoint_args:
        inferred_aux_encoder = infer_aux_encoder_from_checkpoint(checkpoint)
        if inferred_aux_encoder:
            args_dict["aux_encoder"] = inferred_aux_encoder

    return Namespace(**args_dict)


def apply_cli_overrides(args: Namespace, cli: argparse.Namespace) -> None:
    args.device = cli.device
    args.seed = cli.seed
    args.use_ema = not cli.no_ema
    args.aux_cond_mode = cli.aux_cond_mode
    args.aux_cond_seed = cli.aux_cond_seed

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


def validate_args(args: Namespace) -> None:
    if args.win_length != args.n_fft:
        raise ValueError(
            f"win_length must match n_fft for this preprocess pipeline. "
            f"got win_length={args.win_length}, n_fft={args.n_fft}"
        )

    expected_target_length = (args.num_frames - 1) * args.hop_length
    if args.target_length != expected_target_length:
        raise ValueError(
            f"target_length mismatch: got {args.target_length}, "
            f"expected {expected_target_length}"
        )

    if args.use_aux_cond and args.aux_cond_dim != 8:
        raise ValueError(
            f"aux_cond_dim must be 8 for the current aux path. got {args.aux_cond_dim}"
        )

    if args.use_aux_cond and args.backbone != "ncsnpp_v2":
        raise ValueError(
            f"use_aux_cond=True currently requires backbone='ncsnpp_v2'. "
            f"got {args.backbone}"
        )


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def state_dict_candidates(state_dict: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    yield "as-is", state_dict

    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    for prefix in ("module.", "model."):
        if keys and all(k.startswith(prefix) for k in keys):
            stripped = {
                (k[len(prefix):] if isinstance(k, str) else k): v
                for k, v in state_dict.items()
            }
            yield f"strip:{prefix}", stripped


def extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), dict):
            return checkpoint["model"], "model"
        if isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"], "state_dict"
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint, "raw"
    raise ValueError(
        "Could not find model weights. Expected checkpoint['model'], "
        "checkpoint['state_dict'], or a raw state_dict."
    )


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Any,
    strict: bool,
) -> Tuple[str, Any]:
    state_dict, source = extract_state_dict(checkpoint)
    last_error = None

    for candidate_name, candidate in state_dict_candidates(state_dict):
        try:
            incompatible = model.load_state_dict(candidate, strict=strict)
            return f"{source}/{candidate_name}", incompatible
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load model weights: {last_error}") from last_error


def load_ema_if_available(model: Any, checkpoint: Any, args: Namespace) -> bool:
    if not args.use_ema:
        return False

    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        if hasattr(model, "load_ema_state_dict"):
            model.load_ema_state_dict(checkpoint["ema"])
        else:
            model.ema.load_state_dict(checkpoint["ema"])
        return True

    print("[ema] EMA weights were requested but not found. Using raw model weights.")
    args.use_ema = False
    return False


def resample_waveform(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav

    try:
        import torchaudio.functional as AF

        return AF.resample(
            wav.unsqueeze(0),
            orig_freq=orig_sr,
            new_freq=target_sr,
        ).squeeze(0).contiguous()
    except Exception:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd
        audio = resample_poly(wav.detach().cpu().numpy(), up, down).astype(np.float32)
        return torch.from_numpy(audio).contiguous()


def load_noisy_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    wav = torch.from_numpy(audio.mean(axis=1).astype(np.float32)).contiguous()
    wav = resample_waveform(wav, sr, target_sr)
    return wav, sr


def default_output_path(noisy_path: Path, out_dir: Path, suffix: str) -> Path:
    return out_dir / f"{noisy_path.stem}{suffix}.wav"


def comparison_stem(output_path: Path, enhanced_suffix: str) -> str:
    if enhanced_suffix and output_path.stem.endswith(enhanced_suffix):
        return output_path.stem[: -len(enhanced_suffix)]
    return output_path.stem


def comparison_output_path(output_path: Path, enhanced_suffix: str, label: str) -> Path:
    return output_path.with_name(f"{comparison_stem(output_path, enhanced_suffix)}_{label}.wav")


def comparison_metrics_path(output_path: Path, enhanced_suffix: str) -> Path:
    return output_path.with_name(f"{comparison_stem(output_path, enhanced_suffix)}_metrics.json")


def waveform_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> Dict[str, float]:
    n = min(reference.numel(), estimate.numel())
    if n <= 0:
        raise ValueError("Cannot compute metrics for empty waveforms.")

    reference = reference[:n].detach().cpu().to(dtype=torch.float64)
    estimate = estimate[:n].detach().cpu().to(dtype=torch.float64)
    diff = estimate - reference

    mse = torch.mean(diff.pow(2)).item()
    mae = torch.mean(diff.abs()).item()
    rmse = math.sqrt(mse)
    ref_power = torch.mean(reference.pow(2)).item()
    snr_db = 10.0 * math.log10(ref_power / max(mse, 1e-12)) if ref_power > 0 else float("nan")

    return {
        "num_samples": int(n),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "snr_db": float(snr_db),
    }


def build_audio_cfg(args: Namespace) -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        sample_rate=args.target_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_frames=args.num_frames,
        center=True,
        spec_factor=args.spec_factor,
        spec_abs_exponent=args.spec_abs_exponent,
        normalize=args.normalize,
    )


def stft_data_loss(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    norm_source: torch.Tensor,
    args: Namespace,
) -> Dict[str, float]:
    """
    Direct clean-vs-estimate loss in the transformed STFT domain.

    This is not the diffusion score-matching training loss. It is a deterministic
    data-space loss after the same STFT/spec_fwd transform used by training.
    """
    n = min(reference.numel(), estimate.numel(), norm_source.numel())
    if n <= 0:
        raise ValueError("Cannot compute STFT loss for empty waveforms.")

    cfg = build_audio_cfg(args)
    normfac = norm_source[:n].detach().cpu().abs().max().clamp_min(cfg.eps)

    ref = ensure_mono_2d(reference[:n].detach().cpu().float() / normfac)
    est = ensure_mono_2d(estimate[:n].detach().cpu().float() / normfac)

    ref_spec = spec_fwd(stft(ref, cfg), cfg)
    est_spec = spec_fwd(stft(est, cfg), cfg)
    diff = est_spec - ref_spec
    abs_diff = diff.abs()

    complex_mse = abs_diff.pow(2).mean().item()
    complex_l1 = abs_diff.mean().item()
    realimag_mse = torch.stack([diff.real, diff.imag], dim=0).pow(2).mean().item()

    return {
        "num_samples": int(n),
        "num_freq_bins": int(ref_spec.shape[-2]),
        "num_frames": int(ref_spec.shape[-1]),
        "normfac": float(normfac.item()),
        "complex_mse": float(complex_mse),
        "complex_l1": float(complex_l1),
        "realimag_mse": float(realimag_mse),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise noisy wav files with a trained LRDSE SGMSE checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to latest.pt/best.pt/etc.")
    parser.add_argument(
        "--noisy-wav",
        required=True,
        nargs="+",
        help="One or more noisy wav/flac files to denoise.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output wav path. Only valid when a single --noisy-wav is given.",
    )
    parser.add_argument("--out-dir", default="./outputs/denoise_sgmse")
    parser.add_argument("--suffix", default="_enhanced")
    parser.add_argument(
        "--args-json",
        default="",
        help="Optional training args JSON. Overrides checkpoint/adjacent args.json values.",
    )
    parser.add_argument("--run-dir", default="", help="Aux condition run_dir override.")
    parser.add_argument(
        "--clean-wav",
        default=[],
        nargs="*",
        help="Optional clean reference wav/flac path(s). Count must match --noisy-wav.",
    )
    parser.add_argument(
        "--save-noisy",
        action="store_true",
        help="Also save the resampled noisy input next to the enhanced output.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-sec", type=float, default=0.0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-clamp", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--aux-cond-mode",
        default="real",
        choices=["real", "zero", "zero_padded", "random"],
        help="Inference-only aux condition override for ablation tests.",
    )
    parser.add_argument(
        "--aux-cond-seed",
        type=int,
        default=1234,
        help="Seed used when --aux-cond-mode random.",
    )

    parser.add_argument("--sampling-N", type=int, default=None)
    parser.add_argument("--sampler-type", choices=["auto", "pc", "ode", "sde"], default=None)
    parser.add_argument("--predictor", default="")
    parser.add_argument("--corrector", default="")
    parser.add_argument("--corrector-steps", type=int, default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--sample-chunk-hop", type=int, default=None)
    parser.add_argument("--sample-max-rms-ratio", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    noisy_paths = [Path(p).expanduser().resolve() for p in cli.noisy_wav]
    for noisy_path in noisy_paths:
        if not noisy_path.exists():
            raise FileNotFoundError(f"noisy wav not found: {noisy_path}")

    if cli.out and len(noisy_paths) != 1:
        raise ValueError("--out can be used only with a single --noisy-wav")

    clean_paths = [Path(p).expanduser().resolve() for p in cli.clean_wav]
    if clean_paths and len(clean_paths) != len(noisy_paths):
        raise ValueError("--clean-wav count must match --noisy-wav count")
    for clean_path in clean_paths:
        if not clean_path.exists():
            raise FileNotFoundError(f"clean wav not found: {clean_path}")

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    args = load_training_args(checkpoint_path, checkpoint, cli.args_json)
    apply_cli_overrides(args, cli)
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    model = build_model(args)
    weights_source, incompatible = load_model_weights(
        model=model,
        checkpoint=checkpoint,
        strict=(not cli.non_strict),
    )
    ema_loaded = load_ema_if_available(model, checkpoint, args)
    model = model.to(device)

    step = checkpoint.get("step", "?") if isinstance(checkpoint, dict) else "?"
    loss = checkpoint.get("loss", None) if isinstance(checkpoint, dict) else None
    print("--------------------------------------------------")
    print("[denoise config]")
    print(f"checkpoint         : {checkpoint_path}")
    print(f"weights            : {weights_source}")
    print(f"step               : {step}")
    if loss is not None:
        print(f"loss               : {float(loss):.8f}")
    print(f"device             : {device}")
    print(f"use_ema            : {args.use_ema} (loaded={ema_loaded})")
    print(f"target_sr          : {args.target_sr}")
    print(f"target_length      : {args.target_length}")
    print(f"backbone           : {args.backbone}")
    print(f"sde                : {args.sde}")
    print(f"sampler_type       : {args.sampler_type}")
    print(f"sampling_N         : {args.sampling_N}")
    print(f"use_aux_cond       : {args.use_aux_cond}")
    print(f"aux_encoder        : {args.aux_encoder}")
    print(f"aux_cond_mode      : {args.aux_cond_mode}")
    if cli.non_strict:
        print(f"missing_keys       : {len(incompatible.missing_keys)}")
        print(f"unexpected_keys    : {len(incompatible.unexpected_keys)}")
    print("--------------------------------------------------")

    out_dir = Path(cli.out_dir).expanduser().resolve()

    for file_idx, noisy_path in enumerate(noisy_paths):
        output_path = (
            Path(cli.out).expanduser().resolve()
            if cli.out
            else default_output_path(noisy_path, out_dir, cli.suffix)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        noisy_wav, original_sr = load_noisy_audio(noisy_path, target_sr=args.target_sr)
        if cli.max_sec > 0:
            noisy_wav = noisy_wav[: int(cli.max_sec * args.target_sr)]
        if noisy_wav.numel() == 0:
            raise ValueError(f"No audio samples left after trimming: {noisy_path}")

        run_dir = None
        if args.use_aux_cond:
            run_dir = str(Path(cli.run_dir).expanduser().resolve()) if cli.run_dir else str(noisy_path.parent)

        print(
            f"[input] {noisy_path} "
            f"original_sr={original_sr} "
            f"samples={noisy_wav.numel()} "
            f"duration={noisy_wav.numel() / args.target_sr:.2f}s "
            f"run_dir={run_dir}"
        )

        enhanced_wav = enhance_full_wav(
            model=model,
            noisy_wav=noisy_wav,
            args=args,
            device=device,
            run_dir=run_dir,
        )
        if not cli.no_clamp:
            enhanced_wav = torch.clamp(enhanced_wav, min=-1.0, max=1.0)

        sf.write(str(output_path), enhanced_wav.detach().cpu().numpy(), args.target_sr)
        print(f"[saved] {output_path}")

        if cli.save_noisy:
            noisy_out = comparison_output_path(output_path, cli.suffix, "noisy")
            sf.write(str(noisy_out), noisy_wav.detach().cpu().numpy(), args.target_sr)
            print(f"[saved] {noisy_out}")

        if clean_paths:
            clean_wav, clean_original_sr = load_noisy_audio(
                clean_paths[file_idx],
                target_sr=args.target_sr,
            )
            if cli.max_sec > 0:
                clean_wav = clean_wav[: int(cli.max_sec * args.target_sr)]
            if clean_wav.numel() > enhanced_wav.numel():
                clean_wav = clean_wav[: enhanced_wav.numel()]
            elif clean_wav.numel() < enhanced_wav.numel():
                clean_wav = torch.nn.functional.pad(
                    clean_wav,
                    (0, enhanced_wav.numel() - clean_wav.numel()),
                    mode="constant",
                )

            clean_out = comparison_output_path(output_path, cli.suffix, "clean")
            sf.write(str(clean_out), clean_wav.detach().cpu().numpy(), args.target_sr)
            print(f"[saved] {clean_out} (original_sr={clean_original_sr})")

            metrics = {
                "sample_rate": int(args.target_sr),
                "noisy_wav": str(noisy_path),
                "clean_wav": str(clean_paths[file_idx]),
                "enhanced_wav": str(output_path),
                "waveform_error_note": (
                    "Direct waveform error against clean. This is not the diffusion "
                    "score-matching training loss."
                ),
                "waveform_error_noisy_vs_clean": waveform_metrics(clean_wav, noisy_wav),
                "waveform_error_enhanced_vs_clean": waveform_metrics(clean_wav, enhanced_wav),
                "stft_data_loss_note": (
                    "Direct clean-vs-estimate loss after LRDSE STFT/spec_fwd, "
                    "normalized by the original noisy max amplitude. This is still "
                    "not the random-timestep diffusion score-matching training loss."
                ),
                "stft_data_loss_noisy_vs_clean": stft_data_loss(
                    reference=clean_wav,
                    estimate=noisy_wav,
                    norm_source=noisy_wav,
                    args=args,
                ),
                "stft_data_loss_enhanced_vs_clean": stft_data_loss(
                    reference=clean_wav,
                    estimate=enhanced_wav,
                    norm_source=noisy_wav,
                    args=args,
                ),
            }
            metrics_out = comparison_metrics_path(output_path, cli.suffix)
            with metrics_out.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            print("[waveform error vs clean]")
            print(
                "  noisy    "
                f"mse={metrics['waveform_error_noisy_vs_clean']['mse']:.10f} "
                f"mae={metrics['waveform_error_noisy_vs_clean']['mae']:.10f} "
                f"rmse={metrics['waveform_error_noisy_vs_clean']['rmse']:.10f} "
                f"snr_db={metrics['waveform_error_noisy_vs_clean']['snr_db']:.4f}"
            )
            print(
                "  enhanced "
                f"mse={metrics['waveform_error_enhanced_vs_clean']['mse']:.10f} "
                f"mae={metrics['waveform_error_enhanced_vs_clean']['mae']:.10f} "
                f"rmse={metrics['waveform_error_enhanced_vs_clean']['rmse']:.10f} "
                f"snr_db={metrics['waveform_error_enhanced_vs_clean']['snr_db']:.4f}"
            )
            print("[stft data loss vs clean]")
            print(
                "  noisy    "
                f"complex_mse={metrics['stft_data_loss_noisy_vs_clean']['complex_mse']:.10f} "
                f"complex_l1={metrics['stft_data_loss_noisy_vs_clean']['complex_l1']:.10f} "
                f"realimag_mse={metrics['stft_data_loss_noisy_vs_clean']['realimag_mse']:.10f}"
            )
            print(
                "  enhanced "
                f"complex_mse={metrics['stft_data_loss_enhanced_vs_clean']['complex_mse']:.10f} "
                f"complex_l1={metrics['stft_data_loss_enhanced_vs_clean']['complex_l1']:.10f} "
                f"realimag_mse={metrics['stft_data_loss_enhanced_vs_clean']['realimag_mse']:.10f}"
            )
            print(f"[saved] {metrics_out}")


if __name__ == "__main__":
    main()
