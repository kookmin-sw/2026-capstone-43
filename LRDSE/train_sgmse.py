import argparse
import ast
import gc
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataset import SpeechEnhancementDataset, load_mono_audio

from src.audio.preprocess import (
    AudioPreprocessConfig,
    ensure_mono_2d,
    normalize_noisy,
    stft,
    istft,
    spec_fwd,
    spec_back,
)

from src.condition.preprocess import (
    ConditionPreprocessConfig,
    preprocess_condition_for_train,
)

from src.models.model_sgmse import ScoreModel
from src.lrdse_sgmse.util.other import pad_spec


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_sequence(s: str):
    value = ast.literal_eval(s)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    raise ValueError(f"invalid int sequence: {s}")


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def autocast_context(device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=True)


def build_preprocess_cfg(args):
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


def build_condition_cfg(args):
    return ConditionPreprocessConfig(
        raw_force_scale=args.raw_force_scale,
        d_force_scale=args.d_force_scale,
        smooth_win=args.condition_smooth_win,
    )


def twoch_to_complex_batch(x_2ch: torch.Tensor) -> torch.Tensor:
    if x_2ch.dim() != 4 or x_2ch.size(1) != 2:
        raise ValueError(f"Expected [B, 2, F, T], got {tuple(x_2ch.shape)}")
    return torch.complex(x_2ch[:, 0], x_2ch[:, 1]).unsqueeze(1)


def build_model(args):
    ch_mult = parse_int_sequence(args.ch_mult)
    attn_resolutions = parse_int_sequence(args.attn_resolutions)

    if args.network_scaling == "auto":
        # SGMSE+ ncsnpp_v2 + score_matching commonly uses 1/t scaling.
        if args.backbone == "ncsnpp_v2" and args.loss_type == "score_matching":
            network_scaling = "1/t"
        else:
            network_scaling = None
    else:
        network_scaling = None if args.network_scaling == "none" else args.network_scaling
    if args.sampler_type == "auto":
        sde_sampler_type = "pc" if args.sde == "ouve" else "ode"
    else:
        sde_sampler_type = args.sampler_type

    args._resolved_network_scaling = network_scaling

    model = ScoreModel(
        backbone=args.backbone,
        sde=args.sde,
        lr=args.lr,
        ema_decay=args.ema_decay,
        t_eps=args.t_eps,
        num_eval_files=0,
        loss_type=args.loss_type,
        loss_weighting=args.loss_weighting,
        network_scaling=network_scaling,
        c_in=args.c_in,
        c_out=args.c_out,
        c_skip=args.c_skip,
        sigma_data=args.sigma_data,
        l1_weight=args.l1_weight,
        pesq_weight=args.pesq_weight,
        sr=args.target_sr,
        use_aux_cond=args.use_aux_cond,
        aux_cond_dim=args.aux_cond_dim,
        aux_hidden_dim=args.aux_hidden_dim,
        aux_scale_init=args.aux_scale_init,
        nf=args.nf,
        ch_mult=ch_mult,
        num_res_blocks=args.num_res_blocks,
        attn_resolutions=attn_resolutions,
        theta=args.theta,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        N=args.sde_N,
        sampler_type=sde_sampler_type,
        k=args.k,
        c=args.c,
        eps=args.sb_eps,
    )
    return model


def stabilize_sample_chunk(
    enhanced_chunk: torch.Tensor,
    noisy_chunk: torch.Tensor,
    max_rms_ratio: float = 0.0,
) -> torch.Tensor:
    """
    Limit extreme energy bursts in generated chunks.
    This only affects sampling-time waveform export, not training loss.
    """
    if max_rms_ratio <= 0:
        return enhanced_chunk

    noisy_rms = torch.sqrt(torch.mean(noisy_chunk.pow(2)) + 1e-8)
    enh_rms = torch.sqrt(torch.mean(enhanced_chunk.pow(2)) + 1e-8)
    max_allowed = noisy_rms * float(max_rms_ratio)

    if enh_rms > max_allowed:
        scale = (max_allowed / enh_rms).clamp(min=0.0, max=1.0)
        return enhanced_chunk * scale
    return enhanced_chunk


def save_checkpoint(path, model, optimizer, scaler, args, step, loss_value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": float(loss_value),
        "args": vars(args),
    }

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    ckpt["ema"] = model.ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if "ema" in ckpt:
        model.ema.load_state_dict(ckpt["ema"])
    else:
        # Keep behavior sane for older checkpoints without explicit ema payload.
        model.ema = model.ema.__class__(model.dnn.parameters(), decay=model.ema_decay)

    return int(ckpt.get("step", 0)), float(ckpt.get("loss", 0.0))


@torch.no_grad()
def build_aux_cond_for_chunk(
    run_dir,
    chunk_start_sample,
    noisy_spec,
    args,
    device,
):
    if run_dir is None or str(run_dir) == "":
        raise RuntimeError(
            "use_aux_cond=True but run_dir is missing. "
            "Dataset meta must contain run_dir or noisy_wav parent must contain anchors/lowstate."
        )

    cond_cfg = build_condition_cfg(args)
    cond_out = preprocess_condition_for_train(
        run_dir=str(run_dir),
        crop_start_sample=int(chunk_start_sample),
        num_frames=noisy_spec.shape[-1],
        hop_length=args.hop_length,
        freq_bins=noisy_spec.shape[-2],
        cfg=cond_cfg,
    )
    return cond_out["cond_8ch"].unsqueeze(0).to(device, non_blocking=True)


@torch.no_grad()
def build_sampler(model, y_spec, aux_cond, args):
    N = args.sampling_N if args.sampling_N > 0 else model.sde.N
    sampler_type = args.sampler_type if args.sampler_type != "auto" else model.sde.sampler_type

    sde_name = model.sde.__class__.__name__

    if sde_name == "OUVESDE":
        if sampler_type == "pc":
            return model.get_pc_sampler(
                predictor_name=args.predictor,
                corrector_name=args.corrector,
                y=y_spec,
                N=N,
                corrector_steps=args.corrector_steps,
                snr=args.snr,
                intermediate=False,
                aux_cond=aux_cond,
            )
        if sampler_type == "ode":
            return model.get_ode_sampler(
                y=y_spec,
                N=N,
                aux_cond=aux_cond,
            )
        raise ValueError(f"Invalid sampler_type={sampler_type} for OUVESDE")

    if sde_name == "SBVESDE":
        return model.get_sb_sampler(
            sde=model.sde,
            y=y_spec,
            sampler_type=sampler_type,
            N=N,
            aux_cond=aux_cond,
        )

    raise ValueError(f"Unsupported SDE type: {sde_name}")


@torch.no_grad()
def enhance_full_wav(model, noisy_wav, args, device, run_dir=None):
    was_training = model.training
    model.eval(no_ema=(not args.use_ema))

    cfg = build_preprocess_cfg(args)

    orig_len = noisy_wav.numel()
    chunk_len = args.target_length
    if args.sample_chunk_hop > 0:
        chunk_hop = args.sample_chunk_hop
    else:
        chunk_hop = chunk_len // 2

    noisy_norm_full, normfac = normalize_noisy(ensure_mono_2d(noisy_wav), cfg)
    noisy_norm_full = noisy_norm_full.squeeze(0)

    if orig_len <= chunk_len:
        starts = [0]
    else:
        starts = list(range(0, orig_len - chunk_len + 1, chunk_hop))
        last_start = orig_len - chunk_len
        if starts[-1] != last_start:
            starts.append(last_start)

    out = torch.zeros(orig_len, dtype=torch.float32)
    weight = torch.zeros(orig_len, dtype=torch.float32)
    window = torch.hann_window(chunk_len, periodic=False)
    window = torch.clamp(window, min=1e-4)

    print(
        f"[full sample] "
        f"orig_len={orig_len}, "
        f"duration={orig_len / args.target_sr:.2f}s, "
        f"chunk_len={chunk_len}, "
        f"hop={chunk_hop}, "
        f"chunks={len(starts)}, "
        f"normfac={normfac.item():.6f}, "
        f"use_aux_cond={args.use_aux_cond}"
    )

    for ci, start in enumerate(starts):
        chunk = noisy_norm_full[start:start + chunk_len]
        if chunk.numel() < chunk_len:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_len - chunk.numel()), mode="constant")

        chunk_2d = ensure_mono_2d(chunk)
        y_spec = spec_fwd(stft(chunk_2d, cfg), cfg).unsqueeze(0).to(device)
        orig_frames = y_spec.shape[-1]
        y_spec = pad_spec(y_spec).to(device)

        aux_cond = None
        if args.use_aux_cond:
            aux_cond = build_aux_cond_for_chunk(
                run_dir=run_dir,
                chunk_start_sample=start,
                noisy_spec=y_spec.squeeze(0),
                args=args,
                device=device,
            )

        print(
            f"[sample chunk] "
            f"{ci + 1}/{len(starts)} "
            f"input={tuple(y_spec.shape)} "
            f"aux={None if aux_cond is None else tuple(aux_cond.shape)} "
            f"start={start}"
        )

        sampler = build_sampler(model=model, y_spec=y_spec, aux_cond=aux_cond, args=args)
        sample, _ = sampler()

        enhanced_spec = sample.squeeze(0)[..., :orig_frames].detach().cpu()
        enhanced_chunk_norm = istft(spec_back(enhanced_spec, cfg), cfg, length=chunk_len).squeeze(0)
        enhanced_chunk_norm = stabilize_sample_chunk(
            enhanced_chunk=enhanced_chunk_norm,
            noisy_chunk=chunk,
            max_rms_ratio=args.sample_max_rms_ratio,
        )

        valid_len = min(chunk_len, orig_len - start)
        out[start:start + valid_len] += enhanced_chunk_norm[:valid_len] * window[:valid_len]
        weight[start:start + valid_len] += window[:valid_len]

        del sampler
        del sample
        del y_spec
        del enhanced_spec
        del enhanced_chunk_norm
        if aux_cond is not None:
            del aux_cond
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    enhanced_norm = out / torch.clamp(weight, min=1e-8)
    enhanced_wav = enhanced_norm * normfac.cpu()

    if was_training:
        model.train()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return enhanced_wav


@torch.no_grad()
def save_sample_wavs(model, batch, args, device, step):
    was_training = model.training
    model.eval(no_ema=(not args.use_ema))

    sample_dir = Path(args.save_dir) / "samples" / f"step_{step:07d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = batch.get("meta", None)
    if meta is None:
        raise RuntimeError("batch does not contain meta. Cannot save full sample wavs.")

    noisy_paths = meta.get("noisy_wav", None)
    clean_paths = meta.get("clean_wav", None)
    ids = meta.get("id", None)
    run_dirs = meta.get("run_dir", None)

    if noisy_paths is None or clean_paths is None:
        raise RuntimeError("batch meta does not contain noisy_wav / clean_wav paths.")

    max_items = min(args.num_sample_wavs, len(noisy_paths))

    for i in range(max_items):
        noisy_path = noisy_paths[i]
        clean_path = clean_paths[i]

        sample_id = ids[i] if ids is not None else f"{i:02d}"
        safe_id = str(sample_id).replace("/", "_").replace(" ", "_")

        run_dir = None
        if run_dirs is not None:
            run_dir = run_dirs[i]
        elif args.use_aux_cond:
            run_dir = str(Path(noisy_path).parent)

        noisy_wav, _ = load_mono_audio(noisy_path, target_sr=args.target_sr)
        clean_wav, _ = load_mono_audio(clean_path, target_sr=args.target_sr)

        min_len = min(noisy_wav.numel(), clean_wav.numel())
        noisy_wav = noisy_wav[:min_len]
        clean_wav = clean_wav[:min_len]

        if args.sample_max_sec > 0:
            max_len = int(args.sample_max_sec * args.target_sr)
            noisy_wav = noisy_wav[:max_len]
            clean_wav = clean_wav[:max_len]

        enhanced_wav = enhance_full_wav(
            model=model,
            noisy_wav=noisy_wav,
            args=args,
            device=device,
            run_dir=run_dir,
        )
        enhanced_wav = torch.clamp(enhanced_wav, min=-1.0, max=1.0)

        sf.write(sample_dir / f"{i:02d}_{safe_id}_noisy_full.wav", noisy_wav.numpy(), args.target_sr)
        sf.write(sample_dir / f"{i:02d}_{safe_id}_clean_full.wav", clean_wav.numpy(), args.target_sr)
        sf.write(sample_dir / f"{i:02d}_{safe_id}_enhanced_full.wav", enhanced_wav.numpy(), args.target_sr)

        print(
            f"[sample full] {i:02d} "
            f"id={sample_id} "
            f"duration={noisy_wav.numel() / args.target_sr:.2f}s "
            f"run_dir={run_dir} "
            f"enhanced_range={enhanced_wav.min().item():.6f}/{enhanced_wav.max().item():.6f} "
            f"saved={sample_dir}"
        )
        del noisy_wav
        del clean_wav
        del enhanced_wav
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if was_training:
        model.train()

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--save-dir", default="./checkpoints/sgmse_se")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=32640)
    parser.add_argument("--n-fft", type=int, default=510)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=510)
    parser.add_argument("--num-frames", type=int, default=256)

    parser.add_argument("--spec-factor", type=float, default=0.15)
    parser.add_argument("--spec-abs-exponent", type=float, default=0.5)
    parser.add_argument("--normalize", default="noisy", choices=["noisy", "clean", "not"])

    parser.add_argument("--backbone", default="ncsnpp_v2")
    parser.add_argument("--sde", default="ouve", choices=["ouve", "sbve"])

    parser.add_argument("--nf", type=int, default=128)
    parser.add_argument("--ch-mult", default="(1, 1, 2, 2, 2, 2, 2)")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attn-resolutions", default="(16,)")

    parser.add_argument("--theta", type=float, default=1.5)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--sde-N", type=int, default=30)
    parser.add_argument("--k", type=float, default=2.6)
    parser.add_argument("--c", type=float, default=0.4)
    parser.add_argument("--sb-eps", type=float, default=1e-8)
    parser.add_argument("--sampler-type", default="auto", choices=["auto", "pc", "ode", "sde"])

    parser.add_argument("--t-eps", type=float, default=0.03)
    parser.add_argument("--loss-type", default="score_matching", choices=["score_matching", "denoiser", "data_prediction"])
    parser.add_argument("--loss-weighting", default="sigma^2")
    parser.add_argument("--network-scaling", default="auto", choices=["auto", "none", "1/sigma", "1/t"])
    parser.add_argument("--c-in", default="1", choices=["1", "edm"])
    parser.add_argument("--c-out", default="1", choices=["1", "sigma", "1/sigma", "edm"])
    parser.add_argument("--c-skip", default="0", choices=["0", "edm"])
    parser.add_argument("--sigma-data", type=float, default=0.1)
    parser.add_argument("--l1-weight", type=float, default=0.001)
    parser.add_argument("--pesq-weight", type=float, default=0.0)

    parser.add_argument("--use-aux-cond", action="store_true")
    parser.add_argument("--aux-cond-dim", type=int, default=8)
    parser.add_argument("--aux-hidden-dim", type=int, default=128)
    parser.add_argument("--aux-scale-init", type=float, default=0.1)
    parser.add_argument("--raw-force-scale", type=float, default=220.0)
    parser.add_argument("--d-force-scale", type=float, default=9220.325595510363)
    parser.add_argument("--condition-smooth-win", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument(
        "--max-epochs",
        type=float,
        default=0.0,
        help="0보다 크면 max_steps 대신 epoch 기준으로 총 optimizer step 수를 계산.",
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--num-sample-wavs", type=int, default=2)
    parser.add_argument("--sample-max-sec", type=float, default=0.0)
    parser.add_argument("--sample-chunk-hop", type=int, default=0)
    parser.add_argument(
        "--sample-max-rms-ratio",
        type=float,
        default=3.0,
        help="샘플 저장 시 enhanced chunk RMS가 noisy chunk RMS의 이 배수를 넘으면 자동 스케일 다운. "
             "0 이하면 비활성화.",
    )
    parser.add_argument("--predictor", default="reverse_diffusion")
    parser.add_argument("--corrector", default="ald")
    parser.add_argument("--corrector-steps", type=int, default=1)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--sampling-N", type=int, default=0)

    parser.add_argument("--resume", default="")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.loss_type == "data_prediction":
        raise ValueError(
            "train_sgmse.py currently supports loss_type in {'score_matching', 'denoiser'} only. "
            "data_prediction requires a dedicated data_module implementation."
        )

    if args.win_length != args.n_fft:
        raise ValueError(
            f"현재 preprocess.py는 win_length == n_fft 기준으로 사용 중. "
            f"got win_length={args.win_length}, n_fft={args.n_fft}"
        )

    expected_target_length = (args.num_frames - 1) * args.hop_length
    if args.target_length != expected_target_length:
        raise ValueError(
            f"target_length mismatch: got {args.target_length}, "
            f"expected {(args.num_frames - 1)} * {args.hop_length} = {expected_target_length}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.overfit_samples > 0:
        dataset_limit = args.overfit_samples
        shuffle = False
        random_crop = False
    elif args.limit_samples > 0:
        dataset_limit = args.limit_samples
        shuffle = True
        random_crop = True
    else:
        dataset_limit = None
        shuffle = True
        random_crop = True

    dataset = SpeechEnhancementDataset(
        manifest_path=args.manifest,
        target_sr=args.target_sr,
        target_length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        num_frames=args.num_frames,
        spec_factor=args.spec_factor,
        spec_abs_exponent=args.spec_abs_exponent,
        normalize=args.normalize,
        random_crop=random_crop,
        valid_only=True,
        limit=dataset_limit,
        use_condition=args.use_aux_cond,
        raw_force_scale=args.raw_force_scale,
        d_force_scale=args.d_force_scale,
        condition_smooth_win=args.condition_smooth_win,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty. Check manifest or batch size.")

    updates_per_epoch = max(1, math.ceil(len(loader) / max(args.grad_accum, 1)))
    if args.max_epochs > 0:
        total_steps = max(1, int(math.ceil(updates_per_epoch * args.max_epochs)))
    else:
        total_steps = int(args.max_steps)

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
        start_step, loaded_loss = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        print(f"[resume] path={args.resume}")
        print(f"[resume] step={start_step}, loss={loaded_loss:.8f}")

    loader_iter = cycle(loader)

    print("--------------------------------------------------")
    print("[train config]")
    print(f"device             : {device}")
    print(f"samples            : {len(dataset)}")
    print(f"batches_per_epoch  : {len(loader)}")
    print(f"updates_per_epoch  : {updates_per_epoch}")
    print(f"batch_size         : {args.batch_size}")
    print(f"grad_accum         : {args.grad_accum}")
    print(f"max_steps          : {total_steps}")
    print(f"max_epochs         : {args.max_epochs}")
    print(f"lr                 : {args.lr}")
    print(f"amp                : {args.amp}")
    print(f"save_dir           : {save_dir}")
    print("--------------------------------------------------")
    print("[preprocess config]")
    print(f"target_sr          : {args.target_sr}")
    print(f"target_length      : {args.target_length}")
    print(f"n_fft              : {args.n_fft}")
    print(f"hop_length         : {args.hop_length}")
    print(f"num_frames         : {args.num_frames}")
    print(f"spec_factor        : {args.spec_factor}")
    print(f"spec_abs_exponent  : {args.spec_abs_exponent}")
    print(f"normalize          : {args.normalize}")
    print("--------------------------------------------------")
    print("[condition config]")
    print(f"use_aux_cond       : {args.use_aux_cond}")
    print(f"aux_cond_dim       : {args.aux_cond_dim}")
    print(f"aux_hidden_dim     : {args.aux_hidden_dim}")
    print(f"aux_scale_init     : {args.aux_scale_init}")
    print("--------------------------------------------------")
    print("[model config]")
    print(f"backbone           : {args.backbone}")
    print(f"sde                : {args.sde}")
    print(f"nf                 : {args.nf}")
    print(f"ch_mult            : {parse_int_sequence(args.ch_mult)}")
    print(f"num_res_blocks     : {args.num_res_blocks}")
    print(f"attn_resolutions   : {parse_int_sequence(args.attn_resolutions)}")
    print(f"loss_type          : {args.loss_type}")
    print(f"loss_weighting     : {args.loss_weighting}")
    print(f"network_scaling    : {args.network_scaling} -> {args._resolved_network_scaling}")
    print(f"sampler_type(train): {args.sampler_type}")
    print(f"params             : {count_params(model):,}")
    print("--------------------------------------------------")
    print("[ema config]")
    print(f"use_ema            : {args.use_ema}")
    print(f"ema_decay          : {args.ema_decay}")
    print("--------------------------------------------------")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_count = 0
    last_time = time.time()

    for step in range(start_step + 1, total_steps + 1):
        total_loss = 0.0
        aux_cond = None

        for _ in range(args.grad_accum):
            batch = next(loader_iter)

            clean = twoch_to_complex_batch(
                batch["clean_stft"].to(device, non_blocking=True)
            )
            noisy = twoch_to_complex_batch(
                batch["noisy_stft"].to(device, non_blocking=True)
            )

            aux_cond = None
            if args.use_aux_cond:
                if "cond" not in batch:
                    raise RuntimeError("use_aux_cond=True but batch does not contain 'cond'")

                aux_cond = batch["cond"].to(device, non_blocking=True)
                if aux_cond.dim() != 3:
                    raise RuntimeError(
                        f"Expected aux_cond shape [B, 8, K], got {tuple(aux_cond.shape)}"
                    )
                if aux_cond.size(1) != args.aux_cond_dim:
                    raise RuntimeError(
                        f"Expected aux_cond channel={args.aux_cond_dim}, got {aux_cond.size(1)}"
                    )

            with autocast_context(device, enabled=(args.amp and device.type == "cuda")):
                loss = model._step({"x": clean, "y": noisy, "aux_cond": aux_cond}, batch_idx=0)
                loss_for_backward = loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"loss is NaN or Inf at step {step}")

            scaler.scale(loss_for_backward).backward()
            total_loss += float(loss.item())

        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if args.use_ema:
            model.ema.update(model.dnn.parameters())

        avg_loss = total_loss / args.grad_accum
        running_loss += avg_loss
        running_count += 1

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            elapsed = now - last_time
            last_time = now

            mean_loss = running_loss / max(running_count, 1)
            running_loss = 0.0
            running_count = 0

            aux_shape = None if aux_cond is None else tuple(aux_cond.shape)
            print(
                f"[step {step:07d}/{total_steps:07d}] "
                f"epoch~{step / updates_per_epoch:.2f} "
                f"loss={avg_loss:.8f} "
                f"mean_loss={mean_loss:.8f} "
                f"aux={aux_shape} "
                f"time={elapsed:.2f}s"
            )

        if step % args.save_every == 0 or step == total_steps:
            latest_path = save_dir / "latest.pt"
            step_path = save_dir / f"step_{step:07d}.pt"

            save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                step=step,
                loss_value=avg_loss,
            )
            save_checkpoint(
                path=step_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                step=step,
                loss_value=avg_loss,
            )

            print(f"[save] {latest_path}")
            print(f"[save] {step_path}")

        if args.sample_every > 0 and (step % args.sample_every == 0 or step == total_steps):
            sample_batch = next(loader_iter)
            save_sample_wavs(
                model=model,
                batch=sample_batch,
                args=args,
                device=device,
                step=step,
            )
            del sample_batch
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[sample] saved full sample wavs at step {step}")

    print("--------------------------------------------------")
    print("[DONE] training completed")


if __name__ == "__main__":
    main()
