import argparse
import ast
import copy
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
    spec_fwd,
    complex_to_channels,
    reconstruct_from_2ch,
)

from src.condition.preprocess import (
    ConditionPreprocessConfig,
    preprocess_condition_for_train,
)

from LRDSE.src.models.model_rddm import UnetRes, ResidualDiffusion


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dim_mults(s: str):
    value = ast.literal_eval(s)

    if isinstance(value, int):
        return (value,)

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    raise ValueError(f"invalid dim_mults: {s}")


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


class EMA:
    def __init__(
        self,
        model,
        beta=0.995,
        update_every=10,
        update_after_step=100,
    ):
        self.beta = beta
        self.update_every = max(1, int(update_every))
        self.update_after_step = max(0, int(update_after_step))
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)

    def state_dict(self):
        return {
            "beta": self.beta,
            "update_every": self.update_every,
            "update_after_step": self.update_after_step,
            "model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.beta = float(state_dict.get("beta", self.beta))
        self.update_every = int(state_dict.get("update_every", self.update_every))
        self.update_after_step = int(
            state_dict.get("update_after_step", self.update_after_step)
        )
        self.ema_model.load_state_dict(state_dict["model"], strict=True)

    def to(self, device):
        self.ema_model.to(device)
        return self

    @torch.no_grad()
    def copy_from_model(self, model):
        self.ema_model.load_state_dict(model.state_dict(), strict=True)

    @torch.no_grad()
    def update(self, model, step):
        if step % self.update_every != 0:
            return

        if step <= self.update_after_step:
            self.copy_from_model(model)
            return

        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()

        for name, value in model_state.items():
            ema_value = ema_state[name]

            if torch.is_floating_point(value):
                ema_value.mul_(self.beta).add_(value, alpha=1.0 - self.beta)
            else:
                ema_value.copy_(value)


def normalize_stft_for_model(x, args):
    scale = max(float(args.stft_scale), 1e-8)
    x = x / scale

    if args.model_norm == "none":
        return x

    if args.model_norm == "tanh":
        return torch.tanh(x)

    raise ValueError(f"unknown model_norm: {args.model_norm}")


def denormalize_stft_from_model(x, args):
    scale = max(float(args.stft_scale), 1e-8)

    if args.model_norm == "none":
        return x * scale

    if args.model_norm == "tanh":
        limit = 1.0 - float(args.model_norm_eps)
        x = torch.clamp(x, min=-limit, max=limit)
        return torch.atanh(x) * scale

    raise ValueError(f"unknown model_norm: {args.model_norm}")


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


def build_model(args):
    dim_mults = parse_dim_mults(args.dim_mults)

    aux_context_dim = args.aux_context_dim
    if aux_context_dim <= 0:
        aux_context_dim = None

    net = UnetRes(
        dim=args.dim,
        dim_mults=dim_mults,
        channels=args.channels,
        share_encoder=args.share_encoder,
        condition=True,
        input_condition=False,
        use_aux_cond=args.use_aux_cond,
        aux_cond_dim=args.aux_cond_dim,
        aux_context_dim=aux_context_dim,
    )

    diffusion = ResidualDiffusion(
        model=net,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
        loss_type=args.loss_type,
        condition=True,
        sum_scale=args.sum_scale,
        input_condition=False,
        input_condition_mask=False,
        clip_denoised=args.clip_denoised,
        sampling_type=getattr(args, "sampling_type", "use_pred_noise"),
        sampling_init=getattr(args, "sampling_init", "input"),
        sampling_init_noise_scale=getattr(args, "sampling_init_noise_scale", 1.0),
    )

    return diffusion


def save_checkpoint(path, model, optimizer, scaler, ema, args, step, loss_value):
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

    if ema is not None:
        ckpt["ema"] = ema.state_dict()

    torch.save(ckpt, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scaler=None,
    ema=None,
    device="cpu",
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if ema is not None:
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        else:
            ema.copy_from_model(model)

    return int(ckpt.get("step", 0)), float(ckpt.get("loss", 0.0))


def get_meta_item(meta, key, index, default=None):
    if meta is None:
        return default

    value = meta.get(key, default)

    if value is None:
        return default

    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return default

    if torch.is_tensor(value):
        if value.dim() == 0:
            return value.item()
        if index < value.numel():
            return value[index].item()
        return default

    return value


@torch.no_grad()
def build_aux_cond_for_chunk(
    run_dir,
    chunk_start_sample,
    noisy_2ch,
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
        num_frames=noisy_2ch.shape[-1],
        hop_length=args.hop_length,
        freq_bins=noisy_2ch.shape[-2],
        cfg=cond_cfg,
    )

    aux_cond = cond_out["cond_8ch"].unsqueeze(0).to(device, non_blocking=True)
    return aux_cond


@torch.no_grad()
def enhance_full_wav(model, noisy_wav, args, device, run_dir=None):
    """
    전체 wav를 utterance-level normalization 한 번만 적용한 뒤,
    target_length chunk 단위로 RDDM sample 후 overlap-add.

    aux condition 사용 시:
        각 chunk start sample에 맞춰 foot_force condition [1, 8, 256] 생성 후
        model.sample(..., aux_cond=aux_cond)로 전달.
    """
    was_training = model.training
    model.eval()

    cfg = build_preprocess_cfg(args)

    orig_len = noisy_wav.numel()
    chunk_len = args.target_length

    if args.sample_chunk_hop > 0:
        chunk_hop = args.sample_chunk_hop
    else:
        chunk_hop = chunk_len // 2

    normfac = torch.clamp(noisy_wav.abs().max(), min=cfg.eps)
    noisy_norm_full = noisy_wav / normfac

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
            chunk = torch.nn.functional.pad(
                chunk,
                (0, chunk_len - chunk.numel()),
                mode="constant",
            )

        chunk_2d = ensure_mono_2d(chunk)

        noisy_spec = spec_fwd(stft(chunk_2d, cfg), cfg)
        noisy_2ch = complex_to_channels(noisy_spec).float()

        expected_shape = (2, args.n_fft // 2 + 1, args.num_frames)
        if noisy_2ch.shape != expected_shape:
            raise RuntimeError(
                f"Expected noisy_2ch shape {expected_shape}, "
                f"but got {tuple(noisy_2ch.shape)}"
            )

        noisy_input = normalize_stft_for_model(
            noisy_2ch.unsqueeze(0).to(device),
            args,
        )

        aux_cond = None
        if args.use_aux_cond:
            aux_cond = build_aux_cond_for_chunk(
                run_dir=run_dir,
                chunk_start_sample=start,
                noisy_2ch=noisy_2ch,
                args=args,
                device=device,
            )

        print(
            f"[sample chunk] "
            f"{ci + 1}/{len(starts)} "
            f"input={tuple(noisy_input.shape)} "
            f"aux={None if aux_cond is None else tuple(aux_cond.shape)} "
            f"start={start}"
        )

        with autocast_context(device, enabled=(args.amp and device.type == "cuda")):
            sample_out = model.sample(
                noisy_input,
                last=True,
                aux_cond=aux_cond,
            )

        if isinstance(sample_out, list):
            enhanced_2ch = sample_out[-1]
        else:
            enhanced_2ch = sample_out

        enhanced_2ch = denormalize_stft_from_model(
            enhanced_2ch.squeeze(0),
            args,
        )

        enhanced_chunk_norm = reconstruct_from_2ch(
            pred_2ch=enhanced_2ch.detach().cpu(),
            normfac=torch.tensor(1.0),
            orig_len=chunk_len,
            pad_frames=0,
            cfg=cfg,
        ).squeeze(0)

        valid_len = min(chunk_len, orig_len - start)

        out[start:start + valid_len] += enhanced_chunk_norm[:valid_len] * window[:valid_len]
        weight[start:start + valid_len] += window[:valid_len]

        del noisy_input
        del noisy_2ch
        del noisy_spec
        del sample_out
        del enhanced_2ch
        del enhanced_chunk_norm

        if aux_cond is not None:
            del aux_cond

        if device.type == "cuda":
            torch.cuda.empty_cache()

    enhanced_norm = out / torch.clamp(weight, min=1e-8)
    enhanced_wav = enhanced_norm * normfac

    if was_training:
        model.train()

    return enhanced_wav


@torch.no_grad()
def save_sample_wavs(model, batch, args, device, step):
    """
    sample 저장:
        batch crop 결과가 아니라,
        meta에 저장된 원본 noisy/clean wav 경로를 다시 읽고,
        chunk + overlap-add로 full enhanced wav 저장.

    aux condition 사용 시:
        meta["run_dir"]를 사용해 chunk별 condition 생성.
    """
    was_training = model.training
    model.eval()

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

        sf.write(
            sample_dir / f"{i:02d}_{safe_id}_noisy_full.wav",
            noisy_wav.numpy(),
            args.target_sr,
        )
        sf.write(
            sample_dir / f"{i:02d}_{safe_id}_clean_full.wav",
            clean_wav.numpy(),
            args.target_sr,
        )
        sf.write(
            sample_dir / f"{i:02d}_{safe_id}_enhanced_full.wav",
            enhanced_wav.numpy(),
            args.target_sr,
        )

        print(
            f"[sample full] {i:02d} "
            f"id={sample_id} "
            f"duration={noisy_wav.numel() / args.target_sr:.2f}s "
            f"run_dir={run_dir} "
            f"enhanced_range={enhanced_wav.min().item():.6f}/{enhanced_wav.max().item():.6f} "
            f"saved={sample_dir}"
        )

    if was_training:
        model.train()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--save-dir", default="./checkpoints/rddm_se")

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

    parser.add_argument("--stft-scale", type=float, default=1.0)
    parser.add_argument(
        "--model-norm",
        default="tanh",
        choices=["none", "tanh"],
    )
    parser.add_argument("--model-norm-eps", type=float, default=1e-4)
    parser.add_argument(
        "--no-clip-denoised",
        dest="clip_denoised",
        action="store_false",
    )
    parser.set_defaults(clip_denoised=True)

    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--dim-mults", default="(1, 2, 4)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--share-encoder", type=int, default=0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=20)
    parser.add_argument(
        "--sampling-type",
        default="use_pred_noise",
        choices=[
            "use_pred_noise",
            "use_x_start",
            "special_eta_0",
            "special_eta_1",
        ],
    )
    parser.add_argument(
        "--sampling-init",
        default="input",
        choices=["input", "input_plus_noise"],
    )
    parser.add_argument("--sampling-init-noise-scale", type=float, default=1.0)
    parser.add_argument("--objective", default="pred_res_noise")
    parser.add_argument("--loss-type", default="l1")
    parser.add_argument("--sum-scale", type=float, default=0.01)

    parser.add_argument("--use-aux-cond", action="store_true")
    parser.add_argument("--aux-cond-dim", type=int, default=8)
    parser.add_argument(
        "--aux-context-dim",
        type=int,
        default=0,
        help="0이면 U-Net mid_dim 사용.",
    )
    parser.add_argument("--raw-force-scale", type=float, default=220.0)
    parser.add_argument("--d-force-scale", type=float, default=9220.325595510363)
    parser.add_argument("--condition-smooth-win", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
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
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
    )
    parser.set_defaults(use_ema=True)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema-update-every", type=int, default=10)
    parser.add_argument("--ema-update-after-step", type=int, default=100)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--num-sample-wavs", type=int, default=2)

    parser.add_argument(
        "--sample-max-sec",
        type=float,
        default=0.0,
        help="0이면 전체 길이 sample 저장. OOM이나 시간이 길면 예: 8 또는 10으로 제한.",
    )
    parser.add_argument(
        "--sample-chunk-hop",
        type=int,
        default=0,
        help="Full sample 저장 시 chunk hop samples. 0이면 target_length//2.",
    )

    parser.add_argument("--resume", default="")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

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

    loader_iter = cycle(loader)

    model = build_model(args).to(device)
    ema = None
    if args.use_ema:
        ema = EMA(
            model=model,
            beta=args.ema_decay,
            update_every=args.ema_update_every,
            update_after_step=args.ema_update_after_step,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    start_step = 0

    if args.resume:
        start_step, loaded_loss = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
        )
        print(f"[resume] path={args.resume}")
        print(f"[resume] step={start_step}, loss={loaded_loss:.8f}")

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
    print(f"stft_scale         : {args.stft_scale}")
    print(f"model_norm         : {args.model_norm}")
    print(f"clip_denoised      : {args.clip_denoised}")
    print(f"sample_max_sec     : {args.sample_max_sec}")
    print(f"sample_chunk_hop   : {args.sample_chunk_hop}")
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
    print(f"aux_context_dim    : {args.aux_context_dim}")
    print(f"raw_force_scale    : {args.raw_force_scale}")
    print(f"d_force_scale      : {args.d_force_scale}")
    print(f"cond_smooth_win    : {args.condition_smooth_win}")
    print("--------------------------------------------------")
    print("[model config]")
    print(f"dim                : {args.dim}")
    print(f"dim_mults          : {parse_dim_mults(args.dim_mults)}")
    print(f"timesteps          : {args.timesteps}")
    print(f"sampling_timesteps : {args.sampling_timesteps}")
    print(f"sampling_type      : {args.sampling_type}")
    print(f"sampling_init      : {args.sampling_init}")
    print(f"sampling_init_scale: {args.sampling_init_noise_scale}")
    print(f"objective          : {args.objective}")
    print(f"loss_type          : {args.loss_type}")
    print(f"sum_scale          : {args.sum_scale}")
    print(f"params             : {count_params(model):,}")
    print("--------------------------------------------------")
    print("[ema config]")
    print(f"use_ema            : {args.use_ema}")
    print(f"ema_decay          : {args.ema_decay}")
    print(f"ema_update_every   : {args.ema_update_every}")
    print(f"ema_update_after   : {args.ema_update_after_step}")
    print("--------------------------------------------------")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_count = 0
    last_time = time.time()

    for step in range(start_step + 1, total_steps + 1):
        total_loss = 0.0

        for _ in range(args.grad_accum):
            batch = next(loader_iter)

            noisy = normalize_stft_for_model(
                batch["noisy_stft"].to(device, non_blocking=True),
                args,
            )
            clean = normalize_stft_for_model(
                batch["clean_stft"].to(device, non_blocking=True),
                args,
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
                        f"Expected aux_cond channel={args.aux_cond_dim}, "
                        f"got {aux_cond.size(1)}"
                    )

            with autocast_context(device, enabled=(args.amp and device.type == "cuda")):
                loss = model([clean, noisy], aux_cond=aux_cond)
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

        if ema is not None:
            ema.update(model, step)

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

            aux_shape = None
            if args.use_aux_cond and aux_cond is not None:
                aux_shape = tuple(aux_cond.shape)

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
                ema=ema,
                args=args,
                step=step,
                loss_value=avg_loss,
            )

            save_checkpoint(
                path=step_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                args=args,
                step=step,
                loss_value=avg_loss,
            )

            print(f"[save] {latest_path}")
            print(f"[save] {step_path}")

        if args.sample_every > 0 and (step % args.sample_every == 0 or step == total_steps):
            sample_batch = next(loader_iter)
            sample_model = ema.ema_model if ema is not None else model

            save_sample_wavs(
                model=sample_model,
                batch=sample_batch,
                args=args,
                device=device,
                step=step,
            )

            print(f"[sample] saved full sample wavs at step {step}")

    print("--------------------------------------------------")
    print("[DONE] training completed")


if __name__ == "__main__":
    main()
