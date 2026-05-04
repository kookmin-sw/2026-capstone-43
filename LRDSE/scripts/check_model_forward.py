import argparse
import ast
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import SpeechEnhancementDataset
from LRDSE.src.models.model_rddm import UnetRes, ResidualDiffusion


def parse_dim_mults(s: str):
    try:
        value = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"invalid --dim-mults: {s}, error={e}")

    if isinstance(value, int):
        return (value,)

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    raise ValueError(f"--dim-mults must be tuple/list/int, got: {value}")


def has_nan_or_inf(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item() or torch.isinf(x).any().item())


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_grad_stats(model):
    grad_count = 0
    grad_abs_sum = 0.0
    grad_max = 0.0

    for p in model.parameters():
        if p.grad is None:
            continue

        grad_count += 1
        g = p.grad.detach()
        grad_abs_sum += g.abs().mean().item()
        grad_max = max(grad_max, g.abs().max().item())

    if grad_count == 0:
        return {
            "grad_count": 0,
            "grad_abs_mean": 0.0,
            "grad_abs_max": 0.0,
        }

    return {
        "grad_count": grad_count,
        "grad_abs_mean": grad_abs_sum / grad_count,
        "grad_abs_max": grad_max,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=32640)
    parser.add_argument("--n-fft", type=int, default=510)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=510)

    parser.add_argument("--stft-scale", type=float, default=64.0)

    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--dim-mults", default="(1, 2, 4)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--share-encoder", type=int, default=0)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=20)
    parser.add_argument("--loss-type", default="l1")
    parser.add_argument("--objective", default="pred_res_noise")
    parser.add_argument("--sum-scale", type=float, default=0.01)

    parser.add_argument("--check-sample", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dim_mults = parse_dim_mults(args.dim_mults)

    dataset = SpeechEnhancementDataset(
        manifest_path=args.manifest,
        target_sr=args.target_sr,
        target_length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        random_crop=True,
        valid_only=True,
        limit=args.batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )

    batch = next(iter(loader))

    noisy = batch["noisy_stft"].to(device) / args.stft_scale
    clean = batch["clean_stft"].to(device) / args.stft_scale

    print("--------------------------------------------------")
    print("[batch]")
    print(f"device      : {device}")
    print(f"noisy shape : {tuple(noisy.shape)}")
    print(f"clean shape : {tuple(clean.shape)}")
    print(f"noisy range : {noisy.min().item():.6f} / {noisy.max().item():.6f}")
    print(f"clean range : {clean.min().item():.6f} / {clean.max().item():.6f}")
    print(f"noisy nan   : {has_nan_or_inf(noisy)}")
    print(f"clean nan   : {has_nan_or_inf(clean)}")
    print(f"stft_scale  : {args.stft_scale}")

    net = UnetRes(
        dim=args.dim,
        dim_mults=dim_mults,
        channels=args.channels,
        share_encoder=args.share_encoder,
        condition=True,
        input_condition=False,
    )

    diffusion = ResidualDiffusion(
        model=net,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        loss_type=args.loss_type,
        objective=args.objective,
        condition=True,
        sum_scale=args.sum_scale,
        input_condition=False,
    ).to(device)

    print("--------------------------------------------------")
    print("[model]")
    print(f"net                 : UnetRes")
    print(f"diffusion           : ResidualDiffusion")
    print(f"dim                 : {args.dim}")
    print(f"dim_mults           : {dim_mults}")
    print(f"channels            : {args.channels}")
    print(f"share_encoder       : {args.share_encoder}")
    print(f"timesteps           : {args.timesteps}")
    print(f"sampling_timesteps  : {args.sampling_timesteps}")
    print(f"loss_type           : {args.loss_type}")
    print(f"objective           : {args.objective}")
    print(f"sum_scale           : {args.sum_scale}")
    print(f"params              : {count_params(diffusion):,}")

    diffusion.train()

    print("--------------------------------------------------")
    print("[loss forward]")
    loss = diffusion([clean, noisy])

    print(f"loss shape : {tuple(loss.shape)}")
    print(f"loss value : {loss.item():.8f}")
    print(f"loss nan   : {bool(torch.isnan(loss).item())}")
    print(f"loss inf   : {bool(torch.isinf(loss).item())}")

    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError("loss is NaN or Inf")

    print("--------------------------------------------------")
    print("[backward]")
    diffusion.zero_grad(set_to_none=True)
    loss.backward()

    grad_stats = get_grad_stats(diffusion)

    print(f"grad_count    : {grad_stats['grad_count']}")
    print(f"grad_abs_mean : {grad_stats['grad_abs_mean']:.12f}")
    print(f"grad_abs_max  : {grad_stats['grad_abs_max']:.12f}")

    if grad_stats["grad_count"] == 0:
        raise RuntimeError("no gradient found")

    if args.optimizer_step:
        print("--------------------------------------------------")
        print("[optimizer step]")
        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            loss_after = diffusion([clean, noisy])

        print(f"loss before : {loss.item():.8f}")
        print(f"loss after  : {loss_after.item():.8f}")

    if args.check_sample:
        print("--------------------------------------------------")
        print("[sample check]")
        diffusion.eval()

        with torch.no_grad():
            sample_out = diffusion.sample(noisy, last=True)

        if isinstance(sample_out, list):
            enhanced = sample_out[-1]
        else:
            enhanced = sample_out

        print(f"enhanced shape : {tuple(enhanced.shape)}")
        print(f"enhanced range : {enhanced.min().item():.6f} / {enhanced.max().item():.6f}")
        print(f"enhanced nan   : {has_nan_or_inf(enhanced)}")

        if enhanced.shape != noisy.shape:
            raise RuntimeError(
                f"sample output shape mismatch: enhanced={tuple(enhanced.shape)}, noisy={tuple(noisy.shape)}"
            )

    print("--------------------------------------------------")
    print("[OK] native RDDM loss/backward check passed")


if __name__ == "__main__":
    main()