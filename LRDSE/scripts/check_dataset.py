import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import SpeechEnhancementDataset, stft_2ch_to_wav


def has_nan_or_inf(x):
    return bool(torch.isnan(x).any() or torch.isinf(x).any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--target-length", type=int, default=32640)
    parser.add_argument("--n-fft", type=int, default=510)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=510)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-dir", default="outputs/debug_dataset")
    args = parser.parse_args()

    dataset = SpeechEnhancementDataset(
        manifest_path=args.manifest,
        target_sr=args.target_sr,
        target_length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        random_crop=False,
        valid_only=True,
        limit=args.num_samples,
    )

    print("--------------------------------------------------")
    print("[dataset]")
    print(f"samples       : {len(dataset)}")
    print(f"target_sr     : {args.target_sr}")
    print(f"target_length : {args.target_length}")
    print(f"n_fft         : {args.n_fft}")
    print(f"hop_length    : {args.hop_length}")
    print(f"win_length    : {args.win_length}")

    sample = dataset[0]

    noisy_stft = sample["noisy_stft"]
    clean_stft = sample["clean_stft"]
    noisy_wav = sample["noisy_wav"]
    clean_wav = sample["clean_wav"]

    print("--------------------------------------------------")
    print("[single sample]")
    print(f"id                 : {sample['meta']['id']}")
    print(f"noisy_wav shape    : {tuple(noisy_wav.shape)}")
    print(f"clean_wav shape    : {tuple(clean_wav.shape)}")
    print(f"noisy_stft shape   : {tuple(noisy_stft.shape)}")
    print(f"clean_stft shape   : {tuple(clean_stft.shape)}")
    print(f"noisy_stft min/max : {noisy_stft.min().item():.6f} / {noisy_stft.max().item():.6f}")
    print(f"clean_stft min/max : {clean_stft.min().item():.6f} / {clean_stft.max().item():.6f}")
    print(f"noisy nan/inf      : {has_nan_or_inf(noisy_stft)}")
    print(f"clean nan/inf      : {has_nan_or_inf(clean_stft)}")

    recon_noisy = stft_2ch_to_wav(
        noisy_stft,
        length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    recon_clean = stft_2ch_to_wav(
        clean_stft,
        length=args.target_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    noisy_mse = torch.mean((noisy_wav - recon_noisy) ** 2).item()
    clean_mse = torch.mean((clean_wav - recon_clean) ** 2).item()

    print("--------------------------------------------------")
    print("[stft inverse check]")
    print(f"noisy recon mse : {noisy_mse:.12e}")
    print(f"clean recon mse : {clean_mse:.12e}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    batch = next(iter(loader))

    print("--------------------------------------------------")
    print("[batch check]")
    print(f"batch noisy_stft : {tuple(batch['noisy_stft'].shape)}")
    print(f"batch clean_stft : {tuple(batch['clean_stft'].shape)}")
    print(f"batch noisy_wav  : {tuple(batch['noisy_wav'].shape)}")
    print(f"batch clean_wav  : {tuple(batch['clean_wav'].shape)}")
    print(f"batch noisy nan  : {has_nan_or_inf(batch['noisy_stft'])}")
    print(f"batch clean nan  : {has_nan_or_inf(batch['clean_stft'])}")

    if args.save_debug:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        sf.write(debug_dir / "noisy_original.wav", noisy_wav.numpy(), args.target_sr)
        sf.write(debug_dir / "clean_original.wav", clean_wav.numpy(), args.target_sr)
        sf.write(debug_dir / "noisy_reconstructed.wav", recon_noisy.numpy(), args.target_sr)
        sf.write(debug_dir / "clean_reconstructed.wav", recon_clean.numpy(), args.target_sr)

        print("--------------------------------------------------")
        print("[saved debug wavs]")
        print(f"saved dir: {debug_dir}")

    print("--------------------------------------------------")
    print("[OK] dataset pipeline looks valid")


if __name__ == "__main__":
    main()