# scripts/plot_stft.py

"""
Input:
    --wav path/to/audio.wav

Output:
    outputs/debug/stft_plot.png

Checks:
    1. train-style cropped waveform
    2. raw STFT log magnitude
    3. transformed STFT log magnitude
"""

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


from src.audio.preprocess import (
    AudioPreprocessConfig,
    load_wav,
    crop_or_pad_for_train,
    normalize_noisy,
    stft,
    spec_fwd,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/debug/stft_plot.png")
    args = parser.parse_args()

    cfg = AudioPreprocessConfig()

    wav, sr = load_wav(args.wav, cfg)

    wav_crop, start = crop_or_pad_for_train(
        wav,
        cfg,
        random_crop=False,
    )

    wav_norm, normfac = normalize_noisy(wav_crop, cfg)

    raw_spec = stft(wav_norm, cfg)          # [1, 256, 256]
    trans_spec = spec_fwd(raw_spec, cfg)    # [1, 256, 256]

    raw_mag = torch.log1p(raw_spec.abs().squeeze(0)).cpu()
    trans_mag = torch.log1p(trans_spec.abs().squeeze(0)).cpu()

    wav_np = wav_norm.squeeze(0).cpu().numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(wav_np)
    plt.title("Train-style cropped waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.imshow(raw_mag, origin="lower", aspect="auto")
    plt.title("Raw STFT log magnitude")
    plt.xlabel("Frame")
    plt.ylabel("Frequency bin")
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(trans_mag, origin="lower", aspect="auto")
    plt.title("Transformed STFT log magnitude")
    plt.xlabel("Frame")
    plt.ylabel("Frequency bin")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    print("wav shape:", tuple(wav.shape))
    print("cropped wav shape:", tuple(wav_crop.shape))
    print("raw STFT shape:", tuple(raw_spec.shape))
    print("transformed STFT shape:", tuple(trans_spec.shape))
    print("crop start:", start)
    print("saved:", args.out)


if __name__ == "__main__":
    main()