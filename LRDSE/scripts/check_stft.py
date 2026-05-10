# scripts/check_stft.py

"""
Check SGMSE+-style audio preprocessing.

This script checks two paths:

1. Train-style preprocessing
   Input:
       --wav path/to/noisy.wav
   Process:
       load wav
       -> crop/pad to 32640 samples
       -> normalize by noisy max
       -> STFT
       -> spec_fwd
       -> spec_back
       -> iSTFT
   Output:
       shape logs
       reconstruction error
       outputs/debug/train_reconstructed.wav

2. Inference-style preprocessing
   Input:
       --wav path/to/noisy.wav
   Process:
       load full wav
       -> normalize by noisy max
       -> STFT
       -> spec_fwd
       -> real/imag 2ch
       -> pad time frames to multiple of 64
       -> unpad
       -> spec_back
       -> iSTFT with original length
   Output:
       shape logs
       reconstruction error
       outputs/debug/inference_reconstructed.wav

Usage:
    python scripts/check_stft.py --wav noisy.wav

Optional:
    python scripts/check_stft.py --wav noisy.wav --out_dir outputs/debug
"""

import argparse
import os
import sys

import torch
import torchaudio
import soundfile as sf


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


from src.audio.preprocess import (
    AudioPreprocessConfig,
    load_wav,
    crop_or_pad_for_train,
    normalize_noisy,
    stft,
    istft,
    spec_fwd,
    spec_back,
    complex_to_channels,
    preprocess_noisy_for_inference,
    reconstruct_from_2ch,
)


def print_line():
    print("-" * 70)


def save_wav(path: str, wav: torch.Tensor, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    wav = wav.detach().cpu()

    # [1, T] -> [T]
    if wav.dim() == 2 and wav.size(0) == 1:
        wav = wav.squeeze(0)

    # soundfile expects numpy array
    sf.write(path, wav.numpy(), sr)

def check_train_style(wav_path: str, out_dir: str, cfg: AudioPreprocessConfig):
    """
    Input:
        wav_path: noisy wav path
        out_dir: output directory
        cfg: AudioPreprocessConfig

    Output:
        Saves:
            train_original_crop.wav
            train_reconstructed.wav

        Prints:
            waveform shape
            STFT shape
            2-channel shape
            reconstruction MSE
            reconstruction max abs error
    """

    print_line()
    print("[1] Train-style check")
    print_line()

    wav, sr = load_wav(wav_path, cfg)
    print("loaded wav shape:", tuple(wav.shape))
    print("sample rate:", sr)

    wav_crop, start = crop_or_pad_for_train(
        wav,
        cfg,
        start=None,
        random_crop=False,
    )

    print("crop start:", start)
    print("train target length:", cfg.train_target_len)
    print("cropped/padded wav shape:", tuple(wav_crop.shape))

    wav_norm, normfac = normalize_noisy(wav_crop, cfg)

    raw_spec = stft(wav_norm, cfg)
    print("raw STFT shape:", tuple(raw_spec.shape))
    print("expected raw STFT shape:", "(1, 256, 256)")

    transformed_spec = spec_fwd(raw_spec, cfg)
    print("transformed STFT shape:", tuple(transformed_spec.shape))

    spec_2ch = complex_to_channels(transformed_spec)
    print("2ch STFT shape:", tuple(spec_2ch.shape))
    print("expected 2ch shape:", "(2, 256, 256)")

    restored_spec = spec_back(transformed_spec, cfg)
    wav_recon_norm = istft(
        restored_spec,
        cfg,
        length=cfg.train_target_len,
    )

    mse = torch.mean((wav_norm - wav_recon_norm) ** 2).item()
    max_abs_err = torch.max(torch.abs(wav_norm - wav_recon_norm)).item()

    print("reconstruction MSE:", mse)
    print("reconstruction max abs error:", max_abs_err)

    wav_crop_denorm = wav_norm * normfac
    wav_recon_denorm = wav_recon_norm * normfac

    save_wav(
        os.path.join(out_dir, "train_original_crop.wav"),
        wav_crop_denorm,
        cfg.sample_rate,
    )
    save_wav(
        os.path.join(out_dir, "train_reconstructed.wav"),
        wav_recon_denorm,
        cfg.sample_rate,
    )

    print("saved:", os.path.join(out_dir, "train_original_crop.wav"))
    print("saved:", os.path.join(out_dir, "train_reconstructed.wav"))


def check_inference_style(wav_path: str, out_dir: str, cfg: AudioPreprocessConfig):
    """
    Input:
        wav_path: noisy wav path
        out_dir: output directory
        cfg: AudioPreprocessConfig

    Output:
        Saves:
            inference_reconstructed.wav

        Prints:
            original waveform length
            original STFT frame count
            padded STFT frame count
            pad frame count
            reconstruction MSE
            reconstruction max abs error
    """

    print_line()
    print("[2] Inference-style check")
    print_line()

    original_wav, sr = load_wav(wav_path, cfg)

    result = preprocess_noisy_for_inference(
        wav_path,
        cfg=cfg,
        pad_to_64=True,
    )

    noisy_wave = result["noisy_wave"]
    noisy_2ch = result["noisy_2ch"]
    noisy_2ch_padded = result["noisy_2ch_padded"]
    normfac = result["normfac"]
    orig_len = int(result["orig_len"].item())
    orig_frames = int(result["orig_frames"].item())
    pad_frames = int(result["pad_frames"].item())

    print("original wav shape:", tuple(original_wav.shape))
    print("normalized full wav shape:", tuple(noisy_wave.shape))
    print("original length:", orig_len)
    print("2ch STFT shape before padding:", tuple(noisy_2ch.shape))
    print("original STFT frames:", orig_frames)
    print("pad frames:", pad_frames)
    print("2ch STFT shape after padding:", tuple(noisy_2ch_padded.shape))

    wav_recon = reconstruct_from_2ch(
        pred_2ch=noisy_2ch_padded,
        normfac=normfac,
        orig_len=orig_len,
        pad_frames=pad_frames,
        cfg=cfg,
    )

    mse = torch.mean((original_wav - wav_recon) ** 2).item()
    max_abs_err = torch.max(torch.abs(original_wav - wav_recon)).item()

    print("reconstruction MSE:", mse)
    print("reconstruction max abs error:", max_abs_err)

    save_wav(
        os.path.join(out_dir, "inference_reconstructed.wav"),
        wav_recon,
        cfg.sample_rate,
    )

    print("saved:", os.path.join(out_dir, "inference_reconstructed.wav"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to 16 kHz mono or multi-channel wav file.")
    parser.add_argument("--out_dir", type=str, default="outputs/debug")
    args = parser.parse_args()

    cfg = AudioPreprocessConfig()

    os.makedirs(args.out_dir, exist_ok=True)

    check_train_style(args.wav, args.out_dir, cfg)
    check_inference_style(args.wav, args.out_dir, cfg)

    print_line()
    print("Done.")
    print_line()


if __name__ == "__main__":
    main()