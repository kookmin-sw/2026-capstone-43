# src/audio/preprocess.py

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import soundfile as sf

import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class AudioPreprocessConfig:
    """
    SGMSE+ 공식 구현 기반 audio preprocessing 설정.

    기본값:
        sample_rate = 16000
        n_fft = 510
        hop_length = 128
        num_frames = 256
        center = True
        window = periodic Hann
        spec_factor = 0.15
        spec_abs_exponent = 0.5
        normalize = "noisy"

    train target_len:
        (num_frames - 1) * hop_length
        = 32640 samples
        ≈ 2.04 sec at 16 kHz
    """
    sample_rate: int = 16000
    n_fft: int = 510
    hop_length: int = 128
    num_frames: int = 256
    center: bool = True

    spec_factor: float = 0.15
    spec_abs_exponent: float = 0.5

    normalize: str = "noisy"  # "noisy", "clean", "not"
    eps: float = 1e-8

    @property
    def train_target_len(self) -> int:
        return (self.num_frames - 1) * self.hop_length


# Input:
#   cfg: preprocessing config
#   device: torch device
#   dtype: torch dtype
# Output:
#   periodic Hann window, shape [n_fft]
def get_window(
    cfg: AudioPreprocessConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.hann_window(
        cfg.n_fft,
        periodic=True,
        device=device,
        dtype=dtype,
    )


# Input:
#   wav: waveform, shape [T] or [C, T]
# Output:
#   mono waveform, shape [1, T]
def ensure_mono_2d(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if wav.dim() != 2:
        raise ValueError(f"Expected wav shape [T] or [C, T], got {wav.shape}")

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav


# Input:
#   path: wav file path
#   cfg: preprocessing config
# Output:
#   wav: mono waveform tensor, shape [1, T]
#   sr: sample rate
#
# Note:
#   torchaudio.load()가 torchcodec/ffmpeg 문제를 일으킬 수 있어서
#   soundfile로 wav를 읽는다.
def load_wav(path: str, cfg: AudioPreprocessConfig) -> Tuple[torch.Tensor, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    # soundfile: [T, C] -> torch: [C, T]
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()
    wav = ensure_mono_2d(wav)

    if sr != cfg.sample_rate:
        raise ValueError(
            f"Expected {cfg.sample_rate} Hz audio, but got {sr} Hz: {path}"
        )

    return wav, sr


# Input:
#   wav: waveform, shape [1, T]
#   cfg: preprocessing config
#   start: crop start index. None이면 random/center로 자동 선택
#   random_crop: True면 random crop, False면 center crop
# Output:
#   wav: cropped or padded waveform, shape [1, train_target_len]
#   start: 실제 crop start index. padding인 경우 0
#
# Used for:
#   training / validation / test dataset preprocessing
def crop_or_pad_for_train(
    wav: torch.Tensor,
    cfg: AudioPreprocessConfig,
    start: Optional[int] = None,
    random_crop: bool = True,
) -> Tuple[torch.Tensor, int]:
    wav = ensure_mono_2d(wav)

    target_len = cfg.train_target_len
    current_len = wav.size(-1)

    if current_len >= target_len:
        max_start = current_len - target_len

        if start is None:
            if random_crop:
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = max_start // 2

        wav = wav[..., start:start + target_len]
        return wav, start

    pad = target_len - current_len
    left = pad // 2
    right = pad - left
    wav = F.pad(wav, (left, right), mode="constant")

    return wav, 0


# Input:
#   clean: clean waveform, shape [1, T]
#   noisy: noisy waveform, shape [1, T]
#   cfg: preprocessing config
# Output:
#   clean_norm: normalized clean waveform, shape [1, T]
#   noisy_norm: normalized noisy waveform, shape [1, T]
#   normfac: scalar normalization factor
#
# Used for:
#   paired clean/noisy training
def normalize_pair(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    cfg: AudioPreprocessConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.normalize == "noisy":
        normfac = noisy.abs().max()
    elif cfg.normalize == "clean":
        normfac = clean.abs().max()
    elif cfg.normalize == "not":
        normfac = torch.tensor(1.0, device=clean.device, dtype=clean.dtype)
    else:
        raise ValueError(f"Unknown normalize mode: {cfg.normalize}")

    normfac = torch.clamp(normfac, min=cfg.eps)

    return clean / normfac, noisy / normfac, normfac


# Input:
#   noisy: noisy waveform, shape [1, T]
#   cfg: preprocessing config
# Output:
#   noisy_norm: normalized noisy waveform, shape [1, T]
#   normfac: scalar normalization factor
#
# Used for:
#   inference
def normalize_noisy(
    noisy: torch.Tensor,
    cfg: AudioPreprocessConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normfac = torch.clamp(noisy.abs().max(), min=cfg.eps)
    return noisy / normfac, normfac


# Input:
#   wav: waveform, shape [1, T]
#   cfg: preprocessing config
# Output:
#   complex STFT, shape [1, F, K]
#
# Default expected:
#   F = 256
#   K depends on waveform length
def stft(wav: torch.Tensor, cfg: AudioPreprocessConfig) -> torch.Tensor:
    wav = ensure_mono_2d(wav)

    window = get_window(
        cfg,
        device=wav.device,
        dtype=wav.dtype,
    )

    return torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=window,
        center=cfg.center,
        return_complex=True,
    )


# Input:
#   spec: complex STFT, shape [1, F, K]
#   cfg: preprocessing config
#   length: output waveform length
# Output:
#   waveform, shape [1, length]
def istft(
    spec: torch.Tensor,
    cfg: AudioPreprocessConfig,
    length: int,
) -> torch.Tensor:
    window = get_window(
        cfg,
        device=spec.device,
        dtype=spec.real.dtype,
    )

    return torch.istft(
        spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=window,
        center=cfg.center,
        length=length,
    )


# Input:
#   spec: raw complex STFT, shape [1, F, K]
#   cfg: preprocessing config
# Output:
#   transformed complex STFT, shape [1, F, K]
#
# Formula:
#   beta * |c|^alpha * exp(j * angle(c))
#
# Default:
#   alpha = 0.5
#   beta = 0.15
def spec_fwd(spec: torch.Tensor, cfg: AudioPreprocessConfig) -> torch.Tensor:
    e = cfg.spec_abs_exponent

    if e != 1.0:
        spec = spec.abs().pow(e) * torch.exp(1j * spec.angle())

    spec = spec * cfg.spec_factor
    return spec


# Input:
#   spec: transformed complex STFT, shape [1, F, K]
#   cfg: preprocessing config
# Output:
#   raw complex STFT, shape [1, F, K]
#
# Inverse of:
#   beta * |c|^alpha * exp(j * angle(c))
def spec_back(spec: torch.Tensor, cfg: AudioPreprocessConfig) -> torch.Tensor:
    spec = spec / cfg.spec_factor
    e = cfg.spec_abs_exponent

    if e != 1.0:
        spec = spec.abs().pow(1.0 / e) * torch.exp(1j * spec.angle())

    return spec


# Input:
#   spec: complex STFT, shape [1, F, K] or [F, K]
# Output:
#   real/imag tensor, shape [2, F, K]
#
# channel:
#   0 = real
#   1 = imag
def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    if spec.dim() == 3:
        if spec.size(0) != 1:
            raise ValueError(f"Expected mono spec shape [1, F, K], got {spec.shape}")
        spec = spec.squeeze(0)

    if spec.dim() != 2:
        raise ValueError(f"Expected spec shape [F, K], got {spec.shape}")

    return torch.stack([spec.real, spec.imag], dim=0)


# Input:
#   x: real/imag tensor, shape [2, F, K]
# Output:
#   complex STFT, shape [1, F, K]
def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3 or x.size(0) != 2:
        raise ValueError(f"Expected x shape [2, F, K], got {x.shape}")

    spec = torch.complex(x[0], x[1])
    return spec.unsqueeze(0)


# Input:
#   x: tensor with time dimension at the last axis
#      supported shapes:
#        [2, F, K]
#        [B, 2, F, K]
#   multiple: padding multiple, default 64
# Output:
#   padded_x: same rank as input, time dimension padded to multiple of 64
#   pad_frames: number of padded frames
#
# Used for:
#   inference
#
# Note:
#   SGMSE+ pad_spec pads the last time dimension so that K % 64 == 0.
def pad_time_to_multiple(
    x: torch.Tensor,
    multiple: int = 64,
) -> Tuple[torch.Tensor, int]:
    K = x.size(-1)
    pad_frames = (multiple - K % multiple) % multiple

    if pad_frames == 0:
        return x, 0

    if x.dim() == 3:
        # [2, F, K]
        x = F.pad(x, (0, pad_frames, 0, 0), mode="constant")
    elif x.dim() == 4:
        # [B, 2, F, K]
        x = F.pad(x, (0, pad_frames, 0, 0), mode="constant")
    else:
        raise ValueError(f"Expected x shape [2,F,K] or [B,2,F,K], got {x.shape}")

    return x, pad_frames


# Input:
#   x: tensor with time dimension at the last axis
#   pad_frames: number of padded frames
# Output:
#   x without padded time frames
def unpad_time(x: torch.Tensor, pad_frames: int) -> torch.Tensor:
    if pad_frames == 0:
        return x

    return x[..., :-pad_frames]


# Input:
#   clean_path: clean wav path
#   noisy_path: noisy wav path
#   cfg: preprocessing config
#   random_crop:
#       True  -> random crop, train
#       False -> center crop, valid/test
# Output:
#   dict:
#       clean_wave: normalized clean waveform, shape [1, 32640]
#       noisy_wave: normalized noisy waveform, shape [1, 32640]
#       clean_spec: transformed complex clean STFT, shape [1, 256, 256]
#       noisy_spec: transformed complex noisy STFT, shape [1, 256, 256]
#       clean_2ch: real/imag clean, shape [2, 256, 256]
#       noisy_2ch: real/imag noisy, shape [2, 256, 256]
#       residual_2ch: noisy_2ch - clean_2ch, shape [2, 256, 256]
#       normfac: scalar normalization factor
#       start: crop start index
#
# Used for:
#   training / validation / test dataset
def preprocess_pair_for_train(
    clean_path: str,
    noisy_path: str,
    cfg: Optional[AudioPreprocessConfig] = None,
    random_crop: bool = True,
) -> Dict[str, torch.Tensor]:
    if cfg is None:
        cfg = AudioPreprocessConfig()

    clean, _ = load_wav(clean_path, cfg)
    noisy, _ = load_wav(noisy_path, cfg)

    clean, start = crop_or_pad_for_train(
        clean,
        cfg,
        start=None,
        random_crop=random_crop,
    )

    noisy, _ = crop_or_pad_for_train(
        noisy,
        cfg,
        start=start,
        random_crop=random_crop,
    )

    clean, noisy, normfac = normalize_pair(clean, noisy, cfg)

    clean_spec = spec_fwd(stft(clean, cfg), cfg)
    noisy_spec = spec_fwd(stft(noisy, cfg), cfg)

    clean_2ch = complex_to_channels(clean_spec)
    noisy_2ch = complex_to_channels(noisy_spec)

    return {
        "clean_wave": clean,
        "noisy_wave": noisy,
        "clean_spec": clean_spec,
        "noisy_spec": noisy_spec,
        "clean_2ch": clean_2ch,
        "noisy_2ch": noisy_2ch,
        "residual_2ch": noisy_2ch - clean_2ch,
        "normfac": normfac,
        "start": torch.tensor(start, dtype=torch.long),
    }


# Input:
#   noisy_path: noisy wav path
#   cfg: preprocessing config
#   pad_to_64: True면 time frame을 64 배수로 padding
# Output:
#   dict:
#       noisy_wave: normalized full noisy waveform, shape [1, T_orig]
#       noisy_spec: transformed complex STFT before padding, shape [1, 256, K]
#       noisy_2ch: real/imag noisy before padding, shape [2, 256, K]
#       noisy_2ch_padded: real/imag noisy after padding, shape [2, 256, K_pad]
#       normfac: scalar normalization factor
#       orig_len: original waveform length T_orig
#       orig_frames: original STFT frame count K
#       pad_frames: padded STFT frame count
#
# Used for:
#   inference
def preprocess_noisy_for_inference(
    noisy_path: str,
    cfg: Optional[AudioPreprocessConfig] = None,
    pad_to_64: bool = True,
) -> Dict[str, torch.Tensor]:
    if cfg is None:
        cfg = AudioPreprocessConfig()

    noisy, _ = load_wav(noisy_path, cfg)

    orig_len = noisy.size(-1)

    noisy, normfac = normalize_noisy(noisy, cfg)

    noisy_spec = spec_fwd(stft(noisy, cfg), cfg)
    noisy_2ch = complex_to_channels(noisy_spec)

    if pad_to_64:
        noisy_2ch_padded, pad_frames = pad_time_to_multiple(noisy_2ch, multiple=64)
    else:
        noisy_2ch_padded, pad_frames = noisy_2ch, 0

    return {
        "noisy_wave": noisy,
        "noisy_spec": noisy_spec,
        "noisy_2ch": noisy_2ch,
        "noisy_2ch_padded": noisy_2ch_padded,
        "normfac": normfac,
        "orig_len": torch.tensor(orig_len, dtype=torch.long),
        "orig_frames": torch.tensor(noisy_2ch.size(-1), dtype=torch.long),
        "pad_frames": torch.tensor(pad_frames, dtype=torch.long),
    }


# Input:
#   pred_2ch: model output, shape [2, F, K] or padded [2, F, K_pad]
#   normfac: scalar normalization factor
#   orig_len: original waveform length
#   pad_frames: number of padded time frames
#   cfg: preprocessing config
# Output:
#   reconstructed waveform, shape [1, orig_len]
#
# Used for:
#   inference reconstruction
def reconstruct_from_2ch(
    pred_2ch: torch.Tensor,
    normfac: torch.Tensor,
    orig_len: int,
    pad_frames: int = 0,
    cfg: Optional[AudioPreprocessConfig] = None,
) -> torch.Tensor:
    if cfg is None:
        cfg = AudioPreprocessConfig()

    pred_2ch = unpad_time(pred_2ch, pad_frames)

    pred_spec = channels_to_complex(pred_2ch)
    pred_spec = spec_back(pred_spec, cfg)

    wav = istft(pred_spec, cfg, length=orig_len)
    wav = wav * normfac

    return wav