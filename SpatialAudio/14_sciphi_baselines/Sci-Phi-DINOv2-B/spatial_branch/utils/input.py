"""FOA spatial feature extractor for SELD-style inputs.

Implements the core feature logic used in the DCASE SELD baseline:
- 4-channel mel spectrogram maps
- 3-channel FOA intensity vector maps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal
import torch


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 2 ** (x - 1).bit_length()


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))


def _mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2.0

    mel_points = np.linspace(
        _hz_to_mel(np.array([fmin]))[0],
        _hz_to_mel(np.array([fmax]))[0],
        n_mels + 2,
        dtype=np.float64,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = min(left + 1, n_fft // 2)
        if right == center:
            right = min(center + 1, n_fft // 2)

        for k in range(left, center):
            filterbank[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, right):
            filterbank[m - 1, k] = (right - k) / float(right - center)
    return filterbank.T  # [freq_bins, mel_bins]


def _power_to_db(power: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    power = np.maximum(power, amin)
    ref = max(ref, amin)
    return 10.0 * np.log10(power) - 10.0 * np.log10(ref)


@dataclass
class FOASpatialFeatureExtractor:
    sample_rate: int = 24000
    hop_len_s: float = 0.02
    nb_mel_bins: int = 64
    nb_channels: int = 4
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.hop_len = int(self.sample_rate * self.hop_len_s)
        self.win_len = 2 * self.hop_len
        self.nfft = _next_power_of_two(self.win_len)
        self.mel_wts = _build_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.nfft,
            n_mels=self.nb_mel_bins,
        ).astype(np.float32)

    def _ensure_float(self, waveform: np.ndarray) -> np.ndarray:
        if np.issubdtype(waveform.dtype, np.integer):
            scale = float(np.iinfo(waveform.dtype).max)
            waveform = waveform.astype(np.float32) / scale
        else:
            waveform = waveform.astype(np.float32)
        return waveform

    def _resample(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.sample_rate:
            return waveform
        gcd = np.gcd(sample_rate, self.sample_rate)
        up = self.sample_rate // gcd
        down = sample_rate // gcd
        return scipy.signal.resample_poly(waveform, up, down, axis=0).astype(np.float32)

    def _spectrogram(self, audio_input: np.ndarray) -> np.ndarray:
        spectra = []
        for ch_idx in range(self.nb_channels):
            _, _, stft_ch = scipy.signal.stft(
                audio_input[:, ch_idx],
                fs=self.sample_rate,
                window="hann",
                nperseg=self.win_len,
                noverlap=self.win_len - self.hop_len,
                nfft=self.nfft,
                boundary=None,
                padded=False,
            )
            spectra.append(stft_ch)
        # [channels, bins, frames] -> [frames, bins, channels]
        return np.array(spectra).transpose(2, 1, 0)

    def _mel_maps(self, linear_spectra: np.ndarray) -> np.ndarray:
        frames, _, channels = linear_spectra.shape
        mel_feat = np.zeros((frames, channels, self.nb_mel_bins), dtype=np.float32)
        for ch_idx in range(channels):
            mag = np.abs(linear_spectra[:, :, ch_idx]) ** 2
            mel = np.dot(mag, self.mel_wts)
            mel_db = _power_to_db(mel)
            mel_feat[:, ch_idx, :] = mel_db
        return mel_feat

    def _foa_intensity_vector_maps(self, linear_spectra: np.ndarray) -> np.ndarray:
        # W channel is channel 0, directional channels ar
        # 이전에 채널이 꼬였을 때 나왔던 "Down Back-Left"와 비교해보면, 상하(Up/Down)와 좌우(Left/Right)가 완벽하게 뒤집혀서 이제야 진짜 제자리를 찾은 것을 알 수 있습니다.

        w = linear_spectra[:, :, 0]
        directional = linear_spectra[:, :, 1:]
        intensity = np.real(np.conj(w)[:, :, None] * directional)
        energy = self.eps + (np.abs(w) ** 2 + (np.abs(directional) ** 2).sum(-1) / 3.0)
        intensity_norm = intensity / energy[:, :, None]

        intensity_mel = np.transpose(
            np.dot(np.transpose(intensity_norm, (0, 2, 1)), self.mel_wts),
            (0, 2, 1),
        )
        iv_maps = intensity_mel.transpose((0, 2, 1)).astype(np.float32)
        if np.isnan(iv_maps).any():
            raise ValueError("NaN detected while computing FOA intensity vectors.")
        return iv_maps

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return FOA spatial maps with shape [frames, 7, mel_bins].

        The caller must provide canonical 4-channel FOA input in ``WYZX`` order
        when using DCASE/STARSS FOA checkpoints.
        This extractor does not reorder channels internally.
        """
        if waveform.ndim != 2:
            raise ValueError(
                f"Expected 2D waveform [samples, channels], got shape={waveform.shape}"
            )
        if waveform.shape[1] != self.nb_channels:
            raise ValueError(
                f"Expected exactly {self.nb_channels} FOA channels already in WYZX order, got {waveform.shape[1]}"
            )
        waveform = self._ensure_float(waveform)
        waveform = self._resample(waveform, sample_rate)

        spect = self._spectrogram(waveform)
        mel_maps = self._mel_maps(spect)  # [frames, 4, mel]
        iv_maps = self._foa_intensity_vector_maps(spect)  # [frames, 3, mel]
        return np.concatenate((mel_maps, iv_maps), axis=1).astype(np.float32)


def feature_maps_to_seld_tensor(
    feature_maps: np.ndarray | torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """Convert [frames, 7, mel] maps to [1, 7, frames, mel] tensor."""
    if isinstance(feature_maps, np.ndarray):
        tensor = torch.from_numpy(feature_maps)
    else:
        tensor = feature_maps

    if tensor.ndim == 3:
        tensor = tensor.permute(1, 0, 2).unsqueeze(0).contiguous()
    elif tensor.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Expected 3D [frames, channels, mel] or 4D tensor, got shape={tuple(tensor.shape)}"
        )

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor
