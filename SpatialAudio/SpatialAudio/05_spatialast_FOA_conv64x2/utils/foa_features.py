import torch
import torch.nn as nn
import numpy as np


class FOANativeFeatureExtractor(nn.Module):
    """
    DCASE-style FOA front-end:
    - 4-channel log-mel from canonical WXYZ
    - 3-channel normalized FOA intensity vectors

    This follows the DCASE2024 SELD baseline idea more closely than the
    earlier proxy stem:
    mel = power_to_db(mel(|STFT|^2))
    IV = mel(Re(conj(W) * XYZ) / (|W|^2 + mean(|XYZ|^2) + eps))
    """
    def __init__(
            self,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            use_diffuseness=False,
            use_beam_proxy=False,
            eps=1e-8,
        ):
        super().__init__()
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.use_diffuseness = use_diffuseness
        self.use_beam_proxy = use_beam_proxy
        self.eps = eps

    @property
    def channels_num(self):
        channels = 7
        if self.use_diffuseness:
            channels += 1
        if self.use_beam_proxy:
            channels += 1
        return channels

    def _mel_project(self, feature, melW):
        return torch.matmul(feature, melW)

    def _power_to_db(self, feature):
        log_spec = 10.0 * torch.log10(torch.clamp(feature, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, self.ref))
        if self.top_db is not None:
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)
        return log_spec

    def forward(self, real, imag, melW):
        """
        Args:
            real: STFT real part, shape [B, 4, frames, freq_bins] in canonical WXYZ order.
            imag: STFT imaginary part, shape [B, 4, frames, freq_bins] in canonical WXYZ order.

        Returns:
            stacked FOA-native features, shape [B, C, frames, mel_bins].
        """
        assert real.ndim == 4 and imag.ndim == 4, 'FOA STFT tensors must be [B, 4, frames, freq_bins].'
        assert real.shape == imag.shape, f'FOA real/imag shapes must match, got {real.shape} and {imag.shape}.'
        assert real.shape[1] == 4, f'FOA STFT must be canonical WXYZ with 4 channels, got {real.shape[1]}.'

        power = real ** 2 + imag ** 2
        log_mel = self._power_to_db(self._mel_project(power, melW))
        log_mel = (log_mel - log_mel.mean(dim=(-2, -1), keepdim=True)) / (
            log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        )
        log_mel = torch.clamp(log_mel, -5.0, 5.0)

        w_real, x_real, y_real, z_real = real[:, 0], real[:, 1], real[:, 2], real[:, 3]
        w_imag, x_imag, y_imag, z_imag = imag[:, 0], imag[:, 1], imag[:, 2], imag[:, 3]

        # DCASE SELD baseline-style normalized FOA intensity vectors.
        intensity_x = w_real * x_real + w_imag * x_imag
        intensity_y = w_real * y_real + w_imag * y_imag
        intensity_z = w_real * z_real + w_imag * z_imag
        energy = self.eps + (power[:, 0] + (power[:, 1] + power[:, 2] + power[:, 3]) / 3.0)
        i_norm = torch.stack([
            intensity_x / energy,
            intensity_y / energy,
            intensity_z / energy,
        ], dim=1)
        i_norm = i_norm / (i_norm.std(dim=(-2, -1), keepdim=True) + 1e-6)
        i_norm = i_norm * 3.0
        iv = self._mel_project(i_norm, melW)
        iv = iv / (iv.std(dim=(-2, -1), keepdim=True) + 1e-6)
        iv = iv * 3.0
        iv = torch.clamp(iv, -5.0, 5.0)

        if not hasattr(self, "_debug_print"):
            print("logmel mean/std:", log_mel.mean().item(), log_mel.std().item())
            print("IV mean/std:", iv.mean().item(), iv.std().item())
            self._debug_print = True

        features = [log_mel, iv]
        debug_shapes = {
            'foa_log_mel': tuple(log_mel.shape),
            'foa_iv': tuple(iv.shape),
            'foa_aiv': tuple(iv.shape),
        }

        if self.use_diffuseness:
            intensity_mag = torch.sqrt(intensity_x ** 2 + intensity_y ** 2 + intensity_z ** 2 + self.eps)
            diffuseness = 1.0 - torch.clamp(intensity_mag / energy, min=0.0, max=1.0)
            diffuseness = self._mel_project(diffuseness, melW).clamp(0.0, 1.0).unsqueeze(1)
            features.append(diffuseness)
            debug_shapes['foa_diffuseness'] = tuple(diffuseness.shape)

        if self.use_beam_proxy:
            beam_proxy = torch.sqrt(torch.sum(iv ** 2, dim=1, keepdim=True) + self.eps)
            features.append(beam_proxy)
            debug_shapes['foa_beam_proxy'] = tuple(beam_proxy.shape)

        stacked = torch.cat(features, dim=1)
        debug_shapes['foa_stacked'] = tuple(stacked.shape)
        return stacked, debug_shapes
