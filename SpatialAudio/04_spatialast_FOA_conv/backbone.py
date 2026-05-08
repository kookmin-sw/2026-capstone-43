from functools import partial

import torch
import torch.nn as nn
import torchaudio
from utils.foa_features import FOANativeFeatureExtractor
from utils.stft import STFT, LogmelFilterBank
from utils.torch_layers import to_2tuple
from utils.vision_transformer import VisionTransformer as _VisionTransformer


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


FOA_STEM_VARIANTS = {
    "baseline": (16, 1),
    "conv32_out4": (32, 4),
    "conv32_out8": (32, 8),
    "conv64_out8": (64, 8),
    "conv64_out16": (64, 16),
}


def build_foa_stem(in_chans, hidden_chans, out_chans):
    return nn.Sequential(
        nn.GroupNorm(1, in_chans),
        conv3x3(in_chans, hidden_chans),
        nn.BatchNorm2d(hidden_chans),
        nn.GELU(),
        conv3x3(hidden_chans, out_chans),
        nn.BatchNorm2d(out_chans),
        nn.GELU(),
    )


def resolve_foa_stem_channels(variant, default_hidden, hidden_override=0, out_override=0):
    if variant not in FOA_STEM_VARIANTS:
        raise ValueError(f"Unsupported FOA stem variant: {variant}")

    variant_hidden, variant_out = FOA_STEM_VARIANTS[variant]
    hidden_chans = hidden_override or (default_hidden if variant == "baseline" else variant_hidden)
    out_chans = out_override or variant_out
    return hidden_chans, out_chans


class PatchEmbedNew(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class SpatialASTFOABackbone(_VisionTransformer):
    def __init__(
            self,
            num_cls_tokens=3,
            reverb_type="foa",
            foa_stem_type="foa_native",
            foa_use_diffuseness=False,
            foa_use_beam_proxy=False,
            foa_stem_channels=16,
            foa_stem_variant="baseline",
            foa_stem_hidden_channels=0,
            foa_stem_out_channels=0,
            patch_in_from_stem=True,
            **kwargs,
        ):
        kwargs.setdefault("num_classes", 0)
        super().__init__(**kwargs)

        img_size = (1024, 128)
        emb_dim = kwargs["embed_dim"]
        self.reverb_type = reverb_type
        self.foa_stem_type = foa_stem_type
        self.embed_dim = emb_dim
        self.foa_stem_variant = foa_stem_variant
        self.patch_in_from_stem = patch_in_from_stem

        del self.cls_token
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))
        torch.nn.init.normal_(self.cls_tokens, std=0.02)

        self.spectrogram_extractor = STFT(
            n_fft=1024,
            hop_length=320,
            win_length=1024,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=32000,
            n_fft=1024,
            n_mels=128,
            fmin=50,
            fmax=14000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.conv_downsample = nn.Sequential(
            conv3x3(4, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)

        self.bn = nn.BatchNorm2d(2, affine=False)
        self.foa_bn = nn.BatchNorm2d(4, affine=False)
        self.foa_feature_extractor = FOANativeFeatureExtractor(
            ref=1.0,
            amin=1e-10,
            top_db=None,
            use_diffuseness=foa_use_diffuseness,
            use_beam_proxy=foa_use_beam_proxy,
        )
        self.foa_stem_hidden_channels, self.foa_stem_out_channels = resolve_foa_stem_channels(
            variant=foa_stem_variant,
            default_hidden=foa_stem_channels,
            hidden_override=foa_stem_hidden_channels,
            out_override=foa_stem_out_channels,
        )

        # Default FOA input follows the DCASE SELD baseline:
        # 4-channel log-mel (WXYZ) + 3-channel normalized intensity vectors.
        self.foa_native_stem = build_foa_stem(
            in_chans=self.foa_feature_extractor.channels_num,
            hidden_chans=self.foa_stem_hidden_channels,
            out_chans=self.foa_stem_out_channels,
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(self.foa_stem_out_channels, self.foa_stem_out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.foa_stem_out_channels),
        )

        if self.foa_stem_type == "foa_native":
            patch_in_chans = self.foa_stem_out_channels if patch_in_from_stem else 1
            if not patch_in_from_stem and self.foa_stem_out_channels != 1:
                raise ValueError(
                    "patch_in_from_stem=False is only supported when foa_native stem out channels stay at 1"
                )
        else:
            patch_in_chans = 1

        self.patch_embed = PatchEmbedNew(
            img_size=img_size,
            patch_size=(16, 16),
            in_chans=patch_in_chans,
            embed_dim=emb_dim,
            stride=16,
        )
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_dim),
            requires_grad=False,
        )

        del self.norm
        self.target_frame = 1024
        self._foa_debug_printed = False
        self.last_debug_shapes = {}

        norm_layer = kwargs["norm_layer"]
        self.dis_norm = norm_layer(emb_dim)
        self.doa_norm = norm_layer(emb_dim)
        self.fc_norm = norm_layer(emb_dim)

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        n, _, d = x.shape
        t, f = 64, 8

        x = x.reshape(n, t, f, d)
        len_keep_t = int(t * (1 - mask_t_prob))
        noise = torch.rand(n, t, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep_t]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, f, d)
        x = torch.gather(x, dim=1, index=index)

        x = x.permute(0, 2, 1, 3)
        len_keep_f = int(f * (1 - mask_f_prob))
        noise = torch.rand(n, f, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep_f]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_t, d)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3)
        x_masked = x_masked.reshape(n, len_keep_f * len_keep_t, d)
        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        batch = x.shape[0]
        x = x + self.pos_embed[:, 1:, :]

        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x, _, _ = self.random_masking_2d(x, mask_t_prob, mask_f_prob)

        cls_tokens = self.cls_tokens.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def _forward_stem(self, waveforms, reverbs):
        self.last_debug_shapes = {
            "waveform_input": tuple(waveforms.shape),
        }

        if self.reverb_type == "foa":
            assert waveforms.ndim == 3, f"FOA waveforms must have shape [B, 4, T], got {tuple(waveforms.shape)}"
            assert waveforms.shape[1] == 4, f"FOA input must have exactly 4 channels, got {waveforms.shape[1]}"
            batch, channels, time = waveforms.shape
            if not self._foa_debug_printed:
                print(f"[FOA] backbone input waveform shape (canonical WXYZ): {tuple(waveforms.shape)}")
        else:
            waveforms = torchaudio.functional.fftconvolve(
                waveforms,
                reverbs,
                mode="full",
            )[..., :waveforms.shape[-1]]
            batch, channels, time = waveforms.shape

        waveforms = waveforms.reshape(batch * channels, time)
        real, imag = self.spectrogram_extractor(waveforms)

        if self.reverb_type == "foa":
            real_foa = real.reshape(batch, channels, -1, real.shape[-1])
            imag_foa = imag.reshape(batch, channels, -1, imag.shape[-1])

            if self.foa_stem_type == "foa_native":
                x, foa_debug_shapes = self.foa_feature_extractor(
                    real_foa,
                    imag_foa,
                    self.logmel_extractor.melW,
                )
                self.last_debug_shapes.update(foa_debug_shapes)
                if x.shape[2] < self.target_frame:
                    x = nn.functional.interpolate(
                        x,
                        (self.target_frame, x.shape[3]),
                        mode="bicubic",
                        align_corners=True,
                    )
                self.last_debug_shapes["foa_stacked_after_resize"] = tuple(x.shape)
                x = self.foa_native_stem(x)
                x = self.adapter(x)
                self.last_debug_shapes["stem_output"] = tuple(x.shape)
                self.last_debug_shapes["adapter_output"] = tuple(x.shape)
                if not hasattr(self, "_debug_printed"):
                    print("[DEBUG] stem+adapter output mean/std:", x.mean().item(), x.std().item())
                    self._debug_printed = True
                if not self._foa_debug_printed:
                    print(f"[FOA] native stacked input shape: {self.last_debug_shapes['foa_stacked']}")
                    print(f"[FOA] stem output shape: {tuple(x.shape)}")
                    self._foa_debug_printed = True
            elif self.foa_stem_type == "logmel_only":
                log_mel = self.logmel_extractor(torch.sqrt(real_foa ** 2 + imag_foa ** 2)).reshape(batch, channels, -1, 128)
                x = self.foa_bn(log_mel)
                self.last_debug_shapes["foa_log_mel"] = tuple(log_mel.shape)
                if x.shape[2] < self.target_frame:
                    x = nn.functional.interpolate(
                        x,
                        (self.target_frame, x.shape[3]),
                        mode="bicubic",
                        align_corners=True,
                    )
                x = self.conv_downsample(x)
                self.last_debug_shapes["stem_output"] = tuple(x.shape)
                if not self._foa_debug_printed:
                    print(f"[FOA] log-mel-only feature shape before conv_downsample: {tuple(log_mel.shape)}")
                    print(f"[FOA] stem output shape: {tuple(x.shape)}")
                    self._foa_debug_printed = True
            else:
                raise ValueError(f"Unsupported FOA stem type: {self.foa_stem_type}")
        else:
            log_mel = self.logmel_extractor(torch.sqrt(real ** 2 + imag ** 2)).reshape(batch, channels, -1, 128)
            log_mel = self.bn(log_mel)
            ipd = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
            x = torch.cat(
                [
                    log_mel,
                    torch.matmul(
                        torch.cat([torch.cos(ipd), torch.sin(ipd)], dim=1),
                        self.logmel_extractor.melW,
                    ),
                ],
                dim=1,
            )
            if x.shape[2] < self.target_frame:
                x = nn.functional.interpolate(
                    x,
                    (self.target_frame, x.shape[3]),
                    mode="bicubic",
                    align_corners=True,
                )
            x = self.conv_downsample(x)
            self.last_debug_shapes["stem_output"] = tuple(x.shape)

        return x

    def forward(self, waveforms, reverbs=None, mask_t_prob=0.0, mask_f_prob=0.0):
        if reverbs is None:
            reverbs = torch.zeros(waveforms.shape[0], 1, 1, device=waveforms.device, dtype=waveforms.dtype)

        x = self._forward_stem(waveforms, reverbs)

        if self.training:
            x = x.transpose(-2, -1)
            x = self.freqm(x)
            x = self.timem(x)
            x = x.transpose(-2, -1)

        x = self.patch_embed(x)
        self.last_debug_shapes["patch_embed_output"] = tuple(x.shape)
        x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)
        self.last_debug_shapes["backbone_output"] = tuple(x.shape)

        distance_token = self.dis_norm(x[:, 0])
        doa_token = self.doa_norm(x[:, 1])
        class_token = self.fc_norm(x[:, 2])

        return {
            "sequence": x,
            "distance_token": distance_token,
            "doa_token": doa_token,
            "class_token": class_token,
            "debug_shapes": dict(self.last_debug_shapes),
        }

    def get_debug_shapes(self):
        return dict(self.last_debug_shapes)


def build_backbone(**kwargs):
    return SpatialASTFOABackbone(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
