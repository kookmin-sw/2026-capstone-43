# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


def monotonic_sinusoidal_embedding(times, embedding_dim, max_period=10000.0, time_scale=1000.0):
  """Deterministic sinusoidal embedding for absolute monotonic times."""
  if embedding_dim <= 0:
    raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
  if times.dim() != 2:
    raise ValueError(f"Expected times shape [B, L], got {tuple(times.shape)}")
  if max_period <= 1.0:
    raise ValueError(f"max_period must be > 1.0, got {max_period}")

  out_dtype = times.dtype
  device = times.device
  calc_dtype = torch.float64
  scaled = times.to(dtype=calc_dtype) / max(float(time_scale), 1e-8)
  half_dim = embedding_dim // 2
  if half_dim <= 0:
    return scaled.new_zeros((*scaled.shape, 1))

  exponent = torch.arange(half_dim, device=device, dtype=calc_dtype)
  exponent = exponent / max(half_dim - 1, 1)
  freqs = torch.exp(
    -torch.log(torch.tensor(float(max_period), device=device, dtype=calc_dtype)) * exponent
  )
  args = (2.0 * torch.pi) * scaled.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
  emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
  if embedding_dim % 2 == 1:
    emb = F.pad(emb, (0, 1))
  return emb.to(dtype=out_dtype)


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(
      self,
      channels,
      skip_rescale=False,
      init_scale=0.,
      context_dim=None,
      aux_time_scale=1000.0,
      aux_time_embed_dim=128,
      aux_time_max_period=10000.0,
  ):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.context_dim = context_dim
    self.channels = channels
    self.aux_time_scale = float(aux_time_scale)
    self.aux_time_embed_dim = int(aux_time_embed_dim)
    self.aux_time_max_period = float(aux_time_max_period)
    self.use_monotonic_time = context_dim is not None and self.aux_time_embed_dim > 0
    if context_dim is not None:
      self.ContextK = nn.Conv1d(context_dim, channels, kernel_size=1, bias=True)
      self.ContextV = nn.Conv1d(context_dim, channels, kernel_size=1, bias=True)
      self.ContextOut = NIN(channels, channels, init_scale=init_scale)
      if self.use_monotonic_time:
        self.QueryTimeQ = nn.Linear(self.aux_time_embed_dim, channels)
        self.ContextTimeK = nn.Linear(self.aux_time_embed_dim, channels)
        self.ContextTimeV = nn.Linear(self.aux_time_embed_dim, channels)
    self.skip_rescale = skip_rescale

  @staticmethod
  def _resize_1d(values, target_len, mode="linear"):
    if values is None:
      return None
    if values.dim() != 2:
      raise ValueError(f"Expected shape [B, L], got {tuple(values.shape)}")
    if values.size(1) == target_len:
      return values
    if mode == "linear" and values.size(1) == 1:
      return values.expand(values.size(0), target_len)
    if mode == "linear":
      resized = F.interpolate(
        values.unsqueeze(1),
        size=target_len,
        mode=mode,
        align_corners=True,
      )
    else:
      resized = F.interpolate(
        values.unsqueeze(1),
        size=target_len,
        mode=mode,
      )
    return resized.squeeze(1)

  def forward(self, x, context=None, context_times=None, query_times=None, context_mask=None):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)

    # Optional cross-attention to external condition tokens.
    # context: [B, context_dim, L]
    if context is not None and self.context_dim is not None:
      if context.dim() != 3:
        raise ValueError(f"Expected context [B, C_ctx, L], got {tuple(context.shape)}")
      if context.size(0) != B:
        raise ValueError(f"Context batch mismatch: {context.size(0)} vs {B}")
      if context.size(1) != self.context_dim:
        raise ValueError(f"Context channel mismatch: {context.size(1)} vs {self.context_dim}")

      ctx = context.to(dtype=q.dtype, device=q.device)
      k_ctx = self.ContextK(ctx)  # [B, C, L]
      v_ctx = self.ContextV(ctx)  # [B, C, L]

      q_cross = q

      if self.use_monotonic_time and query_times is not None:
        q_times = query_times
        if q_times.dim() == 3 and q_times.size(1) == 1:
          q_times = q_times.squeeze(1)
        q_times = q_times.to(device=q.device)
        q_times = self._resize_1d(q_times, W, mode="linear")
        q_time_emb = monotonic_sinusoidal_embedding(
          q_times,
          embedding_dim=self.aux_time_embed_dim,
          max_period=self.aux_time_max_period,
          time_scale=self.aux_time_scale,
        ).to(dtype=q.dtype)
        q_time_bias = self.QueryTimeQ(q_time_emb).permute(0, 2, 1).unsqueeze(2).expand(-1, -1, H, -1)
        q_cross = q_cross + q_time_bias

      resized_ctx_mask = None
      if context_mask is not None:
        resized_ctx_mask = context_mask
        if resized_ctx_mask.dim() == 3 and resized_ctx_mask.size(1) == 1:
          resized_ctx_mask = resized_ctx_mask.squeeze(1)
        resized_ctx_mask = resized_ctx_mask.to(device=q.device, dtype=q.dtype)
        resized_ctx_mask = self._resize_1d(resized_ctx_mask, k_ctx.size(-1), mode="nearest") > 0.5

      if self.use_monotonic_time and context_times is not None:
        ctx_times = context_times
        if ctx_times.dim() == 3 and ctx_times.size(1) == 1:
          ctx_times = ctx_times.squeeze(1)
        ctx_times = ctx_times.to(device=q.device)
        ctx_times = self._resize_1d(ctx_times, k_ctx.size(-1), mode="linear")
        ctx_time_emb = monotonic_sinusoidal_embedding(
          ctx_times,
          embedding_dim=self.aux_time_embed_dim,
          max_period=self.aux_time_max_period,
          time_scale=self.aux_time_scale,
        ).to(dtype=q.dtype)
        if resized_ctx_mask is not None:
          ctx_time_emb = ctx_time_emb * resized_ctx_mask.unsqueeze(-1).to(dtype=ctx_time_emb.dtype)
        k_ctx = k_ctx + self.ContextTimeK(ctx_time_emb).permute(0, 2, 1)
        v_ctx = v_ctx + self.ContextTimeV(ctx_time_emb).permute(0, 2, 1)

      q_flat = q_cross.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
      attn_ctx = torch.bmm(q_flat, k_ctx) * (int(C) ** (-0.5))  # [B, HW, L]
      if resized_ctx_mask is not None:
        attn_ctx = attn_ctx.masked_fill(~resized_ctx_mask.unsqueeze(1), -1e4)
      attn_ctx = F.softmax(attn_ctx, dim=-1)
      h_ctx = torch.bmm(attn_ctx, v_ctx.transpose(1, 2))  # [B, HW, C]
      h_ctx = h_ctx.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
      h_ctx = self.ContextOut(h_ctx)

      h = h + h_ctx

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
