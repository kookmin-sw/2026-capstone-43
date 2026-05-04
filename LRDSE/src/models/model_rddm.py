import math
import random
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn
from tqdm.auto import tqdm


ModelResPrediction = namedtuple(
    "ModelResPrediction",
    ["pred_res", "pred_noise", "pred_x_start"],
)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(
            weight,
            "o ... -> o 1 1 1",
            partial(torch.var, unbiased=False),
        )
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0

        half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(half_dim),
            requires_grad=not is_random,
        )

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (heads c) x y -> b heads c (x y)",
                heads=self.heads,
            ),
            qkv,
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        out = rearrange(
            out,
            "b heads c (x y) -> b (heads c) x y",
            heads=self.heads,
            x=h,
            y=w,
        )

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (heads c) x y -> b heads c (x y)",
                heads=self.heads,
            ),
            qkv,
        )

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(
            out,
            "b heads (x y) d -> b (heads d) x y",
            x=h,
            y=w,
        )

        return self.to_out(out)


class CrossAttention(nn.Module):
    """
    U-Net feature는 query로 쓰고,
    foot_force condition embedding은 key/value로 쓰는 cross-attention.

    x:
        [B, C, H, W]

    context:
        [B, N, context_dim]
    """

    def __init__(self, dim, context_dim, heads=4, dim_head=32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads

        hidden_dim = heads * dim_head

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context):
        b, c, height, width = x.shape

        q = self.to_q(x)
        q = rearrange(
            q,
            "b (h d) x y -> b h d (x y)",
            h=self.heads,
        )

        k = self.to_k(context)
        v = self.to_v(context)

        k = rearrange(
            k,
            "b n (h d) -> b h d n",
            h=self.heads,
        )
        v = rearrange(
            v,
            "b n (h d) -> b h d n",
            h=self.heads,
        )

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(
            out,
            "b h (x y) d -> b (h d) x y",
            x=height,
            y=width,
        )

        return self.to_out(out)


class AuxConditionEncoder(nn.Module):
    """
    foot_force condition encoder.

    Input:
        aux_cond: [B, 8, 1024]

    Output:
        context: [B, 256, context_dim]

    1024 -> 512 -> 256 으로 시간축을 줄인 뒤,
    cross-attention의 key/value context로 사용한다.
    """

    def __init__(
        self,
        aux_cond_dim=8,
        context_dim=256,
        kernel_size=5,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                aux_cond_dim,
                context_dim,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.SiLU(),
            nn.Conv1d(
                context_dim,
                context_dim,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.SiLU(),
            nn.Conv1d(
                context_dim,
                context_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.SiLU(),
        )

    def forward(self, aux_cond):
        if aux_cond.dim() != 3:
            raise ValueError(
                f"Expected aux_cond shape [B, 8, K], got {aux_cond.shape}"
            )

        context = self.net(aux_cond)
        context = context.permute(0, 2, 1).contiguous()

        return context


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False,
        use_aux_cond=False,
        aux_cond_dim=8,
        aux_context_dim=None,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        self.use_aux_cond = use_aux_cond
        self.aux_cond_dim = aux_cond_dim

        input_channels = (
            channels
            + channels * (1 if self_condition else 0)
            + channels * (1 if condition else 0)
            + channels * (1 if input_condition else 0)
        )

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim,
                random_fourier_features,
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        aux_context_dim = default(aux_context_dim, mid_dim)

        self.aux_cond_encoder = (
            AuxConditionEncoder(
                aux_cond_dim=aux_cond_dim,
                context_dim=aux_context_dim,
                kernel_size=5,
            )
            if use_aux_cond
            else None
        )

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))

        self.mid_cross_attn = (
            Residual(
                PreNorm(
                    mid_dim,
                    CrossAttention(
                        mid_dim,
                        context_dim=aux_context_dim,
                    ),
                )
            )
            if use_aux_cond
            else None
        )

        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                        ),
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(
            dim * 2,
            dim,
            time_emb_dim=time_dim,
        )
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def build_aux_context(self, aux_cond):
        if not self.use_aux_cond or aux_cond is None:
            return None

        if aux_cond.dim() != 3:
            raise ValueError(
                f"Expected aux_cond shape [B, 8, K], got {aux_cond.shape}"
            )

        if aux_cond.size(1) != self.aux_cond_dim:
            raise ValueError(
                f"Expected aux_cond channel dim = {self.aux_cond_dim}, "
                f"got {aux_cond.size(1)}"
            )

        context = self.aux_cond_encoder(aux_cond)

        return context

    def forward(self, x, time, x_self_cond=None, aux_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        context = self.build_aux_context(aux_cond)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)

        if self.use_aux_cond and context is not None:
            x = self.mid_cross_attn(x, context)

        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        share_encoder=0,
        condition=True,
        input_condition=False,
        use_aux_cond=False,
        aux_cond_dim=8,
        aux_context_dim=None,
    ):
        super().__init__()

        self.condition = condition
        self.input_condition = input_condition
        self.share_encoder = share_encoder
        self.channels = channels
        self.self_condition = self_condition
        self.use_aux_cond = use_aux_cond
        self.aux_cond_dim = aux_cond_dim

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.share_encoder == 1:
            input_channels = (
                channels
                + channels * (1 if self_condition else 0)
                + channels * (1 if condition else 0)
                + channels * (1 if input_condition else 0)
            )

            init_dim = default(init_dim, dim)
            self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

            dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            in_out = list(zip(dims[:-1], dims[1:]))

            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

            time_dim = dim * 4

            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim,
                    random_fourier_features,
                )
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            self.ups_no_skip = nn.ModuleList([])

            num_resolutions = len(in_out)

            for ind, (dim_in, dim_out) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)

                self.downs.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                        ]
                    )
                )

            mid_dim = dims[-1]
            aux_context_dim = default(aux_context_dim, mid_dim)

            self.aux_cond_encoder = (
                AuxConditionEncoder(
                    aux_cond_dim=aux_cond_dim,
                    context_dim=aux_context_dim,
                    kernel_size=5,
                )
                if use_aux_cond
                else None
            )

            self.mid_block1 = block_klass(
                mid_dim,
                mid_dim,
                time_emb_dim=time_dim,
            )
            self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))

            self.mid_cross_attn = (
                Residual(
                    PreNorm(
                        mid_dim,
                        CrossAttention(
                            mid_dim,
                            context_dim=aux_context_dim,
                        ),
                    )
                )
                if use_aux_cond
                else None
            )

            self.mid_block2 = block_klass(
                mid_dim,
                mid_dim,
                time_emb_dim=time_dim,
            )

            for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
                is_last = ind == (len(in_out) - 1)

                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(
                                dim_out + dim_in,
                                dim_out,
                                time_emb_dim=time_dim,
                            ),
                            block_klass(
                                dim_out + dim_in,
                                dim_out,
                                time_emb_dim=time_dim,
                            ),
                            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                        ]
                    )
                )

                self.ups_no_skip.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                            block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                        ]
                    )
                )

            self.final_res_block_1 = block_klass(
                dim,
                dim,
                time_emb_dim=time_dim,
            )
            self.final_conv_1 = nn.Conv2d(dim, self.out_dim, 1)

            self.final_res_block_2 = block_klass(
                dim * 2,
                dim,
                time_emb_dim=time_dim,
            )
            self.final_conv_2 = nn.Conv2d(dim, self.out_dim, 1)

        elif self.share_encoder == 0:
            self.unet0 = Unet(
                dim,
                init_dim=init_dim,
                out_dim=out_dim,
                dim_mults=dim_mults,
                channels=channels,
                self_condition=self_condition,
                resnet_block_groups=resnet_block_groups,
                learned_variance=learned_variance,
                learned_sinusoidal_cond=learned_sinusoidal_cond,
                random_fourier_features=random_fourier_features,
                learned_sinusoidal_dim=learned_sinusoidal_dim,
                condition=condition,
                input_condition=input_condition,
                use_aux_cond=use_aux_cond,
                aux_cond_dim=aux_cond_dim,
                aux_context_dim=aux_context_dim,
            )

            self.unet1 = Unet(
                dim,
                init_dim=init_dim,
                out_dim=out_dim,
                dim_mults=dim_mults,
                channels=channels,
                self_condition=self_condition,
                resnet_block_groups=resnet_block_groups,
                learned_variance=learned_variance,
                learned_sinusoidal_cond=learned_sinusoidal_cond,
                random_fourier_features=random_fourier_features,
                learned_sinusoidal_dim=learned_sinusoidal_dim,
                condition=condition,
                input_condition=input_condition,
                use_aux_cond=use_aux_cond,
                aux_cond_dim=aux_cond_dim,
                aux_context_dim=aux_context_dim,
            )

        elif self.share_encoder == -1:
            self.unet0 = Unet(
                dim,
                init_dim=init_dim,
                out_dim=out_dim,
                dim_mults=dim_mults,
                channels=channels,
                self_condition=self_condition,
                resnet_block_groups=resnet_block_groups,
                learned_variance=learned_variance,
                learned_sinusoidal_cond=learned_sinusoidal_cond,
                random_fourier_features=random_fourier_features,
                learned_sinusoidal_dim=learned_sinusoidal_dim,
                condition=condition,
                input_condition=input_condition,
                use_aux_cond=use_aux_cond,
                aux_cond_dim=aux_cond_dim,
                aux_context_dim=aux_context_dim,
            )

    def build_aux_context(self, aux_cond):
        if not self.use_aux_cond or aux_cond is None:
            return None

        if aux_cond.dim() != 3:
            raise ValueError(
                f"Expected aux_cond shape [B, 8, K], got {aux_cond.shape}"
            )

        if aux_cond.size(1) != self.aux_cond_dim:
            raise ValueError(
                f"Expected aux_cond channel dim = {self.aux_cond_dim}, "
                f"got {aux_cond.size(1)}"
            )

        context = self.aux_cond_encoder(aux_cond)

        return context

    def forward(self, x, time, x_self_cond=None, aux_cond=None):
        if self.share_encoder == 1:
            if self.self_condition:
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
                x = torch.cat((x_self_cond, x), dim=1)

            context = self.build_aux_context(aux_cond)

            x = self.init_conv(x)
            r = x.clone()

            t = self.time_mlp(time)

            h = []

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)

                x = block2(x, t)
                x = attn(x)
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)

            if self.use_aux_cond and context is not None:
                x = self.mid_cross_attn(x, context)

            x = self.mid_block2(x, t)

            out_res = x

            for block1, block2, attn, upsample in self.ups_no_skip:
                out_res = block1(out_res, t)
                out_res = block2(out_res, t)
                out_res = attn(out_res)
                out_res = upsample(out_res)

            out_res = self.final_res_block_1(out_res, t)
            out_res = self.final_conv_1(out_res)

            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)

                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)

                x = upsample(x)

            x = torch.cat((x, r), dim=1)

            x = self.final_res_block_2(x, t)
            out_res_add_noise = self.final_conv_2(x)

            return out_res, out_res_add_noise

        elif self.share_encoder == 0:
            return (
                self.unet0(
                    x,
                    time,
                    x_self_cond=x_self_cond,
                    aux_cond=aux_cond,
                ),
                self.unet1(
                    x,
                    time,
                    x_self_cond=x_self_cond,
                    aux_cond=aux_cond,
                ),
            )

        elif self.share_encoder == -1:
            return [
                self.unet0(
                    x,
                    time,
                    x_self_cond=x_self_cond,
                    aux_cond=aux_cond,
                )
            ]


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1):
    if schedule == "increased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale

    elif schedule == "decreased":
        x = torch.linspace(1, timesteps, timesteps, dtype=torch.float64)
        x = torch.flip(x, dims=[0])
        scale = 0.5 * timesteps * (timesteps + 1)
        alphas = x / scale

    elif schedule == "average":
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float64)

    else:
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float64)

    assert alphas.sum() - torch.tensor(1) < torch.tensor(1e-10)

    return alphas * sum_scale


class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type="l1",
        objective="pred_res_noise",
        ddim_sampling_eta=0.0,
        condition=True,
        sum_scale=None,
        input_condition=False,
        input_condition_mask=False,
        clip_denoised=True,
        sampling_type="use_pred_noise",
        sampling_init="input",
        sampling_init_noise_scale=1.0,
    ):
        super().__init__()

        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim
        )
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask
        self.clip_denoised = clip_denoised
        self.sampling_type = sampling_type
        self.sampling_init = sampling_init
        self.sampling_init_noise_scale = float(sampling_init_noise_scale)

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.0
        else:
            self.sum_scale = sum_scale if sum_scale else 1.0

        alphas = gen_coefficients(timesteps, schedule="decreased")
        alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.0)

        betas2 = gen_coefficients(
            timesteps,
            schedule="increased",
            sum_scale=self.sum_scale,
        )
        betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
        betas_cumsum = torch.sqrt(betas2_cumsum)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.0)

        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps

        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("alphas", alphas)
        register_buffer("alphas_cumsum", alphas_cumsum)
        register_buffer("one_minus_alphas_cumsum", 1 - alphas_cumsum)
        register_buffer("betas2", betas2)
        register_buffer("betas", torch.sqrt(betas2))
        register_buffer("betas2_cumsum", betas2_cumsum)
        register_buffer("betas_cumsum", betas_cumsum)

        register_buffer(
            "posterior_mean_coef1",
            betas2_cumsum_prev / betas2_cumsum,
        )
        register_buffer(
            "posterior_mean_coef2",
            (betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas)
            / betas2_cumsum,
        )
        register_buffer(
            "posterior_mean_coef3",
            betas2 / betas2_cumsum,
        )
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (
                x_t
                - x_input
                - (extract(self.alphas_cumsum, t, x_t.shape) - 1)
                * pred_res
            )
            / extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (
                x_t
                - extract(self.alphas_cumsum, t, x_t.shape) * x_input
                - extract(self.betas_cumsum, t, x_t.shape) * noise
            )
            / extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t
            - extract(self.alphas_cumsum, t, x_t.shape) * x_res
            - extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (
            x_t
            - extract(self.alphas, t, x_t.shape) * x_res
            - (
                extract(self.betas2, t, x_t.shape)
                / extract(self.betas_cumsum, t, x_t.shape)
            )
            * noise
        )

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res
            + extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped,
            t,
            x_t.shape,
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x_input,
        x,
        t,
        x_input_condition=0,
        x_self_cond=None,
        aux_cond=None,
        clip_denoised=None,
    ):
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)

        model_output = self.model(
            x_in,
            t,
            x_self_cond=x_self_cond,
            aux_cond=aux_cond,
        )

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0)
            if clip_denoised
            else identity
        )

        if self.objective == "pred_res_noise":
            pred_res = model_output[0]
            pred_noise = model_output[1]

            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x,
                t,
                pred_res,
                pred_noise,
            )
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_res_add_noise":
            pred_res = model_output[0]
            pred_noise = model_output[1] - model_output[0]

            pred_res = maybe_clip(pred_res)
            x_start = self.predict_start_from_res_noise(
                x,
                t,
                pred_res,
                pred_noise,
            )
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0_noise":
            pred_res = x_input - model_output[0]
            pred_noise = model_output[1]

            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])

        elif self.objective == "pred_x0_add_noise":
            x_start = model_output[0]
            pred_noise = model_output[1] - model_output[0]
            pred_res = x_input - x_start

            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])

        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x,
                t,
                x_input,
                pred_noise,
            )
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)

        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)

            pred_noise = self.predict_noise_from_res(
                x,
                t,
                x_input,
                pred_res,
            )
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        else:
            raise ValueError(f"unknown objective {self.objective}")

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(
        self,
        x_input,
        x,
        t,
        x_input_condition=0,
        x_self_cond=None,
        aux_cond=None,
    ):
        preds = self.model_predictions(
            x_input=x_input,
            x=x,
            t=t,
            x_input_condition=x_input_condition,
            x_self_cond=x_self_cond,
            aux_cond=aux_cond,
            clip_denoised=self.clip_denoised,
        )

        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res,
            x_start=x_start,
            x_t=x,
            t=t,
        )

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x_input,
        x,
        t: int,
        x_input_condition=0,
        x_self_cond=None,
        aux_cond=None,
    ):
        batched_times = torch.full(
            (x.shape[0],),
            t,
            device=x.device,
            dtype=torch.long,
        )

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input=x_input,
            x=x,
            t=batched_times,
            x_input_condition=x_input_condition,
            x_self_cond=x_self_cond,
            aux_cond=aux_cond,
        )

        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    @torch.no_grad()
    def init_sampling_state(self, x_input, shape, device):
        if self.condition:
            if self.sampling_init == "input":
                img = x_input.clone()
            elif self.sampling_init == "input_plus_noise":
                img = x_input + (
                    math.sqrt(self.sum_scale)
                    * self.sampling_init_noise_scale
                    * torch.randn(shape, device=device)
                )
            else:
                raise ValueError(f"unknown sampling_init {self.sampling_init}")
        else:
            img = torch.randn(shape, device=device)

        return img

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True, aux_cond=None):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0

        x_input = x_input[0]

        device = self.betas.device

        img = self.init_sampling_state(x_input, shape, device)
        input_add_noise = img

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None

            img, x_start = self.p_sample(
                x_input=x_input,
                x=img,
                t=t,
                x_input_condition=x_input_condition,
                x_self_cond=self_cond,
                aux_cond=aux_cond,
            )

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise] + img_list
            else:
                img_list = [input_add_noise, img]

            return img_list

        else:
            if not last:
                return img_list
            return [img]

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True, aux_cond=None):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0

        x_input = x_input[0]

        batch = shape[0]
        device = self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(
            -1,
            total_timesteps - 1,
            steps=sampling_timesteps + 1,
        )
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = self.init_sampling_state(x_input, shape, device)
        input_add_noise = img

        x_start = None
        sample_type = self.sampling_type

        if not last:
            img_list = []

        for time, time_next in tqdm(
            time_pairs,
            desc="sampling loop time step",
        ):
            time_cond = torch.full(
                (batch,),
                time,
                device=device,
                dtype=torch.long,
            )

            self_cond = x_start if self.self_condition else None

            preds = self.model_predictions(
                x_input=x_input,
                x=img,
                t=time_cond,
                x_input_condition=x_input_condition,
                x_self_cond=self_cond,
                aux_cond=aux_cond,
            )

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start

                if not last:
                    img_list.append(img)

                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas = betas2.sqrt()

            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]

            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)

            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                (betas2_cumsum_next - sigma2).sqrt() / betas_cumsum
            )

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if sample_type == "use_pred_noise":
                img = (
                    img
                    - alpha * pred_res
                    - (
                        betas_cumsum
                        - (betas2_cumsum_next - sigma2).sqrt()
                    )
                    * pred_noise
                    + sigma2.sqrt() * noise
                )

            elif sample_type == "use_x_start":
                img = (
                    sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum
                    * img
                    + (
                        1
                        - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum
                    )
                    * x_start
                    + (
                        alpha_cumsum_next
                        - alpha_cumsum
                        * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum
                    )
                    * pred_res
                    + sigma2.sqrt() * noise
                )

            elif sample_type == "special_eta_0":
                img = (
                    img
                    - alpha * pred_res
                    - (betas_cumsum - betas_cumsum_next) * pred_noise
                )

            elif sample_type == "special_eta_1":
                img = (
                    img
                    - alpha * pred_res
                    - betas2 / betas_cumsum * pred_noise
                    + betas
                    * betas2_cumsum_next.sqrt()
                    / betas_cumsum
                    * noise
                )

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise] + img_list
            else:
                img_list = [input_add_noise, img]

            return img_list

        else:
            if not last:
                return img_list
            return [img]

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True, aux_cond=None):
        image_size = self.image_size
        channels = self.channels

        sample_fn = (
            self.p_sample_loop
            if not self.is_ddim_sampling
            else self.ddim_sample
        )

        if self.condition:
            if torch.is_tensor(x_input):
                x_input = [x_input]

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)

        else:
            size = (batch_size, channels, image_size, image_size)

        return sample_fn(
            x_input,
            size,
            last=last,
            aux_cond=aux_cond,
        )

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start
            + extract(self.alphas_cumsum, t, x_start.shape) * x_res
            + extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss

        elif self.loss_type == "l2":
            return F.mse_loss

        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, imgs, t, noise=None, aux_cond=None):
        if isinstance(imgs, list):
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0

            x_input = imgs[1]
            x_start = imgs[0]

        else:
            x_input = 0
            x_start = imgs

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        x = self.q_sample(x_start, x_res, t, noise=noise)

        x_self_cond = None

        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input=x_input,
                    x=x,
                    t=t,
                    x_input_condition=x_input_condition
                    if self.input_condition
                    else 0,
                    aux_cond=aux_cond,
                ).pred_x_start

                x_self_cond.detach_()

        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(
            x_in,
            t,
            x_self_cond=x_self_cond,
            aux_cond=aux_cond,
        )

        target = []

        if self.objective == "pred_res_noise":
            target.append(x_res)
            target.append(noise)

        elif self.objective == "pred_res_add_noise":
            target.append(x_res)
            target.append(x_res + noise)

        elif self.objective == "pred_x0_noise":
            target.append(x_start)
            target.append(noise)

        elif self.objective == "pred_x0_add_noise":
            target.append(x_start)
            target.append(x_start + noise)

        elif self.objective == "pred_noise":
            target.append(noise)

        elif self.objective == "pred_res":
            target.append(x_res)

        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = 0

        for i in range(len(model_out)):
            loss = loss + self.loss_fn(
                model_out[i],
                target[i],
                reduction="none",
            )

        loss = reduce(loss, "b ... -> b (...)", "mean")

        return loss.mean()

    def forward(self, img, *args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size = (
                *img[0].shape,
                img[0].device,
                self.image_size,
            )
        else:
            b, c, h, w, device, img_size = (
                *img.shape,
                img.device,
                self.image_size,
            )

        t = torch.randint(
            0,
            self.num_timesteps,
            (b,),
            device=device,
        ).long()

        return self.p_losses(img, t, *args, **kwargs)
