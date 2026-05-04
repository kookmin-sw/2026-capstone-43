"""
SGMSE+ style diffusion speech-enhancement model.

This file replaces the RDDM-style `UnetRes + ResidualDiffusion` wrapper with
an SGMSE+/score-based wrapper:

    x, y = clean_spec, noisy_spec
    t ~ Uniform(t_eps, T)
    mean, std = sde.marginal_prob(x, y, t)
    x_t = mean + std * z
    model predicts score / denoiser / data depending on loss_type

Optional aux_cond support is included for the user's foot_force condition.
It is encoded as context tokens and consumed by attention blocks in the
backbone (cross-attention style), instead of direct additive injection into `y`.
"""

import time
from math import ceil
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    # train_sgmse.py uses this model in a pure PyTorch loop.
    # Provide a minimal compatibility shim so pytorch_lightning is optional.
    class _LightningModuleCompat(nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    class _PLCompat:
        LightningModule = _LightningModuleCompat

    pl = _PLCompat()
from librosa import resample
from pesq import pesq
from pystoi import stoi
from torch_ema import ExponentialMovingAverage
from torch_pesq import PesqLoss
from torchaudio import load

from ..lrdse_sgmse import sampling
from ..lrdse_sgmse.backbones import BackboneRegistry
from ..lrdse_sgmse.sdes import SDERegistry
from ..lrdse_sgmse.util.other import pad_spec, si_sdr


class AuxConditionContextEncoder(nn.Module):
    """
    Encode auxiliary condition tokens for cross-attention.

    Expected input:
        aux_cond: [B, aux_cond_dim, K]

    Output:
        context: [B, hidden_dim, K]
    """

    def __init__(
        self,
        aux_cond_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        aux_scale_init: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_ch = aux_cond_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
            ]
            in_ch = hidden_dim

        layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim
        self.context_scale = nn.Parameter(torch.tensor(float(aux_scale_init)))

    def forward(self, aux_cond: torch.Tensor) -> torch.Tensor:
        if aux_cond.dim() != 3:
            raise ValueError(
                f"Expected aux_cond shape [B, C_aux, K], got {tuple(aux_cond.shape)}"
            )
        aux = aux_cond.to(dtype=torch.float32)
        return self.context_scale * self.net(aux)


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--t_eps", type=float, default=0.03)
        parser.add_argument("--num_eval_files", type=int, default=20)
        parser.add_argument("--loss_type", type=str, default="score_matching")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2")
        parser.add_argument("--network_scaling", type=str, default=None)
        parser.add_argument("--c_in", type=str, default="1")
        parser.add_argument("--c_out", type=str, default="1")
        parser.add_argument("--c_skip", type=str, default="0")
        parser.add_argument("--sigma_data", type=float, default=0.1)
        parser.add_argument("--l1_weight", type=float, default=0.001)
        parser.add_argument("--pesq_weight", type=float, default=0.0)
        parser.add_argument("--sr", type=int, default=16000)
        parser.add_argument("--use_aux_cond", action="store_true")
        parser.add_argument("--aux_cond_dim", type=int, default=8)
        parser.add_argument("--aux_hidden_dim", type=int, default=128)
        parser.add_argument("--aux_scale_init", type=float, default=0.1)
        return parser

    def __init__(
        self,
        backbone: str,
        sde: str,
        lr: float = 1e-4,
        ema_decay: float = 0.999,
        t_eps: float = 0.03,
        num_eval_files: int = 20,
        loss_type: str = "score_matching",
        loss_weighting: str = "sigma^2",
        network_scaling: Optional[str] = None,
        c_in: str = "1",
        c_out: str = "1",
        c_skip: str = "0",
        sigma_data: float = 0.1,
        l1_weight: float = 0.001,
        pesq_weight: float = 0.0,
        sr: int = 16000,
        data_module_cls=None,
        use_aux_cond: bool = False,
        aux_cond_dim: int = 8,
        aux_hidden_dim: int = 128,
        aux_scale_init: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        dnn_kwargs = dict(kwargs)
        if use_aux_cond and backbone == "ncsnpp_v2":
            dnn_kwargs["aux_context_dim"] = aux_hidden_dim
        self.dnn = dnn_cls(**dnn_kwargs)

        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)

        self.use_aux_cond = bool(use_aux_cond)
        self.aux_context_encoder = (
            AuxConditionContextEncoder(
                aux_cond_dim=aux_cond_dim,
                hidden_dim=aux_hidden_dim,
                aux_scale_init=aux_scale_init,
            )
            if self.use_aux_cond
            else None
        )

        self.lr = lr
        self.ema_decay = ema_decay
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr

        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False

        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.save_hyperparameters(ignore=["data_module_cls", "no_wandb"])
        self.data_module = (
            data_module_cls(**kwargs, gpu=kwargs.get("gpus", 0) > 0)
            if data_module_cls is not None
            else None
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(ema)
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def train(self, mode: bool = True, no_ema: bool = False):
        res = super().train(mode)
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                self.ema.store(self.dnn.parameters())
                self.ema.copy_to(self.dnn.parameters())
            else:
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())
                    # torch_ema keeps cloned params after restore(); free them to avoid
                    # persistent extra GPU memory after eval/sample.
                    self.ema.collected_params = None
        return res

    def eval(self, no_ema: bool = False):
        return self.train(False, no_ema=no_ema)

    def _split_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Supported batch formats:
            (clean_spec, noisy_spec)
            (clean_spec, noisy_spec, aux_cond)
            {"x"/"clean": clean_spec, "y"/"noisy": noisy_spec, "aux_cond": aux}
        """
        if isinstance(batch, dict):
            x = batch.get("x", batch.get("clean", batch.get("clean_spec")))
            y = batch.get("y", batch.get("noisy", batch.get("noisy_spec")))
            aux_cond = batch.get("aux_cond", batch.get("condition", None))
            if x is None or y is None:
                raise KeyError(
                    "Dict batch must contain x/clean/clean_spec and y/noisy/noisy_spec."
                )
            return x, y, aux_cond

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x, y = batch
                return x, y, None
            if len(batch) >= 3:
                x, y, aux_cond = batch[:3]
                return x, y, aux_cond

        raise TypeError(
            "Batch must be (x, y), (x, y, aux_cond), or a dict containing clean/noisy tensors."
        )

    def _augment_condition(
        self,
        y: torch.Tensor,
        aux_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Keep noisy condition unchanged. Aux information is injected via
        # cross-attention context inside the backbone attention blocks.
        return y

    def _encode_aux_context(
        self,
        aux_cond: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.use_aux_cond or aux_cond is None or self.aux_context_encoder is None:
            return None
        aux_cond = aux_cond.to(device=device, non_blocking=True)
        aux_ctx = self.aux_context_encoder(aux_cond)
        return aux_ctx.to(device=device, dtype=dtype)

    def _make_score_fn_with_aux(self, aux_cond: Optional[torch.Tensor]):
        if aux_cond is None:
            return self

        def score_fn(x_t, y, t, *args):
            return self(x_t, y, t, aux_cond=aux_cond)

        return score_fn

    def _loss(self, forward_out, x_t, z, t, mean, x):
        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting != "sigma^2":
                raise ValueError(
                    f"Invalid loss_weighting for score_matching: {self.loss_weighting}"
                )
            losses = torch.square(torch.abs(score * sigma + z))
            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "denoiser":
            score = forward_out
            denoised = score * sigma.pow(2) + x_t
            losses = torch.square(torch.abs(denoised - mean))

            if self.loss_weighting == "1":
                pass
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma.pow(2)
            elif self.loss_weighting == "edm":
                losses = (
                    (sigma.pow(2) + self.sigma_data**2)
                    / ((sigma * self.sigma_data).pow(2))
                ) * losses
            else:
                raise ValueError(
                    f"Invalid loss_weighting for denoiser: {self.loss_weighting}"
                )

            loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "data_prediction":
            if self.data_module is None:
                raise RuntimeError("data_prediction loss requires data_module_cls/data_module.")

            x_hat = forward_out
            _, _, freq_bins, time_frames = x.shape

            losses_tf = (1 / (freq_bins * time_frames)) * torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(
                0.5 * torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1)
            )

            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(
                0.5 * torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1)
            )

            if self.pesq_weight > 0.0:
                losses_pesq = torch.mean(self.pesq_loss(x_td, x_hat_td))
                loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq
            else:
                loss = losses_tf + self.l1_weight * losses_l1

        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

        return loss

    def _step(self, batch, batch_idx):
        x, y, aux_cond = self._split_batch(batch)
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps

        # Keep the diffusion bridge identical to the original SGMSE+ formulation:
        # p_t(x_t | x_0, y) uses the observed noisy spectrogram y.
        # Aux condition is injected only into the score network via forward(..., aux_cond).
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z

        forward_out = self(x_t, y, t, aux_cond=aux_cond)
        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.num_eval_files != 0 and self.data_module is not None:
            self._log_eval_metrics()

        loss = self._step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _log_eval_metrics(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        clean_files = self.data_module.valid_set.clean_files[: self.num_eval_files]
        noisy_files = self.data_module.valid_set.noisy_files[: self.num_eval_files]

        eval_files_per_gpu = max(1, self.num_eval_files // world_size)
        if world_size > 1:
            if rank == world_size - 1:
                clean_files = clean_files[rank * eval_files_per_gpu :]
                noisy_files = noisy_files[rank * eval_files_per_gpu :]
            else:
                clean_files = clean_files[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]
                noisy_files = noisy_files[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]

        if len(clean_files) == 0:
            return

        pesq_sum = 0.0
        si_sdr_sum = 0.0
        estoi_sum = 0.0

        for clean_file, noisy_file in zip(clean_files, noisy_files):
            x, sr_x = load(clean_file)
            y, sr_y = load(noisy_file)
            assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

            x_np = x.squeeze().numpy()
            if sr_x != 16000:
                x_16k = resample(x_np, orig_sr=sr_x, target_sr=16000).squeeze()
            else:
                x_16k = x_np

            x_hat = self.enhance(y, N=self.sde.N)
            if self.sr != 16000:
                x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
            else:
                x_hat_16k = x_hat

            pesq_sum += pesq(16000, x_16k, x_hat_16k, "wb")
            si_sdr_sum += si_sdr(x_np, x_hat)
            estoi_sum += stoi(x_np, x_hat, self.sr, extended=True)

        denom = len(clean_files)
        self.log("pesq", pesq_sum / denom, on_step=False, on_epoch=True, sync_dist=True)
        self.log("si_sdr", si_sdr_sum / denom, on_step=False, on_epoch=True, sync_dist=True)
        self.log("estoi", estoi_sum / denom, on_step=False, on_epoch=True, sync_dist=True)

    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        aux_cond: Optional[torch.Tensor] = None,
    ):
        y_cond = self._augment_condition(y, aux_cond)
        aux_context = self._encode_aux_context(
            aux_cond=aux_cond,
            device=y.device,
            dtype=y.real.dtype if torch.is_complex(y) else y.dtype,
        )

        if self.backbone == "ncsnpp_v2":
            dnn_in_x = self._c_in(t) * x_t
            dnn_in_y = self._c_in(t) * y_cond
            dnn_out = self.dnn(dnn_in_x, dnn_in_y, t, aux_context=aux_context)

            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                dnn_out = dnn_out / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                dnn_out = dnn_out / t[:, None, None, None]

            if self.loss_type == "score_matching":
                return self._c_skip(t) * x_t + self._c_out(t) * dnn_out
            if self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                return (dnn_out - x_t) / sigmas.pow(2)
            if self.loss_type == "data_prediction":
                return self._c_skip(t) * x_t + self._c_out(t) * dnn_out

            raise ValueError(f"Invalid loss_type for ncsnpp_v2: {self.loss_type}")

        dnn_input = torch.cat([x_t, y_cond], dim=1)
        return -self.dnn(dnn_input, t)

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        if self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        raise ValueError(f"Invalid c_in type: {self.c_in}")

    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        if self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        if self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None]
        if self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        raise ValueError(f"Invalid c_out type: {self.c_out}")

    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        if self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        raise ValueError(f"Invalid c_skip type: {self.c_skip}")

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(
        self,
        predictor_name,
        corrector_name,
        y,
        N=None,
        minibatch=None,
        aux_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        kwargs = {"eps": self.t_eps, **kwargs}
        score_fn = self._make_score_fn_with_aux(aux_cond)

        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name,
                corrector_name,
                sde=sde,
                score_fn=score_fn,
                y=y,
                **kwargs,
            )

        total = y.shape[0]

        def batched_sampling_fn():
            samples, ns = [], []
            for i in range(int(ceil(total / minibatch))):
                y_mini = y[i * minibatch : (i + 1) * minibatch]
                aux_mini = None if aux_cond is None else aux_cond[i * minibatch : (i + 1) * minibatch]
                score_fn_mini = self._make_score_fn_with_aux(aux_mini)
                sampler = sampling.get_pc_sampler(
                    predictor_name,
                    corrector_name,
                    sde=sde,
                    score_fn=score_fn_mini,
                    y=y_mini,
                    **kwargs,
                )
                sample, n = sampler()
                samples.append(sample)
                ns.append(n)
            return torch.cat(samples, dim=0), ns

        return batched_sampling_fn

    def get_ode_sampler(
        self,
        y,
        N=None,
        minibatch=None,
        aux_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        kwargs = {"eps": self.t_eps, **kwargs}
        score_fn = self._make_score_fn_with_aux(aux_cond)

        if minibatch is None:
            return sampling.get_ode_sampler(sde, score_fn, y=y, **kwargs)

        total = y.shape[0]

        def batched_sampling_fn():
            samples, ns = [], []
            for i in range(int(ceil(total / minibatch))):
                y_mini = y[i * minibatch : (i + 1) * minibatch]
                aux_mini = None if aux_cond is None else aux_cond[i * minibatch : (i + 1) * minibatch]
                score_fn_mini = self._make_score_fn_with_aux(aux_mini)
                sampler = sampling.get_ode_sampler(sde, score_fn_mini, y=y_mini, **kwargs)
                sample, n = sampler()
                samples.append(sample)
                ns.append(n)
            return torch.cat(samples, dim=0), ns

        return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, aux_cond=None, **kwargs):
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N
        score_fn = self._make_score_fn_with_aux(aux_cond)
        return sampling.get_sb_sampler(sde, score_fn, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        if self.data_module is None:
            raise RuntimeError("data_module is None.")
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        if self.data_module is None:
            raise RuntimeError("data_module is None.")
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        if self.data_module is None:
            raise RuntimeError("data_module is None.")
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        if self.data_module is None:
            return None
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        if self.data_module is None:
            raise RuntimeError("data_module is required for STFT transform.")
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        if self.data_module is None:
            raise RuntimeError("data_module is required for inverse STFT transform.")
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        if self.data_module is None:
            raise RuntimeError("data_module is required for STFT.")
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        if self.data_module is None:
            raise RuntimeError("data_module is required for ISTFT.")
        return self.data_module.istft(spec, length)

    @torch.no_grad()
    def enhance(
        self,
        y,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        timeit=False,
        aux_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """One-call speech enhancement of noisy waveform `y`."""
        start = time.time()
        device = next(self.parameters()).device

        y = y.to(device)
        T_orig = y.size(1)
        norm_factor = y.abs().max().clamp_min(1e-8)
        y = y / norm_factor

        Y = torch.unsqueeze(self._forward_transform(self._stft(y)), 0)
        Y = pad_spec(Y).to(device)

        if aux_cond is not None:
            aux_cond = aux_cond.to(device)

        if self.sde.__class__.__name__ == "OUVESDE":
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(
                    predictor,
                    corrector,
                    Y,
                    N=N,
                    corrector_steps=corrector_steps,
                    snr=snr,
                    intermediate=False,
                    aux_cond=aux_cond,
                    **kwargs,
                )
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y, N=N, aux_cond=aux_cond, **kwargs)
            else:
                raise ValueError(f"Invalid sampler type for OUVESDE: {sampler_type}")

        elif self.sde.__class__.__name__ == "SBVESDE":
            sampler = self.get_sb_sampler(
                sde=self.sde,
                y=Y,
                sampler_type=self.sde.sampler_type,
                aux_cond=aux_cond,
            )
        else:
            raise ValueError(
                f"Invalid SDE type for speech enhancement: {self.sde.__class__.__name__}"
            )

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().detach().cpu().numpy()

        if timeit:
            end = time.time()
            rtf = (end - start) / (len(x_hat) / self.sr)
            return x_hat, nfe, rtf
        return x_hat


# Backward-compatible name for code that imports `SGMSEPlusModel` explicitly.
SGMSEPlusModel = ScoreModel
