"""Dual-branch helpers for Conformer + SELD spatial integration."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchAudioEncoder(nn.Module):
    """Wrap an existing conformer encoder with a spatial branch and fusion."""

    def __init__(
        self,
        conformer_encoder: nn.Module,
        spatial_encoder: nn.Module,
        conformer_dim: int,
        spatial_dim: Optional[int] = None,
        freeze_conformer: bool = True,
    ) -> None:
        super().__init__()
        self.conformer_encoder = conformer_encoder
        self.spatial_encoder = spatial_encoder
        self.freeze_conformer = freeze_conformer

        if spatial_dim is None:
            spatial_dim = getattr(spatial_encoder, "output_dim", None)
        if spatial_dim is None:
            raise ValueError(
                "Could not infer spatial encoder output_dim. "
                "Pass spatial_dim explicitly or set spatial_encoder.output_dim."
            )

        self.spatial_projection = nn.Sequential(
            nn.Linear(spatial_dim, conformer_dim),
            nn.Linear(conformer_dim, conformer_dim),
        )
        self._spatial_cache: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None

    def set_spatial_features(
        self,
        spatial_features: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._spatial_cache = (spatial_features, spatial_mask)

    def clear_spatial_features(self) -> None:
        self._spatial_cache = None

    def _run_conformer(
        self,
        input_embeds: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.freeze_conformer:
            with torch.no_grad():
                return self.conformer_encoder(input_embeds, audio_attention_mask)
        return self.conformer_encoder(input_embeds, audio_attention_mask)

    def _run_spatial(self, spatial_features: torch.Tensor) -> torch.Tensor:
        out = self.spatial_encoder(spatial_features)
        if isinstance(out, tuple):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError("spatial_encoder must return Tensor or Tuple[Tensor, ...].")
        return out

    def forward(
        self,
        input_embeds: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        conf_out, conf_mask = self._run_conformer(input_embeds, audio_attention_mask)

        if self._spatial_cache is None:
            return conf_out, conf_mask

        spatial_features, _spatial_mask = self._spatial_cache
        self._spatial_cache = None

        spatial_features = spatial_features.to(conf_out.device)
        spatial_out = self._run_spatial(spatial_features)
        spatial_out = self.spatial_projection(spatial_out)

        # Match sequence length to conformer output.
        if spatial_out.size(1) != conf_out.size(1):
            spatial_out = F.interpolate(
                spatial_out.transpose(1, 2),
                size=conf_out.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        spatial_out = spatial_out.to(dtype=conf_out.dtype, device=conf_out.device)
        return conf_out + spatial_out, conf_mask


class SpatialFusionProjector(nn.Sequential):
    """Fuse frozen audio projector path with trainable spatial projector path.

    This class subclasses ``nn.Sequential`` intentionally so that the vendor
    ``phi4mm`` code can still index ``[0].bias`` to infer target device/dtype.
    """

    def __init__(
        self,
        audio_projector: nn.Module,
        spatial_encoder: nn.Module,
        hidden_size: int,
        spatial_dim: Optional[int] = None,
        freeze_audio_projector: bool = True,
    ) -> None:
        # Anchor layer for vendor compatibility: audio_projection["speech"][0].bias
        super().__init__(nn.Linear(1, 1))

        self.audio_projector = audio_projector
        self.spatial_encoder = spatial_encoder

        if spatial_dim is None:
            spatial_dim = getattr(spatial_encoder, "output_dim", None)
        if spatial_dim is None:
            raise ValueError(
                "Could not infer spatial encoder output_dim. "
                "Pass spatial_dim explicitly or set spatial_encoder.output_dim."
            )

        # Paper-style 2-layer linear spatial projector to LLM hidden size.
        self.spatial_projector = nn.Sequential(
            nn.Linear(spatial_dim, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        if freeze_audio_projector:
            for param in self.audio_projector.parameters():
                param.requires_grad = False

        self._spatial_cache: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None

    def set_spatial_features(
        self,
        spatial_features: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._spatial_cache = (spatial_features, spatial_mask)

    def clear_spatial_features(self) -> None:
        self._spatial_cache = None

    def _run_spatial_encoder(self, spatial_features: torch.Tensor) -> torch.Tensor:
        ref = next(self.spatial_encoder.parameters(), None)
        if ref is not None:
            spatial_features = spatial_features.to(device=ref.device, dtype=ref.dtype)
        out = self.spatial_encoder(spatial_features)
        if isinstance(out, tuple):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError("spatial_encoder must return Tensor or Tuple[Tensor, ...].")
        return out

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_tokens = self.audio_projector(audio_features)

        if self._spatial_cache is None:
            return audio_tokens

        spatial_features, _spatial_mask = self._spatial_cache
        self._spatial_cache = None

        spatial_out = self._run_spatial_encoder(spatial_features)
        spatial_tokens = self.spatial_projector(spatial_out)

        if spatial_tokens.size(0) != audio_tokens.size(0):
            if spatial_tokens.size(0) == 1:
                spatial_tokens = spatial_tokens.expand(audio_tokens.size(0), -1, -1)
            else:
                raise ValueError(
                    f"Batch mismatch: audio batch={audio_tokens.size(0)}, "
                    f"spatial batch={spatial_tokens.size(0)}"
                )

        if spatial_tokens.size(1) != audio_tokens.size(1):
            spatial_tokens = F.interpolate(
                spatial_tokens.transpose(1, 2),
                size=audio_tokens.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        spatial_tokens = spatial_tokens.to(dtype=audio_tokens.dtype, device=audio_tokens.device)
        return audio_tokens + spatial_tokens
