"""Model attach helpers for adding spatial branch without editing phi4mm_clean."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .projector import DualBranchAudioEncoder, SpatialFusionProjector


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def get_audio_embed_module(model: nn.Module) -> nn.Module:
    """Return ``model.model.embed_tokens_extend.audio_embed`` safely."""
    core = _unwrap_model(model)
    try:
        return core.model.embed_tokens_extend.audio_embed
    except AttributeError as exc:
        raise RuntimeError(
            "Could not locate audio embedding module at model.model.embed_tokens_extend.audio_embed"
        ) from exc


def _count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _get_speech_projection(audio_embed: nn.Module) -> tuple[nn.Module, str]:
    proj = audio_embed.audio_projection
    if isinstance(proj, nn.ModuleDict):
        if "speech" not in proj:
            raise RuntimeError("audio_projection ModuleDict does not have 'speech' key.")
        return proj["speech"], "moduledict"
    return proj, "direct"


def _set_speech_projection(audio_embed: nn.Module, speech_proj: nn.Module, mode: str) -> None:
    if mode == "moduledict":
        audio_embed.audio_projection["speech"] = speech_proj
        return
    audio_embed.audio_projection = speech_proj


def setup_spatial_lora(
    model: nn.Module,
    rank: int = 320,
    lora_alpha_override: Optional[int] = None,
    adapter_name: str = "spatial",
    freeze_mono_audio_lora: bool = True,
    set_active_adapter: bool = True,
) -> Dict[str, int | str]:
    """Create spatial LoRA with rank=320 initialized from speech LoRA config."""
    try:
        from peft import LoraConfig
    except Exception as exc:
        raise RuntimeError("peft is required to set up spatial LoRA.") from exc

    core = _unwrap_model(model)
    active_before = []
    if hasattr(model, "active_adapters"):
        try:
            active_before = list(model.active_adapters())
        except Exception:
            active_before = []

    speech_template_cfg = None
    peft_cfg = getattr(model, "peft_config", None)
    if isinstance(peft_cfg, dict):
        speech_template_cfg = peft_cfg.get("speech", None)
    if speech_template_cfg is None:
        base_peft_cfg = getattr(core, "peft_config", None)
        if isinstance(base_peft_cfg, dict):
            speech_template_cfg = base_peft_cfg.get("speech", None)

    # Prefer exact target modules from loaded mono-audio LoRA config.
    target_modules = None
    lora_alpha = None
    lora_dropout = None
    bias = "none"
    if speech_template_cfg is not None:
        target_modules = getattr(speech_template_cfg, "target_modules", None)
        lora_alpha = getattr(speech_template_cfg, "lora_alpha", None)
        lora_dropout = getattr(speech_template_cfg, "lora_dropout", None)
        bias = getattr(speech_template_cfg, "bias", "none")

    # Fallback to config file fields if speech adapter is not loaded.
    speech_cfg = getattr(core.config, "speech_lora", None) or {}
    if target_modules is None:
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    if lora_alpha is None:
        lora_alpha = speech_cfg.get("lora_alpha", 640)
    if lora_dropout is None:
        lora_dropout = speech_cfg.get("dp", 0.01)
    if lora_alpha_override is not None:
        if int(lora_alpha_override) <= 0:
            raise ValueError(f"lora_alpha_override must be > 0, got {lora_alpha_override}.")
        lora_alpha = int(lora_alpha_override)

    if isinstance(target_modules, set):
        target_modules = list(target_modules)
    elif isinstance(target_modules, tuple):
        target_modules = list(target_modules)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    spatial_cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        bias=bias,
    )

    has_spatial = any(f".{adapter_name}." in n for n, _ in model.named_parameters())
    if not has_spatial:
        add_targets = [model]
        if core is not model:
            add_targets.append(core)
        added = False
        last_err: Optional[Exception] = None
        for target in add_targets:
            if not hasattr(target, "add_adapter"):
                continue
            try:
                # transformers PeftAdapterMixin: add_adapter(adapter_config, adapter_name=None)
                # peft PeftModel variants can differ by version/signature.
                target.add_adapter(spatial_cfg, adapter_name=adapter_name)
                added = True
                break
            except Exception:
                pass
            try:
                target.add_adapter(spatial_cfg, adapter_name)
                added = True
                break
            except Exception:
                pass
            try:
                # Legacy/alternate order
                target.add_adapter(adapter_name, spatial_cfg)
                added = True
                break
            except Exception as exc:  # noqa: PERF203
                last_err = exc
        if not added:
            raise RuntimeError(f"Failed to add spatial LoRA adapter '{adapter_name}'.") from last_err

    mono_frozen = 0
    spatial_trainable = 0
    if freeze_mono_audio_lora:
        # Freeze only mono-audio LoRA tensors, not unrelated module names containing ".speech.".
        for name, param in model.named_parameters():
            if f".speech." in name and ".lora_" in name:
                param.requires_grad = False
                mono_frozen += param.numel()
    for name, param in model.named_parameters():
        if f".{adapter_name}." in name and ".lora_" in name:
            param.requires_grad = True
            spatial_trainable += param.numel()

    if hasattr(model, "set_adapter"):
        if set_active_adapter:
            merged = []
            for name in active_before:
                if name != adapter_name:
                    merged.append(name)
            merged.append(adapter_name)
            if len(merged) == 1:
                model.set_adapter(merged[0])
            elif len(merged) > 1:
                model.set_adapter(merged)
        elif active_before:
            # add_adapter may switch active adapter to the newly added one; restore previous.
            if len(active_before) == 1:
                model.set_adapter(active_before[0])
            else:
                model.set_adapter(active_before)

    return {
        "adapter_name": adapter_name,
        "rank": rank,
        "lora_alpha": int(lora_alpha),
        "target_modules_count": len(target_modules),
        "frozen_mono_lora_params": mono_frozen,
        "trainable_spatial_lora_params": spatial_trainable,
    }


def attach_spatial_branch(
    model: nn.Module,
    spatial_encoder: nn.Module,
    freeze_phi: bool = True,
    freeze_conformer: bool = True,
    freeze_audio_projector: bool = True,
    train_spatial: bool = True,
) -> Dict[str, int]:
    """Attach paper-style spatial branch with separate spatial projector."""
    core = _unwrap_model(model)
    if freeze_phi:
        for param in core.parameters():
            param.requires_grad = False

    audio_embed = get_audio_embed_module(model)
    conformer_encoder = audio_embed.encoder
    speech_projector, proj_mode = _get_speech_projection(audio_embed)

    hidden_size = getattr(core.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(core.config, "n_embd", None)
    if hidden_size is None:
        raise RuntimeError("Could not infer LLM hidden size from config.")

    fusion_projector = SpatialFusionProjector(
        audio_projector=speech_projector,
        spatial_encoder=spatial_encoder,
        hidden_size=hidden_size,
        spatial_dim=getattr(spatial_encoder, "output_dim", None),
        freeze_audio_projector=freeze_audio_projector,
    )

    ref_param = next(speech_projector.parameters(), None)
    if ref_param is not None:
        fusion_projector = fusion_projector.to(device=ref_param.device, dtype=ref_param.dtype)

    _set_speech_projection(audio_embed, fusion_projector, proj_mode)
    audio_embed.freeze_audio_processor = freeze_conformer

    if freeze_conformer:
        for param in conformer_encoder.parameters():
            param.requires_grad = False

    if train_spatial:
        for param in fusion_projector.spatial_encoder.parameters():
            param.requires_grad = True
        for param in fusion_projector.spatial_projector.parameters():
            param.requires_grad = True
        if hasattr(fusion_projector, "fusion_gate"):
            for param in fusion_projector.fusion_gate.parameters():
                param.requires_grad = True

    return {"trainable_params": _count_trainable_params(core)}


def set_spatial_features(
    model: nn.Module,
    spatial_features: torch.Tensor,
    spatial_mask: torch.Tensor | None = None,
) -> None:
    audio_embed = get_audio_embed_module(model)
    proj = audio_embed.audio_projection
    speech_proj: Optional[nn.Module] = None
    if isinstance(proj, nn.ModuleDict):
        speech_proj = proj["speech"] if "speech" in proj else None
    else:
        speech_proj = proj

    if isinstance(speech_proj, SpatialFusionProjector):
        speech_proj.set_spatial_features(spatial_features, spatial_mask)
        return

    # Backward compatibility with old encoder-level wrapper.
    encoder = audio_embed.encoder
    if isinstance(encoder, DualBranchAudioEncoder):
        encoder.set_spatial_features(spatial_features, spatial_mask)
        return

    raise RuntimeError(
        "Spatial branch is not attached. Call attach_spatial_branch(...) first."
    )
