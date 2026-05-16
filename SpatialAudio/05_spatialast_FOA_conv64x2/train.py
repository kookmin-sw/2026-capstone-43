import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone import FOA_STEM_VARIANTS
from dataset import FOAWaveDataset, SyntheticFOADataset, foa_collate_fn
from losses import compute_losses, labels_to_unit_vectors
from model import build_model


ROOT = Path(__file__).resolve().parent
STEM_VARIANT_CHOICES = list(FOA_STEM_VARIANTS.keys())
TRANSFORMER_PATCH_KEYS = {
    "pos_embed",
    "cls_tokens",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
}


def is_transformer_patch_key(key):
    return key in TRANSFORMER_PATCH_KEYS or key.startswith("blocks.")


def extract_transformer_patch_checkpoint(source_path, output_path):
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    transformer_state = {
        "backbone." + key: value
        for key, value in source_state.items()
        if is_transformer_patch_key(key)
    }
    torch.save({"model": transformer_state}, output_path)
    print(f"saved transformer/patch checkpoint to {output_path}")
    return output_path


def adapt_patch_embed_proj_weight(target_value, source_value):
    if target_value.ndim != 4 or source_value.ndim != 4:
        return None, None
    if target_value.shape[0] != source_value.shape[0]:
        return None, None
    if target_value.shape[2:] != source_value.shape[2:]:
        return None, None

    src_in = source_value.shape[1]
    tgt_in = target_value.shape[1]
    if src_in == tgt_in:
        return source_value, "exact"

    if tgt_in > src_in:
        repeats = math.ceil(tgt_in / src_in)
        adapted = source_value.repeat(1, repeats, 1, 1)[:, :tgt_in, :, :]
        adapted = adapted * (src_in / tgt_in)
        strategy = f"repeat_scale_{src_in}_to_{tgt_in}"
        return adapted, strategy

    mean_weight = source_value.mean(dim=1, keepdim=True)
    adapted = mean_weight.repeat(1, tgt_in, 1, 1)
    adapted = adapted * (src_in / tgt_in)
    strategy = f"mean_repeat_scale_{src_in}_to_{tgt_in}"
    return adapted, strategy


def load_frozen_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state must be a dict, got {type(state)}")

    if all(name.startswith("backbone.") for name in state.keys()):
        target_module = model
        target_state = model.state_dict()
    else:
        target_module = model.backbone
        target_state = model.backbone.state_dict()

    compatible_state = {}
    dropped_keys = []
    adapted_keys = []
    for name, value in state.items():
        if name in target_state and target_state[name].shape == value.shape:
            compatible_state[name] = value
        elif name in target_state and name.endswith("patch_embed.proj.weight"):
            adapted_value, strategy = adapt_patch_embed_proj_weight(target_state[name], value)
            if adapted_value is not None:
                compatible_state[name] = adapted_value
                adapted_keys.append(
                    {
                        "name": name,
                        "source_shape": tuple(value.shape),
                        "target_shape": tuple(target_state[name].shape),
                        "strategy": strategy,
                    }
                )
            else:
                dropped_keys.append(name)
        else:
            dropped_keys.append(name)

    message = target_module.load_state_dict(compatible_state, strict=False)
    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"loaded keys: {len(compatible_state)}")
    print(f"dropped keys: {len(dropped_keys)}")
    print(f"missing keys: {len(message.missing_keys)}")
    print(f"unexpected keys: {len(message.unexpected_keys)}")
    if adapted_keys:
        print(f"shape-adapted keys: {len(adapted_keys)}")
        for item in adapted_keys[:10]:
            print(
                f"adapted {item['name']}: {item['source_shape']} -> {item['target_shape']} "
                f"via {item['strategy']}"
            )
    if dropped_keys:
        print(f"first dropped keys: {dropped_keys[:10]}")
    if message.missing_keys:
        print(f"first missing keys: {message.missing_keys[:10]}")
    if message.unexpected_keys:
        print(f"first unexpected keys: {message.unexpected_keys[:10]}")


def move_batch_to_device(batch, device):
    moved = dict(batch)
    moved["waveforms"] = batch["waveforms"].to(device, non_blocking=True)
    moved["class_target"] = batch["class_target"].to(device, non_blocking=True)
    moved["distance_target"] = batch["distance_target"].to(device, non_blocking=True)
    moved["azimuth_target"] = batch["azimuth_target"].to(device, non_blocking=True)
    moved["elevation_target"] = batch["elevation_target"].to(device, non_blocking=True)
    return moved


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def circular_azimuth_error_deg(pred, target):
    diff = (pred.float() - target.float() + 180.0) % 360.0 - 180.0
    return diff.abs()


def angular_error_deg(pred_azimuth, pred_elevation, target_azimuth, target_elevation):
    pred_vector = labels_to_unit_vectors(pred_azimuth, pred_elevation)
    target_vector = labels_to_unit_vectors(target_azimuth, target_elevation)
    cosine = (pred_vector * target_vector).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def get_unfrozen_blocks(model, unfreeze_last_n_blocks):
    if unfreeze_last_n_blocks <= 0:
        return []
    total_blocks = len(model.backbone.blocks)
    count = min(unfreeze_last_n_blocks, total_blocks)
    return list(model.backbone.blocks[total_blocks - count:])


def build_unfreeze_strategy_name(args):
    if getattr(args, "full_tuning", False):
        return "full_tuning"
    prefix = "patch_plus_" if args.unfreeze_patch_embed else ""
    if args.unfreeze_last_n_blocks == 2 and not args.unfreeze_patch_embed:
        return "stage3_last2"
    return f"{prefix}last{args.unfreeze_last_n_blocks}"


def save_model_checkpoint(model, checkpoint_path, epoch, summary, args, trainable_parameter_names):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": summary,
        "config": vars(args),
        "trainable_parameter_names": trainable_parameter_names,
    }
    torch.save(payload, checkpoint_path)


def is_better_metric(summary, best_summary, select_best_by):
    if best_summary is None:
        return True
    if select_best_by == "vector_cosine":
        return summary["val_vector_cosine"] > best_summary["val_vector_cosine"]
    return summary["val_angular_error"] < best_summary["val_angular_error"]


def freeze_for_stage4(model, args):
    if getattr(args, "full_tuning", False):
        for param in model.parameters():
            param.requires_grad = True
        return model

    for param in model.parameters():
        param.requires_grad = False

    for param in model.backbone.foa_native_stem.parameters():
        param.requires_grad = True

    if hasattr(model.backbone, "adapter"):
        for param in model.backbone.adapter.parameters():
            param.requires_grad = True

    for block in get_unfrozen_blocks(model, args.unfreeze_last_n_blocks):
        for param in block.parameters():
            param.requires_grad = True

    if args.unfreeze_patch_embed:
        for param in model.backbone.patch_embed.parameters():
            param.requires_grad = True

    if hasattr(model.heads, "azimuth_head") and model.heads.azimuth_head is not None:
        for param in model.heads.azimuth_head.parameters():
            param.requires_grad = True

    if hasattr(model.heads, "elevation_head") and model.heads.elevation_head is not None:
        for param in model.heads.elevation_head.parameters():
            param.requires_grad = True

    if hasattr(model.heads, "vector_head") and model.heads.vector_head is not None:
        for param in model.heads.vector_head.parameters():
            param.requires_grad = True

    return model


def print_trainable_parameters(model):
    total = 0
    trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            trainable_names.append(name)
            print(f"[TRAINABLE] {name} {tuple(param.shape)}")
    print(f"Trainable params: {trainable}/{total}")
    return trainable_names


def set_stage4_mode(model, train, args):
    if getattr(args, "full_tuning", False):
        if train:
            model.train()
        else:
            model.eval()
        return

    model.eval()
    if train:
        model.backbone.foa_native_stem.train()
        if hasattr(model.backbone, "adapter"):
            model.backbone.adapter.train()
        for block in get_unfrozen_blocks(model, args.unfreeze_last_n_blocks):
            block.train()
        if args.unfreeze_patch_embed:
            model.backbone.patch_embed.train()
        if hasattr(model.heads, "azimuth_head") and model.heads.azimuth_head is not None:
            model.heads.azimuth_head.train()
        if hasattr(model.heads, "elevation_head") and model.heads.elevation_head is not None:
            model.heads.elevation_head.train()
        if hasattr(model.heads, "vector_head") and model.heads.vector_head is not None:
            model.heads.vector_head.train()


def build_epoch_lr_scale(epoch, args):
    if args.scheduler == "none":
        return 1.0

    warmup_epochs = max(args.warmup_epochs, 0)
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)

    cosine_epochs = max(args.epochs - warmup_epochs, 1)
    cosine_index = max(epoch - warmup_epochs, 0)
    cosine_denominator = max(cosine_epochs - 1, 1)
    progress = min(float(cosine_index) / float(cosine_denominator), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_epoch_learning_rates(optimizer, args, epoch):
    lr_scale = build_epoch_lr_scale(epoch, args)
    current_lrs = {}
    for idx, group in enumerate(optimizer.param_groups):
        base_lr = group.get("base_lr", group["lr"])
        group["base_lr"] = base_lr
        group["lr"] = base_lr * lr_scale
        group_name = group.get("group_name", f"group_{idx}")
        current_lrs[group_name] = group["lr"]
    return lr_scale, current_lrs


def collect_optimizer_param_groups(model, args):
    named_params = list(model.named_parameters())
    used = set()

    def take(group_name, prefix, lr):
        params = []
        for name, param in named_params:
            if not param.requires_grad or id(param) in used:
                continue
            if name.startswith(prefix):
                params.append(param)
                used.add(id(param))
        if not params:
            return None
        return {
            "params": params,
            "lr": lr,
            "base_lr": lr,
            "group_name": group_name,
        }

    groups = []
    for maybe_group in [
        take("stem", "backbone.foa_native_stem.", args.lr_stem),
        take("adapter", "backbone.adapter.", args.lr_adapter),
        take("heads", "heads.", args.lr_heads),
        take("transformer", "backbone.blocks.", args.lr_transformer),
        take("patch_embed", "backbone.patch_embed.", args.lr_transformer),
    ]:
        if maybe_group is not None:
            groups.append(maybe_group)

    misc_params = []
    for _, param in named_params:
        if not param.requires_grad or id(param) in used:
            continue
        misc_params.append(param)
        used.add(id(param))
    if misc_params:
        groups.append(
            {
                "params": misc_params,
                "lr": args.lr_transformer,
                "base_lr": args.lr_transformer,
                "group_name": "backbone_misc",
            }
        )

    return groups


def build_datasets(args):
    if args.debug_overfit_subset_size > 0:
        subset_size = args.debug_overfit_subset_size
        print(f"[DEBUG] overfit-subset mode enabled with subset_size={subset_size}")
        if args.train_json:
            train_dataset = FOAWaveDataset(
                json_path=args.train_json,
                audio_path_root=args.audio_path_root,
                num_classes=args.num_classes,
                label_csv=args.label_csv,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                normalize=args.audio_normalize,
                limit_samples=subset_size,
            )
            # Use the same subset for train/val to measure memorization, not generalization.
            val_dataset = FOAWaveDataset(
                json_path=args.train_json,
                audio_path_root=args.audio_path_root,
                num_classes=args.num_classes,
                label_csv=args.label_csv,
                sample_rate=args.sample_rate,
                clip_seconds=args.clip_seconds,
                normalize=args.audio_normalize,
                limit_samples=subset_size,
            )
        else:
            train_dataset = SyntheticFOADataset(
                num_samples=subset_size,
                num_classes=args.num_classes,
                sample_rate=args.sample_rate,
                clip_seconds=args.synthetic_clip_seconds,
            )
            val_dataset = SyntheticFOADataset(
                num_samples=subset_size,
                num_classes=args.num_classes,
                sample_rate=args.sample_rate,
                clip_seconds=args.synthetic_clip_seconds,
            )
        return train_dataset, val_dataset

    if args.train_json:
        train_dataset = FOAWaveDataset(
            json_path=args.train_json,
            audio_path_root=args.audio_path_root,
            num_classes=args.num_classes,
            label_csv=args.label_csv,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            normalize=args.audio_normalize,
            limit_samples=args.limit_train_samples,
        )
    else:
        train_dataset = SyntheticFOADataset(
            num_samples=args.synthetic_train_samples,
            num_classes=args.num_classes,
            sample_rate=args.sample_rate,
            clip_seconds=args.synthetic_clip_seconds,
        )

    if args.val_json:
        val_dataset = FOAWaveDataset(
            json_path=args.val_json,
            audio_path_root=args.audio_path_root,
            num_classes=args.num_classes,
            label_csv=args.label_csv,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            normalize=args.audio_normalize,
            limit_samples=args.limit_val_samples,
        )
    else:
        val_dataset = SyntheticFOADataset(
            num_samples=args.synthetic_val_samples,
            num_classes=args.num_classes,
            sample_rate=args.sample_rate,
            clip_seconds=args.synthetic_clip_seconds,
        )

    return train_dataset, val_dataset


def prepare_debug_targets(batch, train, step):
    azimuth_target = batch["azimuth_target"]
    elevation_target = batch["elevation_target"]

    print_looks_like_degrees = False
    if azimuth_target.min().item() < 0 or azimuth_target.max().item() >= 360:
        print("Looks like degrees, converting to class index")
        azimuth_target = (azimuth_target % 360).long()
        print_looks_like_degrees = True
    elif azimuth_target.max().item() > 180 and step == 0 and train:
        print("azimuth_target has values above 180; treating them as valid class indices in [0, 359].")

    if elevation_target.min().item() < 0 or elevation_target.max().item() >= 180:
        if not print_looks_like_degrees:
            print("Looks like degrees, converting to class index")
        elevation_target = (elevation_target + 90).clamp(0, 179).long()

    if azimuth_target.dtype != torch.long:
        azimuth_target = azimuth_target.long()
    if elevation_target.dtype != torch.long:
        elevation_target = elevation_target.long()

    batch["azimuth_target"] = azimuth_target
    batch["elevation_target"] = elevation_target

    assert azimuth_target.min() >= 0
    assert azimuth_target.max() < 360
    assert elevation_target.min() >= 0
    assert elevation_target.max() < 180

    if step == 0 and train:
        unique_az = torch.unique(azimuth_target)
        unique_el = torch.unique(elevation_target)
        print("\n[TRAIN DEBUG - TARGET]")
        print("azimuth_target shape:", azimuth_target.shape)
        print("elevation_target shape:", elevation_target.shape)
        print("azimuth_target sample:", azimuth_target[:20])
        print("elevation_target sample:", elevation_target[:20])
        print("azimuth min/max:", azimuth_target.min().item(), azimuth_target.max().item())
        print("elevation min/max:", elevation_target.min().item(), elevation_target.max().item())
        print("azimuth dtype:", azimuth_target.dtype)
        print("elevation dtype:", elevation_target.dtype)
        print("unique azimuth count:", len(unique_az))
        print("unique elevation count:", len(unique_el))


def run_epoch(model, loader, optimizer, device, loss_weights, args, train=True, max_steps=0):
    set_stage4_mode(model, train=train, args=args)
    stats = {
        "loss_sum": 0.0,
        "azimuth_loss_sum": 0.0,
        "elevation_loss_sum": 0.0,
        "vector_loss_sum": 0.0,
        "azimuth_correct": 0,
        "elevation_correct": 0,
        "vector_cosine_sum": 0.0,
        "sample_count": 0,
        "azimuth_count": 0,
        "elevation_count": 0,
        "vector_count": 0,
        "azimuth_mae_sum": 0.0,
        "elevation_mae_sum": 0.0,
        "angular_error_sum": 0.0,
        "error_count": 0,
        "steps": 0,
    }
    first_debug_shapes = None
    last_debug = {}

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        batch = move_batch_to_device(batch, device)
        prepare_debug_targets(batch, train=train, step=step)
        reverbs = torch.zeros(batch["waveforms"].shape[0], 1, 1, device=device, dtype=batch["waveforms"].dtype)

        with torch.set_grad_enabled(train):
            return_backbone_outputs = step == 0 and train
            model_outputs = model(
                batch["waveforms"],
                reverbs=reverbs,
                mask_t_prob=0.0,
                mask_f_prob=0.0,
                return_backbone_outputs=return_backbone_outputs,
            )
            if return_backbone_outputs:
                outputs = model_outputs["outputs"]
                backbone_outputs = model_outputs["backbone"]
            else:
                outputs = model_outputs
                backbone_outputs = None
            loss, loss_dict = compute_losses(outputs, batch, loss_weights)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if step == 0 and train and "azimuth_logits" in outputs and "elevation_logits" in outputs:
            pred_az = outputs["azimuth_logits"].argmax(-1)
            pred_el = outputs["elevation_logits"].argmax(-1)
            print("\n[TRAIN DEBUG - PRED]")
            print("pred azimuth:", pred_az[:20])
            print("pred elevation:", pred_el[:20])
            print(
                "[DEBUG] azimuth logits mean/std:",
                outputs["azimuth_logits"].mean().item(),
                outputs["azimuth_logits"].std().item(),
            )
            print(
                "[DEBUG] elevation logits mean/std:",
                outputs["elevation_logits"].mean().item(),
                outputs["elevation_logits"].std().item(),
            )
            if backbone_outputs is not None and "doa_token" in backbone_outputs:
                print(
                    "[DEBUG] doa_token mean/std:",
                    backbone_outputs["doa_token"].mean().item(),
                    backbone_outputs["doa_token"].std().item(),
                )
            if hasattr(model.heads, "azimuth_head") and model.heads.azimuth_head is not None:
                az_weight = model.heads.azimuth_head.proj.weight
                print("[DEBUG] azimuth_head weight mean/std:", az_weight.mean().item(), az_weight.std().item())
            if hasattr(model.heads, "elevation_head") and model.heads.elevation_head is not None:
                el_weight = model.heads.elevation_head.proj.weight
                print("[DEBUG] elevation_head weight mean/std:", el_weight.mean().item(), el_weight.std().item())

        if first_debug_shapes is None:
            first_debug_shapes = model.get_debug_shapes()
            print(f"first batch waveforms: {tuple(batch['waveforms'].shape)}")
            print(f"first batch outputs: { {key: tuple(value.shape) for key, value in outputs.items()} }")
            print(f"first batch debug shapes: {first_debug_shapes}")

        batch_size = batch["waveforms"].shape[0]
        stats["loss_sum"] += loss.item() * batch_size
        stats["azimuth_loss_sum"] += loss_dict["azimuth"].item() * batch_size
        stats["elevation_loss_sum"] += loss_dict["elevation"].item() * batch_size
        stats["vector_loss_sum"] += loss_dict["vector"].item() * batch_size
        stats["sample_count"] += batch_size
        stats["steps"] += 1

        if "azimuth_logits" in outputs:
            azimuth_pred = outputs["azimuth_logits"].argmax(dim=-1)
            stats["azimuth_correct"] += (azimuth_pred == batch["azimuth_target"]).sum().item()
            stats["azimuth_count"] += batch_size
            stats["azimuth_mae_sum"] += circular_azimuth_error_deg(azimuth_pred, batch["azimuth_target"]).sum().item()

        if "elevation_logits" in outputs:
            elevation_pred = outputs["elevation_logits"].argmax(dim=-1)
            stats["elevation_correct"] += (elevation_pred == batch["elevation_target"]).sum().item()
            stats["elevation_count"] += batch_size
            stats["elevation_mae_sum"] += (elevation_pred.float() - batch["elevation_target"].float()).abs().sum().item()

        if "azimuth_logits" in outputs and "elevation_logits" in outputs:
            angular_error = angular_error_deg(
                outputs["azimuth_logits"].argmax(dim=-1),
                outputs["elevation_logits"].argmax(dim=-1),
                batch["azimuth_target"],
                batch["elevation_target"],
            )
            stats["angular_error_sum"] += angular_error.sum().item()
            stats["error_count"] += batch_size

        if "vector" in outputs:
            target_vector = labels_to_unit_vectors(
                batch["azimuth_target"],
                batch["elevation_target"],
            ).to(outputs["vector"].device)
            cosine = F.cosine_similarity(outputs["vector"], target_vector, dim=-1)
            stats["vector_cosine_sum"] += cosine.sum().item()
            stats["vector_count"] += batch_size
            last_debug["vector_cosine"] = cosine.mean().item()

        last_debug["azimuth_target"] = batch["azimuth_target"][:10].detach().cpu().tolist()
        last_debug["elevation_target"] = batch["elevation_target"][:10].detach().cpu().tolist()
        if "azimuth_logits" in outputs:
            last_debug["pred_azimuth"] = outputs["azimuth_logits"].argmax(dim=-1)[:10].detach().cpu().tolist()
        if "elevation_logits" in outputs:
            last_debug["pred_elevation"] = outputs["elevation_logits"].argmax(dim=-1)[:10].detach().cpu().tolist()
        last_debug["azimuth_loss"] = loss_dict["azimuth"].item()
        last_debug["elevation_loss"] = loss_dict["elevation"].item()

        print(
            f"{'train' if train else 'val'} step={step} "
            f"loss={loss.item():.6f} "
            f"azimuth={loss_dict['azimuth'].item():.6f} "
            f"elevation={loss_dict['elevation'].item():.6f} "
            f"vector={loss_dict['vector'].item():.6f}"
        )

    return {
        "loss": stats["loss_sum"] / max(stats["sample_count"], 1),
        "azimuth_loss": stats["azimuth_loss_sum"] / max(stats["sample_count"], 1),
        "elevation_loss": stats["elevation_loss_sum"] / max(stats["sample_count"], 1),
        "vector_loss": stats["vector_loss_sum"] / max(stats["sample_count"], 1),
        "azimuth_acc": stats["azimuth_correct"] / max(stats["azimuth_count"], 1),
        "elevation_acc": stats["elevation_correct"] / max(stats["elevation_count"], 1),
        "vector_cosine": stats["vector_cosine_sum"] / max(stats["vector_count"], 1),
        "azimuth_mae": stats["azimuth_mae_sum"] / max(stats["azimuth_count"], 1),
        "elevation_mae": stats["elevation_mae_sum"] / max(stats["elevation_count"], 1),
        "angular_error": stats["angular_error_sum"] / max(stats["error_count"], 1),
        "steps": stats["steps"],
        "debug_shapes": first_debug_shapes or {},
        "last_debug": last_debug,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="")
    parser.add_argument("--val_json", default="")
    parser.add_argument("--audio_path_root", default="")
    parser.add_argument("--label_csv", default="")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_seconds", type=int, default=10)
    parser.add_argument("--audio_normalize", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_class_head", action="store_true", default=True)
    parser.add_argument("--no_use_class_head", action="store_false", dest="use_class_head")
    parser.add_argument("--use_distance_head", action="store_true", default=True)
    parser.add_argument("--no_use_distance_head", action="store_false", dest="use_distance_head")
    parser.add_argument("--use_azimuth_head", action="store_true", default=True)
    parser.add_argument("--no_use_azimuth_head", action="store_false", dest="use_azimuth_head")
    parser.add_argument("--use_elevation_head", action="store_true", default=True)
    parser.add_argument("--no_use_elevation_head", action="store_false", dest="use_elevation_head")
    parser.add_argument("--use_vector_head", action="store_true", default=True)
    parser.add_argument("--no_use_vector_head", action="store_false", dest="use_vector_head")
    parser.add_argument("--limit_train_samples", type=int, default=0)
    parser.add_argument("--limit_val_samples", type=int, default=0)
    parser.add_argument("--synthetic_train_samples", type=int, default=8)
    parser.add_argument("--synthetic_val_samples", type=int, default=4)
    parser.add_argument("--synthetic_clip_seconds", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_val_steps", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--class_loss_weight", type=float, default=0.0)
    parser.add_argument("--distance_loss_weight", type=float, default=0.0)
    parser.add_argument("--azimuth_loss_weight", type=float, default=2.0)
    parser.add_argument("--elevation_loss_weight", type=float, default=2.0)
    parser.add_argument("--vector_loss_weight", type=float, default=0.5)
    parser.add_argument("--foa_stem_type", default="foa_native", choices=["foa_native", "logmel_only"])
    parser.add_argument("--foa_use_diffuseness", action="store_true", default=False)
    parser.add_argument("--foa_use_beam_proxy", action="store_true", default=False)
    parser.add_argument("--foa_stem_channels", type=int, default=16)
    parser.add_argument(
        "--foa_stem_variant",
        default="baseline",
        choices=STEM_VARIANT_CHOICES,
    )
    parser.add_argument("--foa_stem_hidden_channels", type=int, default=0)
    parser.add_argument("--foa_stem_out_channels", type=int, default=0)
    parser.add_argument("--patch_in_from_stem", action="store_true", dest="patch_in_from_stem")
    parser.add_argument("--no_patch_in_from_stem", action="store_false", dest="patch_in_from_stem")
    parser.set_defaults(patch_in_from_stem=True)
    parser.add_argument("--pretrained_backbone_ckpt", default="")
    parser.add_argument("--default_full_ckpt", default=str(ROOT / "finetuned.pth"))
    parser.add_argument("--default_transformer_ckpt", default=str(ROOT / "finetuned_transformer_only.pth"))
    parser.add_argument("--full_tuning", action="store_true", default=False)
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=2)
    parser.add_argument("--unfreeze_patch_embed", action="store_true", default=False)
    parser.add_argument("--freeze_patch_embed", action="store_false", dest="unfreeze_patch_embed")
    parser.add_argument("--lr_stem", type=float, default=1e-3)
    parser.add_argument("--lr_adapter", type=float, default=1e-3)
    parser.add_argument("--lr_heads", type=float, default=1e-3)
    parser.add_argument("--lr_transformer", type=float, default=1e-5)
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--select_best_by", default="angular_error", choices=["angular_error", "vector_cosine"])
    parser.add_argument("--recipe_name", default="default_recipe")
    parser.add_argument("--debug_overfit_one_sample", action="store_true", default=False)
    parser.add_argument("--debug_overfit_subset_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.debug_overfit_one_sample:
        print("[DEBUG] overfit-one-sample mode enabled")
        args.limit_train_samples = 1
        args.limit_val_samples = 1
        args.synthetic_train_samples = 1
        args.synthetic_val_samples = 1
        args.batch_size = 1
        args.epochs = max(args.epochs, 200)

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=foa_collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=foa_collate_fn,
    )

    model = build_model(
        num_classes=args.num_classes,
        reverb_type="foa",
        foa_stem_type=args.foa_stem_type,
        foa_use_diffuseness=args.foa_use_diffuseness,
        foa_use_beam_proxy=args.foa_use_beam_proxy,
        foa_stem_channels=args.foa_stem_channels,
        foa_stem_variant=args.foa_stem_variant,
        foa_stem_hidden_channels=args.foa_stem_hidden_channels,
        foa_stem_out_channels=args.foa_stem_out_channels,
        patch_in_from_stem=args.patch_in_from_stem,
        use_class_head=args.use_class_head,
        use_distance_head=args.use_distance_head,
        use_azimuth_head=args.use_azimuth_head,
        use_elevation_head=args.use_elevation_head,
        use_vector_head=args.use_vector_head,
    ).to(device)

    checkpoint_path = args.pretrained_backbone_ckpt
    if checkpoint_path:
        load_frozen_checkpoint(model, checkpoint_path)
    else:
        default_transformer_ckpt = Path(args.default_transformer_ckpt)
        if not default_transformer_ckpt.exists():
            extract_transformer_patch_checkpoint(args.default_full_ckpt, default_transformer_ckpt)
        load_frozen_checkpoint(model, default_transformer_ckpt)

    freeze_for_stage4(model, args)
    trainable_parameter_names = print_trainable_parameters(model)
    optimizer_param_groups = collect_optimizer_param_groups(model, args)
    optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=args.weight_decay)

    loss_weights = {
        "class": args.class_loss_weight,
        "distance": args.distance_loss_weight,
        "azimuth": args.azimuth_loss_weight,
        "elevation": args.elevation_loss_weight,
        "vector": args.vector_loss_weight,
    }
    print(f"loss_weights: {loss_weights}")
    print(f"unfreeze_strategy: {build_unfreeze_strategy_name(args)}")

    history = []
    best_summary = None
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    for epoch in range(args.epochs):
        print(f"epoch {epoch}")
        lr_scale, current_lrs = apply_epoch_learning_rates(optimizer, args, epoch)
        print(f"lr_scale: {lr_scale:.6f}")
        print(f"current_lrs: {current_lrs}")
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            args,
            train=True,
            max_steps=args.max_train_steps,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            loss_weights,
            args,
            train=False,
            max_steps=args.max_val_steps,
        )
        summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_azimuth_loss": train_stats["azimuth_loss"],
            "train_elevation_loss": train_stats["elevation_loss"],
            "train_vector_loss": train_stats["vector_loss"],
            "train_azimuth_acc": train_stats["azimuth_acc"],
            "train_elevation_acc": train_stats["elevation_acc"],
            "train_vector_cosine": train_stats["vector_cosine"],
            "train_azimuth_mae": train_stats["azimuth_mae"],
            "train_elevation_mae": train_stats["elevation_mae"],
            "train_angular_error": train_stats["angular_error"],
            "val_azimuth_loss": val_stats["azimuth_loss"],
            "val_elevation_loss": val_stats["elevation_loss"],
            "val_vector_loss": val_stats["vector_loss"],
            "val_azimuth_acc": val_stats["azimuth_acc"],
            "val_elevation_acc": val_stats["elevation_acc"],
            "val_vector_cosine": val_stats["vector_cosine"],
            "val_azimuth_mae": val_stats["azimuth_mae"],
            "val_elevation_mae": val_stats["elevation_mae"],
            "val_angular_error": val_stats["angular_error"],
            "train_steps": train_stats["steps"],
            "val_steps": val_stats["steps"],
            "lr_scale": lr_scale,
            "lr_stem": current_lrs.get("stem", 0.0),
            "lr_adapter": current_lrs.get("adapter", 0.0),
            "lr_heads": current_lrs.get("heads", 0.0),
            "lr_transformer": current_lrs.get("transformer", 0.0),
            "lr_patch_embed": current_lrs.get("patch_embed", 0.0),
        }
        history.append(summary)
        print(summary)
        if is_better_metric(summary, best_summary, args.select_best_by):
            best_summary = dict(summary)
            save_model_checkpoint(
                model,
                best_checkpoint_path,
                epoch,
                best_summary,
                args,
                trainable_parameter_names,
            )
            print(f"saved best checkpoint to {best_checkpoint_path}")

    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_summary = history[-1] if history else {}
    metrics_summary = {
        "config": {
            "foa_stem_type": args.foa_stem_type,
            "foa_stem_variant": args.foa_stem_variant,
            "foa_stem_hidden_channels": args.foa_stem_hidden_channels,
            "foa_stem_out_channels": args.foa_stem_out_channels,
            "patch_in_from_stem": args.patch_in_from_stem,
            "debug_overfit_one_sample": args.debug_overfit_one_sample,
            "debug_overfit_subset_size": args.debug_overfit_subset_size,
            "limit_train_samples": args.limit_train_samples,
            "limit_val_samples": args.limit_val_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
            "unfreeze_patch_embed": args.unfreeze_patch_embed,
            "full_tuning": args.full_tuning,
            "unfreeze_strategy": build_unfreeze_strategy_name(args),
            "recipe_name": args.recipe_name,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "select_best_by": args.select_best_by,
            "lr_stem": args.lr_stem,
            "lr_adapter": args.lr_adapter,
            "lr_heads": args.lr_heads,
            "lr_transformer": args.lr_transformer,
            "resolved_stem_hidden_channels": list(getattr(model.backbone, "foa_stem_hidden_channels", [])),
            "resolved_stem_out_channels": getattr(model.backbone, "foa_stem_out_channels", 0),
            "loss_weights": loss_weights,
        },
        "trainable_parameter_names": trainable_parameter_names,
        "final": final_summary,
        "best": best_summary or {},
    }
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    if final_summary:
        print("\n[RUN SUMMARY]")
        print("final train azimuth acc:", final_summary["train_azimuth_acc"])
        print("final train elevation acc:", final_summary["train_elevation_acc"])
        print("final val azimuth acc:", final_summary["val_azimuth_acc"])
        print("final val elevation acc:", final_summary["val_elevation_acc"])
        print("final train vector cosine:", final_summary["train_vector_cosine"])
        print("final val vector cosine:", final_summary["val_vector_cosine"])
        if args.debug_overfit_subset_size == 16 and (
            final_summary["train_azimuth_acc"] < 0.25 or final_summary["train_elevation_acc"] < 0.25
        ):
            print("[WARNING] 16-sample overfit failed badly; inspect training/pipeline before larger runs.")
        if best_summary:
            print("best val azimuth acc:", best_summary["val_azimuth_acc"])
            print("best val elevation acc:", best_summary["val_elevation_acc"])
            print("best val vector cosine:", best_summary["val_vector_cosine"])
            print("best val angular error:", best_summary["val_angular_error"])

    if args.debug_overfit_one_sample and history:
        debug_dir = ROOT / "outputs_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_payload = {
            "final_target_azimuth": train_stats["last_debug"].get("azimuth_target", []),
            "final_target_elevation": train_stats["last_debug"].get("elevation_target", []),
            "final_predicted_azimuth": train_stats["last_debug"].get("pred_azimuth", []),
            "final_predicted_elevation": train_stats["last_debug"].get("pred_elevation", []),
            "final_azimuth_ce_loss": train_stats["last_debug"].get("azimuth_loss"),
            "final_elevation_ce_loss": train_stats["last_debug"].get("elevation_loss"),
            "final_vector_cosine": final_summary["train_vector_cosine"],
            "trainable_parameter_names": trainable_parameter_names,
        }
        debug_path = debug_dir / "one_sample_overfit_debug.json"
        with open(debug_path, "w") as f:
            json.dump(debug_payload, f, indent=2)
        print(f"saved one-sample debug artifact to {debug_path}")
        if (
            final_summary["train_vector_cosine"] > 0.95 and
            (final_summary["train_azimuth_acc"] < 0.99 or final_summary["train_elevation_acc"] < 0.99)
        ):
            print("[WARNING] CE heads still not adapting; inspect logits/token statistics.")

    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    save_model_checkpoint(
        model,
        last_checkpoint_path,
        final_summary.get("epoch", -1),
        final_summary,
        args,
        trainable_parameter_names,
    )
    print(f"saved last checkpoint to {last_checkpoint_path}")

    encoder_weights_path = output_dir / "encoder_weights.pt"
    torch.save(model.backbone.state_dict(), encoder_weights_path)
    print(f"saved encoder weights to {encoder_weights_path}")


if __name__ == "__main__":
    main()
