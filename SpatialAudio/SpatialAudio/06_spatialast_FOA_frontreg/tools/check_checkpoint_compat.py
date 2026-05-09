import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import build_model


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    return checkpoint, state_dict, args


def remap_old_spatialast_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("distance_head."):
            remapped["heads.distance_head.proj." + key.split(".", 1)[1]] = value
        elif key.startswith("azimuth_head."):
            remapped["heads.azimuth_head.proj." + key.split(".", 1)[1]] = value
        elif key.startswith("elevation_head."):
            remapped["heads.elevation_head.proj." + key.split(".", 1)[1]] = value
        elif key.startswith("vector_head."):
            remapped["heads.vector_head.proj." + key.split(".", 1)[1]] = value
        elif key.startswith("head."):
            remapped["heads.class_head.proj." + key.split(".", 1)[1]] = value
        else:
            remapped["backbone." + key] = value
    return remapped


def summarize_by_prefix(model_state, remapped_state):
    groups = {
        "transformer": lambda key: key.startswith("backbone.blocks.") or key in {
            "backbone.pos_embed",
            "backbone.cls_tokens",
            "backbone.patch_embed.proj.weight",
            "backbone.patch_embed.proj.bias",
        },
        "input_frontend": lambda key: key.startswith("backbone.spectrogram_extractor.") or key.startswith("backbone.logmel_extractor.") or key.startswith("backbone.conv_downsample.") or key.startswith("backbone.bn.") or key.startswith("backbone.foa_bn.") or key.startswith("backbone.foa_native_stem."),
        "token_norms": lambda key: key.startswith("backbone.dis_norm.") or key.startswith("backbone.doa_norm.") or key.startswith("backbone.fc_norm."),
        "heads": lambda key: key.startswith("heads."),
    }

    summary = {}
    for name, predicate in groups.items():
        keys = [key for key in model_state if predicate(key)]
        matched = [key for key in keys if key in remapped_state and tuple(remapped_state[key].shape) == tuple(model_state[key].shape)]
        summary[name] = {
            "matched_keys": len(matched),
            "total_keys": len(keys),
            "matched_params": int(sum(model_state[key].numel() for key in matched)),
            "total_params": int(sum(model_state[key].numel() for key in keys)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num_classes", type=int, default=355)
    parser.add_argument("--foa_stem_type", default="foa_native", choices=["foa_native", "logmel_only"])
    args = parser.parse_args()

    checkpoint, state_dict, ckpt_args = load_checkpoint(args.checkpoint)
    remapped_state = remap_old_spatialast_keys(state_dict)

    model = build_model(
        num_classes=args.num_classes,
        reverb_type="foa",
        foa_stem_type=args.foa_stem_type,
    )
    model_state = model.state_dict()

    matched = []
    mismatched = []
    missing = []
    for key, value in remapped_state.items():
        if key in model_state:
            if tuple(value.shape) == tuple(model_state[key].shape):
                matched.append(key)
            else:
                mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
    for key in model_state:
        if key not in remapped_state:
            missing.append(key)

    loadable_state = {
        key: value for key, value in remapped_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    msg = model.load_state_dict(loadable_state, strict=False)

    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_num_keys: {len(state_dict)}")
    if ckpt_args is not None:
        arg_dict = vars(ckpt_args) if hasattr(ckpt_args, "__dict__") else ckpt_args
        interesting = [
            "model",
            "reverb_type",
            "foa_stem_type",
            "nb_classes",
        ]
        print("checkpoint_args:", {key: arg_dict.get(key) for key in interesting})

    print(f"remapped_keys: {len(remapped_state)}")
    print(f"matched: {len(matched)}")
    print(f"mismatched: {len(mismatched)}")
    print(f"missing: {len(missing)}")
    print("prefix_summary:", summarize_by_prefix(model_state, remapped_state))
    print("missing_keys:", msg.missing_keys)
    print("unexpected_keys:", msg.unexpected_keys)

    if mismatched:
        print("first_mismatched:", mismatched[:20])


if __name__ == "__main__":
    main()
