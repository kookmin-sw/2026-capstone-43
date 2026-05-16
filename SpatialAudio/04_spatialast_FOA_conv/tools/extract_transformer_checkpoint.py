import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TRANSFORMER_KEYS = {
    "pos_embed",
    "cls_tokens",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
}


def is_transformer_key(key):
    return key in TRANSFORMER_KEYS or key.startswith("blocks.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(ROOT / "finetuned.pth"),
        help="source checkpoint path",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "finetuned_transformer_only.pth"),
        help="output checkpoint path",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    transformer_state = {
        "backbone." + key: value
        for key, value in source_state.items()
        if is_transformer_key(key)
    }

    num_keys = len(transformer_state)
    num_params = sum(value.numel() for value in transformer_state.values())
    tensor_mb = sum(value.numel() * value.element_size() for value in transformer_state.values()) / 1024 / 1024

    output = {
        "model": transformer_state,
        "meta": {
            "source_checkpoint": args.source,
            "key_style": "11_spatialast_foa_backbone_namespace",
            "included_keys_rule": [
                "backbone.pos_embed",
                "backbone.cls_tokens",
                "backbone.patch_embed.proj.weight",
                "backbone.patch_embed.proj.bias",
                "backbone.blocks.*",
            ],
            "num_keys": num_keys,
            "num_params": num_params,
            "tensor_mb": round(tensor_mb, 2),
        },
    }

    torch.save(output, args.output)

    print(f"saved: {args.output}")
    print(f"num_keys: {num_keys}")
    print(f"num_params: {num_params}")
    print(f"tensor_mb: {tensor_mb:.2f}")


if __name__ == "__main__":
    main()
