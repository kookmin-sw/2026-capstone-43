#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from online_gs_slam.semantic.dataset import load_hash_grid_supervision
from online_gs_slam.semantic.hash_grid import HashGrid4DConfig
from online_gs_slam.semantic.trainer import HashGridTrainingConfig, infer_bounds, train_hash_grid
from online_gs_slam.semantic.visualization import write_labeled_ply


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 4D multi-scale hash grid from Gaussian/point labels.")
    parser.add_argument("--samples", required=True, type=Path, help="npz with xyz/positions, labels, optional time/timestamps")
    parser.add_argument("--output", default="outputs/hash_grid/semantic_hash_grid.pt", type=Path)
    parser.add_argument("--preview-ply", default="outputs/hash_grid/semantic_preview.ply", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-levels", type=int, default=12)
    parser.add_argument("--features-per-level", type=int, default=2)
    parser.add_argument("--log2-hashmap-size", type=int, default=18)
    parser.add_argument("--base-resolution", type=int, default=8)
    parser.add_argument("--finest-resolution", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--bbox-padding", type=float, default=0.05)
    args = parser.parse_args()

    supervision = load_hash_grid_supervision(args.samples, device=args.device)
    bbox_min, bbox_max, time_min, time_max = infer_bounds(supervision, padding=args.bbox_padding)
    model_config = HashGrid4DConfig(
        num_levels=args.num_levels,
        features_per_level=args.features_per_level,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        hidden_dim=args.hidden_dim,
        output_dim=supervision.num_classes,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        time_min=time_min,
        time_max=time_max,
    )
    train_config = HashGridTrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    model = train_hash_grid(supervision, model_config, train_config, args.output)

    with torch.no_grad():
        labels = model.predict_labels(supervision.xyz, supervision.time)
    write_labeled_ply(args.preview_ply, supervision.xyz, labels)
    print(f"Wrote {args.preview_ply}")


if __name__ == "__main__":
    main()
