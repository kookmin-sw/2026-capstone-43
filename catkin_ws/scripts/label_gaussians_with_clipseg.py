#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from gaussian_ply_to_semantic_points import read_gaussian_ply
from online_gs_slam.semantic.visualization import write_labeled_ply


def load_clipseg(device: str):
    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers\n"
            "Install it in the ns310 env with:\n"
            "  python -m pip install transformers\n"
        ) from exc

    model_name = "CIDAS/clipseg-rd64-refined"
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def run_clipseg_masks(image: Image.Image, prompts: list[str], processor, model, device: str) -> np.ndarray:
    inputs = processor(text=prompts, images=[image] * len(prompts), padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        if logits.ndim == 2:
            logits = logits[None, ...]
        masks = torch.sigmoid(logits)
        masks = torch.nn.functional.interpolate(
            masks[:, None, :, :],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    return masks.detach().cpu().numpy().astype(np.float32)


def load_transforms(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_dataparser_transform(path: Optional[Path]) -> tuple[np.ndarray, float]:
    if path is None:
        return np.eye(4, dtype=np.float64), 1.0
    with open(path) as f:
        data = json.load(f)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = np.asarray(data["transform"], dtype=np.float64)
    return transform, float(data.get("scale", 1.0))


def apply_dataparser_transform(c2w: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    out = transform @ c2w
    out[:3, 3] *= scale
    return out


def project_world_to_image(
    xyz: np.ndarray,
    c2w_nerfstudio: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_h = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    w2c = np.linalg.inv(c2w_nerfstudio)
    cam_ns = (w2c @ world_h.T).T[:, :3]

    # Nerfstudio/OpenGL camera: +X right, +Y up, -Z forward.
    z = -cam_ns[:, 2]
    x = cam_ns[:, 0]
    y = -cam_ns[:, 1]
    valid = z > 1e-4

    u = fx * (x / np.maximum(z, 1e-6)) + cx
    v = fy * (y / np.maximum(z, 1e-6)) + cy
    valid &= (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    return u.astype(np.float32), v.astype(np.float32), valid


def sample_masks_at_points(masks: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full((len(u),), -1, dtype=np.int64)
    confidence = np.zeros((len(u),), dtype=np.float32)
    indices = np.where(valid)[0]
    if len(indices) == 0:
        return labels, confidence
    uu = np.clip(np.round(u[indices]).astype(np.int64), 0, masks.shape[2] - 1)
    vv = np.clip(np.round(v[indices]).astype(np.int64), 0, masks.shape[1] - 1)
    scores = masks[:, vv, uu].T
    labels[indices] = scores.argmax(axis=1).astype(np.int64)
    confidence[indices] = scores.max(axis=1).astype(np.float32)
    return labels, confidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use CLIPSeg masks on Nerfstudio views to create Gaussian semantic_points.npz."
    )
    parser.add_argument("--gaussian-ply", required=True, type=Path)
    parser.add_argument("--transforms", required=True, type=Path)
    parser.add_argument(
        "--dataparser-transforms",
        type=Path,
        default=None,
        help="Nerfstudio run dataparser_transforms.json. Required when projecting exported splatfacto PLY.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset root for relative image paths")
    parser.add_argument("--prompts", nargs="+", required=True, help='Open-vocabulary labels, e.g. "floor" "table"')
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview-ply", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-gaussians", type=int, default=120000)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--mask-threshold", type=float, default=0.45)
    parser.add_argument("--unknown-label", default="unknown")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    xyz, colors = read_gaussian_ply(args.gaussian_ply)
    rng = np.random.default_rng(args.seed)
    if args.max_gaussians > 0 and len(xyz) > args.max_gaussians:
        keep = rng.choice(len(xyz), size=args.max_gaussians, replace=False)
        xyz = xyz[keep]
        if colors is not None:
            colors = colors[keep]

    transforms = load_transforms(args.transforms)
    dataparser_transform, dataparser_scale = load_dataparser_transform(args.dataparser_transforms)
    data_dir = args.data_dir or args.transforms.parent
    frames = transforms["frames"][:: max(args.frame_stride, 1)]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("No frames selected from transforms.json")

    processor, model = load_clipseg(args.device)

    prompts = list(args.prompts)
    class_names = np.array(prompts + [args.unknown_label])
    unknown_id = len(prompts)
    vote_scores = np.zeros((len(xyz), len(class_names)), dtype=np.float32)
    vote_counts = np.zeros((len(xyz),), dtype=np.float32)

    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    width = int(transforms["w"])
    height = int(transforms["h"])

    for frame_idx, frame in enumerate(frames, start=1):
        image_path = data_dir / frame["file_path"]
        if not image_path.exists():
            print(f"[WARN] missing image {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        masks = run_clipseg_masks(image, prompts, processor, model, args.device)
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w = apply_dataparser_transform(c2w, dataparser_transform, dataparser_scale)
        u, v, valid = project_world_to_image(xyz, c2w, fx, fy, cx, cy, width, height)
        labels, confidence = sample_masks_at_points(masks, u, v, valid)
        good = valid & (confidence >= args.mask_threshold)
        if np.any(good):
            vote_scores[np.where(good)[0], labels[good]] += confidence[good]
            vote_counts[good] += 1.0
        print(
            f"[{frame_idx:03d}/{len(frames):03d}] {image_path.name} "
            f"visible={int(valid.sum())} labeled={int(good.sum())}"
        )

    labels = vote_scores.argmax(axis=1).astype(np.int64)
    unlabeled = vote_counts <= 0
    labels[unlabeled] = unknown_id
    weights = np.maximum(vote_counts, 1.0).astype(np.float32)
    time = np.zeros((len(xyz),), dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xyz=xyz.astype(np.float32),
        time=time,
        labels=labels,
        weights=weights,
        class_names=class_names,
        vote_scores=vote_scores,
        vote_counts=vote_counts,
    )
    print(f"Wrote {args.output}")
    print("label counts:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {int((labels == idx).sum())}")

    if args.preview_ply is not None:
        write_labeled_ply(args.preview_ply, xyz, labels)
        print(f"Wrote {args.preview_ply}")


if __name__ == "__main__":
    main()
