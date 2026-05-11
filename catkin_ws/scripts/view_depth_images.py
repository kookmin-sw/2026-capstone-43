#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def colorize_depth(depth_m, min_depth, max_depth):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    clipped = np.clip(depth_m, min_depth, max_depth)
    normalized = ((clipped - min_depth) / max(max_depth - min_depth, 1e-6) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def summarize_depth(depth_m):
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        return "valid=0"
    qs = np.percentile(valid, [1, 5, 50, 95, 99])
    return (
        f"valid={valid.size} "
        f"min={valid.min():.3f} max={valid.max():.3f} "
        f"p01={qs[0]:.3f} p05={qs[1]:.3f} "
        f"p50={qs[2]:.3f} p95={qs[3]:.3f} p99={qs[4]:.3f}"
    )


def load_rows(dataset_dir):
    poses_csv = dataset_dir / "poses.csv"
    if not poses_csv.exists():
        return []
    with open(poses_csv, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Visualize saved aligned depth images next to RGB frames.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    rows = load_rows(dataset_dir)
    if not rows:
        depth_files = sorted((dataset_dir / "depth").glob("*.png"))
        rows = [{"filename": f.name, "depth_filename": f.name} for f in depth_files]

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    index = max(args.start, 0)
    print("Controls: n/space=next, p=prev, s=save preview, q/esc=quit")
    while 0 <= index < len(rows):
        row = rows[index]
        depth_name = row.get("depth_filename", "")
        rgb_name = row.get("filename", "")
        depth_path = dataset_dir / "depth" / depth_name
        rgb_path = dataset_dir / "rgb" / rgb_name
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"[{index}] missing depth: {depth_path}")
            index += args.step
            continue

        if depth_raw.dtype == np.uint16 or depth_raw.dtype == np.uint32:
            depth_m = depth_raw.astype(np.float32) * args.depth_scale
        else:
            depth_m = depth_raw.astype(np.float32)

        depth_color = colorize_depth(depth_m, args.min_depth, args.max_depth)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            rgb = np.zeros_like(depth_color)
        if rgb.shape[:2] != depth_color.shape[:2]:
            rgb = cv2.resize(rgb, (depth_color.shape[1], depth_color.shape[0]))

        preview = np.hstack([rgb, depth_color])
        text = f"{index}: rgb={rgb_name} depth={depth_name} {summarize_depth(depth_m)}"
        print(text)
        cv2.putText(preview, text[:160], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("RGB | aligned depth", preview)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            index = max(0, index - args.step)
        else:
            if key == ord("s") or args.save_dir:
                out_dir = args.save_dir or dataset_dir
                out_path = out_dir / f"depth_preview_{index:06d}.png"
                cv2.imwrite(str(out_path), preview)
                print(f"saved {out_path}")
            index += args.step


if __name__ == "__main__":
    main()
