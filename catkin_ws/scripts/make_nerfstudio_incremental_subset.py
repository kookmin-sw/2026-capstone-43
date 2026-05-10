#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cumulative Nerfstudio subset for staged/continual training.")
    parser.add_argument("--source-data-dir", required=True, type=Path)
    parser.add_argument("--output-data-dir", required=True, type=Path)
    parser.add_argument("--sample-stride", type=int, default=50, help="Use every Nth original frame.")
    parser.add_argument("--num-sampled-frames", type=int, required=True, help="Cumulative sampled frame count to expose.")
    args = parser.parse_args()

    source = args.source_data_dir.expanduser().resolve()
    output = args.output_data_dir.expanduser().resolve()
    transforms_path = source / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"{transforms_path} does not exist. Run export_nerfstudio_dataset.sh first.")

    with open(transforms_path) as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    selected = frames[:: args.sample_stride][: args.num_sampled_frames]
    if not selected:
        raise RuntimeError("No frames selected. Check --sample-stride and --num-sampled-frames.")

    output.mkdir(parents=True, exist_ok=True)
    link_or_copy(source / "rgb", output / "rgb")

    subset = dict(transforms)
    subset["frames"] = selected
    with open(output / "transforms.json", "w") as f:
        json.dump(subset, f, indent=2)

    print(
        f"[incremental-subset] wrote {output / 'transforms.json'} "
        f"with {len(selected)} frames from {len(frames)} source frames "
        f"(stride={args.sample_stride})"
    )


if __name__ == "__main__":
    main()
