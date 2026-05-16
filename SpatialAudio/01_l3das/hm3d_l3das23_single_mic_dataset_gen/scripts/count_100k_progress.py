#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced")
DEFAULT_TARGET_TOTAL = 100_000
PROCESS_PATTERN = "hm3d_l3das23_single_mic.main_generate generate"


@dataclass
class ManifestStats:
    rows: int
    unique_scenes: int
    glos: int
    gnlos: int
    in_fov: int
    out_of_fov: int
    same_room: int
    cross_room: int
    difficulty: Counter[str]
    splits: Counter[str]
    latest_sample_id: str | None
    latest_scene_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show generation progress for the 100k HM3D FOA dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=DEFAULT_TARGET_TOTAL,
        help=f"Expected final sample count. Default: {DEFAULT_TARGET_TOTAL}",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh every N seconds. Use 0 to print once.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Also count per-sample metadata files on disk.",
    )
    parser.add_argument(
        "--show-processes",
        action="store_true",
        help="Show currently running generator processes.",
    )
    return parser.parse_args()


def read_manifest_stats(manifest_path: Path) -> ManifestStats:
    rows = 0
    scene_ids: set[str] = set()
    glos = 0
    gnlos = 0
    in_fov = 0
    out_of_fov = 0
    same_room = 0
    cross_room = 0
    difficulty: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    latest_sample_id: str | None = None
    latest_scene_id: str | None = None

    if not manifest_path.exists():
        return ManifestStats(
            rows=0,
            unique_scenes=0,
            glos=0,
            gnlos=0,
            in_fov=0,
            out_of_fov=0,
            same_room=0,
            cross_room=0,
            difficulty=difficulty,
            splits=splits,
            latest_sample_id=None,
            latest_scene_id=None,
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            rows += 1

            scene_id = row.get("scene_id")
            if scene_id:
                scene_ids.add(scene_id)

            geometry_los = row.get("geometry_los")
            if geometry_los == "gLOS":
                glos += 1
            elif geometry_los == "gNLOS":
                gnlos += 1

            if bool(row.get("is_in_fov")):
                in_fov += 1
            if bool(row.get("is_out_of_fov")):
                out_of_fov += 1
            if bool(row.get("same_room")):
                same_room += 1
            if bool(row.get("cross_room")):
                cross_room += 1

            difficulty_tag = row.get("difficulty_tag")
            if difficulty_tag:
                difficulty[str(difficulty_tag)] += 1

            split = row.get("split")
            if split:
                splits[str(split)] += 1

            latest_sample_id = row.get("sample_id")
            latest_scene_id = scene_id

    return ManifestStats(
        rows=rows,
        unique_scenes=len(scene_ids),
        glos=glos,
        gnlos=gnlos,
        in_fov=in_fov,
        out_of_fov=out_of_fov,
        same_room=same_room,
        cross_room=cross_room,
        difficulty=difficulty,
        splits=splits,
        latest_sample_id=latest_sample_id,
        latest_scene_id=latest_scene_id,
    )


def count_sample_metadata_files(dataset_root: Path) -> int:
    return sum(1 for _ in dataset_root.glob("scenes/*/samples/*/metadata/sample.json"))


def get_process_lines() -> list[str]:
    result = subprocess.run(
        ["bash", "-lc", f"ps -ef | grep '{PROCESS_PATTERN}' | grep -v grep"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def format_timestamp(path: Path) -> str:
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def print_report(
    dataset_root: Path,
    target_total: int,
    check_files: bool,
    show_processes: bool,
) -> None:
    manifest_path = dataset_root / "manifests" / "dataset_manifest.jsonl"
    stats = read_manifest_stats(manifest_path)
    progress_pct = (stats.rows / target_total * 100.0) if target_total > 0 else 0.0

    print("=" * 72)
    print(f"time                 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"dataset_root         : {dataset_root}")
    print(f"manifest_path        : {manifest_path}")
    print(f"manifest_updated_at  : {format_timestamp(manifest_path)}")
    print(f"rows                 : {stats.rows:,} / {target_total:,} ({progress_pct:.2f}%)")
    print(f"unique_scenes        : {stats.unique_scenes}")
    print(f"geometry_los         : gLOS={stats.glos:,}, gNLOS={stats.gnlos:,}")
    print(f"fov                  : in_fov={stats.in_fov:,}, out_of_fov={stats.out_of_fov:,}")
    print(f"room_relation        : same_room={stats.same_room:,}, cross_room={stats.cross_room:,}")
    print(f"difficulty           : {format_counter(stats.difficulty)}")
    print(f"splits               : {format_counter(stats.splits)}")
    print(f"latest_sample        : {stats.latest_sample_id or '-'}")
    print(f"latest_scene         : {stats.latest_scene_id or '-'}")

    if check_files:
        metadata_count = count_sample_metadata_files(dataset_root)
        diff = metadata_count - stats.rows
        print(f"sample_json_files    : {metadata_count:,}")
        print(f"rows_minus_files     : {stats.rows - metadata_count:+,}")
        if diff != 0:
            print("file_check_status    : mismatch")
        else:
            print("file_check_status    : ok")

    if show_processes:
        process_lines = get_process_lines()
        print(f"running_processes    : {len(process_lines)}")
        for line in process_lines:
            print(f"process              : {line}")


def main() -> None:
    args = parse_args()

    while True:
        print_report(
            dataset_root=args.dataset_root,
            target_total=args.target_total,
            check_files=args.check_files,
            show_processes=args.show_processes,
        )
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
