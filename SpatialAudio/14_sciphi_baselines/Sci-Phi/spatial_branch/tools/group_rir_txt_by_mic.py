#!/usr/bin/env python3
"""Group mp3d_rir_foa RIR txt files by microphone position.

Default grouping key:
- scene folder name (first path component under root)
- "Mic trans habitat_xyz" position (rounded)

Example:
  python spatial_branch/tools/group_rir_txt_by_mic.py \
    --root /home/yu/Project_git/mp3d_rir_foa \
    --output /home/yu/Project_git/mp3d_rir_foa/mic_groups.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


COORD_LABELS = {
    "habitat": "Mic trans habitat_xyz",
    "legacy": "Mic trans",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group RIR txt files by identical mic position."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory (e.g., /home/yu/Project_git/mp3d_rir_foa).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output json path. Default: <root>/mic_groups_by_scene.json",
    )
    parser.add_argument(
        "--coord-field",
        type=str,
        default="habitat",
        choices=tuple(COORD_LABELS.keys()),
        help="Which coordinate field in txt files to use as mic position key.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Decimal places used when grouping floating point positions.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="rir_*.txt",
        help="Glob pattern for RIR metadata files.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Optional scene folder name filter.",
    )
    parser.add_argument(
        "--omit-files",
        action="store_true",
        help="If set, do not include per-group file list in output json.",
    )
    return parser.parse_args()


def find_value_after_label(lines: List[str], label: str, file_path: Path) -> Tuple[float, float, float]:
    for idx, line in enumerate(lines):
        if line.strip() != label:
            continue
        if idx + 1 >= len(lines):
            raise ValueError(f"Missing value line after '{label}' in {file_path}")
        parts = lines[idx + 1].strip().split()
        if len(parts) < 3:
            raise ValueError(f"Expected 3 values after '{label}' in {file_path}")
        return float(parts[0]), float(parts[1]), float(parts[2])
    raise ValueError(f"Label '{label}' not found in {file_path}")


def iter_rir_txt_files(root: Path, pattern: str):
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        if "_logs" in path.parts:
            continue
        yield path


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"root is not a directory: {root}")

    label = COORD_LABELS[args.coord_field]
    files = sorted(iter_rir_txt_files(root, args.pattern))
    if args.scene:
        files = [p for p in files if p.relative_to(root).parts[0] == args.scene]
    if not files:
        raise ValueError(f"No files matched under {root} with pattern '{args.pattern}'")

    groups: Dict[Tuple[str, Tuple[float, float, float]], List[Path]] = defaultdict(list)
    raw_pos_by_key: Dict[Tuple[str, Tuple[float, float, float]], Tuple[float, float, float]] = {}
    scene_counts: Dict[str, int] = defaultdict(int)
    parse_errors: List[str] = []

    for txt_path in files:
        rel = txt_path.relative_to(root)
        scene = rel.parts[0]
        scene_counts[scene] += 1
        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
            raw_pos = find_value_after_label(lines, label, txt_path)
        except Exception as exc:
            parse_errors.append(f"{txt_path}: {exc}")
            continue

        key_pos = tuple(round(v, args.round_decimals) for v in raw_pos)
        key = (scene, key_pos)
        groups[key].append(txt_path)
        if key not in raw_pos_by_key:
            raw_pos_by_key[key] = raw_pos

    scene_to_groups: Dict[str, List[dict]] = defaultdict(list)
    for (scene, key_pos), paths in groups.items():
        entry = {
            "mic_position": [float(x) for x in raw_pos_by_key[(scene, key_pos)]],
            "count": len(paths),
        }
        if not args.omit_files:
            entry["files"] = [str(p.relative_to(root)) for p in sorted(paths)]
        scene_to_groups[scene].append(entry)

    output = {
        "root": str(root),
        "coord_field": args.coord_field,
        "coord_label": label,
        "round_decimals": args.round_decimals,
        "total_txt_files": len(files),
        "parsed_ok_files": sum(len(v) for v in groups.values()),
        "total_groups": len(groups),
        "num_scenes": len(scene_to_groups),
        "scenes": {},
    }

    for scene in sorted(scene_to_groups.keys()):
        g = scene_to_groups[scene]
        g = sorted(g, key=lambda x: (-x["count"], x["mic_position"]))
        counts = [x["count"] for x in g]
        output["scenes"][scene] = {
            "num_txt_files": scene_counts.get(scene, 0),
            "num_groups": len(g),
            "group_size_min": min(counts) if counts else 0,
            "group_size_max": max(counts) if counts else 0,
            "groups": g,
        }

    if parse_errors:
        output["parse_errors"] = parse_errors

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root / "mic_groups_by_scene.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(
        f"Summary: files={output['total_txt_files']}, parsed={output['parsed_ok_files']}, "
        f"groups={output['total_groups']}, scenes={output['num_scenes']}"
    )
    if "parse_errors" in output:
        print(f"Parse errors: {len(output['parse_errors'])}")


if __name__ == "__main__":
    main()
