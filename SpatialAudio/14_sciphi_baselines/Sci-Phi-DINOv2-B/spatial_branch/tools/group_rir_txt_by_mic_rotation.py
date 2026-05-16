#!/usr/bin/env python3
"""Group RIR txt files by (mic position) -> (same mic rotation).

Target use-case:
- choose one mic position
- pick N source RIRs
- apply 0/90/180/270 rotation augmentation

This script first groups by mic position, then groups inside each position by
same mic rotation (both exact yaw bucket and cardinal 0/90/180/270 bucket).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


POS_LABELS = {
    "habitat": "Mic trans habitat_xyz",
    "legacy": "Mic trans",
}

YAW_LABELS = {
    "habitat": "Mic yaw habitat_rad",
}

CARDINAL_BINS = (0, 90, 180, 270)
RIR_IDX_RE = re.compile(r"rir_IDX(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group RIR txt files by mic position and rotation."
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Default: <root>/mic_position_rotation_groups.json",
    )
    parser.add_argument(
        "--coord-field",
        type=str,
        default="habitat",
        choices=tuple(POS_LABELS.keys()),
    )
    parser.add_argument(
        "--yaw-field",
        type=str,
        default="habitat",
        choices=tuple(YAW_LABELS.keys()),
    )
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument(
        "--yaw-round-decimals",
        type=int,
        default=3,
        help="Precision for exact-yaw grouping key (degrees).",
    )
    parser.add_argument("--pattern", type=str, default="rir_*.txt")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument(
        "--min-src-per-rotation",
        type=int,
        default=3,
        help="For eligibility check: min files required in each 0/90/180/270 bucket.",
    )
    parser.add_argument("--omit-files", action="store_true")
    return parser.parse_args()


def iter_rir_txt_files(root: Path, pattern: str):
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        if "_logs" in path.parts:
            continue
        yield path


def _find_line_idx(lines: List[str], label: str) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == label:
            return idx
    return -1


def find_vec3_after_label(lines: List[str], label: str, file_path: Path) -> Tuple[float, float, float]:
    idx = _find_line_idx(lines, label)
    if idx < 0:
        raise ValueError(f"Label '{label}' not found in {file_path}")
    if idx + 1 >= len(lines):
        raise ValueError(f"Missing line after '{label}' in {file_path}")
    parts = lines[idx + 1].strip().split()
    if len(parts) < 3:
        raise ValueError(f"Expected vec3 after '{label}' in {file_path}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def find_scalar_after_label(lines: List[str], label: str, file_path: Path) -> float:
    idx = _find_line_idx(lines, label)
    if idx < 0:
        raise ValueError(f"Label '{label}' not found in {file_path}")
    if idx + 1 >= len(lines):
        raise ValueError(f"Missing line after '{label}' in {file_path}")
    return float(lines[idx + 1].strip().split()[0])


def parse_rir_idx(path: Path) -> int | None:
    m = RIR_IDX_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def rad_to_deg_360(rad: float) -> float:
    deg = math.degrees(rad) % 360.0
    if deg < 0:
        deg += 360.0
    return deg


def circular_distance_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def quantize_cardinal(deg: float) -> int:
    best = CARDINAL_BINS[0]
    best_dist = circular_distance_deg(deg, float(best))
    for b in CARDINAL_BINS[1:]:
        dist = circular_distance_deg(deg, float(b))
        if dist < best_dist:
            best = b
            best_dist = dist
    return int(best)


def sort_items_for_output(items: List[dict]) -> List[dict]:
    return sorted(
        items,
        key=lambda x: (
            10**9 if x["rir_idx"] is None else x["rir_idx"],
            x["file"],
        ),
    )


def build_rotation_group(items: List[dict], omit_files: bool) -> dict:
    items = sort_items_for_output(items)
    if not items:
        out = {
            "num_files": 0,
            "rir_indices": [],
            "mic_yaw_deg_min": None,
            "mic_yaw_deg_max": None,
        }
        if not omit_files:
            out["files"] = []
        return out

    out = {
        "num_files": len(items),
        "rir_indices": [x["rir_idx"] for x in items if x["rir_idx"] is not None],
        "mic_yaw_deg_min": min(x["mic_yaw_deg_0_360"] for x in items),
        "mic_yaw_deg_max": max(x["mic_yaw_deg_0_360"] for x in items),
    }
    if not omit_files:
        out["files"] = [x["file"] for x in items]
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"root is not a directory: {root}")

    pos_label = POS_LABELS[args.coord_field]
    yaw_label = YAW_LABELS[args.yaw_field]

    files = sorted(iter_rir_txt_files(root, args.pattern))
    if args.scene:
        files = [p for p in files if p.relative_to(root).parts[0] == args.scene]
    if not files:
        raise ValueError(f"No files matched under {root} with pattern '{args.pattern}'")

    # scene + rounded mic position -> list[items]
    pos_groups: Dict[Tuple[str, Tuple[float, float, float]], List[dict]] = defaultdict(list)
    scene_counts: Dict[str, int] = defaultdict(int)
    parse_errors: List[str] = []

    for txt_path in files:
        rel = txt_path.relative_to(root)
        scene = rel.parts[0]
        scene_counts[scene] += 1

        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
            mic_pos = find_vec3_after_label(lines, pos_label, txt_path)
            mic_yaw_rad = find_scalar_after_label(lines, yaw_label, txt_path)
        except Exception as exc:
            parse_errors.append(f"{txt_path}: {exc}")
            continue

        mic_yaw_deg = rad_to_deg_360(mic_yaw_rad)
        mic_yaw_cardinal = quantize_cardinal(mic_yaw_deg)

        key_pos = tuple(round(v, args.round_decimals) for v in mic_pos)
        pos_groups[(scene, key_pos)].append(
            {
                "file": str(rel),
                "rir_idx": parse_rir_idx(txt_path),
                "mic_position_raw": mic_pos,
                "mic_yaw_rad": mic_yaw_rad,
                "mic_yaw_deg_0_360": mic_yaw_deg,
                "mic_yaw_cardinal_deg": mic_yaw_cardinal,
                "mic_yaw_exact_key_deg": round(mic_yaw_deg, args.yaw_round_decimals),
            }
        )

    output = {
        "root": str(root),
        "coord_field": args.coord_field,
        "coord_label": pos_label,
        "yaw_field": args.yaw_field,
        "yaw_label": yaw_label,
        "cardinal_bins_deg": list(CARDINAL_BINS),
        "round_decimals": args.round_decimals,
        "yaw_round_decimals": args.yaw_round_decimals,
        "min_src_per_rotation": args.min_src_per_rotation,
        "total_txt_files": len(files),
        "parsed_ok_files": sum(len(v) for v in pos_groups.values()),
        "total_position_groups": len(pos_groups),
        "num_scenes": len({k[0] for k in pos_groups.keys()}),
        "scenes": {},
    }

    scene_to_positions: Dict[str, List[dict]] = defaultdict(list)
    for (scene, _pos_key), items in pos_groups.items():
        items_sorted = sort_items_for_output(items)
        first = items_sorted[0]

        # Cardinal rotation groups inside this position group.
        cardinal_groups: Dict[int, List[dict]] = {b: [] for b in CARDINAL_BINS}
        for it in items_sorted:
            cardinal_groups[int(it["mic_yaw_cardinal_deg"])].append(it)

        cardinal_out = {}
        for b in CARDINAL_BINS:
            cardinal_out[str(b)] = build_rotation_group(cardinal_groups[b], args.omit_files)

        # Exact-yaw groups inside this position group.
        exact_groups_raw: Dict[float, List[dict]] = defaultdict(list)
        for it in items_sorted:
            exact_groups_raw[float(it["mic_yaw_exact_key_deg"])].append(it)

        exact_out = {}
        for yaw_key in sorted(exact_groups_raw.keys()):
            exact_out[str(yaw_key)] = build_rotation_group(exact_groups_raw[yaw_key], args.omit_files)

        # Eligibility for "N sources x 4 rotations" inside this position group.
        eligible = all(
            cardinal_out[str(b)]["num_files"] >= int(args.min_src_per_rotation)
            for b in CARDINAL_BINS
        )

        pos_entry = {
            "mic_position": [float(x) for x in first["mic_position_raw"]],
            "total_files": len(items_sorted),
            "yaw_cardinal_present": [b for b in CARDINAL_BINS if cardinal_out[str(b)]["num_files"] > 0],
            "yaw_exact_present_deg": sorted(exact_groups_raw.keys()),
            "eligible_for_rotation_aug": bool(eligible),
            "rotation_groups_cardinal": cardinal_out,
            "rotation_groups_exact_deg": exact_out,
        }
        scene_to_positions[scene].append(pos_entry)

    for scene in sorted(scene_to_positions.keys()):
        positions = sorted(
            scene_to_positions[scene],
            key=lambda x: (-x["total_files"], x["mic_position"]),
        )
        eligible_cnt = sum(1 for p in positions if p["eligible_for_rotation_aug"])
        output["scenes"][scene] = {
            "num_txt_files": scene_counts.get(scene, 0),
            "num_position_groups": len(positions),
            "eligible_position_groups": eligible_cnt,
            "position_groups": positions,
        }

    if parse_errors:
        output["parse_errors"] = parse_errors

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root / "mic_position_rotation_groups.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(
        f"Summary: files={output['total_txt_files']}, parsed={output['parsed_ok_files']}, "
        f"position_groups={output['total_position_groups']}, scenes={output['num_scenes']}"
    )
    if "parse_errors" in output:
        print(f"Parse errors: {len(output['parse_errors'])}")


if __name__ == "__main__":
    main()
