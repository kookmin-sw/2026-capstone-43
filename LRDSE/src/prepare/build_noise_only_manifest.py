#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.condition.preprocess import find_anchor_file, find_lowstate_file


AUDIO_EXTS = {".wav", ".flac"}
AUDIO_PRIORITY = (
    "audio.wav",
    "noise.wav",
    "noise_only.wav",
    "robot_noise.wav",
    "source.wav",
)


def find_named_file(run_dir: Path, candidates):
    for name in candidates:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def find_highstate_file(run_dir: Path):
    return find_named_file(
        run_dir,
        (
            "highstate.jsonl",
            "highState.jsonl",
            "highstate_segment.jsonl",
            "highState_segment.jsonl",
            "high_level_state.jsonl",
        ),
    )


def find_audio_file(run_dir: Path):
    files = [
        p for p in sorted(run_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    if not files:
        return None, "missing_audio"

    by_name = {p.name.lower(): p for p in files}
    for name in AUDIO_PRIORITY:
        if name in by_name:
            return by_name[name], "priority"

    if len(files) == 1:
        return files[0], "single"
    return files[0], "multiple_first"


def audio_info(path: Path):
    if path is None:
        return {
            "sr": -1,
            "frames": -1,
            "duration_sec": -1.0,
            "error": "missing_audio",
        }

    try:
        info = sf.info(str(path))
        return {
            "sr": int(info.samplerate),
            "frames": int(info.frames),
            "duration_sec": float(info.frames) / float(info.samplerate),
            "error": "",
        }
    except Exception as e:
        return {
            "sr": -1,
            "frames": -1,
            "duration_sec": -1.0,
            "error": str(e),
        }


def has_run_markers(path: Path):
    return any((path / name).is_file() for name in AUDIO_PRIORITY) or any(
        (path / name).is_file()
        for name in ("anchor.json", "anchors.json", "lowstate.jsonl", "lowState.jsonl")
    )


def iter_candidate_dirs(root: Path, recursive: bool, include_contaminated: bool):
    if has_run_markers(root):
        yield root
        return

    if recursive:
        candidates = sorted(p for p in root.rglob("*") if p.is_dir())
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())

    for d in candidates:
        rel_parts = d.relative_to(root).parts
        if (not include_contaminated) and "contaminated" in rel_parts:
            continue
        if d.name == "contaminated" and not include_contaminated:
            continue
        if not has_run_markers(d):
            continue
        yield d


def write_manifest(path: Path, rows):
    fieldnames = [
        "id",
        "run_dir",
        "noise_audio_path",
        "lowstate_path",
        "anchor_path",
        "highstate_path",
        "relative_run_dir",
        "is_contaminated",
        "sr",
        "frames",
        "duration_sec",
        "valid",
        "reason",
        "audio_pick_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows, val_ratio: float, seed: int):
    if val_ratio <= 0.0 or len(rows) <= 1:
        return rows, []

    import random

    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_ratio))
    n_val = max(1, n_val)
    n_val = min(len(rows) - 1, n_val)
    return rows[n_val:], rows[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-root", default="./data/noise")
    parser.add_argument("--out", default="./data/noise_only_manifest.csv")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-out", default="")
    parser.add_argument("--val-out", default="")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search run directories. By default only direct child directories are scanned.",
    )
    parser.add_argument(
        "--include-contaminated",
        action="store_true",
        help="Include runs under a directory named 'contaminated'. Disabled by default.",
    )
    args = parser.parse_args()

    noise_root = Path(args.noise_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not noise_root.exists():
        raise FileNotFoundError(f"noise_root not found: {noise_root}")

    rows = []
    total_dirs = 0
    valid_rows = []

    for run_dir in iter_candidate_dirs(
        noise_root,
        recursive=args.recursive,
        include_contaminated=args.include_contaminated,
    ):
        total_dirs += 1
        reasons = []

        audio_path, audio_pick_reason = find_audio_file(run_dir)
        if audio_path is None:
            audio_pick_reason = "missing_audio"
            reasons.append("missing_audio")

        try:
            lowstate_path = Path(find_lowstate_file(str(run_dir)))
        except FileNotFoundError:
            lowstate_path = None
            reasons.append("missing_lowstate")

        try:
            anchor_path = Path(find_anchor_file(str(run_dir)))
        except FileNotFoundError:
            anchor_path = None
            reasons.append("missing_anchor")

        highstate_path = find_highstate_file(run_dir)
        if highstate_path is None:
            reasons.append("missing_highstate")

        info = audio_info(audio_path)
        if info["error"]:
            if info["error"] != "missing_audio":
                reasons.append(f"audio_read_error:{info['error']}")
        if info["sr"] >= 0 and info["sr"] != args.target_sr:
            reasons.append(f"sr_mismatch:{info['sr']}")
        if info["duration_sec"] >= 0.0 and info["duration_sec"] < args.min_duration:
            reasons.append(f"too_short:{info['duration_sec']:.4f}")

        rel = run_dir.relative_to(noise_root)
        rel_str = "." if str(rel) == "." else rel.as_posix()
        is_contaminated = int("contaminated" in rel.parts)
        valid = len(reasons) == 0
        row = {
            "id": run_dir.name if rel_str == "." else rel_str.replace("/", "__"),
            "run_dir": str(run_dir),
            "noise_audio_path": str(audio_path) if audio_path is not None else "",
            "lowstate_path": str(lowstate_path) if lowstate_path is not None else "",
            "anchor_path": str(anchor_path) if anchor_path is not None else "",
            "highstate_path": str(highstate_path) if highstate_path is not None else "",
            "relative_run_dir": rel_str,
            "is_contaminated": is_contaminated,
            "sr": info["sr"],
            "frames": info["frames"],
            "duration_sec": f"{info['duration_sec']:.6f}",
            "valid": int(valid),
            "reason": "ok" if valid else ";".join(reasons),
            "audio_pick_reason": audio_pick_reason,
        }
        rows.append(row)
        if valid:
            valid_rows.append(row)

    write_manifest(out_path, rows)

    train_out = Path(args.train_out).expanduser().resolve() if args.train_out else out_path.with_name(out_path.stem + "_train.csv")
    val_out = Path(args.val_out).expanduser().resolve() if args.val_out else out_path.with_name(out_path.stem + "_val.csv")
    train_rows, val_rows = split_rows(valid_rows, args.val_ratio, args.split_seed)
    write_manifest(train_out, train_rows)
    write_manifest(val_out, val_rows)

    print(f"[noise-only manifest] root={noise_root}")
    print(f"[noise-only manifest] recursive={args.recursive}")
    print(f"[noise-only manifest] include_contaminated={args.include_contaminated}")
    print(f"[noise-only manifest] scanned_dirs={total_dirs}")
    print(f"[noise-only manifest] rows={len(rows)} valid={len(valid_rows)} invalid={len(rows) - len(valid_rows)}")
    print(f"[noise-only manifest] out={out_path}")
    print(f"[noise-only manifest] train={train_out} rows={len(train_rows)}")
    print(f"[noise-only manifest] val={val_out} rows={len(val_rows)}")


if __name__ == "__main__":
    main()
