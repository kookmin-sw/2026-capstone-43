import argparse
import csv
import random
from pathlib import Path

import soundfile as sf


AUDIO_EXTS = {".wav", ".flac"}


def get_audio_info(path: Path):
    try:
        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
        return {
            "sr": int(info.samplerate),
            "frames": int(info.frames),
            "duration": duration,
            "error": "",
        }
    except Exception as e:
        return {
            "sr": -1,
            "frames": -1,
            "duration": -1.0,
            "error": str(e),
        }


def build_clean_index(clean_root: Path):
    index = {}

    for ext in AUDIO_EXTS:
        for path in clean_root.rglob(f"*{ext}"):
            if path.is_file():
                index.setdefault(path.stem, []).append(path)

    return index


def find_clean_path(clean_root: Path, clean_index, speaker_id, book_id, source_id):
    candidates = []

    for ext in AUDIO_EXTS:
        candidates.append(clean_root / speaker_id / book_id / f"{source_id}{ext}")
        candidates.append(clean_root / speaker_id / book_id / source_id / f"{source_id}{ext}")

    for p in candidates:
        if p.exists():
            return p, "exact"

    indexed = clean_index.get(source_id, [])
    if len(indexed) == 1:
        return indexed[0], "index"

    if len(indexed) > 1:
        return sorted(indexed)[0], "index_duplicate"

    return None, "not_found"


def choose_noisy_file(files, source_id):
    by_name = {p.name.lower(): p for p in files}

    priority_names = [
        f"{source_id}.wav",
        f"{source_id}.flac",
        "noisy.wav",
        "mixed.wav",
        "source.wav",
        "audio.wav",
    ]

    for name in priority_names:
        if name.lower() in by_name:
            return by_name[name.lower()], "priority"

    source_like = [p for p in files if source_id in p.stem]
    if len(source_like) == 1:
        return source_like[0], "source_like"

    if len(files) == 1:
        return files[0], "single"

    return sorted(files)[0], "multiple_first"


def iter_noisy_source_dirs(noisy_root: Path):
    for d in sorted(noisy_root.rglob("*")):
        if not d.is_dir():
            continue

        audio_files = [
            p for p in sorted(d.iterdir())
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]

        if not audio_files:
            continue

        rel_parts = d.relative_to(noisy_root).parts

        if len(rel_parts) >= 3:
            speaker_id = rel_parts[0]
            book_id = rel_parts[1]
            source_id = rel_parts[2]
        elif len(rel_parts) == 2:
            speaker_id = rel_parts[0]
            book_id = ""
            source_id = rel_parts[1]
        else:
            continue

        yield d, speaker_id, book_id, source_id, audio_files


def write_manifest(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_valid_rows(valid_rows, split_by: str, val_ratio: float, seed: int):
    if val_ratio <= 0.0 or len(valid_rows) <= 1:
        return list(valid_rows), []

    rng = random.Random(seed)

    if split_by == "speaker":
        speakers = sorted({row["speaker_id"] for row in valid_rows})
        if len(speakers) >= 2:
            rng.shuffle(speakers)
            n_val_spk = int(round(len(speakers) * val_ratio))
            n_val_spk = max(1, n_val_spk)
            n_val_spk = min(len(speakers) - 1, n_val_spk)
            val_speakers = set(speakers[:n_val_spk])

            train_rows = [row for row in valid_rows if row["speaker_id"] not in val_speakers]
            val_rows = [row for row in valid_rows if row["speaker_id"] in val_speakers]
            return train_rows, val_rows

    # Fallback: sample-level split
    rows = list(valid_rows)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_ratio))
    n_val = max(1, n_val)
    n_val = min(len(rows) - 1, n_val)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-root", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration-diff", type=float, default=0.1)
    parser.add_argument("--strict-duration", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-by", default="speaker", choices=["speaker", "sample"])
    parser.add_argument("--train-out", default="")
    parser.add_argument("--val-out", default="")
    args = parser.parse_args()

    noisy_root = Path(args.noisy_root).expanduser().resolve()
    clean_root = Path(args.clean_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not noisy_root.exists():
        raise FileNotFoundError(f"noisy_root not found: {noisy_root}")

    if not clean_root.exists():
        raise FileNotFoundError(f"clean_root not found: {clean_root}")

    print(f"[manifest] noisy_root: {noisy_root}")
    print(f"[manifest] clean_root: {clean_root}")
    print("[manifest] building clean index...")

    clean_index = build_clean_index(clean_root)

    rows = []
    total = 0
    valid_count = 0

    for source_dir, speaker_id, book_id, source_id, audio_files in iter_noisy_source_dirs(noisy_root):
        total += 1

        noisy_path, noisy_pick_reason = choose_noisy_file(audio_files, source_id)
        clean_path, clean_find_reason = find_clean_path(
            clean_root=clean_root,
            clean_index=clean_index,
            speaker_id=speaker_id,
            book_id=book_id,
            source_id=source_id,
        )

        reasons = []

        if noisy_pick_reason == "multiple_first":
            reasons.append("multiple_noisy_candidates")

        if clean_path is None:
            reasons.append("clean_not_found")

        noisy_info = get_audio_info(noisy_path)
        clean_info = None

        if noisy_info["error"]:
            reasons.append(f"noisy_read_error:{noisy_info['error']}")

        if clean_path is not None:
            clean_info = get_audio_info(clean_path)

            if clean_info["error"]:
                reasons.append(f"clean_read_error:{clean_info['error']}")

        if noisy_info["sr"] != args.target_sr:
            reasons.append(f"noisy_sr_mismatch:{noisy_info['sr']}")

        if clean_info is not None and clean_info["sr"] != args.target_sr:
            reasons.append(f"clean_sr_mismatch:{clean_info['sr']}")

        noisy_duration = noisy_info["duration"]
        clean_duration = clean_info["duration"] if clean_info is not None else -1.0
        duration_diff = abs(noisy_duration - clean_duration) if clean_duration > 0 else -1.0

        if noisy_duration < args.min_duration:
            reasons.append(f"noisy_too_short:{noisy_duration:.4f}")

        if clean_duration > 0 and clean_duration < args.min_duration:
            reasons.append(f"clean_too_short:{clean_duration:.4f}")

        if args.strict_duration and duration_diff > args.max_duration_diff:
            reasons.append(f"duration_mismatch:{duration_diff:.4f}")

        valid = len(reasons) == 0

        if valid:
            valid_count += 1

        row = {
            "id": f"{speaker_id}_{book_id}_{source_id}",
            "speaker_id": speaker_id,
            "book_id": book_id,
            "source_id": source_id,
            "source_dir": str(source_dir),
            "noisy_wav": str(noisy_path),
            "clean_wav": str(clean_path) if clean_path is not None else "",
            "noisy_sr": noisy_info["sr"],
            "clean_sr": clean_info["sr"] if clean_info is not None else -1,
            "noisy_duration_sec": f"{noisy_duration:.6f}",
            "clean_duration_sec": f"{clean_duration:.6f}",
            "duration_diff_sec": f"{duration_diff:.6f}",
            "valid": int(valid),
            "reason": "ok" if valid else ";".join(reasons),
            "noisy_pick_reason": noisy_pick_reason,
            "clean_find_reason": clean_find_reason,
        }

        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "speaker_id",
        "book_id",
        "source_id",
        "source_dir",
        "noisy_wav",
        "clean_wav",
        "noisy_sr",
        "clean_sr",
        "noisy_duration_sec",
        "clean_duration_sec",
        "duration_diff_sec",
        "valid",
        "reason",
        "noisy_pick_reason",
        "clean_find_reason",
    ]

    valid_rows = [row for row in rows if int(row["valid"]) == 1]
    train_rows, val_rows = split_valid_rows(
        valid_rows=valid_rows,
        split_by=args.split_by,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )

    train_out_path = Path(args.train_out).expanduser().resolve() if args.train_out else out_path.with_name(f"{out_path.stem}_train{out_path.suffix}")
    val_out_path = Path(args.val_out).expanduser().resolve() if args.val_out else out_path.with_name(f"{out_path.stem}_val{out_path.suffix}")

    write_manifest(out_path, fieldnames, rows)
    write_manifest(train_out_path, fieldnames, train_rows)
    write_manifest(val_out_path, fieldnames, val_rows)

    print("--------------------------------------------------")
    print(f"[manifest] total source dirs : {total}")
    print(f"[manifest] valid samples     : {valid_count}")
    print(f"[manifest] invalid samples   : {total - valid_count}")
    print(f"[manifest] saved(all)       : {out_path}")
    print(f"[manifest] split_by         : {args.split_by}")
    print(f"[manifest] val_ratio        : {args.val_ratio}")
    print(f"[manifest] split_seed       : {args.split_seed}")
    print(f"[manifest] train samples    : {len(train_rows)}")
    print(f"[manifest] val samples      : {len(val_rows)}")
    print(f"[manifest] saved(train)     : {train_out_path}")
    print(f"[manifest] saved(val)       : {val_out_path}")


if __name__ == "__main__":
    main()
