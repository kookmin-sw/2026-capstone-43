from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .manifest_io import (
    iter_dataset_rows,
    output_relpath,
    row_azimuth_deg,
    row_in_fov,
    row_visibility_ratio,
)
from .spatial_conventions import (
    EIGHT_WAY_ORDER,
    direction_8way_from_azimuth as azimuth_to_eight_way_label,
    normalize_azimuth_deg,
)


GEOMETRY_ORDER = ("gLOS", "gNLOS")

QUESTION_8WAY = (
    "<audio>\n<image>\n"
    "What is the speaker direction in the camera-centered view? "
    "Choose one of: front-left, front, front-right, right, back-right, back, back-left, left."
)

MIC_TOKEN_RE = re.compile(r"(mic\d+)")


@dataclass(frozen=True)
class Candidate:
    unique_id: str
    sample_id: str
    scene_id: str
    mic_id: str
    mic_uid: str
    audio_src: Path
    image_src: Path
    direction_8way: str
    geometry_los: str
    in_fov: bool
    fov_tag: str
    projected_pixel_xy: Optional[list[float]]
    projection_depth_cam: Optional[float]
    projection_reason: Optional[str]
    visibility_ratio: Optional[float]

def parse_mic_id(sample_id: str) -> str:
    for token in sample_id.split("__"):
        if token.startswith("mic"):
            return token
    match = MIC_TOKEN_RE.search(sample_id)
    if match:
        return match.group(1)
    return "mic_unknown"


def parse_optional_xy(value: object) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if value[0] is None or value[1] is None:
        return None
    return [float(value[0]), float(value[1])]


def parse_optional_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_candidates(
    dataset_roots: list[Path],
    labels: set[str],
) -> tuple[dict[tuple[str, str], list[Candidate]], dict[str, int], Counter[str]]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    root_counts: dict[str, int] = {}
    bucket_counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for root in dataset_roots:
        accepted = 0
        for data in iter_dataset_rows(root):

            geometry = str(data.get("geometry_los", "")).strip()
            if geometry not in GEOMETRY_ORDER:
                continue

            sample_id = str(data.get("sample_id", "")).strip()
            scene_id = str(data.get("scene_id", "")).strip()
            if not sample_id or not scene_id:
                continue

            dedupe_key = (scene_id, sample_id, geometry)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            direction = azimuth_to_eight_way_label(row_azimuth_deg(data))
            if direction not in labels:
                continue

            audio_rel = output_relpath(data, "audio_mic_wav", "foa_audio_path")
            image_rel = output_relpath(data, "rgb_front_png", "rgb_image_path")
            if not audio_rel or not image_rel:
                continue

            audio_src = (root / audio_rel).resolve()
            image_src = (root / image_rel).resolve()
            if not audio_src.exists() or not image_src.exists():
                continue

            mic_id = parse_mic_id(sample_id)
            mic_uid = f"{scene_id}:{mic_id}"
            in_fov = row_in_fov(data)
            fov_tag = "FOV" if in_fov else "OOF"

            candidate = Candidate(
                unique_id=f"{sample_id}__{geometry}",
                sample_id=sample_id,
                scene_id=scene_id,
                mic_id=mic_id,
                mic_uid=mic_uid,
                audio_src=audio_src,
                image_src=image_src,
                direction_8way=direction,
                geometry_los=geometry,
                in_fov=in_fov,
                fov_tag=fov_tag,
                projected_pixel_xy=parse_optional_xy(data.get("projected_pixel_xy")),
                projection_depth_cam=parse_optional_float(data.get("projection_depth_cam")),
                projection_reason=(
                    str(data.get("projection_reason"))
                    if data.get("projection_reason") is not None
                    else None
                ),
                visibility_ratio=row_visibility_ratio(data),
            )
            grouped[(direction, geometry)].append(candidate)
            bucket_counts[f"{direction}|{geometry}"] += 1
            accepted += 1

        root_counts[str(root)] = accepted
    return grouped, root_counts, bucket_counts


def load_excluded_ids(manifest_paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in manifest_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = str(row.get("id", "")).strip()
                if sample_id:
                    excluded.add(sample_id)
    return excluded


def choose_next(
    candidates: list[Candidate],
    used_ids: set[str],
    split_scene_counts: Counter[str],
    split_mic_counts: Counter[str],
    bucket_scene_counts: Counter[str],
    bucket_mic_counts: Counter[str],
    rng: random.Random,
) -> Optional[Candidate]:
    eligible = [c for c in candidates if c.unique_id not in used_ids]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda c: (
            split_scene_counts[c.scene_id],
            bucket_scene_counts[c.scene_id],
            split_mic_counts[c.mic_uid],
            bucket_mic_counts[c.mic_uid],
            rng.random(),
        ),
    )


def select_split_samples(
    grouped_candidates: dict[tuple[str, str], list[Candidate]],
    labels: list[str],
    target_per_bucket: int,
    used_ids: set[str],
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    for values in grouped_candidates.values():
        rng.shuffle(values)

    split_scene_counts: Counter[str] = Counter()
    split_mic_counts: Counter[str] = Counter()
    selected: list[Candidate] = []

    for direction in labels:
        for geometry in GEOMETRY_ORDER:
            bucket = grouped_candidates.get((direction, geometry), [])
            bucket_selected: list[Candidate] = []
            bucket_scene_counts: Counter[str] = Counter()
            bucket_mic_counts: Counter[str] = Counter()

            while len(bucket_selected) < target_per_bucket:
                cand = choose_next(
                    candidates=bucket,
                    used_ids=used_ids,
                    split_scene_counts=split_scene_counts,
                    split_mic_counts=split_mic_counts,
                    bucket_scene_counts=bucket_scene_counts,
                    bucket_mic_counts=bucket_mic_counts,
                    rng=rng,
                )
                if cand is None:
                    break

                bucket_selected.append(cand)
                selected.append(cand)
                used_ids.add(cand.unique_id)
                split_scene_counts[cand.scene_id] += 1
                split_mic_counts[cand.mic_uid] += 1
                bucket_scene_counts[cand.scene_id] += 1
                bucket_mic_counts[cand.mic_uid] += 1

            if len(bucket_selected) != target_per_bucket:
                raise RuntimeError(
                    f"Could not fill bucket {direction}|{geometry}: "
                    f"{len(bucket_selected)} / {target_per_bucket}"
                )

    rng.shuffle(selected)
    return selected


def ensure_uploaded_assets(
    selected: list[Candidate],
    upload_root: Path,
) -> dict[str, dict[str, str]]:
    audio_dir = upload_root / "audio"
    image_dir = upload_root / "image"
    audio_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    id_to_paths: dict[str, dict[str, str]] = {}
    for c in selected:
        audio_ext = c.audio_src.suffix.lower() or ".wav"
        image_ext = c.image_src.suffix.lower() or ".png"
        audio_rel = f"audio/{c.unique_id}{audio_ext}"
        image_rel = f"image/{c.unique_id}{image_ext}"
        audio_dst = upload_root / audio_rel
        image_dst = upload_root / image_rel

        if not audio_dst.exists():
            shutil.copy2(c.audio_src, audio_dst)
        if not image_dst.exists():
            shutil.copy2(c.image_src, image_dst)

        id_to_paths[c.unique_id] = {"audio_path": audio_rel, "image_path": image_rel}
    return id_to_paths


def build_rows(
    selected: list[Candidate],
    id_to_paths: dict[str, dict[str, str]],
    split_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows_master: list[dict[str, object]] = []
    rows_audio: list[dict[str, object]] = []
    rows_av: list[dict[str, object]] = []

    for c in selected:
        mapped = id_to_paths[c.unique_id]
        row_master = {
            "id": c.unique_id,
            "sample_id": c.sample_id,
            "scene_id": c.scene_id,
            "mic_id": c.mic_id,
            "mic_uid": c.mic_uid,
            "split": split_name,
            "audio_path": mapped["audio_path"],
            "image_path": mapped["image_path"],
            "label": c.direction_8way,
            "direction_8way": c.direction_8way,
            "geometry_los": c.geometry_los,
            "fov_tag": c.fov_tag,
            "in_fov": c.in_fov,
            "projected_pixel_xy": c.projected_pixel_xy,
            "projection_depth_cam": c.projection_depth_cam,
            "projection_reason": c.projection_reason,
            "visibility_ratio": c.visibility_ratio,
        }
        rows_master.append(row_master)

        rows_audio.append(
            {
                "id": c.unique_id,
                "audio_path": mapped["audio_path"],
                "label": c.direction_8way,
                "scene_id": c.scene_id,
                "mic_id": c.mic_id,
                "fov_tag": c.fov_tag,
                "in_fov": c.in_fov,
                "geometry_los": c.geometry_los,
            }
        )

        rows_av.append(
            {
                "id": c.unique_id,
                "audio_path": mapped["audio_path"],
                "image_path": mapped["image_path"],
                "geometry_los": c.geometry_los,
                "fov_tag": c.fov_tag,
                "in_fov": c.in_fov,
                "task": "direction_classification",
                "messages": [
                    {"role": "user", "content": QUESTION_8WAY},
                    {"role": "assistant", "content": c.direction_8way},
                ],
            }
        )

    return rows_master, rows_audio, rows_av


def dump_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build val/test sets from unused HM3D pool samples (excluding train manifest ids), "
            "with balanced 8-way x gLOS/gNLOS buckets."
        )
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Pool roots, e.g., hm3d_glos_8way_pool_3200_diverse hm3d_gnlos_pool_4000_diverse",
    )
    parser.add_argument(
        "--exclude-manifests",
        type=Path,
        nargs="+",
        required=True,
        help="Manifest(s) whose ids must be excluded (e.g., train 2400 manifest).",
    )
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--out-manifest-dir", type=Path, required=True)
    parser.add_argument("--per-split-total", type=int, default=400)
    parser.add_argument("--target-per-bucket", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--labels", type=str, default=",".join(EIGHT_WAY_ORDER))
    args = parser.parse_args()

    labels = [item.strip() for item in str(args.labels).split(",") if item.strip()]
    if not labels:
        raise ValueError("No valid labels provided.")
    for label in labels:
        if label not in EIGHT_WAY_ORDER:
            raise ValueError(f"Unknown label in --labels: {label}")

    num_buckets = len(labels) * len(GEOMETRY_ORDER)
    if args.target_per_bucket > 0:
        target_per_bucket = int(args.target_per_bucket)
    else:
        if args.per_split_total % num_buckets != 0:
            raise ValueError(
                f"--per-split-total={args.per_split_total} must be divisible by "
                f"{num_buckets} buckets (labels x geometry)."
            )
        target_per_bucket = args.per_split_total // num_buckets

    grouped, root_counts, bucket_counts = collect_candidates(
        dataset_roots=[path.resolve() for path in args.dataset_roots],
        labels=set(labels),
    )
    excluded = load_excluded_ids([path.resolve() for path in args.exclude_manifests])

    # Remove excluded ids from grouped candidates first.
    for key, values in grouped.items():
        grouped[key] = [v for v in values if v.unique_id not in excluded]

    needed_per_bucket_both_splits = target_per_bucket * 2
    shortage: dict[str, int] = {}
    for direction in labels:
        for geometry in GEOMETRY_ORDER:
            available = len(grouped.get((direction, geometry), []))
            if available < needed_per_bucket_both_splits:
                shortage[f"{direction}|{geometry}"] = available
    if shortage:
        raise RuntimeError(
            "Not enough unused candidates for val+test. "
            f"Need {needed_per_bucket_both_splits} each bucket, got {shortage}"
        )

    used_ids = set(excluded)
    selected_val = select_split_samples(
        grouped_candidates=grouped,
        labels=labels,
        target_per_bucket=target_per_bucket,
        used_ids=used_ids,
        seed=int(args.seed),
    )
    selected_test = select_split_samples(
        grouped_candidates=grouped,
        labels=labels,
        target_per_bucket=target_per_bucket,
        used_ids=used_ids,
        seed=int(args.seed) + 1,
    )

    selected_all = selected_val + selected_test
    id_to_paths = ensure_uploaded_assets(selected_all, upload_root=args.upload_root.resolve())

    val_master, val_audio, val_av = build_rows(selected_val, id_to_paths, split_name="val")
    test_master, test_audio, test_av = build_rows(selected_test, id_to_paths, split_name="test")

    out_dir = args.out_manifest_dir.resolve()
    dump_jsonl(out_dir / "val_master_glos_gnlos_8way_400_unused.jsonl", val_master)
    dump_jsonl(out_dir / "val_audio_only_glos_gnlos_8way_400_unused.jsonl", val_audio)
    dump_jsonl(out_dir / "val_av_messages_glos_gnlos_8way_400_unused.jsonl", val_av)
    dump_jsonl(out_dir / "test_master_glos_gnlos_8way_400_unused.jsonl", test_master)
    dump_jsonl(out_dir / "test_audio_only_glos_gnlos_8way_400_unused.jsonl", test_audio)
    dump_jsonl(out_dir / "test_av_messages_glos_gnlos_8way_400_unused.jsonl", test_av)

    summary = {
        "dataset_roots": root_counts,
        "raw_bucket_counts_before_exclude": dict(sorted(bucket_counts.items())),
        "excluded_id_count": len(excluded),
        "per_split_total": int(args.per_split_total),
        "target_per_bucket": int(target_per_bucket),
        "val_size": len(val_master),
        "test_size": len(test_master),
        "val_bucket_counts": dict(
            sorted(
                Counter(f"{r['direction_8way']}|{r['geometry_los']}" for r in val_master).items()
            )
        ),
        "test_bucket_counts": dict(
            sorted(
                Counter(f"{r['direction_8way']}|{r['geometry_los']}" for r in test_master).items()
            )
        ),
    }
    summary_path = out_dir / "val_test_unused_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
