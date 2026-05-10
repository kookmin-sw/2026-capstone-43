from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

try:
    from .build_val_test_from_unused import (  # type: ignore
        EIGHT_WAY_ORDER,
        Candidate,
        build_rows,
        collect_candidates,
        dump_jsonl,
        ensure_uploaded_assets,
        load_excluded_ids,
    )
except ImportError:
    CURRENT_FILE = Path(__file__).resolve()
    PACKAGE_ROOT = CURRENT_FILE.parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from hm3d_l3das23_single_mic.build_val_test_from_unused import (  # type: ignore
        EIGHT_WAY_ORDER,
        Candidate,
        build_rows,
        collect_candidates,
        dump_jsonl,
        ensure_uploaded_assets,
        load_excluded_ids,
    )


def choose_next(
    candidates: list[Candidate],
    used_ids: set[str],
    split_scene_counts: Counter[str],
    split_mic_counts: Counter[str],
    split_geometry_counts: Counter[str],
    label_scene_counts: Counter[str],
    label_mic_counts: Counter[str],
    label_geometry_counts: Counter[str],
    rng: random.Random,
) -> Optional[Candidate]:
    eligible = [c for c in candidates if c.unique_id not in used_ids]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda c: (
            split_scene_counts[c.scene_id],
            label_scene_counts[c.scene_id],
            split_mic_counts[c.mic_uid],
            label_mic_counts[c.mic_uid],
            split_geometry_counts[c.geometry_los],
            label_geometry_counts[c.geometry_los],
            rng.random(),
        ),
    )


def select_direction_balanced_samples(
    candidates_by_direction: dict[str, list[Candidate]],
    labels: list[str],
    target_per_label: int,
    used_ids: set[str],
    seed: int,
) -> list[Candidate]:
    rng = random.Random(seed)
    split_scene_counts: Counter[str] = Counter()
    split_mic_counts: Counter[str] = Counter()
    split_geometry_counts: Counter[str] = Counter()
    selected: list[Candidate] = []

    for direction in labels:
        bucket = list(candidates_by_direction.get(direction, []))
        rng.shuffle(bucket)
        label_scene_counts: Counter[str] = Counter()
        label_mic_counts: Counter[str] = Counter()
        label_geometry_counts: Counter[str] = Counter()
        label_selected = 0
        while label_selected < target_per_label:
            cand = choose_next(
                candidates=bucket,
                used_ids=used_ids,
                split_scene_counts=split_scene_counts,
                split_mic_counts=split_mic_counts,
                split_geometry_counts=split_geometry_counts,
                label_scene_counts=label_scene_counts,
                label_mic_counts=label_mic_counts,
                label_geometry_counts=label_geometry_counts,
                rng=rng,
            )
            if cand is None:
                raise RuntimeError(
                    f"Could not fill direction bucket {direction}: "
                    f"{label_selected} / {target_per_label}"
                )
            selected.append(cand)
            used_ids.add(cand.unique_id)
            split_scene_counts[cand.scene_id] += 1
            split_mic_counts[cand.mic_uid] += 1
            split_geometry_counts[cand.geometry_los] += 1
            label_scene_counts[cand.scene_id] += 1
            label_mic_counts[cand.mic_uid] += 1
            label_geometry_counts[cand.geometry_los] += 1
            label_selected += 1

    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a direction-balanced validation set from unused HM3D pool samples "
            "(balanced over 8 directions only, not by geometry)."
        )
    )
    parser.add_argument("--dataset-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--out-manifest-dir", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--labels", type=str, default=",".join(EIGHT_WAY_ORDER))
    args = parser.parse_args()

    labels = [item.strip() for item in str(args.labels).split(",") if item.strip()]
    for label in labels:
        if label not in EIGHT_WAY_ORDER:
            raise ValueError(f"Unknown label in --labels: {label}")

    grouped, root_counts, bucket_counts = collect_candidates(
        dataset_roots=[path.resolve() for path in args.dataset_roots],
        labels=set(labels),
    )
    excluded = load_excluded_ids([path.resolve() for path in args.exclude_manifests])

    by_direction: dict[str, list[Candidate]] = defaultdict(list)
    for (direction, _geometry), values in grouped.items():
        by_direction[direction].extend(
            [value for value in values if value.unique_id not in excluded]
        )

    shortage: dict[str, int] = {}
    for direction in labels:
        available = len(by_direction.get(direction, []))
        if available < int(args.per_label):
            shortage[direction] = available
    if shortage:
        raise RuntimeError(
            "Not enough unused candidates for direction-balanced val set. "
            f"Need {int(args.per_label)} per direction, got {shortage}"
        )

    used_ids = set(excluded)
    selected = select_direction_balanced_samples(
        candidates_by_direction=by_direction,
        labels=labels,
        target_per_label=int(args.per_label),
        used_ids=used_ids,
        seed=int(args.seed),
    )

    id_to_paths = ensure_uploaded_assets(selected, upload_root=args.upload_root.resolve())
    rows_master, rows_audio, rows_av = build_rows(selected, id_to_paths, split_name="val")

    out_dir = args.out_manifest_dir.resolve()
    dump_jsonl(out_dir / "val_master_direction_8way_400_unused.jsonl", rows_master)
    dump_jsonl(out_dir / "val_audio_only_direction_8way_400_unused.jsonl", rows_audio)
    dump_jsonl(out_dir / "val_av_messages_direction_8way_400_unused.jsonl", rows_av)

    direction_counts = Counter(row["direction_8way"] for row in rows_master)
    geometry_counts = Counter(row["geometry_los"] for row in rows_master)
    fov_counts = Counter(row["fov_tag"] for row in rows_master)
    summary = {
        "dataset_roots": root_counts,
        "raw_bucket_counts_before_exclude": dict(sorted(bucket_counts.items())),
        "excluded_id_count": len(excluded),
        "per_label": int(args.per_label),
        "total_rows": len(rows_master),
        "direction_counts": dict(sorted(direction_counts.items())),
        "geometry_counts": dict(sorted(geometry_counts.items())),
        "fov_counts": dict(sorted(fov_counts.items())),
        "summary_paths": {
            "master": str(out_dir / "val_master_direction_8way_400_unused.jsonl"),
            "audio_only": str(out_dir / "val_audio_only_direction_8way_400_unused.jsonl"),
            "av_messages": str(out_dir / "val_av_messages_direction_8way_400_unused.jsonl"),
        },
    }
    summary_path = out_dir / "val_direction_8way_400_unused_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
