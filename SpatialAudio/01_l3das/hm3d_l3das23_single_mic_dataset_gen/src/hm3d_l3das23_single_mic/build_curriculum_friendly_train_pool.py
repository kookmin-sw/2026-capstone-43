from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .build_val_test_from_unused import (  # type: ignore
        EIGHT_WAY_ORDER,
        Candidate,
        build_rows,
        collect_candidates,
        dump_jsonl,
        ensure_uploaded_assets,
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
    )


LOGGER = logging.getLogger(__name__)

CONDITION_ORDER = ("FOV|gLOS", "FOV|gNLOS", "OOF|gLOS", "OOF|gNLOS")

# Chosen to minimize worst-stage duplicate pressure while staying within
# currently available source counts and keeping stage ratios compatible.
DEFAULT_POOL_RATIOS = {
    "FOV|gLOS": 0.35,
    "FOV|gNLOS": 0.20,
    "OOF|gLOS": 0.20,
    "OOF|gNLOS": 0.25,
}


def _condition_key(candidate: Candidate) -> str:
    return f"{candidate.fov_tag}|{candidate.geometry_los}"


def _round_condition_targets(ratios: dict[str, float], total_rows: int) -> dict[str, int]:
    raw = {key: ratios[key] * total_rows for key in CONDITION_ORDER}
    rounded = {key: int(raw[key]) for key in CONDITION_ORDER}
    remainder = int(total_rows - sum(rounded.values()))
    if remainder > 0:
        for _, key in sorted(
            ((raw[key] - rounded[key], key) for key in CONDITION_ORDER),
            reverse=True,
        )[:remainder]:
            rounded[key] += 1
    return rounded


def _choose_next_candidate(
    *,
    candidates: list[Candidate],
    used_ids: set[str],
    global_direction_counts: Counter[str],
    global_scene_counts: Counter[str],
    global_mic_counts: Counter[str],
    condition_direction_counts: Counter[str],
    condition_scene_counts: Counter[str],
    condition_mic_counts: Counter[str],
) -> Candidate | None:
    eligible = [candidate for candidate in candidates if candidate.unique_id not in used_ids]
    if not eligible:
        return None

    return min(
        eligible,
        key=lambda candidate: (
            global_direction_counts[candidate.direction_8way],
            condition_direction_counts[candidate.direction_8way],
            global_scene_counts[candidate.scene_id],
            condition_scene_counts[candidate.scene_id],
            global_mic_counts[candidate.mic_uid],
            condition_mic_counts[candidate.mic_uid],
            candidate.unique_id,
        ),
    )


def build_train_pool(
    *,
    grouped_candidates: dict[tuple[str, str], list[Candidate]],
    target_rows: int,
    pool_ratios: dict[str, float],
) -> tuple[list[Candidate], dict[str, Any]]:
    condition_targets = _round_condition_targets(pool_ratios, target_rows)
    condition_groups: dict[str, list[Candidate]] = defaultdict(list)
    direction_condition_counts: Counter[str] = Counter()
    for values in grouped_candidates.values():
        for candidate in values:
            key = _condition_key(candidate)
            condition_groups[key].append(candidate)
            direction_condition_counts[f"{candidate.direction_8way}|{key}"] += 1

    availability = {key: len(condition_groups.get(key, ())) for key in CONDITION_ORDER}
    missing = {
        key: {"available": availability[key], "target": condition_targets[key]}
        for key in CONDITION_ORDER
        if availability[key] < condition_targets[key]
    }
    if missing:
        raise RuntimeError(
            "Not enough candidates to build requested train pool without replacement. "
            f"Shortages: {missing}"
        )

    selected: list[Candidate] = []
    used_ids: set[str] = set()
    global_direction_counts: Counter[str] = Counter()
    global_scene_counts: Counter[str] = Counter()
    global_mic_counts: Counter[str] = Counter()
    per_condition_stats: dict[str, dict[str, int]] = {}

    for condition in CONDITION_ORDER:
        candidates = list(condition_groups[condition])
        condition_direction_counts: Counter[str] = Counter()
        condition_scene_counts: Counter[str] = Counter()
        condition_mic_counts: Counter[str] = Counter()
        condition_selected: list[Candidate] = []

        while len(condition_selected) < condition_targets[condition]:
            candidate = _choose_next_candidate(
                candidates=candidates,
                used_ids=used_ids,
                global_direction_counts=global_direction_counts,
                global_scene_counts=global_scene_counts,
                global_mic_counts=global_mic_counts,
                condition_direction_counts=condition_direction_counts,
                condition_scene_counts=condition_scene_counts,
                condition_mic_counts=condition_mic_counts,
            )
            if candidate is None:
                raise RuntimeError(
                    f"Could not fill condition {condition}: "
                    f"selected={len(condition_selected)} target={condition_targets[condition]}"
                )

            condition_selected.append(candidate)
            selected.append(candidate)
            used_ids.add(candidate.unique_id)
            global_direction_counts[candidate.direction_8way] += 1
            global_scene_counts[candidate.scene_id] += 1
            global_mic_counts[candidate.mic_uid] += 1
            condition_direction_counts[candidate.direction_8way] += 1
            condition_scene_counts[candidate.scene_id] += 1
            condition_mic_counts[candidate.mic_uid] += 1

        per_condition_stats[condition] = {
            "selected": len(condition_selected),
            "unique_scenes": len(condition_scene_counts),
            "unique_mics": len(condition_mic_counts),
            "max_scene_concentration": max(condition_scene_counts.values()) if condition_scene_counts else 0,
            "max_mic_concentration": max(condition_mic_counts.values()) if condition_mic_counts else 0,
        }

    selected.sort(
        key=lambda candidate: (
            CONDITION_ORDER.index(_condition_key(candidate)),
            EIGHT_WAY_ORDER.index(candidate.direction_8way),
            candidate.scene_id,
            candidate.mic_id,
            candidate.sample_id,
        )
    )

    summary = {
        "target_rows": int(target_rows),
        "pool_ratios": {key: float(pool_ratios[key]) for key in CONDITION_ORDER},
        "condition_targets": condition_targets,
        "condition_availability": availability,
        "direction_condition_availability": {
            key: int(value) for key, value in sorted(direction_condition_counts.items())
        },
        "global_unique_scenes": len(global_scene_counts),
        "global_unique_mics": len(global_mic_counts),
        "global_max_scene_concentration": max(global_scene_counts.values()) if global_scene_counts else 0,
        "global_max_mic_concentration": max(global_mic_counts.values()) if global_mic_counts else 0,
        "condition_selection_stats": per_condition_stats,
    }
    return selected, summary


def regroup_candidates(candidates: list[Candidate]) -> dict[tuple[str, str], list[Candidate]]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.direction_8way, candidate.geometry_los)].append(candidate)
    return grouped


def summarize_master_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direction_counts = Counter(row["direction_8way"] for row in rows)
    geometry_counts = Counter(row["geometry_los"] for row in rows)
    fov_counts = Counter(row["fov_tag"] for row in rows)
    condition_counts = Counter(f"{row['fov_tag']}|{row['geometry_los']}" for row in rows)
    unique_scene_ids = {row["scene_id"] for row in rows}
    unique_mic_pairs = {f"{row['scene_id']}|{row['mic_id']}" for row in rows}
    return {
        "total_rows": int(len(rows)),
        "unique_sample_ids": int(len({row['sample_id'] for row in rows})),
        "direction_counts": {key: int(direction_counts[key]) for key in EIGHT_WAY_ORDER if direction_counts[key] > 0},
        "geometry_counts": {key: int(geometry_counts[key]) for key in ("gLOS", "gNLOS") if geometry_counts[key] > 0},
        "fov_counts": {key: int(fov_counts[key]) for key in ("FOV", "OOF") if fov_counts[key] > 0},
        "condition_counts": {key: int(condition_counts[key]) for key in CONDITION_ORDER if condition_counts[key] > 0},
        "unique_scenes": int(len(unique_scene_ids)),
        "unique_mic_pairs": int(len(unique_mic_pairs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a curriculum-friendly train pool and aligned end-to-end manifests from existing HM3D "
            "generated candidate pools."
        )
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Source candidate pool roots, e.g. hm3d_glos_8way_pool_3200_diverse hm3d_gnlos_pool_4000_diverse",
    )
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=1800)
    parser.add_argument(
        "--end-to-end-rows",
        type=int,
        default=None,
        help="Optional row count for the end-to-end train manifest. Defaults to the full pool size.",
    )
    parser.add_argument(
        "--end-to-end-ratios",
        type=str,
        default="",
        help=(
            "Optional ratios for the end-to-end subset. If omitted, the full pool is used as end-to-end. "
            "Example: FOV|gLOS=0.325,FOV|gNLOS=0.225,OOF|gLOS=0.20,OOF|gNLOS=0.25"
        ),
    )
    parser.add_argument(
        "--pool-ratios",
        type=str,
        default="FOV|gLOS=0.35,FOV|gNLOS=0.20,OOF|gLOS=0.20,OOF|gNLOS=0.25",
        help="Comma-separated ratios for the new train pool.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_ratio_string(text: str) -> dict[str, float]:
    ratios = dict(DEFAULT_POOL_RATIOS)
    if not text.strip():
        return ratios
    parsed: dict[str, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid ratio token: {token}")
        key, value = token.split("=", 1)
        key = key.strip()
        if key not in CONDITION_ORDER:
            raise ValueError(f"Unknown condition in pool ratios: {key}")
        parsed[key] = float(value)
    if set(parsed) != set(CONDITION_ORDER):
        missing = sorted(set(CONDITION_ORDER) - set(parsed))
        raise ValueError(f"Missing pool ratio keys: {missing}")
    total = sum(parsed.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Pool ratios must sum to 1.0, got {total}")
    return parsed


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    dataset_roots = [path.resolve() for path in args.dataset_roots]
    upload_root = args.upload_root.resolve()
    manifests_dir = upload_root / "manifests"
    ratios = parse_ratio_string(str(args.pool_ratios))
    end_to_end_rows = int(args.end_to_end_rows) if args.end_to_end_rows is not None else int(args.target_rows)
    end_to_end_ratio_text = str(args.end_to_end_ratios).strip()
    end_to_end_ratios = parse_ratio_string(end_to_end_ratio_text) if end_to_end_ratio_text else None

    grouped_candidates, root_counts, bucket_counts = collect_candidates(
        dataset_roots=dataset_roots,
        labels=set(EIGHT_WAY_ORDER),
    )
    LOGGER.info("Collected roots=%s", root_counts)
    LOGGER.info("Direction x geometry counts=%s", dict(sorted(bucket_counts.items())))

    selected, selection_summary = build_train_pool(
        grouped_candidates=grouped_candidates,
        target_rows=int(args.target_rows),
        pool_ratios=ratios,
    )

    selected_for_end_to_end = list(selected)
    end_to_end_summary: dict[str, Any] = {
        "mode": "full_pool",
        "target_rows": int(len(selected)),
    }
    if end_to_end_rows != int(args.target_rows) or end_to_end_ratios is not None:
        subset_grouped = regroup_candidates(selected)
        subset_ratios = end_to_end_ratios or ratios
        selected_for_end_to_end, subset_summary = build_train_pool(
            grouped_candidates=subset_grouped,
            target_rows=end_to_end_rows,
            pool_ratios=subset_ratios,
        )
        end_to_end_summary = {
            "mode": "subset",
            "target_rows": int(end_to_end_rows),
            "ratios": {key: float(subset_ratios[key]) for key in CONDITION_ORDER},
            "selection": subset_summary,
        }

    id_to_paths = ensure_uploaded_assets(selected, upload_root=upload_root)
    rows_master, _, _ = build_rows(selected, id_to_paths, split_name="train")
    rows_end_to_end_master, rows_audio, rows_av = build_rows(
        selected_for_end_to_end,
        id_to_paths,
        split_name="train",
    )

    master_path = manifests_dir / "00_train_pool_master.jsonl"
    end_to_end_master_path = manifests_dir / "00_train_end_to_end_master.jsonl"
    audio_path = manifests_dir / "00_train_end_to_end_audio_only.jsonl"
    av_path = manifests_dir / "00_train_end_to_end_av_messages.jsonl"

    dump_jsonl(master_path, rows_master)
    dump_jsonl(end_to_end_master_path, rows_end_to_end_master)
    dump_jsonl(audio_path, rows_audio)
    dump_jsonl(av_path, rows_av)

    summary = {
        "dataset_roots": {str(path): int(root_counts.get(str(path), 0)) for path in dataset_roots},
        "target_rows": int(args.target_rows),
        "pool_selection": selection_summary,
        "train_pool_summary": summarize_master_rows(rows_master),
        "end_to_end_summary": summarize_master_rows(rows_end_to_end_master),
        "end_to_end_selection": end_to_end_summary,
        "outputs": {
            "master": str(master_path),
            "end_to_end_master": str(end_to_end_master_path),
            "audio_only": str(audio_path),
            "av_messages": str(av_path),
            "upload_root": str(upload_root),
        },
    }
    summary_path = manifests_dir / "00_train_pool_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
