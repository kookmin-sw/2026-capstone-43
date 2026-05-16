from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .build_val_test_from_unused import QUESTION_8WAY, EIGHT_WAY_ORDER, dump_jsonl  # type: ignore
except ImportError:
    import sys

    CURRENT_FILE = Path(__file__).resolve()
    PACKAGE_ROOT = CURRENT_FILE.parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from hm3d_l3das23_single_mic.build_val_test_from_unused import (  # type: ignore
        QUESTION_8WAY,
        EIGHT_WAY_ORDER,
        dump_jsonl,
    )


LOGGER = logging.getLogger(__name__)

FRONT_DIRS = ("front", "front-right", "front-left")
REAR_DIRS = ("right", "back-right", "back", "back-left", "left")
CONDITION_ORDER = ("FOV|gLOS", "FOV|gNLOS", "OOF|gLOS", "OOF|gNLOS")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    direction = str(normalized.get("direction_8way") or normalized.get("label") or "").strip()
    if not direction:
        raise ValueError(f"Missing direction_8way/label in row: {row}")
    scene_id = str(normalized.get("scene_id") or "").strip()
    mic_id = str(normalized.get("mic_id") or "").strip()
    sample_id = str(normalized.get("sample_id") or normalized.get("id") or "").strip()
    if not sample_id:
        raise ValueError(f"Missing sample_id/id in row: {row}")
    if not scene_id:
        scene_id = sample_id.split("__", 1)[0]
    if not mic_id:
        mic_id = next((token for token in sample_id.split("__") if token.startswith("mic")), "mic_unknown")
    normalized["scene_id"] = scene_id
    normalized["mic_id"] = mic_id
    normalized["sample_id"] = sample_id
    normalized["id"] = str(normalized.get("id") or sample_id).strip()
    normalized["mic_uid"] = str(normalized.get("mic_uid") or f"{scene_id}:{mic_id}")
    normalized["direction_8way"] = direction
    normalized["label"] = str(normalized.get("label") or direction)
    normalized["fov_tag"] = "FOV" if bool(normalized.get("in_fov")) or str(normalized.get("fov_tag", "")).upper() == "FOV" else "OOF"
    normalized["geometry_los"] = str(normalized.get("geometry_los") or "").strip()
    if normalized["geometry_los"] not in {"gLOS", "gNLOS"}:
        raise ValueError(f"Unsupported geometry_los in row: {row}")
    normalized["in_fov"] = normalized["fov_tag"] == "FOV"
    if "audio_path" not in normalized or "image_path" not in normalized:
        raise ValueError(f"Missing audio/image path in row: {row}")
    return normalized


def condition_key(row: dict[str, Any]) -> str:
    return f"{row['fov_tag']}|{row['geometry_los']}"


def build_message_row(row: dict[str, Any], split_name: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "sample_id": row["sample_id"],
        "scene_id": row["scene_id"],
        "mic_id": row["mic_id"],
        "mic_uid": row["mic_uid"],
        "split": split_name,
        "audio_path": row["audio_path"],
        "image_path": row["image_path"],
        "label": row["label"],
        "direction_8way": row["direction_8way"],
        "geometry_los": row["geometry_los"],
        "fov_tag": row["fov_tag"],
        "in_fov": row["in_fov"],
        "projected_pixel_xy": row.get("projected_pixel_xy"),
        "projection_depth_cam": row.get("projection_depth_cam"),
        "projection_reason": row.get("projection_reason"),
        "visibility_ratio": row.get("visibility_ratio"),
        "task": "direction_classification",
        "messages": [
            {"role": "user", "content": QUESTION_8WAY},
            {"role": "assistant", "content": row["direction_8way"]},
        ],
    }


def score_row(
    row: dict[str, Any],
    *,
    global_direction_counts: Counter[str],
    global_scene_counts: Counter[str],
    global_mic_counts: Counter[str],
    local_direction_counts: Counter[str],
    local_scene_counts: Counter[str],
    local_mic_counts: Counter[str],
) -> tuple[Any, ...]:
    direction = row["direction_8way"]
    scene = row["scene_id"]
    mic_uid = row["mic_uid"]
    return (
        int(local_direction_counts[direction]),
        int(global_direction_counts[direction]),
        int(local_scene_counts[scene]),
        int(global_scene_counts[scene]),
        int(local_mic_counts[mic_uid]),
        int(global_mic_counts[mic_uid]),
        row["sample_id"],
    )


def select_exact_targets(
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]],
    target_map: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    global_direction_counts: Counter[str] = Counter()
    global_scene_counts: Counter[str] = Counter()
    global_mic_counts: Counter[str] = Counter()

    for direction in EIGHT_WAY_ORDER:
        for condition in CONDITION_ORDER:
            target = int(target_map.get((direction, condition), 0))
            if target <= 0:
                continue
            bucket = list(grouped_rows.get((direction, condition), []))
            if len(bucket) < target:
                raise RuntimeError(
                    f"Bucket shortage for {direction}|{condition}: need {target}, have {len(bucket)}"
                )
            local_direction_counts: Counter[str] = Counter()
            local_scene_counts: Counter[str] = Counter()
            local_mic_counts: Counter[str] = Counter()
            for _ in range(target):
                eligible = [row for row in bucket if row["id"] not in used_ids]
                if not eligible:
                    raise RuntimeError(
                        f"Ran out of unique rows in {direction}|{condition} after selecting "
                        f"{target - len(eligible)} / {target}"
                    )
                best = min(
                    eligible,
                    key=lambda row: score_row(
                        row,
                        global_direction_counts=global_direction_counts,
                        global_scene_counts=global_scene_counts,
                        global_mic_counts=global_mic_counts,
                        local_direction_counts=local_direction_counts,
                        local_scene_counts=local_scene_counts,
                        local_mic_counts=local_mic_counts,
                    ),
                )
                selected.append(best)
                used_ids.add(best["id"])
                global_direction_counts[best["direction_8way"]] += 1
                global_scene_counts[best["scene_id"]] += 1
                global_mic_counts[best["mic_uid"]] += 1
                local_direction_counts[best["direction_8way"]] += 1
                local_scene_counts[best["scene_id"]] += 1
                local_mic_counts[best["mic_uid"]] += 1
    return selected


def stats_for_rows(rows: list[dict[str, Any]], *, val_direction: list[float], val_condition: list[float]) -> dict[str, Any]:
    direction_counts = Counter(row["direction_8way"] for row in rows)
    fov_counts = Counter(row["fov_tag"] for row in rows)
    geometry_counts = Counter(row["geometry_los"] for row in rows)
    condition_counts = Counter(condition_key(row) for row in rows)
    unique_scenes = {row["scene_id"] for row in rows}
    unique_mic_pairs = {row["mic_uid"] for row in rows}
    unique_ids = {row["id"] for row in rows}
    duplicates = len(rows) - len(unique_ids)
    return {
        "total_rows": int(len(rows)),
        "unique_sample_id": int(len(unique_ids)),
        "duplicates": int(duplicates),
        "direction_counts": {key: int(direction_counts[key]) for key in EIGHT_WAY_ORDER},
        "fov_counts": {key: int(fov_counts[key]) for key in ("FOV", "OOF")},
        "geometry_counts": {key: int(geometry_counts[key]) for key in ("gLOS", "gNLOS")},
        "condition_counts": {key: int(condition_counts[key]) for key in CONDITION_ORDER},
        "unique_scenes": int(len(unique_scenes)),
        "unique_mic_pairs": int(len(unique_mic_pairs)),
        "direction_js_vs_val": js_divergence(probabilities(direction_counts, EIGHT_WAY_ORDER), val_direction),
        "condition_js_vs_val": js_divergence(probabilities(condition_counts, CONDITION_ORDER), val_condition),
    }


def probabilities(counter: Counter[str] | dict[str, int], order: tuple[str, ...] | list[str]) -> list[float]:
    total = sum(int(counter.get(key, 0)) for key in order)
    if total <= 0:
        return [0.0 for _ in order]
    return [int(counter.get(key, 0)) / total for key in order]


def js_divergence(a: list[float], b: list[float]) -> float:
    midpoint = [(x + y) / 2.0 for x, y in zip(a, b)]

    def kl(p: list[float], q: list[float]) -> float:
        score = 0.0
        for x, y in zip(p, q):
            if x > 0 and y > 0:
                from math import log

                score += x * log(x / y, 2)
        return score

    return 0.5 * kl(a, midpoint) + 0.5 * kl(b, midpoint)


def schema_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    messages = first.get("messages", [])
    return {
        "keys": sorted(first.keys()),
        "task": first.get("task"),
        "message_roles": [message.get("role") for message in messages],
        "question": messages[0]["content"] if len(messages) >= 1 else None,
    }


def fixed_target_map() -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    front_fov_glos = {"front": 144, "front-right": 143, "front-left": 143}
    front_fov_gnlos = {"front": 156, "front-right": 157, "front-left": 157}
    for direction in FRONT_DIRS:
        targets[(direction, "FOV|gLOS")] = front_fov_glos[direction]
        targets[(direction, "FOV|gNLOS")] = front_fov_gnlos[direction]
    for direction in REAR_DIRS:
        targets[(direction, "OOF|gLOS")] = 152
        targets[(direction, "OOF|gNLOS")] = 148
    return targets


def source_pool_target_map() -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    for direction in FRONT_DIRS:
        targets[(direction, "FOV|gLOS")] = 400
        targets[(direction, "FOV|gNLOS")] = 200
    for direction in REAR_DIRS:
        targets[(direction, "OOF|gLOS")] = 168
        targets[(direction, "OOF|gNLOS")] = 192
    return targets


def changing_stage_target_maps() -> dict[str, dict[tuple[str, str], int]]:
    stage_maps: dict[str, dict[tuple[str, str], int]] = {}
    specs = {
        "01_train_stage1.jsonl": {
            "front": {"FOV|gLOS": 400, "FOV|gNLOS": 160},
            "rear": {"OOF|gLOS": 96, "OOF|gNLOS": 48},
        },
        "02_train_stage2.jsonl": {
            "front": {"FOV|gLOS": 280, "FOV|gNLOS": 200},
            "rear": {"OOF|gLOS": 96, "OOF|gNLOS": 96},
        },
        "03_train_stage3.jsonl": {
            "front": {"FOV|gLOS": 200, "FOV|gNLOS": 200},
            "rear": {"OOF|gLOS": 96, "OOF|gNLOS": 144},
        },
        "04_train_stage4.jsonl": {
            "front": {"FOV|gLOS": 160, "FOV|gNLOS": 160},
            "rear": {"OOF|gLOS": 96, "OOF|gNLOS": 192},
        },
    }
    for filename, spec in specs.items():
        target_map: dict[tuple[str, str], int] = {}
        for direction in FRONT_DIRS:
            for condition, value in spec["front"].items():
                target_map[(direction, condition)] = value
        for direction in REAR_DIRS:
            for condition, value in spec["rear"].items():
                target_map[(direction, condition)] = value
        stage_maps[filename] = target_map
    return stage_maps


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild curriculum comparison manifests with unified AV messages schema and "
            "direction-balanced fixed/end-to-end splits."
        )
    )
    parser.add_argument(
        "--source-train-pool-manifest",
        type=Path,
        default=Path("/home/yu/Project_git/01_dataset/hm3d_curriculum_compare_5000_upload/manifests/00_train_pool_master.jsonl"),
    )
    parser.add_argument(
        "--old-end-to-end-manifest",
        type=Path,
        default=Path("/home/yu/Project_git/01_dataset/99_archive/hm3d_master_8way_glos_gnlos_2700_upload/manifests/end_to_end_baseline/02_train_av_messages_8way_glos_gnlos_2700_end_to_end.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("/home/yu/Project_git/01_dataset/99_archive/hm3d_glos_gnlos_8way_val400_upload/manifests/val_av_messages_glos_gnlos_8way_400_upload.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/yu/Project_git/01_dataset/hm3d_curriculum_compare_5000_upload/manifests/rebalanced_messages_2400"),
    )
    parser.add_argument("--split-name", type=str, default="train")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    source_rows = [normalize_row(row) for row in read_jsonl(args.source_train_pool_manifest)]
    val_rows_raw = read_jsonl(args.val_manifest)
    old_rows_raw = read_jsonl(args.old_end_to_end_manifest)

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped_rows[(row["direction_8way"], condition_key(row))].append(row)

    val_direction = probabilities(
        Counter(message["content"] for row in val_rows_raw for message in row.get("messages", []) if message.get("role") == "assistant"),
        list(EIGHT_WAY_ORDER),
    )
    val_condition = probabilities(
        Counter(f"{row['fov_tag']}|{row['geometry_los']}" for row in val_rows_raw),
        list(CONDITION_ORDER),
    )

    pool_rows = [build_message_row(row, args.split_name) for row in select_exact_targets(grouped_rows, source_pool_target_map())]
    fixed_rows = [build_message_row(row, args.split_name) for row in select_exact_targets(grouped_rows, fixed_target_map())]

    changing_rows: dict[str, list[dict[str, Any]]] = {}
    for filename, target_map in changing_stage_target_maps().items():
        selected = select_exact_targets(grouped_rows, target_map)
        changing_rows[filename] = [build_message_row(row, args.split_name) for row in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir = args.output_dir / "fixed_ratio"
    changing_dir = args.output_dir / "changing_ratio"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    changing_dir.mkdir(parents=True, exist_ok=True)

    pool_path = args.output_dir / "00_train_pool_master_rebalanced_messages.jsonl"
    end_to_end_path = args.output_dir / "00_train_end_to_end_av_messages_rebalanced.jsonl"
    dump_jsonl(pool_path, pool_rows)
    dump_jsonl(end_to_end_path, fixed_rows)
    for filename in ("01_train_stage1.jsonl", "02_train_stage2.jsonl", "03_train_stage3.jsonl", "04_train_stage4.jsonl"):
        dump_jsonl(fixed_dir / filename, fixed_rows)
        dump_jsonl(changing_dir / filename, changing_rows[filename])

    pool_stats = stats_for_rows(pool_rows, val_direction=val_direction, val_condition=val_condition)
    end_stats = stats_for_rows(fixed_rows, val_direction=val_direction, val_condition=val_condition)
    fixed_stats = {
        filename: stats_for_rows(fixed_rows, val_direction=val_direction, val_condition=val_condition)
        for filename in ("01_train_stage1.jsonl", "02_train_stage2.jsonl", "03_train_stage3.jsonl", "04_train_stage4.jsonl")
    }
    changing_stats = {
        filename: stats_for_rows(rows, val_direction=val_direction, val_condition=val_condition)
        for filename, rows in changing_rows.items()
    }

    old_schema = {
        "keys": sorted(old_rows_raw[0].keys()),
        "task": old_rows_raw[0].get("task"),
        "message_roles": [message.get("role") for message in old_rows_raw[0].get("messages", [])],
        "question": old_rows_raw[0]["messages"][0]["content"],
    }
    schema_checks = {
        "old_end_to_end": old_schema,
        "new_rebalanced_end_to_end": schema_check(fixed_rows),
        "new_fixed_stage1": schema_check(fixed_rows),
        "new_changing_stage1": schema_check(changing_rows["01_train_stage1.jsonl"]),
    }

    fixed_direction_consistent = len({tuple(stats["direction_counts"].items()) for stats in fixed_stats.values()}) == 1
    changing_direction_consistent = len({tuple(stats["direction_counts"].items()) for stats in changing_stats.values()}) == 1

    summary = {
        "inputs": {
            "source_train_pool_manifest": str(args.source_train_pool_manifest),
            "old_end_to_end_manifest": str(args.old_end_to_end_manifest),
            "val_manifest": str(args.val_manifest),
        },
        "outputs": {
            "train_pool": str(pool_path),
            "end_to_end": str(end_to_end_path),
            "fixed_ratio_dir": str(fixed_dir),
            "changing_ratio_dir": str(changing_dir),
        },
        "design": {
            "source_pool_target_rows": 3600,
            "end_to_end_target_rows": 2400,
            "fixed_ratio_target_rows": 2400,
            "changing_ratio_target_rows": 2400,
            "source_pool_bucket_targets": {
                f"{direction}|{condition}": int(target)
                for (direction, condition), target in sorted(source_pool_target_map().items())
            },
            "fixed_bucket_targets": {
                f"{direction}|{condition}": int(target)
                for (direction, condition), target in sorted(fixed_target_map().items())
            },
            "changing_bucket_targets": {
                filename: {
                    f"{direction}|{condition}": int(target)
                    for (direction, condition), target in sorted(target_map.items())
                }
                for filename, target_map in changing_stage_target_maps().items()
            },
        },
        "train_pool_summary": pool_stats,
        "end_to_end_summary": end_stats,
        "fixed_ratio_summary": {
            "direction_counts_identical_across_stages": fixed_direction_consistent,
            "stages": fixed_stats,
        },
        "changing_ratio_summary": {
            "direction_counts_identical_across_stages": changing_direction_consistent,
            "stages": changing_stats,
        },
        "schema_check": schema_checks,
        "judgement": {
            "prompt_confound_removed": all(
                check["task"] == "direction_classification"
                and check["message_roles"] == ["user", "assistant"]
                and check["question"] == QUESTION_8WAY
                for name, check in schema_checks.items()
                if name != "old_end_to_end"
            ),
            "fixed_vs_changing_prompt_match_old": (
                schema_checks["new_rebalanced_end_to_end"]["task"] == schema_checks["old_end_to_end"]["task"]
                and schema_checks["new_rebalanced_end_to_end"]["question"] == schema_checks["old_end_to_end"]["question"]
                and schema_checks["new_fixed_stage1"]["question"] == schema_checks["old_end_to_end"]["question"]
                and schema_checks["new_changing_stage1"]["question"] == schema_checks["old_end_to_end"]["question"]
            ),
            "changing_direction_stage_variation_reason": (
                "FOV samples exist only in front/front-right/front-left within the source pool, so exact stage-invariant "
                "direction counts are mathematically incompatible with changing FOV ratios."
            ),
            "remaining_bottleneck": "FOV is inherently tied to the front trio in the source pool; changing-ratio stages still shift direction counts as FOV ratio changes.",
        },
    }
    write_summary(args.output_dir / "summary.json", summary)

    LOGGER.info("Wrote rebalanced manifests to %s", args.output_dir)
    LOGGER.info("Train pool rows=%d, end-to-end rows=%d", pool_stats["total_rows"], end_stats["total_rows"])


if __name__ == "__main__":
    main()
