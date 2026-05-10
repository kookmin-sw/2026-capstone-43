from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger(__name__)

EIGHT_WAY_ORDER = (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)

CONDITION_ORDER = (
    "FOV|gLOS",
    "FOV|gNLOS",
    "OOF|gLOS",
    "OOF|gNLOS",
)

STAGE_SPECS = (
    (
        "01_train_stage1.jsonl",
        {
            "FOV|gLOS": 0.50,
            "FOV|gNLOS": 0.20,
            "OOF|gLOS": 0.20,
            "OOF|gNLOS": 0.10,
        },
    ),
    (
        "02_train_stage2.jsonl",
        {
            "FOV|gLOS": 0.35,
            "FOV|gNLOS": 0.25,
            "OOF|gLOS": 0.20,
            "OOF|gNLOS": 0.20,
        },
    ),
    (
        "03_train_stage3.jsonl",
        {
            "FOV|gLOS": 0.25,
            "FOV|gNLOS": 0.25,
            "OOF|gLOS": 0.20,
            "OOF|gNLOS": 0.30,
        },
    ),
    (
        "04_train_stage4.jsonl",
        {
            "FOV|gLOS": 0.20,
            "FOV|gNLOS": 0.20,
            "OOF|gLOS": 0.20,
            "OOF|gNLOS": 0.40,
        },
    ),
)


def parse_ratio_spec(text: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid ratio item '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in CONDITION_ORDER:
            raise ValueError(
                f"Unknown condition '{key}'. Expected one of: {', '.join(CONDITION_ORDER)}"
            )
        ratios[key] = float(value.strip())
    missing = [key for key in CONDITION_ORDER if key not in ratios]
    if missing:
        raise ValueError(
            f"Missing ratio(s) for: {', '.join(missing)}"
        )
    total = sum(ratios[key] for key in CONDITION_ORDER)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")
    return ratios


def resolve_stage_specs(fixed_ratios: str | None) -> tuple[tuple[str, dict[str, float]], ...]:
    if not fixed_ratios:
        return STAGE_SPECS
    ratios = parse_ratio_spec(fixed_ratios)
    return tuple(
        (filename, dict(ratios))
        for filename, _ in STAGE_SPECS
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    sample_id = str(normalized.get("sample_id") or normalized.get("id") or "").strip()
    if not sample_id:
        raise ValueError(f"Missing sample_id/id in row: {row}")
    scene_id = str(normalized.get("scene_id") or sample_id.split("__", 1)[0]).strip()
    mic_id = str(normalized.get("mic_id") or _infer_mic_id(sample_id)).strip()
    direction = str(normalized.get("direction_8way") or normalized.get("label") or "").strip()
    if not direction:
        raise ValueError(f"Missing direction_8way/label in row: {row}")
    fov_tag = _normalize_fov_tag(normalized.get("fov_tag"), normalized.get("in_fov"))
    geometry = _normalize_geometry(normalized.get("geometry_los"))
    if fov_tag == "UNK" or geometry == "UNK":
        raise ValueError(f"Could not normalize condition fields for row: {row}")

    normalized["sample_id"] = sample_id
    normalized["scene_id"] = scene_id
    normalized["mic_id"] = mic_id
    normalized["direction_8way"] = direction
    normalized["label"] = str(normalized.get("label") or direction)
    normalized["fov_tag"] = fov_tag
    normalized["geometry_los"] = geometry
    normalized["in_fov"] = fov_tag == "FOV"
    return normalized


def _infer_mic_id(sample_id: str) -> str:
    for token in sample_id.split("__"):
        if token.startswith("mic"):
            return token
    return "mic_unknown"


def _normalize_fov_tag(fov_tag: Any, in_fov: Any) -> str:
    text = str(fov_tag or "").strip().upper()
    if text in {"FOV", "OOF"}:
        return text
    if isinstance(in_fov, bool):
        return "FOV" if in_fov else "OOF"
    if isinstance(in_fov, (int, float)):
        return "FOV" if bool(in_fov) else "OOF"
    return "UNK"


def _normalize_geometry(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"gLOS", "gNLOS"}:
        return text
    return "UNK"


def condition_key(row: dict[str, Any]) -> str:
    return f"{row['fov_tag']}|{row['geometry_los']}"


def round_targets(ratios: dict[str, float], target_rows: int) -> dict[str, int]:
    raw = {key: ratios[key] * target_rows for key in CONDITION_ORDER}
    rounded = {key: int(raw[key]) for key in CONDITION_ORDER}
    deficit = int(target_rows - sum(rounded.values()))
    if deficit != 0:
        remainders = sorted(
            ((raw[key] - rounded[key], key) for key in CONDITION_ORDER),
            reverse=True,
        )
        for _, key in remainders[:deficit]:
            rounded[key] += 1
    return rounded


def build_stage_rows(
    *,
    pool_by_condition: dict[str, list[dict[str, Any]]],
    stage_targets: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    duplicate_by_condition: dict[str, int] = {}

    global_direction_counts: Counter[str] = Counter()
    global_scene_counts: Counter[str] = Counter()
    global_mic_pair_counts: Counter[str] = Counter()

    for condition in CONDITION_ORDER:
        candidates = list(pool_by_condition[condition])
        target = int(stage_targets[condition])
        unique_available = len(candidates)
        duplicate_by_condition[condition] = max(0, target - unique_available)

        use_counts: Counter[str] = Counter()
        local_direction_counts: Counter[str] = Counter()
        local_scene_counts: Counter[str] = Counter()
        local_mic_pair_counts: Counter[str] = Counter()

        chosen_unique = min(target, unique_available)
        if chosen_unique > 0:
            available = list(candidates)
            for _ in range(chosen_unique):
                best_index = min(
                    range(len(available)),
                    key=lambda idx: _candidate_score(
                        available[idx],
                        use_count=use_counts[available[idx]["sample_id"]],
                        global_direction_counts=global_direction_counts,
                        local_direction_counts=local_direction_counts,
                        global_scene_counts=global_scene_counts,
                        local_scene_counts=local_scene_counts,
                        global_mic_pair_counts=global_mic_pair_counts,
                        local_mic_pair_counts=local_mic_pair_counts,
                    ),
                )
                row = available.pop(best_index)
                selected.append(row)
                _update_counters(
                    row=row,
                    use_counts=use_counts,
                    global_direction_counts=global_direction_counts,
                    local_direction_counts=local_direction_counts,
                    global_scene_counts=global_scene_counts,
                    local_scene_counts=local_scene_counts,
                    global_mic_pair_counts=global_mic_pair_counts,
                    local_mic_pair_counts=local_mic_pair_counts,
                )

        remaining = target - chosen_unique
        if remaining > 0:
            if not candidates:
                raise RuntimeError(f"No candidates available for condition {condition}")
            for _ in range(remaining):
                row = min(
                    candidates,
                    key=lambda item: _candidate_score(
                        item,
                        use_count=use_counts[item["sample_id"]],
                        global_direction_counts=global_direction_counts,
                        local_direction_counts=local_direction_counts,
                        global_scene_counts=global_scene_counts,
                        local_scene_counts=local_scene_counts,
                        global_mic_pair_counts=global_mic_pair_counts,
                        local_mic_pair_counts=local_mic_pair_counts,
                    ),
                )
                selected.append(dict(row))
                _update_counters(
                    row=row,
                    use_counts=use_counts,
                    global_direction_counts=global_direction_counts,
                    local_direction_counts=local_direction_counts,
                    global_scene_counts=global_scene_counts,
                    local_scene_counts=local_scene_counts,
                    global_mic_pair_counts=global_mic_pair_counts,
                    local_mic_pair_counts=local_mic_pair_counts,
                )

    selected.sort(
        key=lambda row: (
            condition_key(row),
            EIGHT_WAY_ORDER.index(row["direction_8way"]),
            row["scene_id"],
            row["mic_id"],
            row["sample_id"],
        )
    )
    return selected, duplicate_by_condition


def _candidate_score(
    row: dict[str, Any],
    *,
    use_count: int,
    global_direction_counts: Counter[str],
    local_direction_counts: Counter[str],
    global_scene_counts: Counter[str],
    local_scene_counts: Counter[str],
    global_mic_pair_counts: Counter[str],
    local_mic_pair_counts: Counter[str],
) -> tuple[Any, ...]:
    direction = row["direction_8way"]
    scene = row["scene_id"]
    mic_pair = f"{scene}|{row['mic_id']}"
    return (
        int(use_count),
        int(local_direction_counts[direction]),
        int(global_direction_counts[direction]),
        int(local_scene_counts[scene]),
        int(global_scene_counts[scene]),
        int(local_mic_pair_counts[mic_pair]),
        int(global_mic_pair_counts[mic_pair]),
        row["sample_id"],
    )


def _update_counters(
    *,
    row: dict[str, Any],
    use_counts: Counter[str],
    global_direction_counts: Counter[str],
    local_direction_counts: Counter[str],
    global_scene_counts: Counter[str],
    local_scene_counts: Counter[str],
    global_mic_pair_counts: Counter[str],
    local_mic_pair_counts: Counter[str],
) -> None:
    sample_id = row["sample_id"]
    direction = row["direction_8way"]
    scene = row["scene_id"]
    mic_pair = f"{scene}|{row['mic_id']}"
    use_counts[sample_id] += 1
    global_direction_counts[direction] += 1
    local_direction_counts[direction] += 1
    global_scene_counts[scene] += 1
    local_scene_counts[scene] += 1
    global_mic_pair_counts[mic_pair] += 1
    local_mic_pair_counts[mic_pair] += 1


def summarize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    target_ratios: dict[str, float],
    duplicate_by_condition: dict[str, int],
) -> dict[str, Any]:
    rows = list(rows)
    total_rows = len(rows)
    sample_ids = [row["sample_id"] for row in rows]
    direction_counts = Counter(row["direction_8way"] for row in rows)
    geometry_counts = Counter(row["geometry_los"] for row in rows)
    fov_counts = Counter(row["fov_tag"] for row in rows)
    cond_counts = Counter(condition_key(row) for row in rows)
    scene_ids = {row["scene_id"] for row in rows}
    mic_pairs = {f"{row['scene_id']}|{row['mic_id']}" for row in rows}

    actual_ratios = {
        key: (cond_counts[key] / total_rows) if total_rows else 0.0
        for key in CONDITION_ORDER
    }
    ratio_diff = {
        key: actual_ratios[key] - float(target_ratios[key])
        for key in CONDITION_ORDER
    }

    return {
        "total_rows": int(total_rows),
        "unique_sample_ids": int(len(set(sample_ids))),
        "duplicates": int(total_rows - len(set(sample_ids))),
        "direction_counts": {key: int(direction_counts[key]) for key in EIGHT_WAY_ORDER if direction_counts[key] > 0},
        "geometry_counts": {key: int(geometry_counts[key]) for key in ("gLOS", "gNLOS") if geometry_counts[key] > 0},
        "fov_counts": {key: int(fov_counts[key]) for key in ("FOV", "OOF") if fov_counts[key] > 0},
        "condition_counts": {key: int(cond_counts[key]) for key in CONDITION_ORDER if cond_counts[key] > 0},
        "unique_scenes": int(len(scene_ids)),
        "unique_mic_pairs": int(len(mic_pairs)),
        "target_ratios": {key: float(target_ratios[key]) for key in CONDITION_ORDER},
        "actual_ratios": actual_ratios,
        "ratio_diff": ratio_diff,
        "duplicate_by_condition": {key: int(value) for key, value in duplicate_by_condition.items() if int(value) > 0},
        "bottleneck_conditions": [
            key for key in CONDITION_ORDER
            if duplicate_by_condition.get(key, 0) > 0
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build curriculum stage manifests from an existing train pool while matching the total row "
            "count of a reference end-to-end train manifest."
        )
    )
    parser.add_argument("--train-pool-manifest", type=Path, required=True)
    parser.add_argument("--end-to-end-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--fixed-ratios",
        type=str,
        default=None,
        help=(
            "Optional single ratio spec to use for all four stage files, for example "
            "'FOV|gLOS=0.325,FOV|gNLOS=0.225,OOF|gLOS=0.20,OOF|gNLOS=0.25'."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    train_pool_manifest = args.train_pool_manifest.resolve()
    end_to_end_manifest = args.end_to_end_manifest.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = (
        args.summary_path.resolve()
        if args.summary_path is not None
        else output_dir / "curriculum_step_matched_summary.json"
    )

    pool_rows = [normalize_row(row) for row in read_jsonl(train_pool_manifest)]
    target_rows = len(read_jsonl(end_to_end_manifest))
    LOGGER.info(
        "Loaded train pool rows=%d from %s; end-to-end target rows=%d from %s",
        len(pool_rows),
        train_pool_manifest,
        target_rows,
        end_to_end_manifest,
    )
    stage_specs = resolve_stage_specs(args.fixed_ratios)

    pool_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        pool_by_condition[condition_key(row)].append(row)
    for condition in CONDITION_ORDER:
        pool_by_condition.setdefault(condition, [])

    stage_summaries: dict[str, Any] = {
        "train_pool_manifest": str(train_pool_manifest),
        "end_to_end_manifest": str(end_to_end_manifest),
        "target_rows": int(target_rows),
        "pool_condition_counts": {
            key: int(len(pool_by_condition[key])) for key in CONDITION_ORDER
        },
    }

    stage_summaries["stage_mode"] = "fixed" if args.fixed_ratios else "curriculum"
    if args.fixed_ratios:
        stage_summaries["fixed_ratios"] = parse_ratio_spec(args.fixed_ratios)

    for filename, ratios in stage_specs:
        stage_targets = round_targets(ratios, target_rows)
        rows, duplicate_by_condition = build_stage_rows(
            pool_by_condition=pool_by_condition,
            stage_targets=stage_targets,
        )
        output_path = output_dir / filename
        write_jsonl(output_path, rows)
        summary = summarize_rows(
            rows,
            target_ratios=ratios,
            duplicate_by_condition=duplicate_by_condition,
        )
        summary["path"] = str(output_path)
        summary["targets"] = stage_targets
        stage_summaries[filename] = summary
        LOGGER.info(
            "Wrote %s rows=%d unique=%d duplicates=%d",
            output_path,
            summary["total_rows"],
            summary["unique_sample_ids"],
            summary["duplicates"],
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stage_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stage_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
