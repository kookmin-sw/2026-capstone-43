from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

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


LOGGER = logging.getLogger(__name__)

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
    audio_path: str
    image_path: str
    direction_8way: str
    geometry_los: str
    in_fov: bool
    fov_tag: str
    projected_pixel_xy: Optional[list[float]]
    projection_depth_cam: Optional[float]
    projection_reason: Optional[str]
    visibility_ratio: Optional[float]
    source_dataset_root: str

def _parse_mic_id(sample_id: str) -> str:
    for token in sample_id.split("__"):
        if token.startswith("mic"):
            return token
    match = MIC_TOKEN_RE.search(sample_id)
    if match:
        return match.group(1)
    return "mic_unknown"


def _resolve_output_path(
    *,
    dataset_root: Path,
    relpath: str,
    path_mode: str,
    relative_base: Optional[Path],
) -> str:
    abs_path = (dataset_root / relpath).resolve()
    if path_mode == "relative":
        if relative_base is None:
            raise ValueError("relative_base must be provided when path_mode='relative'")
        return str(abs_path.relative_to(relative_base.resolve()))
    return str(abs_path)
def collect_candidates(
    *,
    dataset_roots: list[Path],
    split: Optional[str],
    audio_output_key: str,
    image_output_key: str,
    path_mode: str,
    relative_base: Optional[Path],
    allowed_labels: set[str],
) -> tuple[dict[tuple[str, str], list[Candidate]], dict[str, object]]:
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    seen_ids: set[tuple[str, str, str]] = set()
    root_counts: dict[str, int] = {}
    raw_counts: Counter[str] = Counter()

    for dataset_root in dataset_roots:
        accepted_for_root = 0
        for data in iter_dataset_rows(dataset_root):
            if split is not None and data.get("split") != split:
                continue

            geometry_los = str(data.get("geometry_los", "")).strip()
            if geometry_los not in GEOMETRY_ORDER:
                continue

            audio_relpath = output_relpath(data, audio_output_key, "foa_audio_path")
            image_relpath = output_relpath(data, image_output_key, "rgb_image_path")
            if not audio_relpath or not image_relpath:
                continue

            scene_id = str(data.get("scene_id", "")).strip()
            sample_id = str(data.get("sample_id", "")).strip()
            if not scene_id or not sample_id:
                continue

            dedupe_key = (scene_id, sample_id, geometry_los)
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)

            direction = azimuth_to_eight_way_label(row_azimuth_deg(data))
            if direction not in allowed_labels:
                continue

            mic_id = _parse_mic_id(sample_id)
            mic_uid = f"{scene_id}:{mic_id}"
            in_fov = row_in_fov(data)
            fov_tag = "FOV" if in_fov else "OOF"
            projected_pixel_xy_raw = data.get("projected_pixel_xy")
            projected_pixel_xy: Optional[list[float]] = None
            if (
                isinstance(projected_pixel_xy_raw, (list, tuple))
                and len(projected_pixel_xy_raw) >= 2
                and projected_pixel_xy_raw[0] is not None
                and projected_pixel_xy_raw[1] is not None
            ):
                projected_pixel_xy = [
                    float(projected_pixel_xy_raw[0]),
                    float(projected_pixel_xy_raw[1]),
                ]

            projection_depth_cam_raw = data.get("projection_depth_cam")
            projection_depth_cam = (
                float(projection_depth_cam_raw)
                if isinstance(projection_depth_cam_raw, (int, float))
                else None
            )
            projection_reason_raw = data.get("projection_reason")
            projection_reason = (
                str(projection_reason_raw) if projection_reason_raw is not None else None
            )
            visibility_ratio = row_visibility_ratio(data)

            candidate = Candidate(
                unique_id=f"{scene_id}:{sample_id}:{geometry_los}",
                sample_id=sample_id,
                scene_id=scene_id,
                mic_id=mic_id,
                mic_uid=mic_uid,
                audio_path=_resolve_output_path(
                    dataset_root=dataset_root,
                    relpath=audio_relpath,
                    path_mode=path_mode,
                    relative_base=relative_base,
                ),
                image_path=_resolve_output_path(
                    dataset_root=dataset_root,
                    relpath=image_relpath,
                    path_mode=path_mode,
                    relative_base=relative_base,
                ),
                direction_8way=direction,
                geometry_los=geometry_los,
                in_fov=in_fov,
                fov_tag=fov_tag,
                projected_pixel_xy=projected_pixel_xy,
                projection_depth_cam=projection_depth_cam,
                projection_reason=projection_reason,
                visibility_ratio=visibility_ratio,
                source_dataset_root=str(dataset_root),
            )
            grouped[(direction, geometry_los)].append(candidate)
            raw_counts[f"{direction}|{geometry_los}"] += 1
            accepted_for_root += 1

        root_counts[str(dataset_root)] = int(accepted_for_root)

    summary = {
        "total_candidates": int(sum(len(values) for values in grouped.values())),
        "raw_counts_by_bucket": {key: int(value) for key, value in sorted(raw_counts.items())},
        "accepted_counts_by_root": root_counts,
    }
    return grouped, summary


def _choose_next_candidate(
    *,
    candidates: list[Candidate],
    used_ids: set[str],
    global_scene_counts: Counter[str],
    global_mic_counts: Counter[str],
    bucket_scene_counts: Counter[str],
    bucket_mic_counts: Counter[str],
    max_per_scene_global: Optional[int],
    max_per_mic_global: Optional[int],
    max_per_scene_per_bucket: Optional[int],
    max_per_mic_per_bucket: Optional[int],
    enforce_new_scene: bool,
    enforce_new_mic: bool,
    rng: random.Random,
) -> Optional[Candidate]:
    eligible: list[Candidate] = []
    for candidate in candidates:
        if candidate.unique_id in used_ids:
            continue
        if max_per_scene_global is not None and global_scene_counts[candidate.scene_id] >= max_per_scene_global:
            continue
        if max_per_mic_global is not None and global_mic_counts[candidate.mic_uid] >= max_per_mic_global:
            continue
        if (
            max_per_scene_per_bucket is not None
            and bucket_scene_counts[candidate.scene_id] >= max_per_scene_per_bucket
        ):
            continue
        if (
            max_per_mic_per_bucket is not None
            and bucket_mic_counts[candidate.mic_uid] >= max_per_mic_per_bucket
        ):
            continue
        eligible.append(candidate)

    if not eligible:
        return None

    if enforce_new_scene:
        fresh_scene = [item for item in eligible if bucket_scene_counts[item.scene_id] == 0]
        if fresh_scene:
            eligible = fresh_scene

    if enforce_new_mic:
        fresh_mic = [item for item in eligible if bucket_mic_counts[item.mic_uid] == 0]
        if fresh_mic:
            eligible = fresh_mic

    return min(
        eligible,
        key=lambda item: (
            global_scene_counts[item.scene_id],
            bucket_scene_counts[item.scene_id],
            global_mic_counts[item.mic_uid],
            bucket_mic_counts[item.mic_uid],
            rng.random(),
        ),
    )


def select_balanced_samples(
    *,
    grouped_candidates: dict[tuple[str, str], list[Candidate]],
    labels: list[str],
    target_per_direction_per_geometry: int,
    seed: int,
    max_per_scene_global: Optional[int],
    max_per_mic_global: Optional[int],
    max_per_scene_per_bucket: Optional[int],
    max_per_mic_per_bucket: Optional[int],
    min_unique_scenes_per_bucket: int,
    min_unique_mics_per_bucket: int,
    strict_diversity: bool,
) -> tuple[list[Candidate], dict[str, object]]:
    missing = {
        f"{direction}|{geometry}": len(grouped_candidates.get((direction, geometry), ()))
        for direction in labels
        for geometry in GEOMETRY_ORDER
        if len(grouped_candidates.get((direction, geometry), ())) < target_per_direction_per_geometry
    }
    if missing:
        raise RuntimeError(
            "Not enough candidates for requested balanced master dataset. "
            f"Need {target_per_direction_per_geometry} per (direction, geometry), got {missing}"
        )

    rng = random.Random(int(seed))
    for values in grouped_candidates.values():
        rng.shuffle(values)

    used_ids: set[str] = set()
    global_scene_counts: Counter[str] = Counter()
    global_mic_counts: Counter[str] = Counter()
    selected: list[Candidate] = []

    bucket_stats: dict[str, dict[str, int]] = {}
    for direction in labels:
        for geometry in GEOMETRY_ORDER:
            bucket_key = f"{direction}|{geometry}"
            bucket_candidates = grouped_candidates[(direction, geometry)]
            bucket_scene_counts: Counter[str] = Counter()
            bucket_mic_counts: Counter[str] = Counter()
            bucket_selected: list[Candidate] = []

            while len(bucket_selected) < target_per_direction_per_geometry:
                enforce_new_scene = (
                    min_unique_scenes_per_bucket > 0
                    and len(bucket_scene_counts) < min_unique_scenes_per_bucket
                )
                enforce_new_mic = (
                    min_unique_mics_per_bucket > 0
                    and len(bucket_mic_counts) < min_unique_mics_per_bucket
                )
                candidate = _choose_next_candidate(
                    candidates=bucket_candidates,
                    used_ids=used_ids,
                    global_scene_counts=global_scene_counts,
                    global_mic_counts=global_mic_counts,
                    bucket_scene_counts=bucket_scene_counts,
                    bucket_mic_counts=bucket_mic_counts,
                    max_per_scene_global=max_per_scene_global,
                    max_per_mic_global=max_per_mic_global,
                    max_per_scene_per_bucket=max_per_scene_per_bucket,
                    max_per_mic_per_bucket=max_per_mic_per_bucket,
                    enforce_new_scene=enforce_new_scene,
                    enforce_new_mic=enforce_new_mic,
                    rng=rng,
                )
                if candidate is None:
                    break

                bucket_selected.append(candidate)
                selected.append(candidate)
                used_ids.add(candidate.unique_id)
                bucket_scene_counts[candidate.scene_id] += 1
                bucket_mic_counts[candidate.mic_uid] += 1
                global_scene_counts[candidate.scene_id] += 1
                global_mic_counts[candidate.mic_uid] += 1

            if len(bucket_selected) != target_per_direction_per_geometry:
                raise RuntimeError(
                    f"Could not fill bucket {bucket_key}: "
                    f"selected={len(bucket_selected)} target={target_per_direction_per_geometry}. "
                    "Relax max-per-scene/mic constraints or add more source data."
                )

            unique_scenes = len(bucket_scene_counts)
            unique_mics = len(bucket_mic_counts)
            if strict_diversity:
                if min_unique_scenes_per_bucket > 0 and unique_scenes < min_unique_scenes_per_bucket:
                    raise RuntimeError(
                        f"Bucket {bucket_key} violates min_unique_scenes_per_bucket: "
                        f"{unique_scenes} < {min_unique_scenes_per_bucket}"
                    )
                if min_unique_mics_per_bucket > 0 and unique_mics < min_unique_mics_per_bucket:
                    raise RuntimeError(
                        f"Bucket {bucket_key} violates min_unique_mics_per_bucket: "
                        f"{unique_mics} < {min_unique_mics_per_bucket}"
                    )

            bucket_stats[bucket_key] = {
                "selected": len(bucket_selected),
                "unique_scenes": unique_scenes,
                "unique_mics": unique_mics,
                "max_scene_concentration": max(bucket_scene_counts.values()) if bucket_scene_counts else 0,
                "max_mic_concentration": max(bucket_mic_counts.values()) if bucket_mic_counts else 0,
            }

    rng.shuffle(selected)
    summary = {
        "num_selected": len(selected),
        "target_per_direction_per_geometry": int(target_per_direction_per_geometry),
        "global_unique_scenes": len(global_scene_counts),
        "global_unique_mics": len(global_mic_counts),
        "global_max_scene_concentration": max(global_scene_counts.values()) if global_scene_counts else 0,
        "global_max_mic_concentration": max(global_mic_counts.values()) if global_mic_counts else 0,
        "bucket_stats": bucket_stats,
    }
    return selected, summary


def write_manifests(
    *,
    output_dir: Path,
    rows_master: list[dict[str, object]],
    rows_audio_only: list[dict[str, object]],
    rows_av_messages: list[dict[str, object]],
    emit_condition_manifests: bool,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    master_path = output_dir / "master_balanced_8way_glos_gnlos_2400.jsonl"
    audio_path = output_dir / "train_audio_only_glos_gnlos_8way_2400.jsonl"
    av_path = output_dir / "train_av_messages_glos_gnlos_8way_2400.jsonl"

    with master_path.open("w", encoding="utf-8") as handle:
        for row in rows_master:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with audio_path.open("w", encoding="utf-8") as handle:
        for row in rows_audio_only:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with av_path.open("w", encoding="utf-8") as handle:
        for row in rows_av_messages:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    outputs = {
        "master": str(master_path),
        "audio_only": str(audio_path),
        "av_messages": str(av_path),
    }

    if not emit_condition_manifests:
        return outputs

    conditions_dir = output_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    condition_map = {
        "fov_glos": lambda row: row["fov_tag"] == "FOV" and row["geometry_los"] == "gLOS",
        "fov_gnlos": lambda row: row["fov_tag"] == "FOV" and row["geometry_los"] == "gNLOS",
        "oof_glos": lambda row: row["fov_tag"] == "OOF" and row["geometry_los"] == "gLOS",
        "oof_gnlos": lambda row: row["fov_tag"] == "OOF" and row["geometry_los"] == "gNLOS",
        "glos_all": lambda row: row["geometry_los"] == "gLOS",
        "gnlos_all": lambda row: row["geometry_los"] == "gNLOS",
        "fov_all": lambda row: row["fov_tag"] == "FOV",
        "oof_all": lambda row: row["fov_tag"] == "OOF",
    }

    for name, predicate in condition_map.items():
        filtered_master = [row for row in rows_master if predicate(row)]
        filtered_audio = [row for row in rows_audio_only if predicate(row)]
        filtered_av = [row for row in rows_av_messages if predicate(row)]

        master_cond = conditions_dir / f"{name}_master.jsonl"
        audio_cond = conditions_dir / f"{name}_audio_only.jsonl"
        av_cond = conditions_dir / f"{name}_av_messages.jsonl"
        with master_cond.open("w", encoding="utf-8") as handle:
            for row in filtered_master:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with audio_cond.open("w", encoding="utf-8") as handle:
            for row in filtered_audio:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with av_cond.open("w", encoding="utf-8") as handle:
            for row in filtered_av:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        outputs[f"cond_{name}_master"] = str(master_cond)
        outputs[f"cond_{name}_audio_only"] = str(audio_cond)
        outputs[f"cond_{name}_av_messages"] = str(av_cond)

    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a master 8-way balanced dataset manifest with per-direction gLOS/gNLOS balancing "
            "and scene/mic concentration control."
        )
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more generated dataset roots (containing scenes/*/samples/*/metadata/sample.json).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--labels", type=str, default=",".join(EIGHT_WAY_ORDER))
    parser.add_argument("--target-per-direction-per-geometry", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-output-key", type=str, default="audio_mic_wav")
    parser.add_argument("--image-output-key", type=str, default="rgb_front_png")
    parser.add_argument("--path-mode", choices=("absolute", "relative"), default="relative")
    parser.add_argument("--relative-base", type=Path, default=None)

    parser.add_argument("--max-per-scene-global", type=int, default=None)
    parser.add_argument("--max-per-mic-global", type=int, default=None)
    parser.add_argument("--max-per-scene-per-bucket", type=int, default=None)
    parser.add_argument("--max-per-mic-per-bucket", type=int, default=None)
    parser.add_argument("--min-unique-scenes-per-bucket", type=int, default=3)
    parser.add_argument("--min-unique-mics-per-bucket", type=int, default=8)
    parser.add_argument("--strict-diversity", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--emit-condition-manifests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    labels = [token.strip() for token in str(args.labels).split(",") if token.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one non-empty label")

    grouped, collect_summary = collect_candidates(
        dataset_roots=[path.resolve() for path in args.dataset_roots],
        split=args.split if str(args.split).strip() else None,
        audio_output_key=args.audio_output_key,
        image_output_key=args.image_output_key,
        path_mode=args.path_mode,
        relative_base=(args.relative_base.resolve() if args.relative_base is not None else None),
        allowed_labels=set(labels),
    )
    LOGGER.info("Collected candidates summary: %s", collect_summary)

    selected, selection_summary = select_balanced_samples(
        grouped_candidates=grouped,
        labels=labels,
        target_per_direction_per_geometry=int(args.target_per_direction_per_geometry),
        seed=int(args.seed),
        max_per_scene_global=args.max_per_scene_global,
        max_per_mic_global=args.max_per_mic_global,
        max_per_scene_per_bucket=args.max_per_scene_per_bucket,
        max_per_mic_per_bucket=args.max_per_mic_per_bucket,
        min_unique_scenes_per_bucket=int(args.min_unique_scenes_per_bucket),
        min_unique_mics_per_bucket=int(args.min_unique_mics_per_bucket),
        strict_diversity=bool(args.strict_diversity),
    )
    LOGGER.info("Selection summary: %s", selection_summary)

    rows_master: list[dict[str, object]] = []
    rows_audio_only: list[dict[str, object]] = []
    rows_av_messages: list[dict[str, object]] = []
    for candidate in selected:
        row_master = {
            "id": f"{candidate.sample_id}__{candidate.geometry_los}",
            "sample_id": candidate.sample_id,
            "scene_id": candidate.scene_id,
            "mic_id": candidate.mic_id,
            "mic_uid": candidate.mic_uid,
            "audio_path": candidate.audio_path,
            "image_path": candidate.image_path,
            "direction_8way": candidate.direction_8way,
            "label": candidate.direction_8way,
            "fov_tag": candidate.fov_tag,
            "in_fov": candidate.in_fov,
            "projected_pixel_xy": candidate.projected_pixel_xy,
            "projection_depth_cam": candidate.projection_depth_cam,
            "projection_reason": candidate.projection_reason,
            "visibility_ratio": candidate.visibility_ratio,
            "fov": {
                "in_fov": candidate.in_fov,
                "tag": candidate.fov_tag,
                "projected_pixel_xy": candidate.projected_pixel_xy,
                "projection_depth_cam": candidate.projection_depth_cam,
                "projection_reason": candidate.projection_reason,
                "visibility_ratio": candidate.visibility_ratio,
            },
            "geometry_los": candidate.geometry_los,
            "task": "direction_classification",
            "source_dataset_root": candidate.source_dataset_root,
        }
        rows_master.append(row_master)

        rows_audio_only.append(
            {
                "id": row_master["id"],
                "audio_path": candidate.audio_path,
                "label": candidate.direction_8way,
                "scene_id": candidate.scene_id,
                "mic_id": candidate.mic_id,
                "fov_tag": candidate.fov_tag,
                "in_fov": candidate.in_fov,
                "projected_pixel_xy": candidate.projected_pixel_xy,
                "projection_depth_cam": candidate.projection_depth_cam,
                "projection_reason": candidate.projection_reason,
                "visibility_ratio": candidate.visibility_ratio,
                "geometry_los": candidate.geometry_los,
            }
        )

        rows_av_messages.append(
            {
                "id": row_master["id"],
                "audio_path": candidate.audio_path,
                "image_path": candidate.image_path,
                "geometry_los": candidate.geometry_los,
                "fov_tag": candidate.fov_tag,
                "in_fov": candidate.in_fov,
                "projected_pixel_xy": candidate.projected_pixel_xy,
                "projection_depth_cam": candidate.projection_depth_cam,
                "projection_reason": candidate.projection_reason,
                "visibility_ratio": candidate.visibility_ratio,
                "task": "direction_classification",
                "messages": [
                    {"role": "user", "content": QUESTION_8WAY},
                    {"role": "assistant", "content": candidate.direction_8way},
                ],
            }
        )

    if bool(args.dry_run):
        LOGGER.info("[dry-run] Skipped writing output files.")
        return

    output_paths = write_manifests(
        output_dir=args.output_dir.resolve(),
        rows_master=rows_master,
        rows_audio_only=rows_audio_only,
        rows_av_messages=rows_av_messages,
        emit_condition_manifests=bool(args.emit_condition_manifests),
    )

    summary = {
        "collect_summary": collect_summary,
        "selection_summary": selection_summary,
        "outputs": output_paths,
    }
    summary_path = args.output_dir.resolve() / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary: %s", summary_path)


if __name__ == "__main__":
    main()
