from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .manifest_io import iter_dataset_rows, output_relpath, row_azimuth_deg, row_in_fov
from .spatial_conventions import (
    direction_8way_from_azimuth as azimuth_to_eight_way_label,
    normalize_azimuth_deg,
)


LOGGER = logging.getLogger(__name__)

QUESTION_8WAY = (
    "<audio>\n<image>\n"
    "What is the speaker direction in the camera-centered view? "
    "Choose one of: front-left, front, front-right, right, back-right, back, back-left, left."
)

def _resolve_output_path(
    dataset_root: Path,
    relpath: str,
    *,
    path_mode: str,
    relative_base: Optional[Path],
) -> str:
    abs_path = (dataset_root / relpath).resolve()
    if path_mode == "relative":
        if relative_base is None:
            raise ValueError("relative_base must be provided when path_mode='relative'")
        return str(abs_path.relative_to(relative_base.resolve()))
    return str(abs_path)


def _collect_candidates(
    *,
    dataset_root: Path,
    split: Optional[str],
    require_geometry_los: Optional[str],
    require_in_fov: Optional[bool],
    audio_output_key: str,
    image_output_key: str,
    path_mode: str,
    relative_base: Optional[Path],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    counts: Counter[str] = Counter()

    for data in iter_dataset_rows(dataset_root):
        if split is not None and data.get("split") != split:
            continue
        if require_geometry_los is not None and data.get("geometry_los") != require_geometry_los:
            continue
        if require_in_fov is not None and row_in_fov(data) != bool(require_in_fov):
            continue

        audio_relpath = output_relpath(data, audio_output_key, "foa_audio_path")
        image_relpath = output_relpath(data, image_output_key, "rgb_image_path")
        if not audio_relpath or not image_relpath:
            continue

        label = azimuth_to_eight_way_label(row_azimuth_deg(data))
        grouped[label].append(
            {
                "id": str(data.get("sample_id", "")),
                "audio_path": _resolve_output_path(
                    dataset_root,
                    audio_relpath,
                    path_mode=path_mode,
                    relative_base=relative_base,
                ),
                "image_path": _resolve_output_path(
                    dataset_root,
                    image_relpath,
                    path_mode=path_mode,
                    relative_base=relative_base,
                ),
                "label": label,
            }
        )
        counts[label] += 1

    return grouped, {key: int(value) for key, value in sorted(counts.items())}


def build_balanced_manifests(
    *,
    dataset_root: Path,
    audio_output_path: Path,
    av_output_path: Path,
    split: Optional[str],
    require_geometry_los: Optional[str],
    require_in_fov: Optional[bool],
    audio_output_key: str,
    image_output_key: str,
    path_mode: str,
    relative_base: Optional[Path],
    labels: list[str],
    per_label_limit: int,
    seed: int,
) -> dict[str, object]:
    grouped, raw_counts = _collect_candidates(
        dataset_root=dataset_root,
        split=split,
        require_geometry_los=require_geometry_los,
        require_in_fov=require_in_fov,
        audio_output_key=audio_output_key,
        image_output_key=image_output_key,
        path_mode=path_mode,
        relative_base=relative_base,
    )

    if not labels:
        raise ValueError("labels must not be empty")
    requested_labels = list(dict.fromkeys(labels))
    missing = {
        label: len(grouped.get(label, ()))
        for label in requested_labels
        if len(grouped.get(label, ())) < per_label_limit
    }
    if missing:
        raise RuntimeError(
            "Not enough samples to build balanced 8-way manifests. "
            f"Need {per_label_limit} per label, got {missing}"
        )

    rng = random.Random(int(seed))
    selected_audio: list[dict[str, str]] = []
    selected_av: list[dict[str, object]] = []

    running_index = 1
    for label in requested_labels:
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        chosen = candidates[:per_label_limit]
        for item in chosen:
            selected_audio.append(
                {
                    "audio_path": item["audio_path"],
                    "label": label,
                }
            )
            selected_av.append(
                {
                    "id": item["id"] or f"sample_{running_index:06d}",
                    "audio_path": item["audio_path"],
                    "image_path": item["image_path"],
                    "task": "direction_classification",
                    "messages": [
                        {
                            "role": "user",
                            "content": QUESTION_8WAY,
                        },
                        {
                            "role": "assistant",
                            "content": label,
                        },
                    ],
                }
            )
            running_index += 1

    rng.shuffle(selected_audio)
    rng.shuffle(selected_av)

    audio_output_path.parent.mkdir(parents=True, exist_ok=True)
    av_output_path.parent.mkdir(parents=True, exist_ok=True)

    with audio_output_path.open("w", encoding="utf-8") as handle:
        for row in selected_audio:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    with av_output_path.open("w", encoding="utf-8") as handle:
        for row in selected_av:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    final_counts = Counter(row["label"] for row in selected_audio)
    return {
        "audio_output": str(audio_output_path),
        "av_output": str(av_output_path),
        "labels": requested_labels,
        "per_label_limit": int(per_label_limit),
        "seed": int(seed),
        "raw_counts": raw_counts,
        "selected_counts": {key: int(value) for key, value in sorted(final_counts.items())},
        "num_audio_rows": len(selected_audio),
        "num_av_rows": len(selected_av),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export balanced 8-way audio-only and AV-message JSONL manifests from generated metadata."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audio-output", type=Path, required=True)
    parser.add_argument("--av-output", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--require-geometry-los", type=str, default="gLOS")
    fov_group = parser.add_mutually_exclusive_group()
    fov_group.add_argument("--require-in-fov", action="store_true")
    fov_group.add_argument("--require-out-of-fov", action="store_true")
    parser.add_argument(
        "--labels",
        type=str,
        default="front,front-right,right,back-right,back,back-left,left,front-left",
        help="Comma-separated direction labels to balance and export.",
    )
    parser.add_argument("--per-label-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audio-output-key",
        type=str,
        default="audio_mic_wav",
    )
    parser.add_argument(
        "--image-output-key",
        type=str,
        default="rgb_front_png",
    )
    parser.add_argument(
        "--path-mode",
        type=str,
        choices=("absolute", "relative"),
        default="relative",
    )
    parser.add_argument("--relative-base", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    requested_labels = [token.strip() for token in str(args.labels).split(",") if token.strip()]
    if not requested_labels:
        raise ValueError("--labels must contain at least one non-empty label")
    require_in_fov: Optional[bool]
    if args.require_in_fov:
        require_in_fov = True
    elif args.require_out_of_fov:
        require_in_fov = False
    else:
        require_in_fov = None

    summary = build_balanced_manifests(
        dataset_root=args.dataset_root.resolve(),
        audio_output_path=args.audio_output.resolve(),
        av_output_path=args.av_output.resolve(),
        split=args.split,
        require_geometry_los=args.require_geometry_los,
        require_in_fov=require_in_fov,
        audio_output_key=args.audio_output_key,
        image_output_key=args.image_output_key,
        path_mode=args.path_mode,
        relative_base=args.relative_base.resolve() if args.relative_base is not None else None,
        labels=requested_labels,
        per_label_limit=args.per_label_limit,
        seed=args.seed,
    )
    LOGGER.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
