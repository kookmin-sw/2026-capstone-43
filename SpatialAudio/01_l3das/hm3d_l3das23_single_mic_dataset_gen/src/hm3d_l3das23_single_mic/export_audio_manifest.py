from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from .manifest_io import iter_dataset_rows, output_relpath, row_azimuth_deg, row_in_fov
from .spatial_conventions import (
    direction_8way_from_azimuth as azimuth_to_eight_way_label,
    normalize_azimuth_deg,
)


LOGGER = logging.getLogger(__name__)
def build_audio_jsonl(
    *,
    dataset_root: Path,
    output_path: Path,
    split: Optional[str],
    require_geometry_los: Optional[str],
    require_in_fov: Optional[bool],
    audio_output_key: str,
    path_mode: str,
    relative_base: Optional[Path],
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter[str] = Counter()
    num_written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for data in iter_dataset_rows(dataset_root):
            if split is not None and data.get("split") != split:
                continue
            if require_geometry_los is not None and data.get("geometry_los") != require_geometry_los:
                continue
            if require_in_fov is not None and row_in_fov(data) != bool(require_in_fov):
                continue

            audio_relpath = output_relpath(data, audio_output_key, "foa_audio_path")
            if not audio_relpath:
                LOGGER.warning("Skipping row %s because %s is missing", data.get("sample_id"), audio_output_key)
                continue

            label = azimuth_to_eight_way_label(row_azimuth_deg(data))
            audio_path = (dataset_root / audio_relpath).resolve()
            if path_mode == "relative":
                if relative_base is None:
                    raise ValueError("relative_base must be provided when path_mode='relative'")
                record_audio_path = str(audio_path.relative_to(relative_base.resolve()))
            else:
                record_audio_path = str(audio_path)

            record = {"audio_path": record_audio_path, "label": label}
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            label_counts[label] += 1
            num_written += 1

    summary = {"num_written": num_written, **{key: int(value) for key, value in sorted(label_counts.items())}}
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an audio-only JSONL manifest with 8-way azimuth labels."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--require-geometry-los", type=str, default=None)
    parser.add_argument("--require-in-fov", action="store_true")
    parser.add_argument(
        "--audio-output-key",
        type=str,
        default="audio_mic_wav",
        help="Metadata output_files key to export. Default: audio_mic_wav",
    )
    parser.add_argument(
        "--path-mode",
        type=str,
        choices=("absolute", "relative"),
        default="absolute",
        help="Whether to write absolute audio paths or paths relative to --relative-base.",
    )
    parser.add_argument(
        "--relative-base",
        type=Path,
        default=None,
        help="Base directory used when --path-mode relative is selected.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    dataset_root = args.dataset_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else dataset_root / "manifests" / "train_audio_only_8way.jsonl"
    )
    summary = build_audio_jsonl(
        dataset_root=dataset_root,
        output_path=output_path,
        split=args.split,
        require_geometry_los=args.require_geometry_los,
        require_in_fov=True if args.require_in_fov else None,
        audio_output_key=args.audio_output_key,
        path_mode=args.path_mode,
        relative_base=args.relative_base.resolve() if args.relative_base is not None else None,
    )
    LOGGER.info("Wrote %s", output_path)
    LOGGER.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
