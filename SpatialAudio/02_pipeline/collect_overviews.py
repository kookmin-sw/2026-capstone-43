from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect only 14_overview.png files into a flat folder.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("/home/yu/Project_git/SpatialAudio/02_pipeline/outputs_strict_indoor_train"),
        help="Root folder containing per-sample output folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/yu/Project_git/SpatialAudio/02_pipeline/collected_overviews"),
        help="Destination folder for flat overview collection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    output_root = ensure_dir(args.output_root.resolve())

    overview_paths = sorted(outputs_root.glob("*/14_overview.png"))
    records: list[dict[str, str]] = []

    for overview_path in overview_paths:
        sample_id = overview_path.parent.name
        destination = output_root / f"{sample_id}__14_overview.png"
        if args.overwrite or not destination.exists():
            shutil.copy2(overview_path, destination)
        records.append(
            {
                "sample_id": sample_id,
                "source_path": str(overview_path.resolve()),
                "destination_path": str(destination),
            }
        )

    summary = {
        "outputs_root": str(outputs_root),
        "output_root": str(output_root),
        "num_overviews": len(records),
        "files": records,
    }
    save_json(output_root / "collection_summary.json", summary)
    print(f"Collected {len(records)} overview files into {output_root}")


if __name__ == "__main__":
    main()
