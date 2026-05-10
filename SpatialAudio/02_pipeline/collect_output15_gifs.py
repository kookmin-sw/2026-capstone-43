from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect output15 GIFs into two flat folders by type.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("/home/yu/Project_git/SpatialAudio/02_pipeline/outputs_strict_indoor_train_with_15"),
        help="Root folder containing per-sample output folders with 15_windowed_beam.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/yu/Project_git/SpatialAudio/02_pipeline/collected_output15_gifs"),
        help="Destination root. Two subfolders will be created under this path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing collected files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def maybe_copy(src: Path, dst: Path, overwrite: bool) -> None:
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    output_root = ensure_dir(args.output_root.resolve())
    maps_root = ensure_dir(output_root / "beam_power_maps_gif")
    overlays_root = ensure_dir(output_root / "beam_filtered_overlays_gif")

    records: list[dict[str, str | None]] = []
    sample_dirs = sorted(path for path in outputs_root.iterdir() if path.is_dir())
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        maps_gif = sample_dir / "15_windowed_beam" / "15_beam_power_maps.gif"
        overlays_gif = sample_dir / "15_windowed_beam" / "15_beam_filtered_overlays.gif"

        maps_dst = None
        overlays_dst = None
        if maps_gif.exists():
            maps_dst_path = maps_root / f"{sample_id}__15_beam_power_maps.gif"
            maybe_copy(maps_gif, maps_dst_path, overwrite=bool(args.overwrite))
            maps_dst = str(maps_dst_path)
        if overlays_gif.exists():
            overlays_dst_path = overlays_root / f"{sample_id}__15_beam_filtered_overlays.gif"
            maybe_copy(overlays_gif, overlays_dst_path, overwrite=bool(args.overwrite))
            overlays_dst = str(overlays_dst_path)

        if maps_dst is not None or overlays_dst is not None:
            records.append(
                {
                    "sample_id": sample_id,
                    "beam_power_maps_gif": maps_dst,
                    "beam_filtered_overlays_gif": overlays_dst,
                }
            )

    summary = {
        "outputs_root": str(outputs_root),
        "output_root": str(output_root),
        "beam_power_maps_gif_root": str(maps_root),
        "beam_filtered_overlays_gif_root": str(overlays_root),
        "num_samples_collected": len(records),
        "num_beam_power_maps_gif": len(list(maps_root.glob("*.gif"))),
        "num_beam_filtered_overlays_gif": len(list(overlays_root.glob("*.gif"))),
        "files": records,
    }
    save_json(output_root / "collection_summary.json", summary)
    print(
        f"Collected {summary['num_beam_power_maps_gif']} beam-map GIFs and "
        f"{summary['num_beam_filtered_overlays_gif']} overlay GIFs into {output_root}"
    )


if __name__ == "__main__":
    main()
