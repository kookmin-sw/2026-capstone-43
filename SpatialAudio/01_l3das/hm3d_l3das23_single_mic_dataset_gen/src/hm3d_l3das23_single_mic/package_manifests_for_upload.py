from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def iter_manifest_rows(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect only the audio/image assets referenced by one or more JSONL manifests and "
            "package them into a compact upload root."
        )
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve()
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    seen_audio: set[str] = set()
    seen_image: set[str] = set()
    copied_manifests: list[str] = []

    for manifest_path in [path.resolve() for path in args.manifest_paths]:
        target_manifest_path = manifests_dir / manifest_path.name
        shutil.copy2(manifest_path, target_manifest_path)
        copied_manifests.append(str(target_manifest_path))

        for row in iter_manifest_rows(manifest_path):
            audio_rel = row.get("audio_path")
            if isinstance(audio_rel, str) and audio_rel.strip():
                audio_rel = audio_rel.strip()
                if audio_rel not in seen_audio:
                    link_or_copy(asset_root / audio_rel, output_root / audio_rel)
                    seen_audio.add(audio_rel)

            image_rel = row.get("image_path")
            if isinstance(image_rel, str) and image_rel.strip():
                image_rel = image_rel.strip()
                if image_rel not in seen_image:
                    link_or_copy(asset_root / image_rel, output_root / image_rel)
                    seen_image.add(image_rel)

    summary = {
        "asset_root": str(asset_root),
        "output_root": str(output_root),
        "manifest_count": len(copied_manifests),
        "audio_files": len(seen_audio),
        "image_files": len(seen_image),
        "manifests": copied_manifests,
    }
    summary_path = output_root / "packaging_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
