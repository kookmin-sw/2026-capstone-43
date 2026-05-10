from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def _load_scene_dataset_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_scene_dataset_config(
    source_payload: dict,
    *,
    output_path: Path,
    scene_ids: list[str],
) -> None:
    payload = dict(source_payload)
    payload["stages"] = dict(source_payload.get("stages", {}))
    payload["stages"]["paths"] = {
        ".glb": [f"{scene_id}/*.basis.glb" for scene_id in scene_ids]
    }

    scene_instances = dict(source_payload.get("scene_instances", {}))
    scene_instances["paths"] = {".json": []}
    payload["scene_instances"] = scene_instances

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def build_merged_root(
    *,
    habitat_root: Path,
    semantic_annots_root: Path,
    output_root: Path,
    source_dataset_config: Path,
    overwrite: bool,
) -> dict[str, int | str]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_ids: list[str] = []
    missing_habitat = 0
    missing_assets = 0

    for annot_dir in sorted(p for p in semantic_annots_root.iterdir() if p.is_dir()):
        scene_id = annot_dir.name
        scene_name = scene_id.split("-", 1)[-1] if "-" in scene_id else scene_id
        habitat_dir = habitat_root / scene_id
        if not habitat_dir.exists():
            missing_habitat += 1
            continue

        basis_glb = habitat_dir / f"{scene_name}.basis.glb"
        basis_navmesh = habitat_dir / f"{scene_name}.basis.navmesh"
        semantic_glb = annot_dir / f"{scene_name}.semantic.glb"
        semantic_txt = annot_dir / f"{scene_name}.semantic.txt"
        if not (basis_glb.exists() and basis_navmesh.exists() and semantic_glb.exists() and semantic_txt.exists()):
            missing_assets += 1
            continue

        out_dir = output_root / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _safe_symlink(basis_glb, out_dir / basis_glb.name)
        _safe_symlink(basis_navmesh, out_dir / basis_navmesh.name)
        _safe_symlink(semantic_glb, out_dir / semantic_glb.name)
        _safe_symlink(semantic_txt, out_dir / semantic_txt.name)

        # Some Habitat-Sim builds derive the semantic asset name from the full
        # stage basename including the `.basis` infix.
        _safe_symlink(semantic_glb, out_dir / f"{scene_name}.basis.semantic.glb")
        _safe_symlink(semantic_txt, out_dir / f"{scene_name}.basis.semantic.txt")
        scene_ids.append(scene_id)

    payload = _load_scene_dataset_config(source_dataset_config)
    output_dataset_config = output_root / source_dataset_config.name
    _write_scene_dataset_config(
        payload,
        output_path=output_dataset_config,
        scene_ids=scene_ids,
    )
    return {
        "output_root": str(output_root),
        "output_dataset_config": str(output_dataset_config),
        "num_scenes": len(scene_ids),
        "missing_habitat_dirs": missing_habitat,
        "missing_assets": missing_assets,
    }


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a merged HM3D semantic habitat root with symlinked assets.")
    parser.add_argument("--habitat-root", type=Path, required=True)
    parser.add_argument("--semantic-annots-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-dataset-config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_arg_parser().parse_args()
    summary = build_merged_root(
        habitat_root=args.habitat_root.resolve(),
        semantic_annots_root=args.semantic_annots_root.resolve(),
        output_root=args.output_root.resolve(),
        source_dataset_config=args.source_dataset_config.resolve(),
        overwrite=bool(args.overwrite),
    )
    print(summary)


if __name__ == "__main__":
    main()
