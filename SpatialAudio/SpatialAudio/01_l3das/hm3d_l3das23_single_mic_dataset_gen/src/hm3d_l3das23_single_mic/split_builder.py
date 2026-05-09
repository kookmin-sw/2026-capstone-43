from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import DatasetGenerationConfig
from .manifest_io import iter_dataset_rows
from .schemas import SceneInfo


def discover_hm3d_scenes(config: DatasetGenerationConfig) -> list[SceneInfo]:
    scene_paths = sorted(config.paths.hm3d_root.glob(config.paths.hm3d_scene_glob))
    scenes: list[SceneInfo] = []
    for scene_path in scene_paths:
        if not scene_path.is_file():
            continue
        scene_id = scene_path.parent.name
        scenes.append(SceneInfo(scene_id=scene_id, scene_path=scene_path))
    return scenes


def _allocate_counts(scene_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if scene_count < 0:
        raise ValueError("scene_count must be non-negative")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-6):
        raise ValueError("split ratios must sum to 1.0")

    raw = [scene_count * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = scene_count - sum(counts)
    order = sorted(
        range(3),
        key=lambda idx: (raw[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def build_scene_splits(
    scene_ids: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    shuffled = list(scene_ids)
    random.Random(int(seed)).shuffle(shuffled)
    n_train, n_val, _ = _allocate_counts(
        len(shuffled),
        (float(train_ratio), float(val_ratio), float(test_ratio)),
    )
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _collect_generated_samples(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    samples_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_dataset_rows(dataset_root):
        split = payload.get("split", "unknown")
        samples_by_split[split].append(
            {
                "sample_id": payload.get("sample_id"),
                "scene_id": payload.get("scene_id"),
                "source_id": payload.get("source_id"),
                "metadata_relpath": payload.get("metadata_path", ""),
            }
        )
    return samples_by_split


def write_split_manifests(
    config: DatasetGenerationConfig,
    split_map: dict[str, list[str]],
) -> None:
    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    samples_by_split = _collect_generated_samples(config.paths.dataset_root)

    for split_name in ("train", "val", "test"):
        payload = {
            "split": split_name,
            "seed": config.splits.seed,
            "scene_ids": split_map.get(split_name, []),
            "scene_count": len(split_map.get(split_name, [])),
            "samples": samples_by_split.get(split_name, []),
            "sample_count": len(samples_by_split.get(split_name, [])),
        }
        output_path = config.manifests_dir / f"{split_name}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
