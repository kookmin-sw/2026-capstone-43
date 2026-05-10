from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


DATASET_MANIFEST_FILENAME = "dataset_manifest.jsonl"


def dataset_manifest_path(dataset_root: Path) -> Path:
    return dataset_root / "manifests" / DATASET_MANIFEST_FILENAME


@contextmanager
def _open_locked_manifest(
    manifest_path: Path,
    mode: str,
    *,
    exclusive: bool,
) -> Iterator[TextIO]:
    with manifest_path.open(mode, encoding="utf-8") as handle:
        if fcntl is not None:
            lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(handle.fileno(), lock_mode)
        try:
            yield handle
            if any(flag in mode for flag in ("a", "w", "+")):
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def iter_dataset_rows(dataset_root: Path) -> Iterable[dict[str, Any]]:
    manifest_path = dataset_manifest_path(dataset_root)
    if manifest_path.exists():
        with _open_locked_manifest(manifest_path, "r", exclusive=False) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    for metadata_path in sorted(dataset_root.glob("scenes/*/samples/*/metadata/sample.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            yield json.load(handle)


def load_manifest_sample_ids(dataset_root: Path) -> set[str]:
    manifest_path = dataset_manifest_path(dataset_root)
    if not manifest_path.exists():
        return set()

    seen: set[str] = set()
    with _open_locked_manifest(manifest_path, "r", exclusive=False) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id:
                seen.add(sample_id)
    return seen


def append_manifest_row(dataset_root: Path, row: dict[str, Any]) -> Path:
    manifest_path = dataset_manifest_path(dataset_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_locked_manifest(manifest_path, "a", exclusive=True) as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return manifest_path


def output_relpath(
    row: dict[str, Any],
    output_key: str,
    *fallback_fields: str,
) -> str:
    output_files = row.get("output_files") or {}
    value = output_files.get(output_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    for field_name in fallback_fields:
        fallback = row.get(field_name)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return ""


def row_azimuth_deg(row: dict[str, Any]) -> float:
    value = row.get("continuous_azimuth_deg", row.get("azimuth_deg", 0.0))
    return float(value)


def row_geometry_los(row: dict[str, Any]) -> str:
    return str(row.get("geometry_los", "")).strip()


def row_in_fov(row: dict[str, Any]) -> bool:
    if "is_in_fov" in row:
        return bool(row.get("is_in_fov"))
    return bool(row.get("in_fov"))


def row_scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id", "")).strip()


def row_sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", "")).strip()


def row_source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_id", "")).strip()


def row_visibility_ratio(row: dict[str, Any]) -> Optional[float]:
    value = row.get("visible_ratio", row.get("visibility_ratio"))
    if isinstance(value, (int, float)):
        return float(value)
    return None
