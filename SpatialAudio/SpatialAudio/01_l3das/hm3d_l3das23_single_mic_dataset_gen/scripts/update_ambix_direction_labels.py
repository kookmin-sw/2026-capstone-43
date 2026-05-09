#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


DEFAULT_DATASET_ROOT = Path("/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced")
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hm3d_l3das23_single_mic.geometry import cylindrical_from_local_xyz, world_to_local  # noqa: E402
from hm3d_l3das23_single_mic.spatial_conventions import (  # noqa: E402
    AZIMUTH_CONVENTION,
    AZIMUTH_REFERENCE,
    FOA_CANONICAL_AXES,
    FOA_CANONICAL_CHANNEL_ORDER,
    FOA_RAW_CHANNEL_ORDER,
    LOCAL_COORDINATE_FRAME,
    ambix_unit_vector_xyz,
    azimuth_raw_deg,
    direction_8way_from_azimuth,
    local_angles_from_relative_xyz,
    local_unit_vector_right_front_up,
)


def rounded_list(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def as_float_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def choose_source_world(row: dict[str, Any]) -> list[float] | None:
    for key in ("source_world_position", "source_pose_world", "speaker_proxy_reference_world"):
        value = as_float_xyz(row.get(key))
        if value is not None:
            return value
    return None


def choose_mic_pose(row: dict[str, Any]) -> tuple[list[float], float] | None:
    pose = row.get("mic_pose_world")
    if isinstance(pose, dict):
        position = as_float_xyz(pose.get("position_xyz"))
        yaw = pose.get("yaw_rad")
        if position is not None and isinstance(yaw, (int, float)):
            return position, float(yaw)
    return None


def choose_relative_xyz(row: dict[str, Any]) -> list[float]:
    source_world = choose_source_world(row)
    mic_pose = choose_mic_pose(row)
    if source_world is not None and mic_pose is not None:
        mic_position, yaw_rad = mic_pose
        return world_to_local(mic_position, yaw_rad, source_world).tolist()

    for key in ("source_mic_relative_position", "source_pose_local_xyz"):
        value = as_float_xyz(row.get(key))
        if value is not None:
            return value
    sources = row.get("sources")
    if isinstance(sources, list) and sources:
        value = as_float_xyz(sources[0].get("source_mic_relative_position"))
        if value is not None:
            return value
    raise KeyError(f"missing source_mic_relative_position for sample {row.get('sample_id', '<unknown>')}")


def update_row(row: dict[str, Any]) -> dict[str, Any]:
    relative_xyz = choose_relative_xyz(row)
    azimuth_deg, elevation_deg, distance_m = local_angles_from_relative_xyz(relative_xyz)
    direction = direction_8way_from_azimuth(azimuth_deg)
    ambix_unit_vector = rounded_list(ambix_unit_vector_xyz(relative_xyz))
    local_unit_vector = rounded_list(local_unit_vector_right_front_up(relative_xyz))

    row["source_mic_relative_position"] = rounded_list(relative_xyz)
    row["source_pose_local_xyz"] = rounded_list(relative_xyz)
    row["source_distance"] = round(distance_m, 6)
    row["distance_to_mic"] = round(distance_m, 6)
    row["euclidean_distance"] = round(distance_m, 6)
    row["direct_path_length"] = round(distance_m, 6)
    row["azimuth_deg"] = round(azimuth_deg, 6)
    row["elevation_deg"] = round(elevation_deg, 6)
    row["continuous_azimuth_deg"] = round(azimuth_deg, 6)
    row["continuous_elevation_deg"] = round(elevation_deg, 6)
    row["local_coordinate_frame"] = LOCAL_COORDINATE_FRAME
    row["azimuth_reference"] = AZIMUTH_REFERENCE
    row["azimuth_convention"] = AZIMUTH_CONVENTION
    row["azimuth_continuous_raw_deg"] = round(azimuth_raw_deg(azimuth_deg), 6)
    row["direction_8way"] = direction
    row["label_8way"] = direction
    row["ambix_unit_vector_xyz"] = ambix_unit_vector
    row["local_unit_vector_right_front_up"] = local_unit_vector
    row["audio_channel_order"] = FOA_RAW_CHANNEL_ORDER
    row["foa_raw_channel_order"] = FOA_RAW_CHANNEL_ORDER
    row["foa_canonical_channel_order"] = FOA_CANONICAL_CHANNEL_ORDER
    row["foa_canonical_axes"] = FOA_CANONICAL_AXES

    spherical = row.get("source_pose_spherical")
    if isinstance(spherical, dict):
        spherical["distance"] = round(distance_m, 6)
        spherical["azimuth_deg"] = round(azimuth_deg, 6)
        spherical["elevation_deg"] = round(elevation_deg, 6)

    cylindrical = row.get("source_pose_cylindrical")
    if isinstance(cylindrical, dict):
        rho, theta_rad, z_local = cylindrical_from_local_xyz(relative_xyz)
        cylindrical["rho"] = round(float(rho), 6)
        cylindrical["theta_rad"] = round(float(theta_rad), 6)
        cylindrical["theta_deg"] = round(azimuth_deg, 6)
        cylindrical["z"] = round(float(z_local), 6)

    sources = row.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            source["source_mic_relative_position"] = rounded_list(relative_xyz)
            source["distance_to_mic"] = round(distance_m, 6)
            source["euclidean_distance"] = round(distance_m, 6)
            source["direct_path_length"] = round(distance_m, 6)
            source["continuous_azimuth_deg"] = round(azimuth_deg, 6)
            source["continuous_elevation_deg"] = round(elevation_deg, 6)
            source["local_coordinate_frame"] = LOCAL_COORDINATE_FRAME
            source["azimuth_reference"] = AZIMUTH_REFERENCE
            source["azimuth_convention"] = AZIMUTH_CONVENTION
            source["azimuth_continuous_raw_deg"] = round(azimuth_raw_deg(azimuth_deg), 6)
            source["direction_8way"] = direction
            source["label_8way"] = direction
            source["ambix_unit_vector_xyz"] = ambix_unit_vector
            source["local_unit_vector_right_front_up"] = local_unit_vector

    return row


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(text)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def update_manifest(dataset_root: Path, *, dry_run: bool) -> tuple[int, dict[str, int]]:
    manifest_path = dataset_root / "manifests" / "dataset_manifest.jsonl"
    counts: dict[str, int] = {}
    updated_lines: list[str] = []
    num_rows = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = update_row(json.loads(line))
            counts[row["direction_8way"]] = counts.get(row["direction_8way"], 0) + 1
            updated_lines.append(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            num_rows += 1

    if not dry_run:
        atomic_write_text(manifest_path, "".join(updated_lines))
    return num_rows, counts


def update_metadata_files(dataset_root: Path, *, dry_run: bool) -> tuple[int, list[str]]:
    count = 0
    skipped: list[str] = []
    for metadata_path in sorted(dataset_root.glob("scenes/*/samples/*/metadata/sample.json")):
        try:
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(str(metadata_path.relative_to(dataset_root)))
            continue
        row = update_row(row)
        if not dry_run:
            atomic_write_text(
                metadata_path,
                json.dumps(row, indent=2, ensure_ascii=True, sort_keys=False) + "\n",
            )
        count += 1
    return count, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    manifest_rows, counts = update_manifest(dataset_root, dry_run=args.dry_run)
    metadata_rows, skipped_metadata = update_metadata_files(dataset_root, dry_run=args.dry_run)
    print(json.dumps(
        {
            "dataset_root": str(dataset_root),
            "dry_run": bool(args.dry_run),
            "manifest_rows": manifest_rows,
            "metadata_files": metadata_rows,
            "skipped_invalid_metadata_files": len(skipped_metadata),
            "skipped_invalid_metadata_preview": skipped_metadata[:20],
            "direction_8way_counts": dict(sorted(counts.items())),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
