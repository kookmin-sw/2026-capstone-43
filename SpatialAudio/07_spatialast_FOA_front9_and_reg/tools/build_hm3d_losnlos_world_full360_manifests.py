import argparse
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path("/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced")
DEFAULT_SOURCE_MANIFEST_DIR = ROOT / "manifests_stage15" / "hm3d_losnlos_100k_balanced_full360"
DEFAULT_OUTPUT_DIR = ROOT / "manifests_stage15" / "hm3d_losnlos_100k_balanced_world_full360"


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_dataset_rows(dataset_root):
    manifest_path = dataset_root / "manifests" / "dataset_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    rows = {}
    for _, row in iter_jsonl(manifest_path):
        rows[str(row["sample_id"])] = row
    return rows


def get_vec3(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(item) for item in value]
    pose = row.get("mic_pose_world")
    if "mic_pose_world" in keys and isinstance(pose, dict):
        value = pose.get("position_xyz")
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return [float(item) for item in value]
    raise KeyError(f"missing vec3 field from candidates: {keys}")


def signed_to_raw_deg(angle_deg):
    return float(angle_deg) % 360.0


def raw_to_signed_deg(angle_deg):
    raw = signed_to_raw_deg(angle_deg)
    return raw - 360.0 if raw > 180.0 else raw


def world_angles_from_row(row):
    mic = get_vec3(row, "mic_world_position", "camera_world_position", "mic_pose_world")
    source = get_vec3(row, "source_world_position", "source_pose_world")
    dx = source[0] - mic[0]
    dy = source[1] - mic[1]
    dz = source[2] - mic[2]
    horizontal = max(math.hypot(dx, dz), 1.0e-12)

    # Habitat is Y-up. This uses the same yaw convention as the generator:
    # world -Z is 0 degrees, world +X is -90 degrees, world -X is +90 degrees.
    azimuth_signed = math.degrees(math.atan2(-dx, -dz))
    elevation = math.degrees(math.atan2(dy, horizontal))
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return raw_to_signed_deg(azimuth_signed), elevation, distance


def convert_split_item(item, source_row):
    world_azimuth_deg, world_elevation_deg, world_distance_m = world_angles_from_row(source_row)
    world_azimuth_raw = signed_to_raw_deg(world_azimuth_deg)

    converted = dict(item)
    converted["local_azimuth"] = item.get("azimuth")
    converted["local_azimuth_deg"] = item.get("azimuth_deg")
    converted["local_elevation"] = item.get("elevation")
    converted["local_elevation_deg"] = item.get("elevation_deg")
    converted["azimuth_reference"] = "world_habitat_minus_z"
    converted["azimuth"] = int(round(world_azimuth_raw)) % 360
    converted["azimuth_deg"] = world_azimuth_deg
    converted["azimuth_continuous_raw_deg"] = world_azimuth_raw
    converted["elevation"] = int(round(world_elevation_deg + 90.0)) % 180
    converted["elevation_deg"] = world_elevation_deg
    converted["distance"] = int(round(world_distance_m * 2.0))
    converted["distance_m"] = round(world_distance_m, 6)
    return converted


def summarize(split_name, items):
    rounded_azimuth_counts = Counter(item["azimuth"] for item in items)
    geometry_counts = Counter(item.get("geometry_los", "unknown") for item in items)
    continuous_values = [float(item["azimuth_deg"]) for item in items]
    local_values = [float(item["local_azimuth_deg"]) for item in items if item.get("local_azimuth_deg") is not None]
    return {
        "split": split_name,
        "sample_count": len(items),
        "azimuth_reference": "world_habitat_minus_z",
        "rounded_azimuth_unique_count": len(rounded_azimuth_counts),
        "rounded_azimuth_counts": dict(sorted(rounded_azimuth_counts.items())),
        "world_azimuth_unique_count_6dp": len({round(value, 6) for value in continuous_values}),
        "local_azimuth_unique_count_6dp": len({round(value, 6) for value in local_values}),
        "world_azimuth_min": min(continuous_values) if continuous_values else None,
        "world_azimuth_max": max(continuous_values) if continuous_values else None,
        "geometry_counts": dict(sorted(geometry_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source_manifest_dir", type=Path, default=DEFAULT_SOURCE_MANIFEST_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_rows = load_dataset_rows(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_path in sorted(args.source_manifest_dir.glob("*.json")):
        if split_path.name.startswith("summary"):
            continue
        split_name = split_path.stem
        source_items = json.loads(split_path.read_text(encoding="utf-8"))
        converted_items = []
        missing_sample_ids = []
        for item in source_items:
            sample_id = str(item["sample_id"])
            source_row = dataset_rows.get(sample_id)
            if source_row is None:
                missing_sample_ids.append(sample_id)
                continue
            converted_items.append(convert_split_item(item, source_row))
        if missing_sample_ids:
            preview = ", ".join(missing_sample_ids[:10])
            raise KeyError(f"{split_name}: missing {len(missing_sample_ids)} sample ids in dataset manifest: {preview}")

        (args.output_dir / split_path.name).write_text(
            json.dumps(converted_items, indent=2),
            encoding="utf-8",
        )
        summary[split_name] = summarize(split_name, converted_items)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()
