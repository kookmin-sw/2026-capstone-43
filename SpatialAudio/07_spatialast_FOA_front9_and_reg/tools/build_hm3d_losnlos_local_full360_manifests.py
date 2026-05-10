import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path("/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced")
DEFAULT_OUTPUT_DIR = ROOT / "manifests_stage15" / "hm3d_losnlos_100k_balanced_local_full360"
DEFAULT_GLOS_OUTPUT_DIR = ROOT / "manifests_stage15" / "hm3d_losnlos_100k_balanced_local_full360_glos"

EXPECTED_AMBIX_CONVENTIONS = {
    "audio_channel_order": "WYZX",
    "foa_raw_channel_order": "WYZX",
    "foa_canonical_channel_order": "WYZX",
    "foa_canonical_axes": "AmbiX_ACN_SN3D_W,Y_left,Z_up,X_front",
    "local_coordinate_frame": "mic_local_right_front_up",
    "azimuth_reference": "mic_local_source_relative",
    "azimuth_convention": "ambix_acn_sn3d_positive_left",
}


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def signed_to_raw_deg(angle_deg):
    return float(angle_deg) % 360.0


def local_angles_from_relative_xyz(relative_xyz):
    # source_mic_relative_position is [right, front, up].
    # AmbiX azimuth is positive toward listener-left, so left = -right.
    x, y, z = [float(value) for value in relative_xyz]
    horizontal = max(math.hypot(x, y), 1.0e-12)
    azimuth_deg = math.degrees(math.atan2(-x, y))
    elevation_deg = math.degrees(math.atan2(z, horizontal))
    distance_m = math.sqrt(x * x + y * y + z * z)
    return azimuth_deg, elevation_deg, distance_m


def circular_diff_deg(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def validate_ambix_conventions(entry, strict):
    mismatches = []
    for key, expected in EXPECTED_AMBIX_CONVENTIONS.items():
        actual = entry.get(key)
        if actual is None and not strict:
            continue
        if actual != expected:
            mismatches.append(f"{key}={actual!r} expected {expected!r}")

    if mismatches:
        sample_id = entry.get("sample_id", "<unknown>")
        raise ValueError(
            f"Sample {sample_id} does not match mic-local AmbiX ACN/SN3D WYZX conventions: "
            + "; ".join(mismatches)
        )


def normalize_azimuth_deg(azimuth_deg):
    return ((float(azimuth_deg) + 180.0) % 360.0) - 180.0


def ambix_direction_8way(azimuth_deg):
    azimuth_norm = normalize_azimuth_deg(azimuth_deg)
    if -22.5 <= azimuth_norm < 22.5:
        return "front"
    if 22.5 <= azimuth_norm < 67.5:
        return "front-left"
    if 67.5 <= azimuth_norm < 112.5:
        return "left"
    if 112.5 <= azimuth_norm < 157.5:
        return "back-left"
    if azimuth_norm >= 157.5 or azimuth_norm < -157.5:
        return "back"
    if -157.5 <= azimuth_norm < -112.5:
        return "back-right"
    if -112.5 <= azimuth_norm < -67.5:
        return "right"
    return "front-right"


def choose_audio_relpath(entry):
    output_files = entry.get("output_files", {})
    audio_relpath = entry.get("foa_audio_path") or output_files.get("audio_foa_wav")
    if not audio_relpath:
        raise KeyError(f"foa audio path missing for sample {entry.get('sample_id', '<unknown>')}")
    return audio_relpath


def choose_metadata_relpath(entry):
    output_files = entry.get("output_files", {})
    return entry.get("metadata_path") or output_files.get("metadata_json", "")


def convert_entry(entry, dataset_root, strict_audio_conventions=False):
    validate_ambix_conventions(entry, strict_audio_conventions)

    relative_xyz = entry.get("source_mic_relative_position")
    if not isinstance(relative_xyz, (list, tuple)) or len(relative_xyz) != 3:
        raise KeyError(f"source_mic_relative_position missing for sample {entry.get('sample_id', '<unknown>')}")

    computed_azimuth_deg, computed_elevation_deg, distance_m = local_angles_from_relative_xyz(relative_xyz)
    azimuth_deg = float(entry.get("continuous_azimuth_deg", entry.get("azimuth_deg", computed_azimuth_deg)))
    elevation_deg = float(entry.get("continuous_elevation_deg", entry.get("elevation_deg", computed_elevation_deg)))
    azimuth_diff = circular_diff_deg(azimuth_deg, computed_azimuth_deg)
    elevation_diff = elevation_deg - computed_elevation_deg
    if abs(azimuth_diff) > 1.0e-3 or abs(elevation_diff) > 1.0e-3:
        sample_id = entry.get("sample_id", "<unknown>")
        raise ValueError(
            f"Sample {sample_id} metadata angles disagree with source_mic_relative_position: "
            f"azimuth diff={azimuth_diff:.6f} deg, elevation diff={elevation_diff:.6f} deg"
        )

    azimuth_raw = signed_to_raw_deg(azimuth_deg)
    direction_8way = ambix_direction_8way(azimuth_deg)

    converted = {
        "sample_id": entry["sample_id"],
        "scene_id": entry["scene_id"],
        "source_dataset": dataset_root.name,
        "audio_path": choose_audio_relpath(entry),
        "source_metadata_path": choose_metadata_relpath(entry),
        "distance": int(round(distance_m * 2.0)),
        "azimuth": int(round(azimuth_raw)) % 360,
        "elevation": int(round(elevation_deg + 90.0)) % 180,
        "distance_m": round(distance_m, 6),
        "azimuth_deg": azimuth_deg,
        "azimuth_continuous_raw_deg": azimuth_raw,
        "elevation_deg": elevation_deg,
        "azimuth_reference": "mic_local_source_relative",
        "azimuth_convention": "ambix_acn_sn3d_positive_left",
        "audio_format": entry.get("audio_format", "foa_ambisonics_4ch"),
        "audio_channel_layout": entry.get("audio_channel_layout", "ambisonics"),
        "audio_channel_order": entry.get("audio_channel_order", "WYZX"),
        "foa_raw_channel_order": entry.get("foa_raw_channel_order", "WYZX"),
        "foa_canonical_channel_order": entry.get("foa_canonical_channel_order", "WYZX"),
        "foa_canonical_axes": entry.get("foa_canonical_axes", "AmbiX_ACN_SN3D_W,Y_left,Z_up,X_front"),
        "local_coordinate_frame": entry.get("local_coordinate_frame", "mic_local_right_front_up"),
        "direction_8way": direction_8way,
        "label_8way": direction_8way,
        "source_mic_relative_position": [round(float(value), 6) for value in relative_xyz],
        "geometry_los": entry.get("geometry_los", "unknown"),
        "in_fov": bool(entry.get("in_fov", entry.get("is_in_fov", False))),
        "is_los": bool(entry.get("is_los", False)),
        "is_nlos": bool(entry.get("is_nlos", False)),
    }

    converted["metadata_local_azimuth_deg"] = float(entry.get("continuous_azimuth_deg", entry.get("azimuth_deg", azimuth_deg)))
    converted["metadata_local_azimuth_diff_deg"] = azimuth_diff
    converted["metadata_local_elevation_deg"] = float(entry.get("continuous_elevation_deg", entry.get("elevation_deg", elevation_deg)))
    converted["metadata_local_elevation_diff_deg"] = elevation_diff
    return converted


def load_entries(dataset_root, require_in_fov, strict_audio_conventions, skip_audio_exists_check):
    manifest_path = dataset_root / "manifests" / "dataset_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")

    entries = []
    for _, raw_entry in iter_jsonl(manifest_path):
        if require_in_fov and not bool(raw_entry.get("in_fov", raw_entry.get("is_in_fov", False))):
            continue
        converted = convert_entry(raw_entry, dataset_root, strict_audio_conventions=strict_audio_conventions)
        if not skip_audio_exists_check:
            audio_path = dataset_root / converted["audio_path"]
            if not audio_path.exists():
                raise FileNotFoundError(f"missing audio file for {converted['sample_id']}: {audio_path}")
        entries.append(converted)

    if not entries:
        raise ValueError("no entries found after filtering")
    return entries


def group_entries_by_scene(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["scene_id"]].append(entry)
    return grouped


def assign_scenes(scene_entries, train_ratio, val_ratio, test_ratio, seed):
    scene_items = list(scene_entries.items())
    rng = random.Random(seed)
    rng.shuffle(scene_items)
    scene_items.sort(key=lambda item: len(item[1]), reverse=True)

    splits = [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]
    splits = [(name, ratio) for name, ratio in splits if ratio > 0.0]
    if len(scene_items) < len(splits):
        raise ValueError("not enough scenes to populate requested splits")

    total_samples = sum(len(items) for _, items in scene_items)
    total_los = sum(sum(1 for item in items if item["is_los"]) for _, items in scene_items)
    target_samples = {name: total_samples * ratio for name, ratio in splits}
    target_los = {name: total_los * ratio for name, ratio in splits}

    assignments = {name: [] for name, _ in splits}
    current_samples = Counter()
    current_los = Counter()

    for (scene_id, items), (split_name, _) in zip(scene_items, splits):
        assignments[split_name].append(scene_id)
        current_samples[split_name] += len(items)
        current_los[split_name] += sum(1 for item in items if item["is_los"])

    for scene_id, items in scene_items[len(splits):]:
        item_count = len(items)
        los_count = sum(1 for item in items if item["is_los"])
        best_split = None
        best_score = None
        for split_name, _ in splits:
            new_sample_count = current_samples[split_name] + item_count
            new_los_count = current_los[split_name] + los_count
            sample_error = (new_sample_count - target_samples[split_name]) / max(target_samples[split_name], 1.0)
            los_error = (new_los_count - target_los[split_name]) / max(target_los[split_name], 1.0)
            score = sample_error * sample_error + 0.35 * los_error * los_error
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assignments[best_split].append(scene_id)
        current_samples[best_split] += item_count
        current_los[best_split] += los_count
    return assignments


def flatten_split(scene_entries, scene_ids, seed):
    data = []
    for scene_id in scene_ids:
        data.extend(scene_entries[scene_id])
    rng = random.Random(seed)
    rng.shuffle(data)
    return data


def summarize_split(split_name, data, scene_ids):
    continuous_values = [float(item["azimuth_deg"]) for item in data]
    metadata_diffs = [abs(float(item.get("metadata_local_azimuth_diff_deg", 0.0))) for item in data]
    return {
        "split": split_name,
        "sample_count": len(data),
        "scene_count": len(scene_ids),
        "scene_ids": sorted(scene_ids),
        "azimuth_reference": "mic_local_source_relative",
        "rounded_azimuth_unique_count": len({int(item["azimuth"]) for item in data}),
        "local_azimuth_unique_count_6dp": len({round(value, 6) for value in continuous_values}),
        "local_azimuth_min": min(continuous_values) if continuous_values else None,
        "local_azimuth_max": max(continuous_values) if continuous_values else None,
        "max_metadata_local_azimuth_abs_diff_deg": max(metadata_diffs) if metadata_diffs else 0.0,
        "geometry_counts": dict(sorted(Counter(item.get("geometry_los", "unknown") for item in data).items())),
        "direction_8way_counts": dict(sorted(Counter(item.get("direction_8way", "unknown") for item in data).items())),
    }


def write_json(path, payload, compact_json):
    if compact_json:
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_split_manifests(output_dir, split_data, split_scene_ids, compact_json=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_name, data in split_data.items():
        write_json(output_dir / f"{split_name}.json", data, compact_json)
        summary[split_name] = summarize_split(split_name, data, split_scene_ids[split_name])
    write_json(output_dir / "summary.json", summary, compact_json)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--glos_output_dir", type=Path, default=DEFAULT_GLOS_OUTPUT_DIR)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--require_in_fov", action="store_true", default=False)
    parser.add_argument(
        "--strict_audio_conventions",
        action="store_true",
        default=False,
        help="Require stored FOA to be mic-local AmbiX ACN/SN3D in WYZX order.",
    )
    parser.add_argument(
        "--skip_audio_exists_check",
        action="store_true",
        default=False,
        help="Trust manifest paths without stat-ing every audio file. Useful for very large external disks.",
    )
    parser.add_argument(
        "--compact_json",
        action="store_true",
        default=False,
        help="Write compact JSON manifests to reduce output size and serialization time.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    entries = load_entries(
        args.dataset_root,
        args.require_in_fov,
        args.strict_audio_conventions,
        args.skip_audio_exists_check,
    )
    scene_entries = group_entries_by_scene(entries)
    assignments = assign_scenes(scene_entries, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    split_data = {
        split_name: flatten_split(scene_entries, scene_ids, args.seed)
        for split_name, scene_ids in assignments.items()
    }
    summary = write_split_manifests(args.output_dir, split_data, assignments, compact_json=args.compact_json)

    glos_data = {
        split_name: [item for item in data if item.get("geometry_los") == "gLOS"]
        for split_name, data in split_data.items()
    }
    glos_scene_ids = {
        split_name: sorted({item["scene_id"] for item in data})
        for split_name, data in glos_data.items()
    }
    glos_summary = write_split_manifests(
        args.glos_output_dir,
        glos_data,
        glos_scene_ids,
        compact_json=args.compact_json,
    )

    print(json.dumps({"all": summary, "glos": glos_summary}, indent=2)[:8000])


if __name__ == "__main__":
    main()
