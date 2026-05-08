import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path("/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced")
DEFAULT_OUTPUT_ROOT = ROOT / "manifests_stage15" / "hm3d_losnlos_100k_balanced_front9"


def iter_jsonl(path):
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def is_front_cone(azimuth_deg):
    azimuth_deg = float(azimuth_deg)
    return -45.0 <= azimuth_deg <= 45.0


def azimuth_deg_to_raw(azimuth_deg):
    return int(round(float(azimuth_deg))) % 360


def elevation_deg_to_raw(elevation_deg):
    return int(round(float(elevation_deg) + 90.0)) % 180


def distance_m_to_bin(distance_m):
    return int(round(float(distance_m) * 2.0))


def choose_distance_m(entry):
    for key in ("distance_to_mic", "source_distance", "euclidean_distance"):
        value = entry.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"distance field missing for sample {entry.get('sample_id', '<unknown>')}")


def choose_audio_relpath(entry):
    output_files = entry.get("output_files", {})
    audio_relpath = entry.get("foa_audio_path") or output_files.get("audio_foa_wav")
    if not audio_relpath:
        raise KeyError(f"foa audio path missing for sample {entry.get('sample_id', '<unknown>')}")
    return audio_relpath


def choose_metadata_relpath(entry):
    output_files = entry.get("output_files", {})
    return entry.get("metadata_path") or output_files.get("metadata_json", "")


def convert_entry(entry, dataset_root):
    azimuth_deg = float(entry.get("continuous_azimuth_deg", entry["azimuth_deg"]))
    elevation_deg = float(entry.get("continuous_elevation_deg", entry["elevation_deg"]))
    distance_m = choose_distance_m(entry)
    audio_relpath = choose_audio_relpath(entry)
    metadata_relpath = choose_metadata_relpath(entry)

    return {
        "sample_id": entry["sample_id"],
        "scene_id": entry["scene_id"],
        "source_dataset": dataset_root.name,
        "audio_path": audio_relpath,
        "source_metadata_path": metadata_relpath,
        "distance": distance_m_to_bin(distance_m),
        "azimuth": azimuth_deg_to_raw(azimuth_deg),
        "elevation": elevation_deg_to_raw(elevation_deg),
        "distance_m": round(distance_m, 6),
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "geometry_los": entry.get("geometry_los", "unknown"),
        "in_fov": bool(entry.get("in_fov", entry.get("is_in_fov", False))),
        "is_los": bool(entry.get("is_los", False)),
        "is_nlos": bool(entry.get("is_nlos", False)),
    }


def load_front_entries(dataset_root, require_in_fov):
    manifest_path = dataset_root / "manifests" / "dataset_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")

    entries = []
    missing_audio = []
    missing_metadata = []

    for line_no, raw_entry in iter_jsonl(manifest_path):
        azimuth_deg = float(raw_entry.get("continuous_azimuth_deg", raw_entry["azimuth_deg"]))
        if not is_front_cone(azimuth_deg):
            continue
        if require_in_fov and not bool(raw_entry.get("in_fov", raw_entry.get("is_in_fov", False))):
            continue

        converted = convert_entry(raw_entry, dataset_root)
        audio_path = dataset_root / converted["audio_path"]
        metadata_path = dataset_root / converted["source_metadata_path"] if converted["source_metadata_path"] else None

        if not audio_path.exists():
            missing_audio.append((line_no, converted["sample_id"], str(audio_path)))
            continue
        if metadata_path is not None and not metadata_path.exists():
            missing_metadata.append((line_no, converted["sample_id"], str(metadata_path)))
            continue

        entries.append(converted)

    if missing_audio:
        preview = "\n".join(f"line {line_no}: {sample_id} -> {path}" for line_no, sample_id, path in missing_audio[:10])
        raise FileNotFoundError(f"missing audio files ({len(missing_audio)} total)\n{preview}")
    if missing_metadata:
        preview = "\n".join(f"line {line_no}: {sample_id} -> {path}" for line_no, sample_id, path in missing_metadata[:10])
        raise FileNotFoundError(f"missing metadata files ({len(missing_metadata)} total)\n{preview}")
    if not entries:
        raise ValueError("no front-cone entries found after filtering")

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
    azimuth_counter = Counter(item["azimuth"] for item in data)
    los_counter = Counter("LOS" if item["is_los"] else "NLOS" for item in data)
    elevation_counter = Counter(item["elevation"] for item in data)
    return {
        "split": split_name,
        "sample_count": len(data),
        "scene_count": len(scene_ids),
        "scene_ids": sorted(scene_ids),
        "azimuth_counts": dict(sorted(azimuth_counter.items())),
        "los_counts": dict(sorted(los_counter.items())),
        "elevation_counts": dict(sorted(elevation_counter.items())),
    }


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--require_in_fov", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    front_entries = load_front_entries(dataset_root, require_in_fov=args.require_in_fov)
    scene_entries = group_entries_by_scene(front_entries)
    assignments = assign_scenes(
        scene_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_payloads = {}
    split_summaries = []
    for offset, split_name in enumerate(("train", "val", "test")):
        scene_ids = assignments.get(split_name, [])
        if not scene_ids:
            continue
        payload = flatten_split(scene_entries, scene_ids, seed=args.seed + offset)
        split_payloads[split_name] = payload
        split_summaries.append(summarize_split(split_name, payload, scene_ids))
        write_json(output_dir / f"{split_name}.json", payload)

    summary = {
        "dataset_root": str(dataset_root),
        "source_manifest": str(dataset_root / "manifests" / "dataset_manifest.jsonl"),
        "require_in_fov": args.require_in_fov,
        "seed": args.seed,
        "front_sample_count": len(front_entries),
        "front_scene_count": len(scene_entries),
        "splits": split_summaries,
    }
    write_json(output_dir / "summary.json", summary)

    for split in split_summaries:
        print(
            f"{split['split']}: "
            f"{split['sample_count']} samples across {split['scene_count']} scenes, "
            f"LOS/NLOS={split['los_counts']}"
        )
    print(f"saved manifests to {output_dir}")


if __name__ == "__main__":
    main()
