import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_MANIFEST_ROOT = Path("/home/yu/Project_git/01_FOV_LOS/manifests")
OUTPUT_ROOT = ROOT / "manifests_stage15"


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def stratum_key(item):
    return (item["azimuth"], item["elevation"], item["geometry_los"])


def allocate_evenly(keys, target_size, capacity_map):
    base = target_size // len(keys)
    remainder = target_size % len(keys)
    allocation = {}
    for idx, key in enumerate(sorted(keys)):
        wanted = base + (1 if idx < remainder else 0)
        capacity = capacity_map[key]
        if wanted > capacity:
            raise ValueError(f"cannot allocate {wanted} samples to azimuth {key}; capacity is {capacity}")
        allocation[key] = wanted
    return allocation


def allocate_counts(groups, target_size):
    strata = list(groups.keys())
    if target_size < len(strata):
        raise ValueError(f"target_size={target_size} is smaller than number of strata={len(strata)}")

    allocation = {key: 1 for key in strata}
    remaining = target_size - len(strata)

    extra_capacity = {key: max(len(groups[key]) - 1, 0) for key in strata}
    total_extra_capacity = sum(extra_capacity.values())
    if remaining == 0 or total_extra_capacity == 0:
        return allocation

    ideal_extra = {key: remaining * extra_capacity[key] / total_extra_capacity for key in strata}
    extra_floor = {
        key: min(int(math.floor(ideal_extra[key])), extra_capacity[key])
        for key in strata
    }
    for key, value in extra_floor.items():
        allocation[key] += value

    used = sum(extra_floor.values())
    left = remaining - used
    remainders = sorted(
        strata,
        key=lambda key: (ideal_extra[key] - extra_floor[key], extra_capacity[key]),
        reverse=True,
    )
    for key in remainders:
        if left <= 0:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            left -= 1

    if left != 0:
        raise RuntimeError(f"failed to allocate all samples, left={left}")
    return allocation


def balanced_azimuth_subset(data, target_size, seed):
    rng = random.Random(seed)
    azimuth_groups = defaultdict(list)
    for item in data:
        azimuth_groups[item["azimuth"]].append(item)

    azimuth_allocation = allocate_evenly(
        azimuth_groups.keys(),
        target_size,
        {azimuth: len(items) for azimuth, items in azimuth_groups.items()},
    )

    subset = []
    for azimuth in sorted(azimuth_groups.keys()):
        nested_groups = defaultdict(list)
        for item in azimuth_groups[azimuth]:
            nested_groups[(item["elevation"], item["geometry_los"])].append(item)
        nested_allocation = allocate_counts(nested_groups, azimuth_allocation[azimuth])
        for key in sorted(nested_groups.keys()):
            items = list(nested_groups[key])
            rng.shuffle(items)
            subset.extend(items[: nested_allocation[key]])

    rng.shuffle(subset)
    if len(subset) != target_size:
        raise RuntimeError(f"subset size mismatch: expected {target_size}, got {len(subset)}")
    return subset, azimuth_allocation


def distribution(items, field):
    return Counter(item[field] for item in items)


def combo_distribution(items):
    return Counter(stratum_key(item) for item in items)


def format_counter(counter):
    return ", ".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_summary(path, split_stats):
    with open(path, "w") as f:
        f.write("# Stage-15 Subset Summary\n\n")
        f.write("Subset goal: compare front9 classification vs front regression while keeping front-cone azimuth support balanced.\n")
        f.write("Sampling mode: azimuth-balanced, with elevation/geometry stratification inside each azimuth.\n")
        f.write("Patch embedding is expected to be pretrained-initialized and trainable in the Stage-15 runs.\n\n")
        for stats in split_stats:
            f.write(f"## {stats['split'].title()}\n\n")
            f.write(f"- Source size: {stats['source_size']}\n")
            f.write(f"- Subset size: {stats['subset_size']}\n")
            f.write(f"- Target azimuth allocation: {format_counter(stats['azimuth_target'])}\n")
            f.write(f"- Source geometry_los: {format_counter(stats['source_geometry'])}\n")
            f.write(f"- Subset geometry_los: {format_counter(stats['subset_geometry'])}\n")
            f.write(f"- Source azimuth: {format_counter(stats['source_azimuth'])}\n")
            f.write(f"- Subset azimuth: {format_counter(stats['subset_azimuth'])}\n")
            f.write(f"- Source elevation: {format_counter(stats['source_elevation'])}\n")
            f.write(f"- Subset elevation: {format_counter(stats['subset_elevation'])}\n")
            f.write(f"- Distinct (azimuth, elevation, geometry_los) strata in source: {stats['source_combo_n']}\n")
            f.write(f"- Distinct (azimuth, elevation, geometry_los) strata in subset: {stats['subset_combo_n']}\n")
            f.write("\n")


def build_split(split, target_size, seed):
    source_path = SOURCE_MANIFEST_ROOT / f"{split}.json"
    data = load_manifest(source_path)
    subset, azimuth_target = balanced_azimuth_subset(data, target_size, seed)
    return {
        "split": split,
        "data": subset,
        "azimuth_target": azimuth_target,
        "source_size": len(data),
        "subset_size": len(subset),
        "source_geometry": distribution(data, "geometry_los"),
        "subset_geometry": distribution(subset, "geometry_los"),
        "source_azimuth": distribution(data, "azimuth"),
        "subset_azimuth": distribution(subset, "azimuth"),
        "source_elevation": distribution(data, "elevation"),
        "subset_elevation": distribution(subset, "elevation"),
        "source_combo_n": len(combo_distribution(data)),
        "subset_combo_n": len(combo_distribution(subset)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=600)
    parser.add_argument("--val_size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_dir", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = build_split("train", args.train_size, args.seed)
    val_stats = build_split("val", args.val_size, args.seed + 1)

    train_path = output_dir / f"train_stage15_subset_{args.train_size}.json"
    val_path = output_dir / f"val_stage15_subset_{args.val_size}.json"
    write_json(train_path, train_stats["data"])
    write_json(val_path, val_stats["data"])

    summary_path = output_dir / "stage15_subset_summary.md"
    write_summary(summary_path, [train_stats, val_stats])

    print(f"saved train subset to {train_path}")
    print(f"saved val subset to {val_path}")
    print(f"saved subset summary to {summary_path}")


if __name__ == "__main__":
    main()
