import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage4"
COMPARE_ROOT = ROOT / "outputs_stage4_comparison"

COLUMNS = [
    "run_name",
    "stem_type",
    "unfreeze_strategy",
    "recipe",
    "seed",
    "final_val_azimuth_acc",
    "final_val_elevation_acc",
    "final_val_vector_cosine",
    "final_val_azimuth_mae",
    "final_val_elevation_mae",
    "final_val_angular_error",
    "best_val_azimuth_acc",
    "best_val_elevation_acc",
    "best_val_vector_cosine",
    "best_val_azimuth_mae",
    "best_val_elevation_mae",
    "best_val_angular_error",
]


def load_run(run_dir):
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    config = metrics.get("config", {})
    final = metrics.get("final", {})
    best = metrics.get("best", {})
    return {
        "run_name": run_dir.name,
        "stem_type": config.get("foa_stem_type", ""),
        "unfreeze_strategy": config.get("unfreeze_strategy", ""),
        "recipe": config.get("recipe_name", ""),
        "seed": config.get("seed", ""),
        "final_val_azimuth_acc": final.get("val_azimuth_acc", ""),
        "final_val_elevation_acc": final.get("val_elevation_acc", ""),
        "final_val_vector_cosine": final.get("val_vector_cosine", ""),
        "final_val_azimuth_mae": final.get("val_azimuth_mae", ""),
        "final_val_elevation_mae": final.get("val_elevation_mae", ""),
        "final_val_angular_error": final.get("val_angular_error", ""),
        "best_val_azimuth_acc": best.get("val_azimuth_acc", ""),
        "best_val_elevation_acc": best.get("val_elevation_acc", ""),
        "best_val_vector_cosine": best.get("val_vector_cosine", ""),
        "best_val_azimuth_mae": best.get("val_azimuth_mae", ""),
        "best_val_elevation_mae": best.get("val_elevation_mae", ""),
        "best_val_angular_error": best.get("val_angular_error", ""),
    }


def is_seed_variant(row):
    return "_seed" in row["run_name"]


def pick_preferred_row(rows, run_name=None, key=None):
    if run_name is not None:
        for row in rows:
            if row["run_name"] == run_name:
                return row

    if key is None:
        return None

    candidates = [row for row in rows if (row["stem_type"], row["unfreeze_strategy"], row["recipe"]) == key]
    if not candidates:
        return None

    non_seed_candidates = [row for row in candidates if not is_seed_variant(row)]
    if non_seed_candidates:
        candidates = non_seed_candidates
    return sorted(candidates, key=lambda row: row["run_name"])[0]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def collect_interpretation(rows):
    lines = []
    canonical_pairs = [
        ("stage3_last2", "foa_stage3_last2", "logmel_stage3_last2"),
        ("last4", "foa_last4", "logmel_last4"),
        ("patch_plus_last4", "foa_patch_last4", "logmel_patch_last4"),
        ("patch_plus_last6", "foa_patch_last6", "logmel_patch_last6"),
    ]
    for strategy, foa_run_name, logmel_run_name in canonical_pairs:
        foa = pick_preferred_row(
            rows,
            run_name=foa_run_name,
            key=("foa_native", strategy, "default_recipe"),
        )
        logmel = pick_preferred_row(
            rows,
            run_name=logmel_run_name,
            key=("logmel_only", strategy, "default_recipe"),
        )
        if logmel and foa:
            if foa["best_val_angular_error"] < logmel["best_val_angular_error"]:
                verdict = "foa_native better"
            elif foa["best_val_angular_error"] > logmel["best_val_angular_error"]:
                verdict = "logmel_only better"
            else:
                verdict = "tie"
            lines.append(
                f"- Same unfreeze `{strategy}`: {verdict} "
                f"(best angular error {foa['best_val_angular_error']} vs {logmel['best_val_angular_error']})."
            )

    foa_stage3 = pick_preferred_row(
        rows,
        run_name="foa_stage3_last2",
        key=("foa_native", "stage3_last2", "default_recipe"),
    )
    foa_last4 = pick_preferred_row(
        rows,
        run_name="foa_last4",
        key=("foa_native", "last4", "default_recipe"),
    )
    foa_patch_last4 = pick_preferred_row(
        rows,
        run_name="foa_patch_last4",
        key=("foa_native", "patch_plus_last4", "default_recipe"),
    )
    logmel_stage3 = pick_preferred_row(
        rows,
        run_name="logmel_stage3_last2",
        key=("logmel_only", "stage3_last2", "default_recipe"),
    )
    logmel_last4 = pick_preferred_row(
        rows,
        run_name="logmel_last4",
        key=("logmel_only", "last4", "default_recipe"),
    )
    logmel_patch_last4 = pick_preferred_row(
        rows,
        run_name="logmel_patch_last4",
        key=("logmel_only", "patch_plus_last4", "default_recipe"),
    )

    if foa_stage3 and foa_last4 and logmel_stage3 and logmel_last4:
        foa_gain = foa_stage3["best_val_angular_error"] - foa_last4["best_val_angular_error"]
        logmel_gain = logmel_stage3["best_val_angular_error"] - logmel_last4["best_val_angular_error"]
        lines.append(
            f"- Larger unfreeze (`stage3_last2` -> `last4`) gain on angular error: "
            f"foa_native {foa_gain:.4f}, logmel_only {logmel_gain:.4f}."
        )

    if foa_last4 and foa_patch_last4:
        lines.append(
            f"- foa_native `patch_plus_last4` vs `last4`: "
            f"{foa_patch_last4['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} best angular error."
        )
    if logmel_last4 and logmel_patch_last4:
        lines.append(
            f"- logmel_only `patch_plus_last4` vs `last4`: "
            f"{logmel_patch_last4['best_val_angular_error']} vs {logmel_last4['best_val_angular_error']} best angular error."
        )

    recipe_variants = [
        ("longer_run", "foa_last4_longer"),
        ("cosine_decay", "foa_last4_cosine"),
        ("lower_head_lr", "foa_last4_lower_head_lr"),
    ]
    for recipe, run_name in recipe_variants:
        variant = pick_preferred_row(
            rows,
            run_name=run_name,
            key=("foa_native", "last4", recipe),
        )
        if foa_last4 and variant:
            lines.append(
                f"- foa_native recipe `{recipe}` vs default: "
                f"{variant['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} best angular error."
            )

    foa_last4_longer = pick_preferred_row(
        rows,
        run_name="foa_last4_longer",
        key=("foa_native", "last4", "longer_run"),
    )
    longer_combo_variants = [
        ("longer_run_cosine", "foa_last4_longer_cosine"),
        ("longer_run_lower_head_lr", "foa_last4_longer_lower_head_lr"),
    ]
    for recipe, run_name in longer_combo_variants:
        variant = pick_preferred_row(
            rows,
            run_name=run_name,
            key=("foa_native", "last4", recipe),
        )
        if foa_last4_longer and variant:
            lines.append(
                f"- foa_native longer recipe `{recipe}` vs longer_run: "
                f"{variant['best_val_angular_error']} vs {foa_last4_longer['best_val_angular_error']} best angular error."
            )

    longer_seed_rows = sorted(
        [
            row
            for row in rows
            if row["stem_type"] == "foa_native"
            and row["unfreeze_strategy"] == "last4"
            and row["recipe"] == "longer_run"
            and is_seed_variant(row)
        ],
        key=lambda row: row["run_name"],
    )
    if longer_seed_rows:
        seed_summary = ", ".join(
            f"{row['run_name']} (seed {row['seed']}): {row['best_val_angular_error']}"
            for row in longer_seed_rows
        )
        lines.append(f"- foa_native `longer_run` seeded repeats: {seed_summary}.")

    return lines


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# Stage-4 Comparison\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in COLUMNS]
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Interpretation Summary\n\n")
        for line in collect_interpretation(rows):
            f.write(line + "\n")

        f.write("\n## Interpretation Rules\n\n")
        f.write("- If foa_native improves with larger unfreeze while logmel_only does not: the richer FOA-native representation likely needs more backbone adaptation.\n")
        f.write("- If foa_native improves with better checkpoint selection or longer schedule: the earlier result was recipe-limited, not representation-limited.\n")
        f.write("- If logmel_only still wins across all Stage-4 variants: foa_native currently helps memorization but not generalization under this backbone/training regime.\n")
        f.write("- If foa_native matches or beats logmel_only on angular error: there is evidence that FOA-native cues help real-data validation too.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["run_name"])

    write_csv(COMPARE_ROOT / "stage4_comparison.csv", rows)
    write_markdown(COMPARE_ROOT / "stage4_comparison.md", rows)
    print(f"saved stage4 comparison to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
