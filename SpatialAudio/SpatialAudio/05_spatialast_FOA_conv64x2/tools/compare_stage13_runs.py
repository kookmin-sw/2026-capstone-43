import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage13"
COMPARE_ROOT = ROOT / "outputs_stage13_comparison"

COLUMNS = [
    "run_name",
    "stem_variant",
    "best_epoch",
    "best_val_loss",
    "best_val_angular_error",
    "best_val_azimuth_mae",
    "best_val_elevation_mae",
    "best_val_vector_cosine",
    "final_val_loss",
    "final_val_angular_error",
]


def load_run(run_dir):
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    config = metrics.get("config", {})
    best = metrics.get("best", {})
    final = metrics.get("final", {})
    return {
        "run_name": run_dir.name,
        "stem_variant": config.get("foa_stem_variant", ""),
        "best_epoch": best.get("epoch", ""),
        "best_val_loss": best.get("val_loss", ""),
        "best_val_angular_error": best.get("val_angular_error", ""),
        "best_val_azimuth_mae": best.get("val_azimuth_mae", ""),
        "best_val_elevation_mae": best.get("val_elevation_mae", ""),
        "best_val_vector_cosine": best.get("val_vector_cosine", ""),
        "final_val_loss": final.get("val_loss", ""),
        "final_val_angular_error": final.get("val_angular_error", ""),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def safe_float(value):
    if value in ("", None):
        return None
    return float(value)


def best_row(rows, prefix=None, stem_variants=None):
    candidates = rows
    if prefix is not None:
        candidates = [row for row in candidates if row["run_name"].startswith(prefix)]
    if stem_variants is not None:
        stem_variants = set(stem_variants)
        candidates = [row for row in candidates if row["stem_variant"] in stem_variants]
    candidates = [row for row in candidates if safe_float(row["best_val_angular_error"]) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: safe_float(row["best_val_angular_error"]))


def collapse_gap(row):
    best_val = safe_float(row["best_val_angular_error"])
    final_val = safe_float(row["final_val_angular_error"])
    if best_val is None or final_val is None:
        return None
    return final_val - best_val


def collect_summary(rows):
    if not rows:
        return ["- No completed Stage-13 runs found yet."]

    lines = []
    subset_rows = [row for row in rows if row["run_name"].startswith("subset_")]
    full_rows = [row for row in rows if row["run_name"].startswith("full_")]
    deeper_variants = {"conv64_64_out8", "conv64_64_out16"}

    subset_baseline = next((row for row in subset_rows if row["stem_variant"] == "baseline"), None)
    subset_conv64_out8 = next((row for row in subset_rows if row["stem_variant"] == "conv64_out8"), None)
    subset_best_deeper = best_row(subset_rows, stem_variants=deeper_variants)
    full_baseline = next((row for row in full_rows if row["stem_variant"] == "baseline"), None)
    full_conv64_out8 = next((row for row in full_rows if row["stem_variant"] == "conv64_out8"), None)
    full_best_deeper = best_row(full_rows, stem_variants=deeper_variants)

    if subset_best_deeper is not None:
        lines.append(
            f"- Best subset 3-block candidate is `{subset_best_deeper['stem_variant']}` "
            f"at {subset_best_deeper['best_val_angular_error']} angular error."
        )
    if subset_baseline is not None:
        lines.append(
            f"- Subset baseline reference is {subset_baseline['best_val_angular_error']} angular error."
        )
    if subset_conv64_out8 is not None:
        lines.append(
            f"- Subset 2-block `conv64_out8` reference is {subset_conv64_out8['best_val_angular_error']} angular error."
        )
    if full_best_deeper is not None:
        lines.append(
            f"- Best full-data 3-block candidate is `{full_best_deeper['stem_variant']}` "
            f"at {full_best_deeper['best_val_angular_error']} angular error."
        )
    if full_baseline is not None:
        lines.append(
            f"- Full baseline reference is {full_baseline['best_val_angular_error']} angular error."
        )
    if full_conv64_out8 is not None:
        lines.append(
            f"- Full 2-block `conv64_out8` reference is {full_conv64_out8['best_val_angular_error']} angular error."
        )

    for row in rows:
        gap = collapse_gap(row)
        if gap is not None and gap > 0:
            lines.append(
                f"- `{row['run_name']}` final collapse gap: {gap:+.4f} angular error from best to final."
            )

    return lines


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# Stage-13 Comparison\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in COLUMNS]
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Interpretation Summary\n\n")
        for line in collect_summary(rows):
            f.write(line + "\n")

        f.write("\n## Interpretation Rules\n\n")
        f.write("- If `conv64_64` variants improve best angular error over both baseline and `conv64_out8`: deeper FOA stem likely preserves/mixes cues better.\n")
        f.write("- If `conv64_64` improves subset but not full: memorization benefit exists, generalization still unconfirmed.\n")
        f.write("- If `conv64_64` also reduces final collapse: deeper stem improves both representation and stability.\n")
        f.write("- If `conv64_64` does not help: bottleneck is more likely recipe/optimization than stem depth.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["run_name"])

    write_csv(COMPARE_ROOT / "stage13_comparison.csv", rows)
    write_markdown(COMPARE_ROOT / "stage13_comparison.md", rows)
    print(f"saved stage13 comparison to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
