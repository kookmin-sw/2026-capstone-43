import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage12"
COMPARE_ROOT = ROOT / "outputs_stage12_comparison"

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


def best_row(rows, prefix=None):
    candidates = rows
    if prefix is not None:
        candidates = [row for row in rows if row["run_name"].startswith(prefix)]
    candidates = [row for row in candidates if safe_float(row["best_val_angular_error"]) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: safe_float(row["best_val_angular_error"]))


def collect_summary(rows):
    lines = []
    subset_rows = [row for row in rows if row["run_name"].startswith("subset_")]
    full_rows = [row for row in rows if row["run_name"].startswith("full_")]

    subset_best = best_row(subset_rows)
    subset_baseline = next((row for row in subset_rows if row["stem_variant"] == "baseline"), None)
    full_best = best_row(full_rows)
    full_baseline = next((row for row in full_rows if row["stem_variant"] == "baseline"), None)

    if subset_best and subset_baseline and subset_best["run_name"] != subset_baseline["run_name"]:
        lines.append(
            f"- Subset best variant is `{subset_best['stem_variant']}` "
            f"({subset_best['best_val_angular_error']} angular error) vs baseline "
            f"({subset_baseline['best_val_angular_error']})."
        )
    elif subset_baseline:
        lines.append(
            f"- Subset baseline remains strongest so far at {subset_baseline['best_val_angular_error']} angular error."
        )

    if full_best and full_baseline and full_best["run_name"] != full_baseline["run_name"]:
        lines.append(
            f"- Full-data best variant is `{full_best['stem_variant']}` "
            f"({full_best['best_val_angular_error']} angular error) vs baseline "
            f"({full_baseline['best_val_angular_error']})."
        )
    elif full_baseline:
        lines.append(
            f"- Full-data baseline remains strongest so far at {full_baseline['best_val_angular_error']} angular error."
        )

    for row in rows:
        best_val = safe_float(row["best_val_angular_error"])
        final_val = safe_float(row["final_val_angular_error"])
        if best_val is not None and final_val is not None and final_val > best_val:
            lines.append(
                f"- `{row['run_name']}` final collapse gap: "
                f"{final_val - best_val:+.4f} angular error from best to final."
            )

    if not rows:
        lines.append("- No completed Stage-12 runs found yet.")
    return lines


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# Stage-12 Comparison\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in COLUMNS]
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Interpretation Summary\n\n")
        for line in collect_summary(rows):
            f.write(line + "\n")

        f.write("\n## Interpretation Rules\n\n")
        f.write("- If a wider stem improves best angular error and reduces final collapse: the current baseline stem is likely over-compressing FOA spatial cues.\n")
        f.write("- If a wider stem improves subset runs but not full runs: memorization benefit exists, but generalization benefit is not yet confirmed.\n")
        f.write("- If a wider stem does not help: the current bottleneck is more likely recipe/optimization than stem width.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["run_name"])

    write_csv(COMPARE_ROOT / "stage12_comparison.csv", rows)
    write_markdown(COMPARE_ROOT / "stage12_comparison.md", rows)
    print(f"saved stage12 comparison to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
