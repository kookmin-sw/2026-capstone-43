import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage14"
COMPARE_ROOT = ROOT / "outputs_stage14_comparison"

COLUMNS = [
    "run_name",
    "stem_variant",
    "azimuth_head_mode",
    "best_epoch",
    "best_val_loss",
    "best_val_angular_error",
    "best_val_azimuth_mae_deg",
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
        "azimuth_head_mode": config.get("azimuth_head_mode", "full360_classification"),
        "best_epoch": best.get("epoch", ""),
        "best_val_loss": best.get("val_loss", ""),
        "best_val_angular_error": best.get("val_angular_error", ""),
        "best_val_azimuth_mae_deg": best.get("val_azimuth_mae_deg", best.get("val_azimuth_mae", "")),
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


def best_row(rows, prefix=None, stem_variant=None, azimuth_head_mode=None):
    candidates = rows
    if prefix is not None:
        candidates = [row for row in candidates if row["run_name"].startswith(prefix)]
    if stem_variant is not None:
        candidates = [row for row in candidates if row["stem_variant"] == stem_variant]
    if azimuth_head_mode is not None:
        candidates = [row for row in candidates if row["azimuth_head_mode"] == azimuth_head_mode]
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
        return ["- No completed Stage-14 runs found yet."]

    lines = []
    subset_cls = best_row(
        rows,
        prefix="subset_",
        stem_variant="baseline",
        azimuth_head_mode="full360_classification",
    )
    subset_reg = best_row(
        rows,
        prefix="subset_",
        stem_variant="baseline",
        azimuth_head_mode="front_regression",
    )
    subset_bestconv_reg = best_row(
        rows,
        prefix="subset_",
        azimuth_head_mode="front_regression",
    )
    full_cls = best_row(
        rows,
        prefix="full_",
        stem_variant="baseline",
        azimuth_head_mode="full360_classification",
    )
    full_reg = best_row(
        rows,
        prefix="full_",
        stem_variant="baseline",
        azimuth_head_mode="front_regression",
    )
    full_bestconv_reg = best_row(
        rows,
        prefix="full_",
        azimuth_head_mode="front_regression",
    )

    if subset_cls is not None and subset_reg is not None:
        lines.append(
            f"- Subset baseline cls vs reg: angular error {subset_cls['best_val_angular_error']} -> "
            f"{subset_reg['best_val_angular_error']}, azimuth MAE {subset_cls['best_val_azimuth_mae_deg']} -> "
            f"{subset_reg['best_val_azimuth_mae_deg']}."
        )
    if subset_bestconv_reg is not None:
        lines.append(
            f"- Best subset front-regression run so far is `{subset_bestconv_reg['run_name']}` "
            f"with stem `{subset_bestconv_reg['stem_variant']}`."
        )
    if full_cls is not None and full_reg is not None:
        lines.append(
            f"- Full baseline cls vs reg: angular error {full_cls['best_val_angular_error']} -> "
            f"{full_reg['best_val_angular_error']}, azimuth MAE {full_cls['best_val_azimuth_mae_deg']} -> "
            f"{full_reg['best_val_azimuth_mae_deg']}."
        )
    if full_bestconv_reg is not None:
        lines.append(
            f"- Best full front-regression run so far is `{full_bestconv_reg['run_name']}` "
            f"with stem `{full_bestconv_reg['stem_variant']}`."
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
        f.write("# Stage-14 Comparison\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in COLUMNS]
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Interpretation Summary\n\n")
        for line in collect_summary(rows):
            f.write(line + "\n")

        f.write("\n## Interpretation Rules\n\n")
        f.write("- If front_regression reduces azimuth MAE and angular error vs 360-class baseline: current front-cone task is better matched by continuous azimuth supervision.\n")
        f.write("- If bestconv plus front_regression further improves results: both stem bottleneck and azimuth discretization were limiting factors.\n")
        f.write("- If regression helps subset but not full: feasibility exists, generalization still uncertain.\n")
        f.write("- If regression does not help: current bottleneck is still more likely optimization / representation than azimuth discretization.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["run_name"])

    write_csv(COMPARE_ROOT / "stage14_comparison.csv", rows)
    write_markdown(COMPARE_ROOT / "stage14_comparison.md", rows)
    print(f"saved stage14 comparison to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
