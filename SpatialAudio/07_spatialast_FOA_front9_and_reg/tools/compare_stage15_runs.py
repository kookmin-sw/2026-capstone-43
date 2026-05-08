import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage15"
COMPARE_ROOT = ROOT / "outputs_stage15_comparison"

COLUMNS = [
    "run_name",
    "stem_variant",
    "azimuth_head_mode",
    "patch_embed_trainable",
    "best_epoch",
    "best_val_loss",
    "best_val_angular_error",
    "best_val_azimuth_mae_deg",
    "best_val_vector_cosine",
    "final_val_loss",
    "final_val_angular_error",
    "final_val_azimuth_mae_deg",
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
        "azimuth_head_mode": config.get("azimuth_head_mode", ""),
        "patch_embed_trainable": config.get("unfreeze_patch_embed", False),
        "best_epoch": best.get("epoch", ""),
        "best_val_loss": best.get("val_loss", ""),
        "best_val_angular_error": best.get("val_angular_error", ""),
        "best_val_azimuth_mae_deg": best.get("val_azimuth_mae_deg", best.get("val_azimuth_mae", "")),
        "best_val_vector_cosine": best.get("val_vector_cosine", ""),
        "final_val_loss": final.get("val_loss", ""),
        "final_val_angular_error": final.get("val_angular_error", ""),
        "final_val_azimuth_mae_deg": final.get("val_azimuth_mae_deg", final.get("val_azimuth_mae", "")),
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


def collect_summary(rows):
    if not rows:
        return ["- No completed Stage-15 runs found yet."]

    lines = []
    subset_front9 = best_row(rows, prefix="subset_", stem_variant="baseline", azimuth_head_mode="front9_classification")
    subset_reg = best_row(rows, prefix="subset_", stem_variant="baseline", azimuth_head_mode="front_regression")
    subset_bestconv_reg = best_row(rows, prefix="subset_", azimuth_head_mode="front_regression")
    full_front9 = best_row(rows, prefix="full_", stem_variant="baseline", azimuth_head_mode="front9_classification")
    full_reg = best_row(rows, prefix="full_", stem_variant="baseline", azimuth_head_mode="front_regression")
    full_bestconv_reg = best_row(rows, prefix="full_", azimuth_head_mode="front_regression")
    full360_reg = best_row(rows, prefix="full_", stem_variant="baseline", azimuth_head_mode="full360_regression")
    full360_sincos_reg = best_row(rows, prefix="full_", stem_variant="baseline", azimuth_head_mode="full360_sincos_regression")
    full_bestconv_full360_reg = best_row(rows, prefix="full_", azimuth_head_mode="full360_regression")
    full_bestconv_full360_sincos_reg = best_row(rows, prefix="full_", azimuth_head_mode="full360_sincos_regression")

    if subset_front9 and subset_reg:
        lines.append(
            f"- Subset baseline front9 vs reg: angular error {subset_front9['best_val_angular_error']} -> {subset_reg['best_val_angular_error']}, "
            f"azimuth MAE {subset_front9['best_val_azimuth_mae_deg']} -> {subset_reg['best_val_azimuth_mae_deg']}."
        )
    if subset_bestconv_reg:
        lines.append(
            f"- Best subset regression run so far is `{subset_bestconv_reg['run_name']}` "
            f"with stem `{subset_bestconv_reg['stem_variant']}`."
        )
    if full_front9 and full_reg:
        lines.append(
            f"- Full baseline front9 vs reg: angular error {full_front9['best_val_angular_error']} -> {full_reg['best_val_angular_error']}, "
            f"azimuth MAE {full_front9['best_val_azimuth_mae_deg']} -> {full_reg['best_val_azimuth_mae_deg']}."
        )
    if full_bestconv_reg:
        lines.append(
            f"- Best full regression run so far is `{full_bestconv_reg['run_name']}` "
            f"with stem `{full_bestconv_reg['stem_variant']}`."
        )
    if (full360_reg or full360_sincos_reg) and full_reg:
        full360_best = full360_sincos_reg or full360_reg
        lines.append(
            f"- Full 360 regression vs front regression: angular error {full_reg['best_val_angular_error']} -> {full360_best['best_val_angular_error']}, "
            f"azimuth MAE {full_reg['best_val_azimuth_mae_deg']} -> {full360_best['best_val_azimuth_mae_deg']}."
        )
    if full_bestconv_full360_sincos_reg:
        lines.append(
            f"- Best full 360 sin/cos regression run so far is `{full_bestconv_full360_sincos_reg['run_name']}` "
            f"with stem `{full_bestconv_full360_sincos_reg['stem_variant']}`."
        )
    elif full_bestconv_full360_reg:
        lines.append(
            f"- Best full 360 regression run so far is `{full_bestconv_full360_reg['run_name']}` "
            f"with stem `{full_bestconv_full360_reg['stem_variant']}`."
        )

    return lines


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# Stage-15 Comparison\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in COLUMNS]
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Interpretation Summary\n\n")
        for line in collect_summary(rows):
            f.write(line + "\n")

        f.write("\n## Interpretation Rules\n\n")
        f.write("- If front_regression beats front9_classification: current front-cone task is better matched by continuous azimuth supervision.\n")
        f.write("- If bestconv plus front_regression beats baseline plus front_regression: stem bottleneck and azimuth supervision mismatch both mattered.\n")
        f.write("- If full360 regression beats front_regression on the full dataset: circular scalar regression is a better fit than front-only supervision.\n")
        f.write("- If patch embed unfreeze helps substantially: widened stem changed patch input semantics enough that frozen patch embed was a bottleneck.\n")
        f.write("- If regression helps subset but not full: feasibility exists but generalization is still uncertain.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["run_name"])

    write_csv(COMPARE_ROOT / "stage15_comparison.csv", rows)
    write_markdown(COMPARE_ROOT / "stage15_comparison.md", rows)
    print(f"saved stage15 comparison to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
