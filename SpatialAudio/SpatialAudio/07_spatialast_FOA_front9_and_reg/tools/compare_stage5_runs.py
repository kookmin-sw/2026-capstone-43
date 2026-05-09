import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage5"
COMPARE_ROOT = ROOT / "outputs_stage5_comparison"

COLUMNS = [
    "run_name",
    "stem_type",
    "unfreeze_strategy",
    "recipe",
    "best_epoch",
    "train_loss_at_best",
    "val_angular_error_at_best",
    "final_train_loss",
    "final_val_angular_error",
    "best_val_angular_error",
    "best_val_azimuth_mae",
    "best_val_elevation_mae",
    "best_val_vector_cosine",
    "best_val_azimuth_acc",
    "best_val_elevation_acc",
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
        "stem_type": config.get("foa_stem_type", ""),
        "unfreeze_strategy": config.get("unfreeze_strategy", ""),
        "recipe": config.get("recipe_name", ""),
        "best_epoch": best.get("epoch", ""),
        "train_loss_at_best": best.get("train_loss", ""),
        "val_angular_error_at_best": best.get("val_angular_error", ""),
        "final_train_loss": final.get("train_loss", ""),
        "final_val_angular_error": final.get("val_angular_error", ""),
        "best_val_angular_error": best.get("val_angular_error", ""),
        "best_val_azimuth_mae": best.get("val_azimuth_mae", ""),
        "best_val_elevation_mae": best.get("val_elevation_mae", ""),
        "best_val_vector_cosine": best.get("val_vector_cosine", ""),
        "best_val_azimuth_acc": best.get("val_azimuth_acc", ""),
        "best_val_elevation_acc": best.get("val_elevation_acc", ""),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def by_name(rows, run_name):
    for row in rows:
        if row["run_name"] == run_name:
            return row
    return None


def best_row_for_stem(rows, stem_type):
    candidates = [row for row in rows if row["stem_type"] == stem_type]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["best_val_angular_error"])


def overfit_signature(row):
    best_epoch = row.get("best_epoch")
    final_val = row.get("final_val_angular_error")
    best_val = row.get("best_val_angular_error")
    if best_epoch == "" or final_val == "" or best_val == "":
        return None
    return best_epoch <= 6 and final_val > best_val


def fmt_delta(a, b):
    if a in ("", None) or b in ("", None):
        return "n/a"
    return f"{a - b:+.4f}"


def collect_summary(rows):
    lines = []
    foa_last4 = by_name(rows, "foa_last4")
    foa_last6 = by_name(rows, "foa_last6")
    foa_patch_last6 = by_name(rows, "foa_patch_last6")
    foa_last4_longer = by_name(rows, "foa_last4_longer")
    foa_last4_lower_head_lr = by_name(rows, "foa_last4_lower_head_lr")
    foa_last4_lower_stem_lr = by_name(rows, "foa_last4_lower_stem_lr")
    foa_last4_lower_head_lr_more = by_name(rows, "foa_last4_lower_head_lr_more")
    foa_last4_lower_stem_lr_more = by_name(rows, "foa_last4_lower_stem_lr_more")
    foa_last4_cosine = by_name(rows, "foa_last4_cosine")
    foa_last4_cosine_warmup = by_name(rows, "foa_last4_cosine_warmup")
    foa_last4_longer_cosine_warmup = by_name(rows, "foa_last4_longer_cosine_warmup")
    logmel_last4 = by_name(rows, "logmel_last4")
    logmel_last6 = by_name(rows, "logmel_last6")
    logmel_last4_cosine_warmup = by_name(rows, "logmel_last4_cosine_warmup")

    if foa_last4 and foa_last6:
        lines.append(
            f"- FOA `last6` vs `last4`: {foa_last6['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last6['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last6 and foa_patch_last6:
        lines.append(
            f"- FOA `patch_plus_last6` vs `last6`: {foa_patch_last6['best_val_angular_error']} vs {foa_last6['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_patch_last6['best_val_angular_error'], foa_last6['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_longer:
        lines.append(
            f"- FOA `longer_run` vs default: {foa_last4_longer['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_longer['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_lower_head_lr:
        lines.append(
            f"- FOA `lower_head_lr` vs default: {foa_last4_lower_head_lr['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_lower_head_lr['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_lower_stem_lr:
        lines.append(
            f"- FOA `lower_stem_lr` vs default: {foa_last4_lower_stem_lr['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_lower_stem_lr['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_lower_head_lr_more:
        lines.append(
            f"- FOA `lower_head_lr_more` vs default: {foa_last4_lower_head_lr_more['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_lower_head_lr_more['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_lower_stem_lr_more:
        lines.append(
            f"- FOA `lower_stem_lr_more` vs default: {foa_last4_lower_stem_lr_more['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_lower_stem_lr_more['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_cosine_warmup:
        lines.append(
            f"- FOA `cosine_warmup` vs default: {foa_last4_cosine_warmup['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_cosine_warmup['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if foa_last4 and foa_last4_longer_cosine_warmup:
        lines.append(
            f"- FOA `longer_cosine_warmup` vs default: {foa_last4_longer_cosine_warmup['best_val_angular_error']} vs {foa_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(foa_last4_longer_cosine_warmup['best_val_angular_error'], foa_last4['best_val_angular_error'])})."
        )
    if logmel_last4 and logmel_last6:
        lines.append(
            f"- logmel `last6` vs `last4`: {logmel_last6['best_val_angular_error']} vs {logmel_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(logmel_last6['best_val_angular_error'], logmel_last4['best_val_angular_error'])})."
        )
    if logmel_last4 and logmel_last4_cosine_warmup:
        lines.append(
            f"- logmel `cosine_warmup` vs default: {logmel_last4_cosine_warmup['best_val_angular_error']} vs {logmel_last4['best_val_angular_error']} "
            f"(delta {fmt_delta(logmel_last4_cosine_warmup['best_val_angular_error'], logmel_last4['best_val_angular_error'])})."
        )

    best_foa = best_row_for_stem(rows, "foa_native")
    best_logmel = best_row_for_stem(rows, "logmel_only")
    if best_foa and best_logmel:
        verdict = "beats" if best_foa["best_val_angular_error"] < best_logmel["best_val_angular_error"] else "does not beat"
        lines.append(
            f"- Best FOA run `{best_foa['run_name']}` {verdict} best logmel run `{best_logmel['run_name']}` "
            f"on angular error ({best_foa['best_val_angular_error']} vs {best_logmel['best_val_angular_error']})."
        )

    for row in rows:
        if overfit_signature(row):
            lines.append(
                f"- `{row['run_name']}` shows early-overfit pattern: best epoch {row['best_epoch']}, "
                f"best angular {row['best_val_angular_error']}, final angular {row['final_val_angular_error']}."
            )

    if not rows:
        lines.append("- No completed Stage-5 runs found yet.")
    return lines


def write_markdown(path, rows):
    with open(path, "w") as f:
        f.write("# Stage-5 Comparison\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in COLUMNS]
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Interpretation Summary\n\n")
        for line in collect_summary(rows):
            f.write(line + "\n")

        f.write("\n## Interpretation Rules\n\n")
        f.write("- If best epoch is very early and train loss keeps falling while validation angular error worsens: early overfitting / too-fast adaptation is likely.\n")
        f.write("- If `lower_head_lr` or `lower_head_lr_more` helps: the CE head was adapting too quickly.\n")
        f.write("- If `lower_stem_lr` or `lower_stem_lr_more` helps: the FOA stem/adapter was adapting too quickly.\n")
        f.write("- If `cosine_warmup` helps: the optimization trajectory was the problem.\n")
        f.write("- If `longer_cosine_warmup` helps most: stable scheduling mattered more than simply training longer.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row["run_name"])

    write_csv(COMPARE_ROOT / "stage5_comparison.csv", rows)
    write_markdown(COMPARE_ROOT / "stage5_comparison.md", rows)
    print(f"saved stage5 comparison to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
