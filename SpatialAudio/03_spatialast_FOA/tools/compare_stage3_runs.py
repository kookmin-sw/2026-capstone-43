import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage3"
COMPARE_ROOT = ROOT / "outputs_stage3_comparison"

BASE_COLUMNS = [
    "run_name",
    "foa_stem_type",
    "epoch",
    "train_azimuth_acc",
    "train_elevation_acc",
    "val_azimuth_acc",
    "val_elevation_acc",
    "train_azimuth_loss",
    "train_elevation_loss",
    "val_azimuth_loss",
    "val_elevation_loss",
    "train_vector_cosine",
    "val_vector_cosine",
]

OPTIONAL_COLUMNS = [
    "train_azimuth_mae",
    "train_elevation_mae",
    "train_angular_error",
    "val_azimuth_mae",
    "val_elevation_mae",
    "val_angular_error",
]


def load_run(run_dir):
    history_path = run_dir / "train_history.json"
    metrics_path = run_dir / "metrics_summary.json"
    if not history_path.exists() or not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(history_path, "r") as f:
        history = json.load(f)

    final = metrics.get("final") or (history[-1] if history else {})
    config = metrics.get("config", {})
    row = {
        "run_name": run_dir.name,
        "foa_stem_type": config.get("foa_stem_type", ""),
    }
    for key in BASE_COLUMNS[2:] + OPTIONAL_COLUMNS:
        row[key] = final.get(key, "")
    return row


def sort_key(row):
    name = row["run_name"]
    subset_order = {"16": 0, "64": 1, "128": 2, "2400": 3}
    stem_order = {"logmel_only": 0, "foa_native": 1}
    subset = next((value for value in subset_order if value in name), "9999")
    return (subset_order.get(subset, 999), stem_order.get(row["foa_stem_type"], 999), name)


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(path, title, rows, columns, include_interpretation=False):
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in columns]
            f.write("| " + " | ".join(values) + " |\n")
        if include_interpretation:
            if "Overfit" in title:
                f.write("\n## Overfit Success Heuristics\n\n")
                f.write("- For 16 / 64 / 128 samples, train azimuth and elevation accuracy should rise clearly above chance.\n")
                f.write("- Train azimuth and elevation CE should decrease significantly across epochs.\n")
                f.write("- Train vector cosine should increase.\n")
                f.write("- If 16-sample memorization fails badly, stop and inspect the pipeline before larger runs.\n")
            f.write("\n## Interpretation Guide\n\n")
            f.write("- If `foa_native` memorizes faster than `logmel_only`: IV helps spatial representation.\n")
            f.write("- If both memorize similarly on small N but `foa_native` is better on 2400: FOA-native cues help generalization.\n")
            f.write("- If 16/64/128 all fail: there is still a training or pipeline issue.\n")
            f.write("- If overfit works but 2400 does not improve: representation is learnable but not yet generalizing.\n")


def main():
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(OUTPUTS_ROOT.glob("*")):
        if run_dir.is_dir():
            row = load_run(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=sort_key)

    overfit_rows = [row for row in rows if row["run_name"].startswith("overfit_")]
    train_rows = [row for row in rows if row["run_name"].startswith("train2400_")]

    overfit_columns = BASE_COLUMNS
    train_columns = BASE_COLUMNS + OPTIONAL_COLUMNS

    write_csv(COMPARE_ROOT / "overfit_comparison.csv", overfit_rows, overfit_columns)
    write_markdown(
        COMPARE_ROOT / "overfit_comparison.md",
        "Stage-3 Overfit Comparison",
        overfit_rows,
        overfit_columns,
        include_interpretation=True,
    )
    write_csv(COMPARE_ROOT / "train2400_comparison.csv", train_rows, train_columns)
    write_markdown(
        COMPARE_ROOT / "train2400_comparison.md",
        "Stage-3 2400-Sample Comparison",
        train_rows,
        train_columns,
        include_interpretation=True,
    )

    print(f"saved comparison files to {COMPARE_ROOT}")


if __name__ == "__main__":
    main()
