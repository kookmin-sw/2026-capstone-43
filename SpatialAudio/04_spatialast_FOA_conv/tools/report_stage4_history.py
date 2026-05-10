import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = ROOT / "outputs_stage4"
REPORT_ROOT = ROOT / "outputs_stage4_history"
DEFAULT_RUNS = [
    "foa_last4",
    "foa_last4_cosine",
    "foa_last4_lower_head_lr",
    "foa_last4_longer",
    "foa_last4_longer_seed2024",
    "foa_last4_longer_seed3407",
    "foa_last4_longer_cosine",
    "foa_last4_longer_lower_head_lr",
    "logmel_last4",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def format_curve(history, first_n_epochs):
    values = [entry["val_angular_error"] for entry in history[:first_n_epochs]]
    return " -> ".join(f"{value:.4f}" for value in values)


def build_row(run_name, first_n_epochs):
    run_dir = OUTPUTS_ROOT / run_name
    metrics_path = run_dir / "metrics_summary.json"
    history_path = run_dir / "train_history.json"
    if not metrics_path.exists() or not history_path.exists():
        return None

    metrics = load_json(metrics_path)
    history = load_json(history_path)
    config = metrics.get("config", {})
    best = metrics.get("best", {})
    final = metrics.get("final", {})
    row = {
        "run_name": run_name,
        "recipe": config.get("recipe_name", ""),
        "seed": config.get("seed", ""),
        "epochs": config.get("epochs", ""),
        "best_epoch": best.get("epoch", ""),
        "best_val_angular_error": best.get("val_angular_error", ""),
        "final_val_angular_error": final.get("val_angular_error", ""),
        "final_minus_best_angular_error": "",
        "first_epochs_val_angular_error": format_curve(history, first_n_epochs),
    }
    if best.get("val_angular_error") is not None and final.get("val_angular_error") is not None:
        row["final_minus_best_angular_error"] = (
            final["val_angular_error"] - best["val_angular_error"]
        )
    for epoch in range(first_n_epochs):
        key = f"epoch_{epoch}_val_angular_error"
        row[key] = history[epoch]["val_angular_error"] if epoch < len(history) else ""
    return row


def write_csv(path, rows, first_n_epochs):
    columns = [
        "run_name",
        "recipe",
        "seed",
        "epochs",
        "best_epoch",
        "best_val_angular_error",
        "final_val_angular_error",
        "final_minus_best_angular_error",
        "first_epochs_val_angular_error",
    ]
    columns.extend(f"epoch_{epoch}_val_angular_error" for epoch in range(first_n_epochs))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_markdown(path, rows, first_n_epochs):
    with open(path, "w") as f:
        f.write("# Stage-4 History Summary\n\n")
        f.write(f"Early-epoch window: first {first_n_epochs} epochs\n\n")
        f.write("| run_name | recipe | seed | best_epoch | best_val_angular_error | final_val_angular_error | final_minus_best_angular_error |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    str(
                        row.get(column, "")
                    )
                    for column in [
                        "run_name",
                        "recipe",
                        "seed",
                        "best_epoch",
                        "best_val_angular_error",
                        "final_val_angular_error",
                        "final_minus_best_angular_error",
                    ]
                )
                + " |\n"
            )
        f.write("\n## Early Curves\n\n")
        for row in rows:
            f.write(
                f"- `{row['run_name']}`: {row['first_epochs_val_angular_error']}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--first_n_epochs", type=int, default=10)
    parser.add_argument("--output_dir", default=str(REPORT_ROOT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_name in args.runs:
        row = build_row(run_name, args.first_n_epochs)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: row["run_name"])
    write_csv(output_dir / "stage4_history_summary.csv", rows, args.first_n_epochs)
    write_markdown(output_dir / "stage4_history_summary.md", rows, args.first_n_epochs)
    print(f"saved stage4 history summary to {output_dir}")


if __name__ == "__main__":
    main()
