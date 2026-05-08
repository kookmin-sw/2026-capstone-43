import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values

    out = []
    run_sum = 0.0
    for i, v in enumerate(values):
        run_sum += v
        if i >= window:
            run_sum -= values[i - window]
        span = min(i + 1, window)
        out.append(run_sum / span)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="train_sgmse.py가 저장한 epoch_loss CSV")
    parser.add_argument("--out", default="outputs/plots/epoch_loss.png")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="resume 직후 partial epoch도 그래프에 포함",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    epochs = []
    losses = []
    val_losses = []
    has_any_val_loss = False

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_partial = int(row.get("is_partial_epoch", "0"))
            if (not args.include_partial) and is_partial == 1:
                continue

            epoch = int(row["epoch"])
            loss = float(row["mean_loss"])
            epochs.append(epoch)
            losses.append(loss)
            val_str = str(row.get("val_loss", "")).strip()
            if val_str == "":
                val_losses.append(float("nan"))
            else:
                val_losses.append(float(val_str))
                has_any_val_loss = True

    if len(epochs) == 0:
        raise RuntimeError("No rows to plot. Check CSV or use --include-partial.")

    smooth_losses = moving_average(losses, max(1, args.smooth_window))
    best_idx = min(range(len(losses)), key=lambda i: losses[i])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label="epoch mean loss", linewidth=1.5, alpha=0.8)
    if args.smooth_window > 1:
        plt.plot(epochs, smooth_losses, label=f"moving avg (w={args.smooth_window})", linewidth=2.0)
    if has_any_val_loss:
        plt.plot(epochs, val_losses, label="validation loss", linewidth=1.5, alpha=0.85)

    plt.scatter([epochs[best_idx]], [losses[best_idx]], s=36, label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"rows: {len(epochs)}")
    print(f"best_epoch: {epochs[best_idx]}")
    print(f"best_loss: {losses[best_idx]:.10f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
