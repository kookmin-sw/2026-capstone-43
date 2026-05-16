import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs_smoke"
    command = [
        sys.executable,
        "train.py",
        "--device", "cpu",
        "--epochs", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--num_classes", "8",
        "--synthetic_train_samples", "6",
        "--synthetic_val_samples", "4",
        "--synthetic_clip_seconds", "1",
        "--max_train_steps", "2",
        "--max_val_steps", "1",
        "--class_loss_weight", "0.0",
        "--distance_loss_weight", "0.0",
        "--azimuth_loss_weight", "2.0",
        "--elevation_loss_weight", "2.0",
        "--vector_loss_weight", "0.5",
        "--output_dir", str(output_dir),
    ]
    print(" ".join(command))
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
