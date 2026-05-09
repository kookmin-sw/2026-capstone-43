from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/yu/Project_git/SpatialAudio/12_overfit_baseline_rerun/analysis_outputs")
PAIR_CSV = ROOT / "overfit_memorization_speed_pairs_excluding_04.csv"
VAL_CSV = ROOT / "overfit_memorization_speed_validation_curve_excluding_04.csv"
DECODE_CSV = ROOT / "overfit_memorization_speed_decode_curve_excluding_04.csv"
OUT_DIR = ROOT / "pairwise_visuals"


TASK_LABELS = {
    "3way_fov_glos_gnlos_300": "3-way FOV gLOS/gNLOS",
    "8way_gnlos_800": "8-way gNLOS",
    "8way_mixed_glos_gnlos_800": "8-way Mixed gLOS/gNLOS",
}

CHECKPOINT_LABELS = {
    "epoch_05": "E05",
    "epoch_10": "E10",
    "epoch_15": "E15",
    "epoch_20": "E20",
}

THRESHOLDS = [
    ("0_90", "90%"),
    ("0_95", "95%"),
    ("0_99", "99%"),
]

COLORS = {
    "audio": "#1f77b4",
    "av": "#d62728",
}


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _as_percent(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def make_pair_dashboard(
    pair_row: pd.Series,
    val_df: pd.DataFrame,
    decode_df: pd.DataFrame,
) -> Path:
    task = pair_row["task"]
    task_label = TASK_LABELS.get(task, task)
    audio_run = pair_row["audio_run"]
    av_run = pair_row["av_run"]

    pair_val = val_df[val_df["run_dir"].isin([audio_run, av_run])].copy()
    pair_decode = decode_df[decode_df["run_dir"].isin([audio_run, av_run])].copy()

    pair_val["epoch"] = pair_val["epoch"].astype(int)
    pair_val["eval_loss"] = pair_val["eval_loss"].astype(float)
    pair_val["eval_token_accuracy_pct"] = _as_percent(pair_val["eval_token_accuracy"])
    pair_decode["accuracy_pct"] = _as_percent(pair_decode["accuracy"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"{task_label}: Audio vs AV", fontsize=16, fontweight="bold")

    # Validation accuracy curve.
    ax = axes[0, 0]
    for modality, run_dir in [("audio", audio_run), ("av", av_run)]:
        run_df = pair_val[pair_val["run_dir"] == run_dir].sort_values("epoch")
        ax.plot(
            run_df["epoch"],
            run_df["eval_token_accuracy_pct"],
            marker="o",
            linewidth=2.2,
            color=COLORS[modality],
            label=modality.upper(),
        )
    for threshold in (90, 95, 99):
        ax.axhline(threshold, color="#888888", linestyle=":", linewidth=1)
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 102)
    _style_axes(ax)
    ax.legend(frameon=False)

    # Validation loss curve.
    ax = axes[0, 1]
    for modality, run_dir in [("audio", audio_run), ("av", av_run)]:
        run_df = pair_val[pair_val["run_dir"] == run_dir].sort_values("epoch")
        ax.plot(
            run_df["epoch"],
            run_df["eval_loss"],
            marker="o",
            linewidth=2.2,
            color=COLORS[modality],
            label=modality.upper(),
        )
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, 20)
    _style_axes(ax)
    ax.legend(frameon=False)

    # Decode checkpoints.
    ax = axes[1, 0]
    ckpt_order = ["epoch_05", "epoch_10", "epoch_15", "epoch_20"]
    x = range(len(ckpt_order))
    width = 0.36
    audio_vals = []
    av_vals = []
    for ckpt in ckpt_order:
        audio_vals.append(
            float(
                pair_decode[
                    (pair_decode["run_dir"] == audio_run)
                    & (pair_decode["checkpoint_name"] == ckpt)
                ]["accuracy_pct"].iloc[0]
            )
        )
        av_vals.append(
            float(
                pair_decode[
                    (pair_decode["run_dir"] == av_run)
                    & (pair_decode["checkpoint_name"] == ckpt)
                ]["accuracy_pct"].iloc[0]
            )
        )
    ax.bar(
        [item - width / 2 for item in x],
        audio_vals,
        width=width,
        color=COLORS["audio"],
        label="AUDIO",
    )
    ax.bar(
        [item + width / 2 for item in x],
        av_vals,
        width=width,
        color=COLORS["av"],
        label="AV",
    )
    ax.set_xticks(list(x), [CHECKPOINT_LABELS[item] for item in ckpt_order])
    ax.set_ylim(0, 102)
    ax.set_ylabel("Decode Accuracy (%)")
    ax.set_title("Decode Checkpoints")
    _style_axes(ax)
    ax.legend(frameon=False)

    # Threshold epochs.
    ax = axes[1, 1]
    labels = [label for _, label in THRESHOLDS]
    audio_epochs = [int(pair_row[f"audio_first_val_epoch_ge_{key}"]) for key, _ in THRESHOLDS]
    av_epochs = [int(pair_row[f"av_first_val_epoch_ge_{key}"]) for key, _ in THRESHOLDS]
    x = range(len(labels))
    ax.bar(
        [item - width / 2 for item in x],
        audio_epochs,
        width=width,
        color=COLORS["audio"],
        label="AUDIO",
    )
    ax.bar(
        [item + width / 2 for item in x],
        av_epochs,
        width=width,
        color=COLORS["av"],
        label="AV",
    )
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("First Epoch Reaching Threshold")
    ax.set_title("Memorization Speed")
    ax.set_ylim(0, 20.5)
    _style_axes(ax)
    ax.legend(frameon=False)

    summary = (
        f"Audio: first 90/95/99 = {audio_epochs[0]}/{audio_epochs[1]}/{audio_epochs[2]}\n"
        f"AV: first 90/95/99 = {av_epochs[0]}/{av_epochs[1]}/{av_epochs[2]}\n"
        f"Decode@20: {audio_vals[-1]:.2f}% vs {av_vals[-1]:.2f}%"
    )
    ax.text(
        0.98,
        0.05,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )

    out_path = OUT_DIR / f"pair_dashboard_{task}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_overview_threshold_plot(pair_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (key, label) in zip(axes, THRESHOLDS):
        tasks = [TASK_LABELS.get(task, task) for task in pair_df["task"]]
        audio_epochs = pair_df[f"audio_first_val_epoch_ge_{key}"].astype(int).tolist()
        av_epochs = pair_df[f"av_first_val_epoch_ge_{key}"].astype(int).tolist()
        x = range(len(tasks))
        width = 0.36
        ax.bar([item - width / 2 for item in x], audio_epochs, width=width, color=COLORS["audio"], label="AUDIO")
        ax.bar([item + width / 2 for item in x], av_epochs, width=width, color=COLORS["av"], label="AV")
        ax.set_xticks(list(x), tasks, rotation=15, ha="right")
        ax.set_ylim(0, 20.5)
        ax.set_title(f"First Epoch Reaching {label}")
        ax.set_ylabel("Epoch")
        _style_axes(ax)
        if ax is axes[0]:
            ax.legend(frameon=False)

    out_path = OUT_DIR / "pair_overview_thresholds.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_overview_decode_plot(pair_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(pair_df), figsize=(15, 4.8), constrained_layout=True)
    if len(pair_df) == 1:
        axes = [axes]

    ckpt_cols = [
        ("audio_decode_e05", "av_decode_e05", "E05"),
        ("audio_decode_e10", "av_decode_e10", "E10"),
        ("audio_decode_e15", "av_decode_e15", "E15"),
        ("audio_decode_e20", "av_decode_e20", "E20"),
    ]

    for ax, (_, row) in zip(axes, pair_df.iterrows()):
        x = range(len(ckpt_cols))
        width = 0.36
        audio_vals = [float(row[a_col]) * 100.0 for a_col, _, _ in ckpt_cols]
        av_vals = [float(row[v_col]) * 100.0 for _, v_col, _ in ckpt_cols]
        ax.bar([item - width / 2 for item in x], audio_vals, width=width, color=COLORS["audio"], label="AUDIO")
        ax.bar([item + width / 2 for item in x], av_vals, width=width, color=COLORS["av"], label="AV")
        ax.set_xticks(list(x), [label for _, _, label in ckpt_cols])
        ax.set_ylim(0, 102)
        ax.set_title(TASK_LABELS.get(row["task"], row["task"]))
        ax.set_ylabel("Decode Accuracy (%)")
        _style_axes(ax)
        ax.legend(frameon=False)

    out_path = OUT_DIR / "pair_overview_decode_checkpoints.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_index(paths: list[Path]) -> Path:
    index_path = OUT_DIR / "README.md"
    lines = ["# Pairwise Visuals", ""]
    lines.append("Comparison-ready plots for `audio vs AV` pairs, excluding `overfit_04_av_8way_glos_800`.")
    lines.append("")
    for path in paths:
        lines.append(f"- [{path.name}]({path.name})")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_df = _load_csv(PAIR_CSV)
    val_df = _load_csv(VAL_CSV)
    decode_df = _load_csv(DECODE_CSV)

    outputs: list[Path] = []
    for _, pair_row in pair_df.iterrows():
        outputs.append(make_pair_dashboard(pair_row, val_df, decode_df))
    outputs.append(make_overview_threshold_plot(pair_df))
    outputs.append(make_overview_decode_plot(pair_df))
    outputs.append(write_index(outputs))

    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
