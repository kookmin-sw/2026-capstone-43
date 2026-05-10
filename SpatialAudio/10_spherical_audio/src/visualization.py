from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .pooling_utils import EIGHT_WAY_LABELS
from .spherical_projection import AngularGrid
from .stft_utils import WindowSTFT

EPSILON = 1.0e-8


def _finalize(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _extent(grid: AngularGrid) -> list[float]:
    return [
        float(np.degrees(grid.azimuth_edges[0])),
        float(np.degrees(grid.azimuth_edges[-1])),
        float(np.degrees(grid.elevation_edges[0])),
        float(np.degrees(grid.elevation_edges[-1])),
    ]


def plot_waveform(audio: np.ndarray, sample_rate: int, output_path: str | Path) -> None:
    max_points = 30000
    step = max(1, audio.shape[0] // max_points)
    time = np.arange(0, audio.shape[0], step, dtype=np.float32) / sample_rate
    channels = ["W", "X", "Y", "Z"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(time, audio[::step, idx], linewidth=0.7)
        ax.set_ylabel(channels[idx])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle("Canonical FOA waveform (WXYZ)")
    _finalize(fig, output_path)


def plot_stft_overview(windows: list[WindowSTFT], output_path: str | Path) -> None:
    if not windows:
        return
    stft = windows[0].stft[0]
    db = 20.0 * np.log10(np.abs(stft) + EPSILON)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(db, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("W-channel STFT overview (first analysis window)")
    ax.set_xlabel("STFT frame")
    ax.set_ylabel("Frequency bin")
    fig.colorbar(im, ax=ax, label="dB")
    _finalize(fig, output_path)


def plot_spherical_heatmap(
    values: np.ndarray,
    grid: AngularGrid,
    title: str,
    output_path: str | Path,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(values, origin="lower", aspect="auto", extent=_extent(grid), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Azimuth deg (0=front, +=right)")
    ax.set_ylabel("Elevation deg (+=up)")
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.grid(True, color="white", alpha=0.18, linewidth=0.5)
    fig.colorbar(im, ax=ax)
    _finalize(fig, output_path)


def plot_audio_sphere_channel_panel(
    tensor: np.ndarray,
    channel_names: list[str],
    grid: AngularGrid,
    output_path: str | Path,
) -> None:
    cols = 3
    rows = int(np.ceil(len(channel_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), squeeze=False)
    for idx, channel_name in enumerate(channel_names):
        ax = axes[idx // cols][idx % cols]
        values = tensor[:, :, idx]
        vmax = 1.0 if channel_name not in {"beam_power"} else max(1.0, float(np.max(values)))
        im = ax.imshow(values, origin="lower", aspect="auto", extent=_extent(grid), cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(channel_name)
        ax.set_xlabel("Azimuth deg")
        ax.set_ylabel("Elevation deg")
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.grid(True, color="white", alpha=0.15, linewidth=0.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for idx in range(len(channel_names), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("A_sphere per-channel spherical maps")
    _finalize(fig, output_path)


def plot_audio_azimuth_multichannel(
    azimuth_tensor: np.ndarray,
    channel_names: list[str],
    grid: AngularGrid,
    output_path: str | Path,
) -> None:
    channels_to_plot = [name for name in ["beam_power", "aiv_score", "diffuseness", "dp_reliability", "energy", "stability"] if name in channel_names]
    centers_deg = np.degrees(grid.azimuth_centers)
    width = 360.0 / grid.num_az_bins * 0.82
    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(11, 2.1 * len(channels_to_plot)), sharex=True)
    if len(channels_to_plot) == 1:
        axes = [axes]
    for ax, channel_name in zip(axes, channels_to_plot):
        idx = channel_names.index(channel_name)
        ax.bar(centers_deg, azimuth_tensor[:, idx], width=width, align="center", alpha=0.82)
        for boundary in np.arange(-157.5, 180.0, 45.0):
            ax.axvline(boundary, color="black", linewidth=0.5, alpha=0.25)
        ax.set_ylabel(channel_name)
        ax.set_ylim(0.0, max(1.0, float(np.max(azimuth_tensor[:, idx])) * 1.05))
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(np.arange(-180, 181, 45))
    axes[-1].set_xlabel("Azimuth deg with 8-way sector boundaries")
    fig.suptitle("Azimuth-only audio evidence channels")
    _finalize(fig, output_path)


def plot_8way_pooled(
    pooled: np.ndarray,
    channel_names: list[str],
    output_path: str | Path,
) -> None:
    score_channels = [name for name in ["beam_power", "aiv_score", "dp_reliability", "energy"] if name in channel_names]
    x = np.arange(len(EIGHT_WAY_LABELS))
    width = 0.8 / max(len(score_channels), 1)
    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, channel_name in enumerate(score_channels):
        channel_idx = channel_names.index(channel_name)
        offset = (idx - (len(score_channels) - 1) / 2.0) * width
        ax.bar(x + offset, pooled[:, channel_idx], width=width, label=channel_name)
    ax.set_xticks(x)
    ax.set_xticklabels(EIGHT_WAY_LABELS, rotation=25, ha="right")
    ax.set_ylim(0.0, max(1.0, float(np.max(pooled)) * 1.05))
    ax.set_ylabel("Pooled score")
    ax.set_title("8-way pooled audio evidence")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    _finalize(fig, output_path)


def plot_global_direction_hist(values: np.ndarray, grid: AngularGrid, output_path: str | Path) -> None:
    normalized = values / max(float(np.max(values)), EPSILON)
    plot_spherical_heatmap(normalized, grid, "Global AIV direction histogram", output_path, cmap="plasma")


def plot_windowwise_direction_track(
    peak_trace: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    if not peak_trace:
        return
    times = np.asarray([(p["start_sec"] + p["end_sec"]) * 0.5 for p in peak_trace], dtype=np.float32)
    az = np.asarray([p["peak_azimuth_deg"] for p in peak_trace], dtype=np.float32)
    el = np.asarray([p["peak_elevation_deg"] for p in peak_trace], dtype=np.float32)
    score = np.asarray([p["peak_dp_reliability"] for p in peak_trace], dtype=np.float32)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(times, az, marker="o")
    axes[0].set_ylabel("Peak az deg")
    axes[1].plot(times, el, marker="o", color="tab:green")
    axes[1].set_ylabel("Peak el deg")
    axes[2].plot(times, score, marker="o", color="tab:red")
    axes[2].set_ylabel("DP reliability")
    axes[2].set_xlabel("Time sec")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Window-wise peak direction track")
    _finalize(fig, output_path)


def plot_summary_panel(
    audio: np.ndarray,
    sample_rate: int,
    windows: list[WindowSTFT],
    tensor: np.ndarray,
    azimuth_tensor: np.ndarray,
    pooled: np.ndarray,
    channel_names: list[str],
    grid: AngularGrid,
    stats: dict[str, Any],
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    step = max(1, audio.shape[0] // 12000)
    time = np.arange(0, audio.shape[0], step, dtype=np.float32) / sample_rate
    axes[0, 0].plot(time, audio[::step, 0], linewidth=0.7)
    axes[0, 0].set_title("W waveform")
    axes[0, 0].set_xlabel("Time sec")
    axes[0, 0].grid(True, alpha=0.25)

    if windows:
        db = 20.0 * np.log10(np.abs(windows[0].stft[0]) + EPSILON)
        axes[0, 1].imshow(db, origin="lower", aspect="auto", cmap="magma")
    axes[0, 1].set_title("W STFT overview")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Freq bin")

    for ax, channel_name, title in [
        (axes[1, 0], "beam_power", "Beam power"),
        (axes[1, 1], "aiv_score", "AIV direction score"),
        (axes[2, 0], "diffuseness", "Diffuseness / uncertainty"),
    ]:
        idx = channel_names.index(channel_name)
        im = ax.imshow(tensor[:, :, idx], origin="lower", aspect="auto", extent=_extent(grid), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Azimuth deg")
        ax.set_ylabel("Elevation deg")
        ax.grid(True, color="white", alpha=0.15, linewidth=0.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    dp_idx = channel_names.index("dp_reliability")
    centers_deg = np.degrees(grid.azimuth_centers)
    axes[2, 1].bar(centers_deg, azimuth_tensor[:, dp_idx], width=360.0 / grid.num_az_bins * 0.8)
    axes[2, 1].set_title(
        "Azimuth DP reliability\n"
        f"top8={stats.get('eight_way_top_label')} peak={stats.get('peak_direction_azimuth_deg')} deg"
    )
    axes[2, 1].set_xlabel("Azimuth deg")
    axes[2, 1].set_ylabel("DP reliability")
    axes[2, 1].set_xticks(np.arange(-180, 181, 45))
    axes[2, 1].grid(True, axis="y", alpha=0.25)

    meta_text = (
        f"sr={stats.get('sample_rate')} duration={stats.get('duration_sec'):.2f}s\n"
        f"bins E/A={grid.num_el_bins}/{grid.num_az_bins} nonzero={stats.get('nonzero_bins')}\n"
        f"peak az/el={stats.get('peak_direction_azimuth_deg')}/{stats.get('peak_direction_elevation_deg')}\n"
        f"8way top={stats.get('eight_way_top_label')}"
    )
    fig.text(0.01, 0.01, meta_text, fontsize=10, va="bottom", ha="left", bbox={"facecolor": "white", "alpha": 0.85})
    fig.suptitle("Audio spherical representation summary")
    _finalize(fig, output_path)

