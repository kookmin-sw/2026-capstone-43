from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .pooling_utils import EIGHT_WAY_LABELS
from .spherical_projection import DEPTH_LIKE_CHANNELS, FeatureBundle


def _subsample_for_plot(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices]


def colorize_depth(
    depth_map: np.ndarray,
    cmap_name: str = "magma_r",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    valid_mask = np.isfinite(depth_map) & (depth_map > 0.0)
    if not np.any(valid_mask):
        return np.full((*depth_map.shape, 3), 160, dtype=np.uint8)

    valid_values = depth_map[valid_mask]
    lo = np.percentile(valid_values, lower_percentile)
    hi = np.percentile(valid_values, upper_percentile)
    if hi <= lo:
        hi = lo + 1.0e-6

    normalized = np.clip((depth_map - lo) / (hi - lo), 0.0, 1.0)
    colormap = plt.get_cmap(cmap_name)
    rgba = colormap(normalized)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~valid_mask] = np.asarray([160, 160, 160], dtype=np.uint8)
    return rgb


def save_depth_visualization(depth_map: np.ndarray, output_path: Path) -> Path:
    depth_rgb = colorize_depth(depth_map)
    Image.fromarray(depth_rgb, mode="RGB").save(output_path)
    return output_path


def _set_equal_3d_axes(ax: plt.Axes, points: np.ndarray) -> None:
    if points.shape[0] == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)


def save_point_cloud_views(
    points: np.ndarray,
    colors: np.ndarray,
    output_3d_path: Path,
    output_topdown_path: Path,
    output_sideview_path: Path,
    max_plot_points: int = 40_000,
    seed: int = 0,
) -> None:
    plot_points, plot_colors = _subsample_for_plot(points, colors, max_plot_points, seed=seed)
    plot_colors_normalized = plot_colors.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        c=plot_colors_normalized,
        s=1.0,
        alpha=0.8,
        linewidths=0.0,
    )
    ax.scatter([0.0], [0.0], [0.0], c="red", s=24.0, label="camera")
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (forward)")
    ax.set_title("Colored Point Cloud (3D)")
    ax.view_init(elev=24.0, azim=-62.0)
    _set_equal_3d_axes(ax, plot_points)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_3d_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 2],
        c=plot_colors_normalized,
        s=2.0,
        alpha=0.7,
        linewidths=0.0,
    )
    ax.scatter([0.0], [0.0], c="red", s=28.0, label="camera")
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Z (forward)")
    ax.set_title("Point Cloud Top-Down (X-Z)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_topdown_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        plot_points[:, 2],
        plot_points[:, 1],
        c=plot_colors_normalized,
        s=2.0,
        alpha=0.7,
        linewidths=0.0,
    )
    ax.scatter([0.0], [0.0], c="red", s=28.0, label="camera")
    ax.set_xlabel("Z (forward)")
    ax.set_ylabel("Y (up)")
    ax.set_title("Point Cloud Side View (Z-Y)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_sideview_path, dpi=180)
    plt.close(fig)


def _channel_display_map(bundle: FeatureBundle, channel_name: str) -> np.ndarray:
    values = bundle.channels[channel_name].astype(np.float32).copy()
    observed_mask = bundle.channels["observed_mask"] > 0.5
    if values.ndim == 1:
        display = values[np.newaxis, :]
        observed = observed_mask[np.newaxis, :]
    else:
        display = values
        observed = observed_mask
    display = display.astype(np.float32)
    display[~observed] = np.nan
    return display


def _heatmap_extent(bundle: FeatureBundle) -> list[float]:
    elevation_edges = bundle.elevation_edges
    if elevation_edges is None:
        elevation_edges = np.asarray([-math.pi / 2.0, math.pi / 2.0], dtype=np.float32)
    return [
        float(np.degrees(bundle.azimuth_edges[0])),
        float(np.degrees(bundle.azimuth_edges[-1])),
        float(np.degrees(elevation_edges[0])),
        float(np.degrees(elevation_edges[-1])),
    ]


def _overlay_bin_centers(ax: plt.Axes, bundle: FeatureBundle) -> None:
    if bundle.elevation_centers is None:
        azimuth_deg = np.degrees(bundle.azimuth_centers)
        ax.scatter(azimuth_deg, np.zeros_like(azimuth_deg), s=6.0, c="white", alpha=0.7, edgecolors="none")
        return

    observed = bundle.channels["observed_mask"] > 0.5
    azimuth_grid, elevation_grid = np.meshgrid(
        np.degrees(bundle.azimuth_centers),
        np.degrees(bundle.elevation_centers),
        indexing="xy",
    )
    ax.scatter(
        azimuth_grid[observed],
        elevation_grid[observed],
        s=6.0,
        c="white",
        alpha=0.6,
        edgecolors="none",
    )


def save_spherical_channel_heatmap(
    bundle: FeatureBundle,
    channel_name: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap_name: str = "viridis",
    overlay_bin_centers: bool = True,
) -> None:
    display = _channel_display_map(bundle, channel_name)
    masked = np.ma.masked_invalid(display)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=_heatmap_extent(bundle),
        cmap=cmap,
    )
    if overlay_bin_centers:
        _overlay_bin_centers(ax, bundle)
    ax.set_xlabel("Azimuth (deg, +right, 0=forward)")
    ax.set_ylabel("Elevation (deg, +up)")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_channel_panel(
    bundle: FeatureBundle,
    channel_names: Sequence[str],
    output_path: Path,
    title_prefix: str = "Vision Sphere",
) -> None:
    cols = 3
    rows = int(math.ceil(len(channel_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.8 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)

    for axis in axes_array.flat:
        axis.axis("off")

    for axis, channel_name in zip(axes_array.flat, channel_names, strict=False):
        display = _channel_display_map(bundle, channel_name)
        masked = np.ma.masked_invalid(display)
        cmap_name = "viridis" if channel_name not in DEPTH_LIKE_CHANNELS else "magma"
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="#d9d9d9")
        image = axis.imshow(masked, origin="lower", aspect="auto", extent=_heatmap_extent(bundle), cmap=cmap)
        _overlay_bin_centers(axis, bundle)
        axis.set_title(f"{title_prefix}: {channel_name}")
        axis.set_xlabel("Azimuth (deg)")
        axis.set_ylabel("Elevation (deg)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.axis("on")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sector_spans() -> list[tuple[str, tuple[float, float]]]:
    return [
        ("front-left", (-67.5, -22.5)),
        ("front", (-22.5, 22.5)),
        ("front-right", (22.5, 67.5)),
        ("right", (67.5, 112.5)),
        ("back-right", (112.5, 157.5)),
        ("back", (157.5, 180.0)),
        ("back-left", (-157.5, -112.5)),
        ("left", (-112.5, -67.5)),
    ]


def _overlay_sector_background(ax: plt.Axes) -> None:
    colors = ["#d8efff", "#eef7d5", "#ffeec9", "#ffe0cc", "#f7d8ff", "#f4d8d8", "#dbe4ff", "#ddebdc"]
    for color, (label, (left_deg, right_deg)) in zip(colors, _sector_spans(), strict=True):
        ax.axvspan(left_deg, right_deg, color=color, alpha=0.35)
        if label == "back":
            ax.axvspan(-180.0, -157.5, color=color, alpha=0.35)


def save_8way_overlay_visualization(
    azimuth_bundle: FeatureBundle,
    output_path: Path,
) -> None:
    azimuth_deg = np.degrees(azimuth_bundle.azimuth_centers)
    occupancy = azimuth_bundle.channels["occupancy"]
    p10_depth = azimuth_bundle.channels["p10_depth"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis in axes:
        _overlay_sector_background(axis)
        axis.grid(True, alpha=0.25)

    axes[0].bar(azimuth_deg, occupancy, width=np.diff(np.degrees(azimuth_bundle.azimuth_edges)) * 0.9, color="#2b7a78")
    axes[0].set_ylabel("Occupancy")
    axes[0].set_title("8-Way Sector Overlay on Azimuth Occupancy")

    axes[1].bar(azimuth_deg, p10_depth, width=np.diff(np.degrees(azimuth_bundle.azimuth_edges)) * 0.9, color="#d95f02")
    axes[1].set_ylabel("P10 Depth")
    axes[1].set_xlabel("Azimuth (deg, +right, 0=forward)")
    axes[1].set_title("8-Way Sector Overlay on Azimuth P10 Depth")

    for axis in axes:
        for label, (left_deg, right_deg) in _sector_spans():
            center_deg = (left_deg + right_deg) * 0.5
            axis.text(center_deg, axis.get_ylim()[1] * 0.93 if axis.get_ylim()[1] > 0 else 0.9, label, ha="center", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_azimuth_multichannel_plot(
    azimuth_bundle: FeatureBundle,
    output_path: Path,
) -> None:
    azimuth_deg = np.degrees(azimuth_bundle.azimuth_centers)
    width = np.diff(np.degrees(azimuth_bundle.azimuth_edges)) * 0.9
    observed_mask = azimuth_bundle.channels["observed_mask"]
    has_points = azimuth_bundle.channels["has_points"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    plots = [
        ("occupancy", "Occupancy", "#2b7a78"),
        ("p10_depth", "P10 Depth", "#d95f02"),
        ("mean_depth", "Mean Depth", "#7570b3"),
        ("valid_ratio", "Valid Ratio", "#1b9e77"),
    ]

    for axis in axes:
        _overlay_sector_background(axis)
        axis.grid(True, alpha=0.25)

    for axis, (channel_name, label, color) in zip(axes, plots, strict=True):
        axis.bar(azimuth_deg, azimuth_bundle.channels[channel_name], width=width, color=color)
        axis.set_ylabel(label)

    axes[3].plot(azimuth_deg, observed_mask, color="#111111", linewidth=1.5, label="observed_mask")
    axes[3].plot(azimuth_deg, has_points, color="#c62828", linewidth=1.5, label="has_points")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("Azimuth (deg, +right, 0=forward)")
    axes[0].set_title("Azimuth Multi-Channel Diagnostics")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metadata_text_box(
    lines: Sequence[str],
    output_path: Path,
    title: str = "Sample Metadata",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.set_title(title, loc="left")
    text = "\n".join(lines)
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.6"},
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_globe_visualization(
    bundle: FeatureBundle,
    channel_name: str,
    output_path: Path,
    title: str,
    cmap_name: str = "viridis",
) -> None:
    if bundle.elevation_centers is None:
        return

    values = bundle.channels[channel_name].astype(np.float32)
    observed_mask = bundle.channels["observed_mask"] > 0.5
    display = values.copy()
    display[~observed_mask] = np.nan

    azimuth_grid, elevation_grid = np.meshgrid(bundle.azimuth_centers, bundle.elevation_centers, indexing="xy")
    x = np.cos(elevation_grid) * np.sin(azimuth_grid)
    y = np.sin(elevation_grid)
    z = np.cos(elevation_grid) * np.cos(azimuth_grid)

    finite_values = display[np.isfinite(display)]
    if finite_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if vmax <= vmin:
            vmax = vmin + 1.0e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    facecolors = cmap(norm(np.nan_to_num(display, nan=vmin)))
    facecolors[~np.isfinite(display)] = np.asarray([0.85, 0.85, 0.85, 1.0])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    ax.set_title(title)
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (forward)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25.0, azim=-55.0)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(finite_values if finite_values.size > 0 else np.asarray([0.0]))
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_panel(panel_items: list[tuple[str, Path]], output_path: Path) -> None:
    cols = 3
    rows = int(math.ceil(len(panel_items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)

    for axis in axes_array.flat:
        axis.axis("off")

    for axis, (title, image_path) in zip(axes_array.flat, panel_items, strict=False):
        image = Image.open(image_path)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
