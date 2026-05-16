from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .geometry import basis_from_yaw


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    return plt, PolyCollection


def _sample_local_navigability_grid(
    pathfinder: object,
    *,
    floor_y: float,
    mic_xz: np.ndarray,
    source_xz: np.ndarray,
    min_window_m: float = 8.0,
    margin_m: float = 2.0,
    meters_per_cell: float = 0.15,
    max_cells: int = 96,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    center_xz = 0.5 * (mic_xz + source_xz)
    pair_dist = float(np.linalg.norm(source_xz - mic_xz))
    window_m = max(float(min_window_m), pair_dist + (2.0 * float(margin_m)))
    n_cells = int(np.clip(np.ceil(window_m / max(meters_per_cell, 1.0e-3)), 32, max_cells))

    half = 0.5 * window_m
    x_min = float(center_xz[0] - half)
    x_max = float(center_xz[0] + half)
    z_min = float(center_xz[1] - half)
    z_max = float(center_xz[1] + half)

    xs = np.linspace(x_min, x_max, n_cells, dtype=np.float64)
    zs = np.linspace(z_min, z_max, n_cells, dtype=np.float64)

    nav = np.zeros((n_cells, n_cells), dtype=np.float32)
    is_navigable = getattr(pathfinder, "is_navigable", None)
    if is_navigable is None:
        return nav, (x_min, x_max, z_min, z_max)

    for zi, z in enumerate(zs):
        for xi, x in enumerate(xs):
            point = np.array([x, float(floor_y), z], dtype=np.float32)
            try:
                nav[zi, xi] = 1.0 if bool(is_navigable(point)) else 0.0
            except Exception:
                nav[zi, xi] = 0.0
    return nav, (x_min, x_max, z_min, z_max)


def _extract_projected_navmesh_triangles(pathfinder: object) -> Optional[np.ndarray]:
    build_vertices = getattr(pathfinder, "build_navmesh_vertices", None)
    build_indices = getattr(pathfinder, "build_navmesh_vertex_indices", None)
    if build_vertices is None or build_indices is None:
        return None

    try:
        vertices_raw = build_vertices()
        indices_raw = build_indices()
    except Exception:
        return None

    if not vertices_raw or not indices_raw:
        return None

    vertices = np.asarray(
        [[float(vertex[0]), float(vertex[2])] for vertex in vertices_raw],
        dtype=np.float64,
    )
    indices = np.asarray(list(indices_raw), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or indices.size < 3:
        return None

    triangle_count = indices.size // 3
    if triangle_count <= 0:
        return None
    indices = indices[: triangle_count * 3].reshape(triangle_count, 3)
    valid_index_mask = np.all((indices >= 0) & (indices < vertices.shape[0]), axis=1)
    if not np.any(valid_index_mask):
        return None

    triangles = vertices[indices[valid_index_mask]]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    doubled_area = np.abs((edge_a[:, 0] * edge_b[:, 1]) - (edge_a[:, 1] * edge_b[:, 0]))
    triangles = triangles[doubled_area > 1.0e-6]
    if triangles.size == 0:
        return None
    return triangles


def _compute_navmesh_anchor(pathfinder: object, point_world: np.ndarray) -> Optional[np.ndarray]:
    snap_point = getattr(pathfinder, "snap_point", None)
    if snap_point is None:
        return None
    try:
        snapped = np.asarray(
            snap_point(point_world.astype(np.float32)),
            dtype=np.float64,
        )
    except Exception:
        return None
    if snapped.shape != (3,) or not np.isfinite(snapped).all():
        return None
    return snapped


def _set_debug_extent(
    ax: object,
    triangles: Optional[np.ndarray],
    mic: np.ndarray,
    source: np.ndarray,
    source_anchor: Optional[np.ndarray],
) -> None:
    x_values = [float(mic[0]), float(source[0])]
    z_values = [float(mic[2]), float(source[2])]
    if source_anchor is not None:
        x_values.append(float(source_anchor[0]))
        z_values.append(float(source_anchor[2]))
    if triangles is not None and triangles.size > 0:
        x_values.extend(triangles[:, :, 0].reshape(-1).astype(float).tolist())
        z_values.extend(triangles[:, :, 1].reshape(-1).astype(float).tolist())

    x_min = min(x_values)
    x_max = max(x_values)
    z_min = min(z_values)
    z_max = max(z_values)
    span = max(x_max - x_min, z_max - z_min, 1.0)
    pad = max(0.5, 0.04 * span)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(z_min - pad, z_max + pad)


def save_topdown_debug(
    output_path: Path,
    mic_position_world: Iterable[float],
    mic_yaw_rad: float,
    source_position_world: Iterable[float],
    *,
    geometry_los: str,
    in_fov: bool,
    title: str,
    pathfinder: Optional[object] = None,
    floor_y: Optional[float] = None,
    source_anchor_world: Optional[Iterable[float]] = None,
) -> None:
    plt, PolyCollection = _load_matplotlib()
    mic = np.asarray(list(mic_position_world), dtype=np.float64)
    source = np.asarray(list(source_position_world), dtype=np.float64)
    right, forward, _ = basis_from_yaw(float(mic_yaw_rad))
    forward_tip = mic + (forward * 0.5)
    if source_anchor_world is not None:
        source_anchor = np.asarray(list(source_anchor_world), dtype=np.float64)
    else:
        source_anchor = _compute_navmesh_anchor(pathfinder, source) if pathfinder is not None else None

    color = "green" if geometry_los == "gLOS" else "red"
    fig, ax = plt.subplots(figsize=(5, 5))
    triangles = _extract_projected_navmesh_triangles(pathfinder) if pathfinder is not None else None

    if triangles is not None:
        ax.add_collection(
            PolyCollection(
                triangles,
                facecolors="#9f9f9f",
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
                antialiaseds=False,
                zorder=0,
            )
        )
    elif pathfinder is not None and floor_y is not None:
        nav, extent = _sample_local_navigability_grid(
            pathfinder,
            floor_y=float(floor_y),
            mic_xz=np.array([mic[0], mic[2]], dtype=np.float64),
            source_xz=np.array([source[0], source[2]], dtype=np.float64),
        )
        ax.imshow(
            nav,
            cmap="Greys",
            origin="lower",
            extent=extent,
            alpha=0.65,
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )

    ax.plot(
        [mic[0], source[0]],
        [mic[2], source[2]],
        color=color,
        linewidth=2.0,
        zorder=2,
        label=f"ray ({geometry_los})",
    )
    ax.scatter([mic[0]], [mic[2]], c="blue", s=50, label="mic", zorder=3)
    ax.scatter(
        [source[0]],
        [source[2]],
        c=color,
        s=55,
        label="source actual",
        zorder=4,
    )
    if source_anchor is not None:
        ax.scatter(
            [source_anchor[0]],
            [source_anchor[2]],
            c="orange",
            s=70,
            marker="x",
            linewidths=2.0,
            label="speaker proxy floor anchor",
            zorder=5,
        )
    ax.arrow(
        mic[0],
        mic[2],
        forward_tip[0] - mic[0],
        forward_tip[2] - mic[2],
        head_width=0.08,
        head_length=0.12,
        fc="black",
        ec="black",
        zorder=4,
    )
    _set_debug_extent(ax, triangles, mic, source, source_anchor)
    ax.set_title(f"{title}\n{geometry_los}, in_fov={in_fov}")
    ax.axis("equal")
    ax.set_axis_off()
    anchor_note = "source anchor: unavailable"
    if source_anchor is not None:
        anchor_note = (
            "source anchor height="
            f"{source_anchor[1]:.2f} m, vertical offset={abs(source[1] - source_anchor[1]):.2f} m"
        )
    ax.text(
        0.01,
        0.01,
        "\n".join(
            [
                "gray: projected navmesh footprint (XZ)",
                "multi-level floors may overlap in this view",
                "blue: mic, orange x: speaker proxy floor anchor",
                f"speaker proxy reference height={source[1]:.2f} m",
                anchor_note,
            ]
        ),
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.legend(loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_front_overlay(
    output_path: Path,
    rgb_image: np.ndarray,
    *,
    projected_pixel_xy: Optional[list[float]],
    geometry_los: str,
    projection_reason: Optional[str],
    title: str,
) -> None:
    plt, _ = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image)
    if projected_pixel_xy is not None:
        ax.scatter(
            [projected_pixel_xy[0]],
            [projected_pixel_xy[1]],
            c="lime" if geometry_los == "gLOS" else "red",
            s=50,
            marker="x",
            linewidths=2.0,
        )
    text = title
    if projection_reason:
        text = f"{text}\nreason={projection_reason}"
    ax.set_title(text)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
