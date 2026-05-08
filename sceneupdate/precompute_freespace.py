#!/usr/bin/env python3
"""
Precompute initial free space from ConceptGraph offline data.

Uses saved RGB, depth, camera poses, and scene graph (obj_json) to
accumulate occupancy grids for each surface (desk, cabinet, counter, etc.)
across all frames, then outputs a surface_json with pre-computed free spaces.

Usage:
    python precompute_freespace.py \
        --data_dir /home/sungbin/extra/data/cg_replica/session_20260202_185355 \
        --exp_dir  exps/r_mapping_stride10 \
        --output   data/surface_json_v2_r_mapping_stride10.json
"""

import os
import sys
import json
import argparse
import time

import cv2
import numpy as np

# ── Surface tags eligible for free space analysis ──
SURFACE_TAGS = {"desk", "table", "dining table", "coffee table",
                "side table", "end table", "workbench", "counter",
                "cabinet", "shelf", "dresser", "nightstand", "bench"}

# ── Grid parameters ──
CELL_SIZE = 0.02          # 2cm cells
HEIGHT_MIN = 0.02         # 2cm above surface = "object"
HEIGHT_MAX = 0.30         # 30cm above surface
EXTENT_SHRINK = 0.95      # shrink bbox extent
EDGE_INSET_RATIO = 0.05
EDGE_INSET_MIN = 0.03
OBJ_DILATE_PX = 3         # morphological dilation for occupied cells
STRIDE = 4                # depth downsample stride


def load_poses(traj_path):
    """Load camera-to-world 4x4 poses from traj.txt."""
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def load_scene_graph(obj_json_path):
    """Load scene graph and extract surfaces."""
    with open(obj_json_path) as f:
        sg = json.load(f)

    surfaces = {}
    all_objects = {}

    for key, info in sg.items():
        obj_id = info.get("id")
        tag = info.get("object_tag", "").lower().strip()
        center = info.get("bbox_center")
        extent = info.get("bbox_extent")
        if center is None or extent is None or obj_id is None:
            continue

        all_objects[obj_id] = {
            "tag": tag,
            "center": np.array(center, dtype=float),
            "extent": np.array(extent, dtype=float),
        }

        # Check if this is a surface
        suitable = any(st in tag or tag in st for st in SURFACE_TAGS)
        if not suitable:
            continue

        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        ew, ed, eh = float(extent[0]), float(extent[1]), float(extent[2])

        surface_z = cz + eh / 2.0 - 0.02
        sw = ew * EXTENT_SHRINK
        sd = ed * EXTENT_SHRINK
        inset_x = max(EDGE_INSET_MIN, sw * EDGE_INSET_RATIO)
        inset_y = max(EDGE_INSET_MIN, sd * EDGE_INSET_RATIO)
        x_min = cx - sw / 2.0 + inset_x
        x_max = cx + sw / 2.0 - inset_x
        y_min = cy - sd / 2.0 + inset_y
        y_max = cy + sd / 2.0 - inset_y

        if x_max - x_min < 0.1 or y_max - y_min < 0.1:
            continue

        nx = max(1, int(round((x_max - x_min) / CELL_SIZE)))
        ny = max(1, int(round((y_max - y_min) / CELL_SIZE)))

        surfaces[obj_id] = {
            "tag": tag,
            "surface_z": surface_z,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "nx": nx, "ny": ny,
            "center": [cx, cy, cz],
            "extent": [ew, ed, eh],
            # Accumulated grids (float for averaging)
            "visibility_count": np.zeros((ny, nx), dtype=np.int32),
            "occupied_count": np.zeros((ny, nx), dtype=np.int32),
        }

    return surfaces, all_objects


def backproject_depth(depth_m, T_cam2world, fx, fy, cx, cy, stride=STRIDE):
    """Back-project depth image to world coordinates using camera pose.

    Args:
        depth_m: depth in meters (H, W)
        T_cam2world: 4x4 camera-to-world transform
        fx, fy, cx, cy: camera intrinsics
        stride: downsample stride
    Returns:
        pts_world: (N, 3) world coordinates
    """
    H, W = depth_m.shape[:2]

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x.ravel().astype(np.float32)
    py = grid_y.ravel().astype(np.float32)

    depths = depth_m[grid_y.ravel(), grid_x.ravel()]

    valid = (depths > 0.1) & (depths < 10.0)
    if valid.sum() < 10:
        return None

    px, py, d = px[valid], py[valid], depths[valid]

    # Back-project to camera frame (standard: X-right, Y-down, Z-forward)
    x_cam = (px - cx) * d / fx
    y_cam = (py - cy) * d / fy
    z_cam = d

    # Stack as (N, 4) homogeneous
    ones = np.ones_like(x_cam)
    pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=1)  # (N, 4)

    # Camera → World
    pts_world = (T_cam2world @ pts_cam.T).T[:, :3]  # (N, 3)

    return pts_world.astype(np.float32)


def collect_z_samples(surf, pts_world):
    """Collect Z values of points falling within a surface's XY bounds.
    Used in pass 1 to determine the true surface Z from point cloud data."""
    x_min, x_max = surf["x_min"], surf["x_max"]
    y_min, y_max = surf["y_min"], surf["y_max"]
    bbox_z = surf["surface_z"]

    mask_xy = ((pts_world[:, 0] >= x_min) & (pts_world[:, 0] <= x_max) &
               (pts_world[:, 1] >= y_min) & (pts_world[:, 1] <= y_max))
    # Broad Z range: from bottom of bbox to well above top
    z_low = bbox_z - 0.5
    z_high = bbox_z + 0.5
    mask_z = (pts_world[:, 2] >= z_low) & (pts_world[:, 2] <= z_high)

    roi_pts = pts_world[mask_xy & mask_z]
    if len(roi_pts) > 0:
        surf["_z_samples"].extend(roi_pts[:, 2].tolist())


def detect_surface_z(surf):
    """Detect true surface Z from collected Z samples using histogram peak.

    The actual top surface of a cabinet/desk shows up as a dense horizontal band
    of points. We find that peak in the Z histogram and use it as the true surface_z.
    """
    samples = np.array(surf["_z_samples"])
    if len(samples) < 50:
        return surf["surface_z"]  # not enough data, keep bbox estimate

    bbox_z = surf["surface_z"]

    # Build fine histogram (1cm bins) around bbox_z
    z_min = bbox_z - 0.3
    z_max = bbox_z + 0.3
    bin_size = 0.01
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    hist, edges = np.histogram(samples, bins=bins)

    if hist.max() == 0:
        return bbox_z

    # Find the peak bin — this is the densest horizontal surface
    peak_idx = np.argmax(hist)
    peak_z = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0

    # The surface Z should be at or just below the peak (the peak is the surface itself)
    # Refine: take the weighted mean of the top 3 bins around peak
    lo = max(0, peak_idx - 1)
    hi = min(len(hist), peak_idx + 2)
    weights = hist[lo:hi]
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(lo, hi)]
    if weights.sum() > 0:
        refined_z = np.average(centers, weights=weights)
    else:
        refined_z = peak_z

    return refined_z


def accumulate_surface(surf, pts_world):
    """Accumulate visibility and occupancy for one surface from one frame's points."""
    sz = surf["surface_z"]
    x_min, x_max = surf["x_min"], surf["x_max"]
    y_min, y_max = surf["y_min"], surf["y_max"]
    nx, ny = surf["nx"], surf["ny"]

    # XY filter
    mask_xy = ((pts_world[:, 0] >= x_min) & (pts_world[:, 0] <= x_max) &
               (pts_world[:, 1] >= y_min) & (pts_world[:, 1] <= y_max))

    # Z band filter: surface level through height_max above
    z_low = sz - 0.10
    z_high = sz + HEIGHT_MAX
    mask_z = (pts_world[:, 2] >= z_low) & (pts_world[:, 2] <= z_high)

    roi_pts = pts_world[mask_xy & mask_z]
    if len(roi_pts) < 3:
        return

    # Separate surface-level vs object points
    z_obj_threshold = sz + HEIGHT_MIN
    obj_pts = roi_pts[roi_pts[:, 2] >= z_obj_threshold]
    surf_pts = roi_pts[roi_pts[:, 2] < z_obj_threshold]

    cs = CELL_SIZE

    # Mark visible cells (any depth point in ROI)
    for pts in [surf_pts, obj_pts]:
        if len(pts) == 0:
            continue
        gx = np.clip(((pts[:, 0] - x_min) / cs).astype(int), 0, nx - 1)
        gy = np.clip(((pts[:, 1] - y_min) / cs).astype(int), 0, ny - 1)
        np.add.at(surf["visibility_count"], (gy, gx), 1)

    # Mark occupied cells (object points only)
    if len(obj_pts) > 0:
        gx = np.clip(((obj_pts[:, 0] - x_min) / cs).astype(int), 0, nx - 1)
        gy = np.clip(((obj_pts[:, 1] - y_min) / cs).astype(int), 0, ny - 1)
        np.add.at(surf["occupied_count"], (gy, gx), 1)


def finalize_grids(surfaces):
    """Convert accumulated counts to binary grids and compute free rects."""
    results = {}
    for sid, surf in surfaces.items():
        vis = surf["visibility_count"]
        occ = surf["occupied_count"]
        nx, ny = surf["nx"], surf["ny"]

        # Visibility: cell observed in at least 2 frames
        visibility = (vis >= 2).astype(np.uint8)

        # Occupied: cell has object points in at least 2 frames
        occupied = (occ >= 2).astype(np.uint8)

        # Dilate occupied
        if OBJ_DILATE_PX > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (OBJ_DILATE_PX * 2 + 1, OBJ_DILATE_PX * 2 + 1))
            occupied = cv2.dilate(occupied, kernel)

        # Free = visible AND not occupied
        free = (visibility & (~occupied.astype(bool)).astype(np.uint8)).astype(np.uint8)

        n_total = nx * ny
        n_visible = int(visibility.sum())
        n_occupied = int(occupied.sum())
        n_free = int(free.sum())
        vis_ratio = n_visible / n_total if n_total > 0 else 0

        # Connected components → free rects
        free_rects = []
        if n_free > 0:
            n_labels, labels = cv2.connectedComponents(free, connectivity=4)
            for label_id in range(1, n_labels):
                ys, xs = np.where(labels == label_id)
                gx0, gx1 = int(xs.min()), int(xs.max()) + 1
                gy0, gy1 = int(ys.min()), int(ys.max()) + 1

                rx_min = surf["x_min"] + gx0 * CELL_SIZE
                rx_max = surf["x_min"] + gx1 * CELL_SIZE
                ry_min = surf["y_min"] + gy0 * CELL_SIZE
                ry_max = surf["y_min"] + gy1 * CELL_SIZE

                w = rx_max - rx_min
                d = ry_max - ry_min
                if w < 0.05 or d < 0.05:
                    continue

                free_rects.append({
                    "direction": "tile",
                    "center": [
                        round((rx_min + rx_max) / 2, 3),
                        round((ry_min + ry_max) / 2, 3),
                        round(surf["surface_z"], 3),
                    ],
                    "available_size": [round(w, 3), round(d, 3)],
                    "area_m2": round(w * d, 4),
                    "bounds": {
                        "x_min": round(rx_min, 3),
                        "y_min": round(ry_min, 3),
                        "x_max": round(rx_max, 3),
                        "y_max": round(ry_max, 3),
                    },
                })

        free_rects.sort(key=lambda r: r["area_m2"], reverse=True)

        results[sid] = {
            "n_visible": n_visible,
            "n_occupied": n_occupied,
            "n_free": n_free,
            "visibility_ratio": round(vis_ratio, 3),
            "free_rects": free_rects,
            "free": free,
            "occupied": occupied,
            "visibility": visibility,
        }

    return results


def build_surface_json(surfaces, results, all_objects, edge_json_path=None):
    """Build the surface_json output format (compatible with surface_analyzer_v2)."""
    # Load edges if available
    edge_list = []
    if edge_json_path and os.path.exists(edge_json_path):
        with open(edge_json_path) as f:
            edge_data = json.load(f)
        if isinstance(edge_data, dict):
            edge_list = edge_data.get("edges", [])
        elif isinstance(edge_data, list):
            edge_list = edge_data

    # Build detection_id → obj_id mapping from edge objects list
    det_to_obj = {}
    if isinstance(edge_data, dict) and "objects" in edge_data:
        for o in edge_data["objects"]:
            det_id = o.get("detection_id")
            if det_id is not None:
                det_to_obj[o.get("pcd_index")] = det_id

    # Determine which objects are ON which surface
    surface_objects = {sid: [] for sid in surfaces}
    for obj_id, obj in all_objects.items():
        if obj_id in surfaces:
            continue  # skip surfaces themselves
        tag = obj["tag"]
        c = obj["center"]
        e = obj["extent"]

        # Check edges first
        placed_by_edge = False
        for edge in edge_list:
            rel = edge.get("object_relation", "").lower()
            # "a on b" means src is on dst
            if "on" in rel:
                src_det = det_to_obj.get(edge.get("src_index"))
                dst_det = det_to_obj.get(edge.get("dst_index"))
                if "a on b" in rel or "a on top" in rel:
                    if src_det == obj_id and dst_det in surfaces:
                        surface_objects[dst_det].append(obj_id)
                        placed_by_edge = True
                        break
                elif "b on a" in rel:
                    if dst_det == obj_id and src_det in surfaces:
                        surface_objects[src_det].append(obj_id)
                        placed_by_edge = True
                        break

        if placed_by_edge:
            continue

        # Fallback: proximity check
        for sid, surf in surfaces.items():
            sz = surf["surface_z"]
            obj_bottom = c[2] - e[2] / 2.0
            if abs(obj_bottom - sz) > 0.15:
                continue
            if (surf["x_min"] <= c[0] <= surf["x_max"] and
                    surf["y_min"] <= c[1] <= surf["y_max"]):
                surface_objects[sid].append(obj_id)
                break

    # Build output
    output_surfaces = []
    for sid, surf in surfaces.items():
        result = results.get(sid, {})
        free_rects = result.get("free_rects", [])
        sw = surf["x_max"] - surf["x_min"]
        sd = surf["y_max"] - surf["y_min"]

        objs_on = []
        for oid in surface_objects.get(sid, []):
            obj = all_objects[oid]
            objs_on.append({
                "obj_id": oid,
                "tag": obj["tag"],
                "footprint": [round(obj["extent"][0] * 0.85, 3),
                               round(obj["extent"][1] * 0.85, 3)],
                "center_xy": [round(obj["center"][0], 3),
                               round(obj["center"][1], 3)],
            })

        total_free_area = sum(r["area_m2"] for r in free_rects)
        occupied_area = sum(fp["footprint"][0] * fp["footprint"][1] for fp in objs_on)

        output_surfaces.append({
            "surface_id": sid,
            "surface_tag": surf["tag"],
            "surface_z": round(surf["surface_z"], 3),
            "surface_size": [round(sw, 3), round(sd, 3)],
            "surface_area_m2": round(sw * sd, 4),
            "objects_on_surface": objs_on,
            "occupied_area_m2": round(occupied_area, 4),
            "free_area_m2": round(total_free_area, 4),
            "occupancy_ratio": round(occupied_area / (sw * sd), 3) if sw * sd > 0 else 0,
            "max_placeable_size": free_rects[0]["available_size"] if free_rects else [0, 0],
            "free_spaces": free_rects,
            "depth_precomputed": True,
            "depth_stats": {
                "n_visible": result.get("n_visible", 0),
                "n_occupied": result.get("n_occupied", 0),
                "n_free": result.get("n_free", 0),
                "visibility_ratio": result.get("visibility_ratio", 0),
            },
        })

    return {"surfaces": output_surfaces}


def save_debug_images(surfaces, results, output_dir):
    """Save per-surface occupancy grid debug images."""
    os.makedirs(output_dir, exist_ok=True)

    for sid, surf in surfaces.items():
        result = results.get(sid)
        if result is None:
            continue

        vis = result["visibility"]
        occ = result["occupied"]
        free = result["free"]
        ny, nx = vis.shape

        # Scale up for visibility
        scale = max(1, 400 // max(nx, ny))
        img = np.zeros((ny * scale, nx * scale, 3), dtype=np.uint8)

        for gy in range(ny):
            for gx in range(nx):
                y0, y1 = gy * scale, (gy + 1) * scale
                x0, x1 = gx * scale, (gx + 1) * scale
                if free[gy, gx]:
                    img[y0:y1, x0:x1] = (0, 200, 0)     # green = free
                elif occ[gy, gx]:
                    img[y0:y1, x0:x1] = (0, 0, 200)     # red = occupied
                elif vis[gy, gx]:
                    img[y0:y1, x0:x1] = (80, 80, 80)    # gray = visible but not free/occ
                # else: black = not observed

        # Grid lines
        for gx in range(0, nx + 1, 10):
            x = gx * scale
            if x < img.shape[1]:
                cv2.line(img, (x, 0), (x, img.shape[0]), (40, 40, 40), 1)
        for gy in range(0, ny + 1, 10):
            y = gy * scale
            if y < img.shape[0]:
                cv2.line(img, (0, y), (img.shape[1], y), (40, 40, 40), 1)

        # Title
        tag = surf["tag"]
        n_vis = result["n_visible"]
        n_free = result["n_free"]
        vis_ratio = result["visibility_ratio"]
        title = f"{tag}(id:{sid}) vis:{vis_ratio:.0%} free:{n_free} cells"
        cv2.putText(img, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        path = os.path.join(output_dir, f"precomputed_s{sid}_{tag}.png")
        cv2.imwrite(path, img)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Precompute free space from ConceptGraph data")
    parser.add_argument("--data_dir", type=str,
                        default="/home/sungbin/extra/data/cg_replica/session_20260202_185355",
                        help="ConceptGraph session directory")
    parser.add_argument("--exp_dir", type=str, default="exps/r_mapping_stride10",
                        help="Experiment directory (relative to data_dir)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output surface JSON path (default: <exp_dir>/surface_json_precomputed.json)")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Depth downsample stride (default: {STRIDE})")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Process every N-th frame (default: 1 = all frames)")
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="Save debug images to this directory")
    args = parser.parse_args()

    exp_path = os.path.join(args.data_dir, args.exp_dir)
    results_path = os.path.join(args.data_dir, "results")
    traj_path = os.path.join(args.data_dir, "traj.txt")

    # Camera intrinsics from zed4.yaml
    fx, fy = 1066.666, 1066.666
    cx, cy = 640.0, 360.0
    depth_scale = 1000.0  # mm -> m

    # ── Load data ──
    print("Loading scene graph...")
    obj_json = os.path.join(exp_path, "obj_json_r_mapping_stride10.json")
    surfaces, all_objects = load_scene_graph(obj_json)
    print(f"  {len(surfaces)} surfaces, {len(all_objects)} total objects")
    for sid, s in surfaces.items():
        print(f"    {s['tag']}(id:{sid}) z={s['surface_z']:.2f} "
              f"bounds=({s['x_min']:.2f},{s['y_min']:.2f})-({s['x_max']:.2f},{s['y_max']:.2f}) "
              f"grid={s['nx']}x{s['ny']}")

    print("\nLoading poses...")
    poses = load_poses(traj_path)
    print(f"  {len(poses)} poses")

    # Count available frames
    frame_files = sorted([f for f in os.listdir(results_path) if f.startswith("depth") and f.endswith(".png")])
    n_frames = min(len(frame_files), len(poses))
    print(f"  {n_frames} frames available")

    # ── Pass 1: Collect Z samples to detect true surface_z ──
    print(f"\nPass 1: Detecting true surface Z from depth data ({n_frames} frames)...")
    for surf in surfaces.values():
        surf["_z_samples"] = []

    t0 = time.time()
    # Use a coarser stride for pass 1 (faster)
    pass1_frame_stride = max(args.frame_stride, 3)
    for i in range(0, n_frames, pass1_frame_stride):
        depth_path = os.path.join(results_path, f"depth{i:06d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) / depth_scale
        T = poses[i]
        pts_world = backproject_depth(depth_m, T, fx, fy, cx, cy, stride=args.stride)
        if pts_world is None:
            continue
        for surf in surfaces.values():
            collect_z_samples(surf, pts_world)

    print(f"  Pass 1 done in {time.time()-t0:.1f}s")

    # Detect and update surface_z
    print("\n  Surface Z correction (bbox → depth-detected):")
    for sid, surf in surfaces.items():
        old_z = surf["surface_z"]
        new_z = detect_surface_z(surf)
        n_samples = len(surf["_z_samples"])
        delta = new_z - old_z
        marker = " ***" if abs(delta) > 0.05 else ""
        print(f"    {surf['tag']}(id:{sid}): bbox_z={old_z:.3f} → detected_z={new_z:.3f} "
              f"(delta={delta:+.3f}, {n_samples} samples){marker}")
        surf["surface_z"] = new_z
        del surf["_z_samples"]  # free memory

    # ── Pass 2: Accumulate occupancy grids with corrected surface_z ──
    print(f"\nPass 2: Accumulating occupancy ({n_frames} frames, stride={args.stride})...")
    t0 = time.time()
    processed = 0

    for i in range(0, n_frames, args.frame_stride):
        depth_path = os.path.join(results_path, f"depth{i:06d}.png")
        if not os.path.exists(depth_path):
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) / depth_scale

        T = poses[i]
        pts_world = backproject_depth(depth_m, T, fx, fy, cx, cy, stride=args.stride)
        if pts_world is None:
            continue

        for sid, surf in surfaces.items():
            accumulate_surface(surf, pts_world)

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {i}/{n_frames} ({processed} processed, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)")

    # ── Finalize ──
    print("\nFinalizing grids...")
    results = finalize_grids(surfaces)

    for sid in sorted(results.keys()):
        r = results[sid]
        tag = surfaces[sid]["tag"]
        n_rects = len(r["free_rects"])
        total_area = sum(rect["area_m2"] for rect in r["free_rects"])
        print(f"  {tag}(id:{sid}): vis={r['visibility_ratio']:.0%} "
              f"free={r['n_free']} cells, {n_rects} rects, {total_area:.4f}m²")

    # ── Build and save output ──
    # Try to find edge JSON
    edge_json = None
    for candidate in ["edge_json_r_mapping_stride10_gemini_v2.json",
                       "edge_json_r_mapping_stride10.json"]:
        p = os.path.join(exp_path, candidate)
        if os.path.exists(p):
            edge_json = p
            break

    output_data = build_surface_json(surfaces, results, all_objects, edge_json)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(exp_path, "surface_json_precomputed.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {output_path}")
    print(f"  {len(output_data['surfaces'])} surfaces with precomputed free space")

    # ── Debug images ──
    debug_dir = args.debug_dir
    if debug_dir is None:
        debug_dir = os.path.join(exp_path, "freespace_debug")
    print(f"\nSaving debug images to {debug_dir}...")
    save_debug_images(surfaces, results, debug_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
