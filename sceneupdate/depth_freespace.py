"""
Depth-based Free Space Detection for Desk Surfaces.

Pipeline (Isaac Sim optimized — no RANSAC/voxel needed):
  1. Depth → Point Cloud (full back-projection)
  2. ROI Filter (surface height band + XY bounds)
  3. Top-view Occupancy Grid (2D histogram)
  4. Morphological Post-processing
  5. Hybrid Merge (depth-visible → depth, non-visible → scene graph)

Usage:
  analyzer = DepthFreeSpaceAnalyzer()
  analyzer.register_from_surface_analyzer(surface_analyzer)
  results = analyzer.process_depth(depth_np, robot_pos, robot_quat)
  hybrid_rects = analyzer.get_hybrid_free_rects(surface_id, sg_objects)
"""

import numpy as np
import cv2
import os
import time
from typing import Dict, List, Optional

from config import (CAMERA_FOCAL, CAMERA_H_APERTURE, CAMERA_V_APERTURE,
                    CAMERA_RESOLUTION, PRECOMPUTED_FREESPACE_FILE)

# Camera mount params (must match sayplan_ui.py)
CAMERA_HEIGHT = 1.5
CAMERA_TILT_DEG = 20.0

# Pipeline defaults
DEFAULT_CELL_SIZE = 0.02       # 2cm grid
DEFAULT_HEIGHT_MIN = 0.02      # 2cm above surface (skip surface itself)
DEFAULT_HEIGHT_MAX = 0.30      # 30cm above surface
DEFAULT_STRIDE = 4             # depth downsample stride (720/4 = 180 rows)
DEFAULT_OBJ_DILATE_PX = 5     # dilate occupied cells (object margin ~10cm)
DEFAULT_SG_MARGIN = 0.08       # scene graph object margin for non-visible areas

# Multi-frame accumulation
ACCUM_WINDOW_SIZE = 5          # sliding window: last N frames
ACCUM_FREE_THRESHOLD = 0.4    # ≥40% of observations say free → free
ACCUM_OCC_THRESHOLD = 0.3     # ≥30% of observations say occupied → occupied

# Surface tags eligible for depth-based analysis
DEPTH_SUITABLE_TAGS = {"desk", "table", "dining table", "coffee table",
                       "side table", "end table", "workbench", "counter",
                       "cabinet", "shelf", "dresser", "nightstand", "bench"}

# Minimum visibility ratio to trust depth data for a surface.
# Below this, depth barely sees the surface → too noisy to use.
MIN_VISIBILITY_RATIO = 0.02    # at least 2% of cells must be observed

# Front-face detection: if camera sees a tall surface (cabinet/shelf) from the side,
# most visible cells will be "occupied" (the front wall) with no "free" surface visible.
# When occupied_ratio > this AND visibility is low → skip (front-face false positive).
FRONT_FACE_OCC_RATIO = 0.95   # >95% of visible cells occupied → likely front face
FRONT_FACE_MAX_VIS = 0.10     # only apply if visibility < 10% (very little surface seen)

# Max rects to export to RViz (prevent JSON bloat)
MAX_RVIZ_RECTS = 50

# Debug
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(_BASE_DIR, "depth_debug")


class DepthFreeSpaceAnalyzer:
    """Depth-based occupancy grid for desk surfaces.

    Hybrid approach:
      - Depth-visible areas: trust depth observation
      - Non-visible areas: fall back to scene graph bounding boxes

    Only processes desk/table surfaces (camera can see their tops).
    Cabinets, shelves, etc. are skipped (camera sees front face → false positives).
    """

    def __init__(self, cell_size=DEFAULT_CELL_SIZE,
                 height_min=DEFAULT_HEIGHT_MIN,
                 height_max=DEFAULT_HEIGHT_MAX,
                 stride=DEFAULT_STRIDE,
                 debug=False):
        self.cell_size = cell_size
        self.height_min = height_min
        self.height_max = height_max
        self.stride = stride
        self.debug = debug

        # Camera intrinsics
        W, H = CAMERA_RESOLUTION
        self.fx = CAMERA_FOCAL * W / CAMERA_H_APERTURE
        self.fy = CAMERA_FOCAL * H / CAMERA_V_APERTURE
        self.ppx = W / 2.0
        self.ppy = H / 2.0

        # Registered surfaces: {surface_id: {...}}
        self.surfaces: Dict[int, dict] = {}

        # Per-frame results: {surface_id: {...}}
        self.results: Dict[int, dict] = {}

        # Multi-frame accumulation: sliding window of recent frames per surface
        # {surface_id: deque of (visibility_mask, occupied_mask, free_mask), maxlen=ACCUM_WINDOW_SIZE}
        self._frame_history: Dict[int, list] = {}

        # Accumulated (merged) grids from sliding window — updated after each process_depth()
        # {surface_id: {"free": ndarray, "occupied": ndarray, "visibility": ndarray,
        #               "nx": int, "ny": int, "x_min": ..., "surface_z": ...}}
        self._accumulated: Dict[int, dict] = {}

        # Persistent cache: survives when surface leaves camera view.
        # Updated from accumulated grids each frame. Used as fallback when no live data.
        # Also updated by mark_cell_occupied() when objects are placed.
        # {surface_id: {"free": ndarray, "occupied": ndarray, "nx": int, "ny": int, ...}}
        self._cached_grids: Dict[int, dict] = {}

        # Precomputed surface Z values from offline depth analysis: {surface_id: float}
        self._precomputed_z: Dict[int, float] = {}

        # Precomputed free rects from offline analysis: {surface_id: list}
        self._precomputed_rects: Dict[int, list] = {}

        # Last placement coordinate for debug overlay (set externally)
        # Format: {"surface_id": int, "x": float, "y": float, "z": float, "tag": str}
        self.last_placement: Optional[dict] = None

        os.makedirs(DEBUG_DIR, exist_ok=True)

    # ================================================================
    # Surface Registration
    # ================================================================

    def register_surface(self, surface_id: int, surface_z: float,
                         x_min: float, x_max: float,
                         y_min: float, y_max: float,
                         tag: str = "desk"):
        """Register a surface for depth analysis."""
        tag_lower = tag.lower().strip()
        suitable = any(st in tag_lower or tag_lower in st
                       for st in DEPTH_SUITABLE_TAGS)
        if not suitable:
            return

        self.surfaces[surface_id] = {
            "z": surface_z,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "tag": tag,
        }

    def register_from_surface_analyzer(self, surface_analyzer):
        """Import desk/table surfaces from existing SurfaceAnalyzer (V2)."""
        self.surfaces.clear()
        for sid, surf in surface_analyzer.surfaces.items():
            self.register_surface(
                sid, surf.surface_z,
                surf.x_min, surf.x_max,
                surf.y_min, surf.y_max,
                surf.tag,
            )
        pass  # registration logging suppressed

    def register_from_scene_graph(self, sg_data: dict,
                                  shrink: float = 0.95,
                                  inset_ratio: float = 0.05,
                                  inset_min: float = 0.03):
        """Register surfaces directly from scene graph dict (react SG or raw_data).

        Uses bbox_center + bbox_extent to compute surface bounds.
        Only registers objects whose tag is in DEPTH_SUITABLE_TAGS.
        """
        self.surfaces.clear()
        for key, info in sg_data.items():
            tag = info.get("object_tag", "").lower().strip()
            suitable = any(st in tag or tag in st for st in DEPTH_SUITABLE_TAGS)
            if not suitable:
                continue
            status = info.get("_status", "present")
            if status == "absent":
                continue
            obj_id = info.get("id")
            if obj_id is None:
                continue
            center = info.get("bbox_center", info.get("center"))
            extent = info.get("bbox_extent")
            if center is None or extent is None:
                continue
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            ew, ed, eh = float(extent[0]), float(extent[1]), float(extent[2])
            # Surface Z: prefer precomputed (depth-detected) over bbox estimate
            bbox_z = cz + eh / 2.0 - 0.02
            surface_z = self._precomputed_z.get(int(obj_id), bbox_z)
            # Shrink extent and apply edge inset
            sw = ew * shrink
            sd = ed * shrink
            inset_x = max(inset_min, sw * inset_ratio)
            inset_y = max(inset_min, sd * inset_ratio)
            x_min = cx - sw / 2.0 + inset_x
            x_max = cx + sw / 2.0 - inset_x
            y_min = cy - sd / 2.0 + inset_y
            y_max = cy + sd / 2.0 - inset_y
            if x_max - x_min < 0.1 or y_max - y_min < 0.1:
                continue  # too small
            self.surfaces[int(obj_id)] = {
                "z": surface_z,
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "tag": tag,
            }
        pass  # SG registration logging suppressed

    def load_precomputed(self, json_path: str = None):
        """Load precomputed freespace from offline depth analysis.

        Extracts:
          - Corrected surface_z values (depth-detected, not bbox-based)
          - Initial free space rectangles (used until live depth overwrites)
        """
        import json
        path = json_path or PRECOMPUTED_FREESPACE_FILE
        if not os.path.exists(path):
            pass  # no precomputed file — silent
            return False

        with open(path) as f:
            data = json.load(f)

        for surf in data.get("surfaces", []):
            sid = surf["surface_id"]
            self._precomputed_z[sid] = surf["surface_z"]
            self._precomputed_rects[sid] = surf.get("free_spaces", [])

        pass  # precomputed loading logging suppressed

        return True

    # ================================================================
    # Main Pipeline
    # ================================================================

    def process_depth(self, depth_np: np.ndarray,
                      robot_pos: np.ndarray,
                      robot_quat: np.ndarray) -> Dict[int, dict]:
        """Process one depth frame → accumulate into sliding window per surface.

        Multi-frame accumulation: each frame's per-cell observations (free/occupied/visible)
        are pushed into a sliding window. The accumulated grid merges recent N frames
        by voting: a cell is free if ≥40% of observations say free, occupied if ≥30% say occupied.

        Returns: {surface_id: {grid, visibility, occupied, free, stats...}}
                 (accumulated results, not single-frame)
        """
        if depth_np is None or depth_np.ndim < 2:
            return {}
        if len(self.surfaces) == 0:
            return {}

        t0 = time.time()

        # Step 1: Depth → World Point Cloud
        pts_world = self._depth_to_world(depth_np, robot_pos, robot_quat)
        if pts_world is None or len(pts_world) == 0:
            return {}

        # Step 2-6: Per-surface single-frame occupancy
        self.results.clear()
        for sid, surf in self.surfaces.items():
            result = self._compute_surface_occupancy(pts_world, surf)
            if result is None:
                continue

            vis_ratio = result["visibility_ratio"]
            n_vis = result["n_visible_cells"]

            # Flag whether this surface is usable for placement
            result["_usable"] = True

            # Skip surfaces with too little visibility
            if vis_ratio < MIN_VISIBILITY_RATIO:
                result["_usable"] = False
            # Front-face detection
            elif n_vis > 0:
                occ_ratio = result["n_occupied_cells"] / n_vis
                if (occ_ratio > FRONT_FACE_OCC_RATIO and
                        vis_ratio < FRONT_FACE_MAX_VIS):
                    result["_usable"] = False

            if not result["_usable"]:
                continue

            # Accumulate this frame into sliding window
            self._accumulate_frame(sid, result)
            self.results[sid] = result

        # Build accumulated grids from sliding windows
        self._rebuild_accumulated()

        return self.results

    def _accumulate_frame(self, sid: int, result: dict):
        """Push single-frame observation into sliding window for surface."""
        nx, ny = result["nx"], result["ny"]

        # Ensure frame history exists with correct dimensions
        if sid in self._frame_history:
            existing = self._frame_history[sid]
            if existing and existing[0][0].shape != (ny, nx):
                # Grid dimensions changed (surface re-registered) — reset
                self._frame_history[sid] = []

        if sid not in self._frame_history:
            self._frame_history[sid] = []

        # Store per-cell observation masks (compact: uint8)
        vis = result["visibility"].copy()
        occ = result["occupied"].copy()
        free = result["free"].copy()

        history = self._frame_history[sid]
        history.append((vis, occ, free))

        # Sliding window: keep only last N frames
        if len(history) > ACCUM_WINDOW_SIZE:
            history.pop(0)

    def _rebuild_accumulated(self):
        """Merge sliding window frames into accumulated grids per surface.

        Voting logic per cell:
          - visibility = fraction of frames that observed this cell
          - If ≥ACCUM_OCC_THRESHOLD of observations say occupied → occupied
          - Elif ≥ACCUM_FREE_THRESHOLD of observations say free → free
          - Else → unknown

        Also updates persistent cache for surfaces with live data.
        """
        for sid, history in self._frame_history.items():
            if not history:
                continue
            surf = self.surfaces.get(sid)
            if surf is None:
                continue

            n_frames = len(history)
            ny, nx = history[0][0].shape

            # Stack frames: (n_frames, ny, nx)
            vis_stack = np.stack([h[0] for h in history], axis=0)
            occ_stack = np.stack([h[1] for h in history], axis=0)
            free_stack = np.stack([h[2] for h in history], axis=0)

            # Count observations per cell
            vis_count = vis_stack.sum(axis=0).astype(np.float32)
            occ_count = occ_stack.sum(axis=0).astype(np.float32)
            free_count = free_stack.sum(axis=0).astype(np.float32)

            # Voting
            total_obs = np.maximum(vis_count, 1.0)  # avoid div by zero
            occ_ratio = occ_count / total_obs
            free_ratio = free_count / total_obs

            acc_visibility = (vis_count > 0).astype(np.uint8)
            acc_occupied = (occ_ratio >= ACCUM_OCC_THRESHOLD).astype(np.uint8)
            acc_free = ((free_ratio >= ACCUM_FREE_THRESHOLD) &
                        (acc_occupied == 0)).astype(np.uint8)

            n_visible = int(acc_visibility.sum())
            n_occupied = int(acc_occupied.sum())
            n_free = int(acc_free.sum())
            n_total = nx * ny

            acc_result = {
                "free": acc_free,
                "occupied": acc_occupied,
                "visibility": acc_visibility,
                "nx": nx, "ny": ny,
                "x_min": surf["x_min"], "x_max": surf["x_max"],
                "y_min": surf["y_min"], "y_max": surf["y_max"],
                "surface_z": surf["z"],
                "n_total_cells": n_total,
                "n_visible_cells": n_visible,
                "n_occupied_cells": n_occupied,
                "n_free_cells": n_free,
                "visibility_ratio": round(n_visible / n_total, 3) if n_total > 0 else 0,
                "n_frames": n_frames,
            }
            self._accumulated[sid] = acc_result

            # Update persistent cache: copy accumulated grids
            self._cached_grids[sid] = {
                "free": acc_free.copy(),
                "occupied": acc_occupied.copy(),
                "visibility": acc_visibility.copy(),
                "nx": nx, "ny": ny,
                "x_min": surf["x_min"], "x_max": surf["x_max"],
                "y_min": surf["y_min"], "y_max": surf["y_max"],
                "surface_z": surf["z"],
                "n_frames": n_frames,
            }

    # ================================================================
    # Step 1: Depth → World Points
    # ================================================================

    def _depth_to_world(self, depth_np, robot_pos, robot_quat):
        """Back-project depth map to world coordinates (downsampled)."""
        H, W = depth_np.shape[:2]
        stride = self.stride

        # Pixel grid (downsampled)
        ys = np.arange(0, H, stride)
        xs = np.arange(0, W, stride)
        grid_x, grid_y = np.meshgrid(xs, ys)
        px = grid_x.ravel().astype(np.float32)
        py = grid_y.ravel().astype(np.float32)

        # Depths at grid points
        depths = depth_np[grid_y.ravel(), grid_x.ravel()]

        # Valid depth filter
        valid = (depths > 0.1) & (depths < 10.0)
        if valid.sum() < 10:
            return None

        px, py, d = px[valid], py[valid], depths[valid].astype(np.float32)

        # Back-project to camera frame
        x_cam = (px - self.ppx) * d / self.fx
        y_cam = (py - self.ppy) * d / self.fy
        z_cam = d

        # Camera → Robot frame (tilt compensation)
        tilt_rad = np.radians(CAMERA_TILT_DEG)
        cos_t, sin_t = np.cos(tilt_rad), np.sin(tilt_rad)
        fwd = z_cam * cos_t - y_cam * sin_t
        up = z_cam * sin_t + y_cam * cos_t
        pts_robot = np.stack([fwd, -x_cam, -up + CAMERA_HEIGHT], axis=1)

        # Robot → World frame (quaternion rotation)
        w_q, qx, qy, qz = robot_quat
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w_q*qz), 2*(qx*qz + w_q*qy)],
            [2*(qx*qy + w_q*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w_q*qx)],
            [2*(qx*qz - w_q*qy), 2*(qy*qz + w_q*qx), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float32)

        pts_world = (R @ pts_robot.T).T + np.array(robot_pos, dtype=np.float32)
        return pts_world

    # ================================================================
    # Steps 2-6: Per-surface Occupancy Grid
    # ================================================================

    def _compute_surface_occupancy(self, pts_world, surf):
        """For one desk/table surface: ROI filter → occupancy grid → morphology."""
        sz = surf["z"]
        x_min, x_max = surf["x_min"], surf["x_max"]
        y_min, y_max = surf["y_min"], surf["y_max"]

        # Step 2: ROI filter — XY STRICTLY within surface bounds, Z in height band
        # No margin expansion — free space must stay within actual surface edges
        mask_xy = ((pts_world[:, 0] >= x_min) &
                   (pts_world[:, 0] <= x_max) &
                   (pts_world[:, 1] >= y_min) &
                   (pts_world[:, 1] <= y_max))

        # Z band: surface level (±10cm) through height_max above surface
        z_surface_low = sz - 0.10
        z_surface_high = sz + self.height_max
        mask_z = ((pts_world[:, 2] >= z_surface_low) &
                  (pts_world[:, 2] <= z_surface_high))

        roi_pts = pts_world[mask_xy & mask_z]
        if len(roi_pts) < 5:
            return None

        # Separate: surface points (at desk level) vs object points (above desk)
        z_obj_threshold = sz + self.height_min
        obj_pts = roi_pts[roi_pts[:, 2] >= z_obj_threshold]
        surf_pts = roi_pts[roi_pts[:, 2] < z_obj_threshold]

        # Step 3: Create grid
        w = x_max - x_min
        d = y_max - y_min
        if w <= 0 or d <= 0:
            return None
        nx = max(1, int(round(w / self.cell_size)))
        ny = max(1, int(round(d / self.cell_size)))

        # 0=unknown, 1=free(visible surface), 2=occupied(object above surface)
        grid = np.zeros((ny, nx), dtype=np.uint8)

        # Mark visible surface cells → free
        if len(surf_pts) > 0:
            gx = np.clip(((surf_pts[:, 0] - x_min) / self.cell_size).astype(int), 0, nx - 1)
            gy = np.clip(((surf_pts[:, 1] - y_min) / self.cell_size).astype(int), 0, ny - 1)
            grid[gy, gx] = 1

        # Mark object cells → occupied
        if len(obj_pts) > 0:
            gx = np.clip(((obj_pts[:, 0] - x_min) / self.cell_size).astype(int), 0, nx - 1)
            gy = np.clip(((obj_pts[:, 1] - y_min) / self.cell_size).astype(int), 0, ny - 1)
            grid[gy, gx] = 2

        # Step 4: Visibility mask (any depth observation)
        visibility = (grid > 0).astype(np.uint8)

        # Step 5: Morphological post-processing
        occupied = (grid == 2).astype(np.uint8)
        if occupied.any():
            # Dilate objects to add safety margin
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (DEFAULT_OBJ_DILATE_PX, DEFAULT_OBJ_DILATE_PX))
            occupied = cv2.dilate(occupied, kernel, iterations=1)

        free = (grid == 1).astype(np.uint8)
        if free.any():
            # Close small gaps in free space
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel_close)

        # Step 6: Final free = visible-free AND not-occupied
        final_free = ((free > 0) & (occupied == 0)).astype(np.uint8)

        n_total = nx * ny
        n_visible = int(visibility.sum())
        n_occupied = int((occupied > 0).sum())
        n_free = int(final_free.sum())

        return {
            "grid": grid,
            "visibility": visibility,
            "occupied": occupied,
            "free": final_free,
            "nx": nx, "ny": ny,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "surface_z": sz,
            "n_total_cells": n_total,
            "n_visible_cells": n_visible,
            "n_occupied_cells": n_occupied,
            "n_free_cells": n_free,
            "visibility_ratio": round(n_visible / n_total, 3) if n_total > 0 else 0,
        }

    # ================================================================
    # Hybrid Free Space (depth + scene graph fallback)
    # ================================================================

    def get_hybrid_free_rects(self, surface_id: int,
                              sg_objects_on: Optional[List[dict]] = None):
        """Combine accumulated depth grids + persistent cache + precomputed + SG fallback.

        Priority per cell:
          1. Accumulated (live depth, multi-frame voted) — highest trust
          2. Cached (last-known from when robot could see) — second
          3. Precomputed (offline analysis) — fallback
          4. SG objects — mark occupied in non-visible areas

        Returns list of FreeRect (compatible with surface_analyzer_v2).
        """
        surf = self.surfaces.get(surface_id)
        if surf is None:
            return self._get_precomputed_rects(surface_id)

        # Try accumulated (live multi-frame) first
        acc = self._accumulated.get(surface_id)
        # Fall back to cached grid (persistent from last observation)
        cached = self._cached_grids.get(surface_id)

        source = acc or cached
        if source is None:
            return self._get_precomputed_rects(surface_id)

        nx, ny = source["nx"], source["ny"]
        x_min = source["x_min"]
        y_min = source["y_min"]
        sz = source["surface_z"]
        cs = self.cell_size

        free_grid = source["free"].copy()
        occupied_grid = source["occupied"].copy()
        visibility = source.get("visibility", np.ones((ny, nx), dtype=np.uint8))

        # For non-visible cells: fill from cached grid (if using accumulated as source)
        if acc is not None and cached is not None and cached is not acc:
            if cached["nx"] == nx and cached["ny"] == ny:
                non_vis = (visibility == 0)
                cached_vis = cached.get("visibility", np.zeros((ny, nx), dtype=np.uint8))
                # Where accumulated doesn't see but cache has data → use cache
                cache_has_data = (cached_vis > 0) & non_vis
                free_grid[cache_has_data & (cached["free"] > 0)] = 1
                occupied_grid[cache_has_data & (cached["occupied"] > 0)] = 1
                # Don't mark cache-free cells as free if they're also cache-occupied
                conflict = cache_has_data & (cached["free"] > 0) & (cached["occupied"] > 0)
                free_grid[conflict] = 0

        # For still-non-visible cells: apply SG objects
        if sg_objects_on:
            sg_occupied = np.zeros((ny, nx), dtype=np.uint8)
            for o in sg_objects_on:
                m = DEFAULT_SG_MARGIN
                ox_min = o["cx"] - o["w"] / 2.0 - m
                ox_max = o["cx"] + o["w"] / 2.0 + m
                oy_min = o["cy"] - o["d"] / 2.0 - m
                oy_max = o["cy"] + o["d"] / 2.0 + m

                gi_x0 = max(0, int((ox_min - x_min) / cs))
                gi_x1 = min(nx, int(np.ceil((ox_max - x_min) / cs)))
                gi_y0 = max(0, int((oy_min - y_min) / cs))
                gi_y1 = min(ny, int(np.ceil((oy_max - y_min) / cs)))

                sg_occupied[gi_y0:gi_y1, gi_x0:gi_x1] = 1

            # Non-visible cells with SG object → occupied
            non_visible = (visibility == 0)
            occupied_grid[non_visible & (sg_occupied > 0)] = 1
            free_grid[non_visible & (sg_occupied > 0)] = 0

        # Final: free must not be occupied
        free_grid = ((free_grid > 0) & (occupied_grid == 0)).astype(np.uint8)

        # Row-run merge into rectangles
        rects = self._merge_to_rects(free_grid, nx, ny, x_min, y_min, sz, cs)
        return rects

    def get_hybrid_stats(self, surface_id: int,
                         sg_objects_on: Optional[List[dict]] = None) -> Optional[dict]:
        """Get summary statistics for hybrid free space."""
        result = self._accumulated.get(surface_id) or self._cached_grids.get(surface_id)
        surf = self.surfaces.get(surface_id)
        if result is None or surf is None:
            return None

        rects = self.get_hybrid_free_rects(surface_id, sg_objects_on)
        total_free_area = sum(r.area for r in rects)
        max_w = max((r.width for r in rects), default=0)
        max_d = max((r.depth for r in rects), default=0)

        return {
            "surface_id": surface_id,
            "surface_tag": surf["tag"],
            "visibility_ratio": result["visibility_ratio"],
            "n_depth_visible": result["n_visible_cells"],
            "n_depth_occupied": result["n_occupied_cells"],
            "n_depth_free": result["n_free_cells"],
            "hybrid_free_area_m2": round(total_free_area, 4),
            "hybrid_n_rects": len(rects),
            "hybrid_max_rect_size": [round(max_w, 3), round(max_d, 3)],
        }

    def _get_precomputed_rects(self, surface_id: int):
        """Convert precomputed free_spaces to FreeRect objects."""
        from surface_analyzer_v2 import FreeRect
        rects = []
        for fs in self._precomputed_rects.get(surface_id, []):
            bounds = fs.get("bounds", {})
            r = FreeRect(
                x_min=bounds.get("x_min", 0),
                y_min=bounds.get("y_min", 0),
                x_max=bounds.get("x_max", 0),
                y_max=bounds.get("y_max", 0),
                direction="precomputed",
                surface_z=fs.get("center", [0, 0, 0])[2] if "center" in fs else 0,
            )
            if r.is_valid:
                rects.append(r)
        return rects

    # ================================================================
    # Cache mutation (pick/place events)
    # ================================================================

    def mark_cells_occupied(self, surface_id: int, world_x: float, world_y: float,
                            radius_m: float = 0.15):
        """Mark cells around (world_x, world_y) as occupied in persistent cache.

        Called when an object is placed on a surface — the cached grid must
        immediately reflect the placement even before the next depth frame.
        """
        cached = self._cached_grids.get(surface_id)
        if cached is None:
            return
        cs = self.cell_size
        nx, ny = cached["nx"], cached["ny"]
        x_min, y_min = cached["x_min"], cached["y_min"]

        r_cells = max(1, int(radius_m / cs))
        cx = int((world_x - x_min) / cs)
        cy = int((world_y - y_min) / cs)
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < nx and 0 <= gy < ny:
                    if dx * dx + dy * dy <= r_cells * r_cells:
                        cached["occupied"][gy, gx] = 1
                        cached["free"][gy, gx] = 0

        # Also update accumulated if present
        acc = self._accumulated.get(surface_id)
        if acc is not None and acc["nx"] == nx and acc["ny"] == ny:
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < nx and 0 <= gy < ny:
                        if dx * dx + dy * dy <= r_cells * r_cells:
                            acc["occupied"][gy, gx] = 1
                            acc["free"][gy, gx] = 0

    def mark_cells_free(self, surface_id: int, world_x: float, world_y: float,
                        radius_m: float = 0.15):
        """Mark cells around (world_x, world_y) as free in persistent cache.

        Called when an object is picked from a surface.
        """
        cached = self._cached_grids.get(surface_id)
        if cached is None:
            return
        cs = self.cell_size
        nx, ny = cached["nx"], cached["ny"]
        x_min, y_min = cached["x_min"], cached["y_min"]

        r_cells = max(1, int(radius_m / cs))
        cx = int((world_x - x_min) / cs)
        cy = int((world_y - y_min) / cs)
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < nx and 0 <= gy < ny:
                    if dx * dx + dy * dy <= r_cells * r_cells:
                        cached["free"][gy, gx] = 1
                        cached["occupied"][gy, gx] = 0
                        cached["visibility"][gy, gx] = 1

        acc = self._accumulated.get(surface_id)
        if acc is not None and acc["nx"] == nx and acc["ny"] == ny:
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < nx and 0 <= gy < ny:
                        if dx * dx + dy * dy <= r_cells * r_cells:
                            acc["free"][gy, gx] = 1
                            acc["occupied"][gy, gx] = 0

    def get_best_grid(self, surface_id: int) -> Optional[dict]:
        """Get the best available grid for a surface (accumulated > cached > None).

        Used by RViz export for per-cell visualization.
        """
        acc = self._accumulated.get(surface_id)
        if acc is not None:
            return acc
        cached = self._cached_grids.get(surface_id)
        if cached is not None:
            return cached
        return None

    def seed_cache_from_precomputed(self, surface_id: int):
        """Initialize persistent cache from precomputed data for a surface.

        Called at startup so surfaces have grid data before any live frames.
        """
        surf = self.surfaces.get(surface_id)
        rects_list = self._precomputed_rects.get(surface_id, [])
        if surf is None or not rects_list:
            return

        cs = self.cell_size
        x_min, x_max = surf["x_min"], surf["x_max"]
        y_min, y_max = surf["y_min"], surf["y_max"]
        nx = max(1, int(round((x_max - x_min) / cs)))
        ny = max(1, int(round((y_max - y_min) / cs)))

        free_grid = np.zeros((ny, nx), dtype=np.uint8)
        vis_grid = np.zeros((ny, nx), dtype=np.uint8)

        for fs in rects_list:
            bounds = fs.get("bounds", {})
            gx0 = max(0, int((bounds.get("x_min", 0) - x_min) / cs))
            gx1 = min(nx, int(np.ceil((bounds.get("x_max", 0) - x_min) / cs)))
            gy0 = max(0, int((bounds.get("y_min", 0) - y_min) / cs))
            gy1 = min(ny, int(np.ceil((bounds.get("y_max", 0) - y_min) / cs)))
            free_grid[gy0:gy1, gx0:gx1] = 1
            vis_grid[gy0:gy1, gx0:gx1] = 1

        self._cached_grids[surface_id] = {
            "free": free_grid,
            "occupied": np.zeros((ny, nx), dtype=np.uint8),
            "visibility": vis_grid,
            "nx": nx, "ny": ny,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "surface_z": surf["z"],
            "n_frames": 0,
            "source": "precomputed",
        }

    # ================================================================
    # Runtime surface Z detection
    # ================================================================

    def detect_surface_z_from_depth(self, surface_id: int, depth_np: np.ndarray,
                                     robot_pos: np.ndarray, robot_quat: np.ndarray) -> Optional[float]:
        """Detect true surface Z from depth using histogram peak detection.

        Same algorithm as precompute_freespace.py but applied to a single live frame.
        Collects Z values of points within surface XY bounds, builds 1cm histogram,
        finds the peak → true surface height.

        Call this during observation of new surfaces to correct bbox-based Z.
        """
        surf = self.surfaces.get(surface_id)
        if surf is None:
            return None

        pts_world = self._depth_to_world(depth_np, robot_pos, robot_quat)
        if pts_world is None or len(pts_world) == 0:
            return None

        x_min, x_max = surf["x_min"], surf["x_max"]
        y_min, y_max = surf["y_min"], surf["y_max"]
        bbox_z = surf["z"]

        # Filter to XY bounds
        mask_xy = ((pts_world[:, 0] >= x_min) & (pts_world[:, 0] <= x_max) &
                   (pts_world[:, 1] >= y_min) & (pts_world[:, 1] <= y_max))
        xy_pts = pts_world[mask_xy]
        if len(xy_pts) < 50:
            return None

        # Z values within a generous range around bbox estimate
        z_vals = xy_pts[:, 2]
        z_range_mask = (z_vals >= bbox_z - 0.4) & (z_vals <= bbox_z + 0.4)
        z_samples = z_vals[z_range_mask]
        if len(z_samples) < 30:
            return None

        # 1cm histogram
        bin_size = 0.01
        bins = np.arange(z_samples.min() - bin_size, z_samples.max() + 2 * bin_size, bin_size)
        if len(bins) < 3:
            return None
        hist, edges = np.histogram(z_samples, bins=bins)
        if hist.max() == 0:
            return None

        # Peak detection with weighted average of ±1 bin
        peak_idx = int(np.argmax(hist))
        lo = max(0, peak_idx - 1)
        hi = min(len(hist), peak_idx + 2)
        weights = hist[lo:hi].astype(np.float64)
        centers = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(lo, hi)])
        if weights.sum() > 0:
            refined_z = float(np.average(centers, weights=weights))
        else:
            refined_z = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)

        return refined_z

    # Z samples collected during multi-frame observation of new surfaces
    # {surface_id: list of detected_z values}
    _observation_z_samples: Dict[int, list] = {}

    def collect_z_sample(self, surface_id: int, depth_np: np.ndarray,
                         robot_pos: np.ndarray, robot_quat: np.ndarray):
        """Collect one Z sample during observation. Call once per frame."""
        z = self.detect_surface_z_from_depth(surface_id, depth_np, robot_pos, robot_quat)
        if z is not None:
            if surface_id not in self._observation_z_samples:
                self._observation_z_samples[surface_id] = []
            self._observation_z_samples[surface_id].append(z)

    def finalize_observed_z(self, surface_id: int) -> Optional[float]:
        """After observation, compute median Z from collected samples and update surface."""
        samples = self._observation_z_samples.pop(surface_id, [])
        if not samples:
            return None
        refined_z = float(np.median(samples))

        # Update registered surface
        if surface_id in self.surfaces:
            self.surfaces[surface_id]["z"] = refined_z

        # Persist into _precomputed_z so register_from_scene_graph picks it up
        # on subsequent re-registrations (every 10 frames)
        self._precomputed_z[surface_id] = refined_z

        # Clear any stale frame history (grid dims might change slightly with new Z)
        self._frame_history.pop(surface_id, None)
        self._accumulated.pop(surface_id, None)
        self._cached_grids.pop(surface_id, None)

        return refined_z

    # ================================================================
    # Grid → FreeRect merge
    # ================================================================

    @staticmethod
    def _merge_to_rects(free_grid, nx, ny, x_min, y_min, sz, cs):
        """Connected-component based merge of free cells into bounding-box rectangles.

        Uses cv2.connectedComponents for robust merging — handles any shape,
        not just exact-width row runs. Each connected free region becomes one
        bounding-box FreeRect.
        """
        from surface_analyzer_v2 import FreeRect

        if not free_grid.any():
            return []

        # Connected components on binary free grid
        n_labels, labels = cv2.connectedComponents(free_grid, connectivity=4)

        result = []
        for label_id in range(1, n_labels):  # skip background (0)
            ys, xs = np.where(labels == label_id)
            if len(xs) == 0:
                continue

            gx0, gx1 = int(xs.min()), int(xs.max()) + 1
            gy0, gy1 = int(ys.min()), int(ys.max()) + 1

            r = FreeRect(
                x_min=x_min + gx0 * cs,
                y_min=y_min + gy0 * cs,
                x_max=x_min + gx1 * cs,
                y_max=y_min + gy1 * cs,
                direction="depth_tile",
                surface_z=sz,
            )
            if r.is_valid:
                result.append(r)

        return result

    # ================================================================
    # Debug Visualization
    # ================================================================

    def save_debug_image(self, surface_id: int, tag: str = ""):
        """Save a clear, readable top-view occupancy grid for debugging.

        Uses accumulated grid (multi-frame) if available, else single-frame result.

        Image layout (600px wide):
        - Top: title with surface info
        - Main: color-coded grid (top-down view matching RViz orientation)
          GREEN = free, RED = occupied, DARK GRAY = not visible, MID GRAY = visible but empty
        - Bottom: legend + world coordinate labels
        - Also saves a 'latest' file that's always overwritten.
        """
        # Use accumulated grid if available, else single-frame
        result = self._accumulated.get(surface_id) or self.results.get(surface_id)
        surf = self.surfaces.get(surface_id)
        if result is None or surf is None:
            return

        nx, ny = result["nx"], result["ny"]
        free = result["free"]
        occupied = result["occupied"]
        visibility = result["visibility"]
        x_min, y_min = result["x_min"], result["y_min"]
        x_max, y_max = result["x_max"], result["y_max"]

        # Color the grid cells
        img = np.zeros((ny, nx, 3), dtype=np.uint8)
        img[visibility == 0] = [50, 50, 50]          # dark gray: not visible
        img[(visibility > 0) & (free == 0) & (occupied == 0)] = [100, 100, 100]  # visible, empty
        img[free > 0] = [80, 220, 80]                 # green: free
        img[occupied > 0] = [80, 80, 220]             # red: occupied

        # Scale to at least 400px on longest side
        target_size = 400
        scale = max(2, target_size // max(nx, ny))
        grid_w, grid_h = nx * scale, ny * scale
        img_big = cv2.resize(img, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

        # Draw grid lines every 10 cells (= 20cm)
        for gx in range(0, nx, 10):
            px = gx * scale
            cv2.line(img_big, (px, 0), (px, grid_h), (70, 70, 70), 1)
        for gy in range(0, ny, 10):
            py = gy * scale
            cv2.line(img_big, (0, py), (grid_w, py), (70, 70, 70), 1)

        # Build full image with info panel
        panel_h = 80
        full_h = grid_h + panel_h
        full_img = np.zeros((full_h, max(grid_w, 400), 3), dtype=np.uint8)
        full_img[panel_h:panel_h + grid_h, :grid_w] = img_big

        # Title
        surface_w = x_max - x_min
        surface_d = y_max - y_min
        title = (f"{surf.get('tag', '?')}(id:{surface_id}) "
                 f"  {surface_w:.2f}m x {surface_d:.2f}m  "
                 f"  z={result['surface_z']:.2f}m")
        cv2.putText(full_img, title, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Stats
        vis_pct = result['visibility_ratio'] * 100
        free_area = result['n_free_cells'] * self.cell_size * self.cell_size
        stats = (f"vis:{vis_pct:.0f}% "
                 f"({result['n_visible_cells']}/{result['n_total_cells']})  "
                 f"occ:{result['n_occupied_cells']} cells  "
                 f"free:{result['n_free_cells']} cells ({free_area:.3f}m2)")
        cv2.putText(full_img, stats, (5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Legend
        cv2.rectangle(full_img, (5, 48), (15, 58), (80, 220, 80), -1)
        cv2.putText(full_img, "FREE", (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 220, 80), 1)
        cv2.rectangle(full_img, (70, 48), (80, 58), (80, 80, 220), -1)
        cv2.putText(full_img, "OCCUPIED", (85, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 220), 1)
        cv2.rectangle(full_img, (175, 48), (185, 58), (50, 50, 50), -1)
        cv2.putText(full_img, "NOT VISIBLE", (190, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

        # Coordinate labels on grid edges
        cv2.putText(full_img, f"x:{x_min:.2f}", (2, panel_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(full_img, f"x:{x_max:.2f}", (grid_w - 55, panel_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(full_img, f"y:{y_min:.2f}", (2, panel_h + grid_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        # Grid = 20cm annotation
        cv2.putText(full_img, "grid=20cm", (5, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

        suffix = f"_{tag}" if tag else ""
        path = os.path.join(DEBUG_DIR, f"occupancy_s{surface_id}{suffix}.png")
        cv2.imwrite(path, full_img)
        # Also save as 'latest' for easy monitoring
        latest = os.path.join(DEBUG_DIR, f"occupancy_s{surface_id}_latest.png")
        cv2.imwrite(latest, full_img)
        return path

    def save_hybrid_debug_image(self, surface_id: int,
                                sg_objects_on: Optional[List[dict]] = None,
                                tag: str = ""):
        """Save hybrid grid with SG object outlines drawn on top."""
        result = self._accumulated.get(surface_id) or self.results.get(surface_id)
        surf = self.surfaces.get(surface_id)
        if result is None or surf is None:
            return

        nx, ny = result["nx"], result["ny"]
        visibility = result["visibility"]
        cs = self.cell_size
        x_min, y_min = result["x_min"], result["y_min"]
        x_max, y_max = result["x_max"], result["y_max"]

        # Recompute hybrid grid (same logic as get_hybrid_free_rects)
        free_grid = result["free"].copy()
        occupied_grid = result["occupied"].copy()

        if sg_objects_on:
            sg_occ = np.zeros((ny, nx), dtype=np.uint8)
            for o in sg_objects_on:
                m = DEFAULT_SG_MARGIN
                gi_x0 = max(0, int((o["cx"] - o["w"]/2 - m - x_min) / cs))
                gi_x1 = min(nx, int(np.ceil((o["cx"] + o["w"]/2 + m - x_min) / cs)))
                gi_y0 = max(0, int((o["cy"] - o["d"]/2 - m - y_min) / cs))
                gi_y1 = min(ny, int(np.ceil((o["cy"] + o["d"]/2 + m - y_min) / cs)))
                sg_occ[gi_y0:gi_y1, gi_x0:gi_x1] = 1

            non_vis = (visibility == 0)
            occupied_grid[non_vis & (sg_occ > 0)] = 1
            free_grid[non_vis & (sg_occ > 0)] = 0

        # Color image
        img = np.zeros((ny, nx, 3), dtype=np.uint8)
        img[visibility == 0] = [50, 50, 50]                              # not visible
        img[(visibility > 0) & (free_grid == 0) & (occupied_grid == 0)] = [100, 100, 100]
        img[free_grid > 0] = [80, 220, 80]                                # free: green
        img[(occupied_grid > 0) & (visibility > 0)] = [80, 80, 220]       # depth-occ: red
        img[(occupied_grid > 0) & (visibility == 0)] = [60, 60, 150]      # sg-occ: dim red

        # Scale up
        target_size = 400
        scale = max(2, target_size // max(nx, ny))
        grid_w, grid_h = nx * scale, ny * scale
        img_big = cv2.resize(img, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

        # Draw SG object outlines (yellow rectangles with labels)
        if sg_objects_on:
            for o in sg_objects_on:
                m = DEFAULT_SG_MARGIN
                px0 = int((o["cx"] - o["w"]/2 - m - x_min) / cs) * scale
                px1 = int(np.ceil((o["cx"] + o["w"]/2 + m - x_min) / cs)) * scale
                py0 = int((o["cy"] - o["d"]/2 - m - y_min) / cs) * scale
                py1 = int(np.ceil((o["cy"] + o["d"]/2 + m - y_min) / cs)) * scale
                px0 = max(0, min(px0, grid_w))
                px1 = max(0, min(px1, grid_w))
                py0 = max(0, min(py0, grid_h))
                py1 = max(0, min(py1, grid_h))
                cv2.rectangle(img_big, (px0, py0), (px1, py1), (0, 220, 220), 2)
                label = o.get("tag", "?")
                cv2.putText(img_big, label, (px0 + 2, py0 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 220, 220), 1)

        # Grid lines every 10 cells
        for gx in range(0, nx, 10):
            px = gx * scale
            cv2.line(img_big, (px, 0), (px, grid_h), (70, 70, 70), 1)
        for gy in range(0, ny, 10):
            py = gy * scale
            cv2.line(img_big, (0, py), (grid_w, py), (70, 70, 70), 1)

        # Build full image with info panel
        panel_h = 80
        full_h = grid_h + panel_h
        full_img = np.zeros((full_h, max(grid_w, 400), 3), dtype=np.uint8)
        full_img[panel_h:panel_h + grid_h, :grid_w] = img_big

        surface_w = x_max - x_min
        surface_d = y_max - y_min
        n_free = int((free_grid > 0).sum())
        n_occ = int((occupied_grid > 0).sum())
        free_area = n_free * cs * cs
        vis_pct = result['visibility_ratio'] * 100

        title = (f"HYBRID {surf['tag']}(id:{surface_id}) "
                 f"  {surface_w:.2f}m x {surface_d:.2f}m")
        cv2.putText(full_img, title, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        stats = (f"vis:{vis_pct:.0f}%  "
                 f"free:{n_free}cells({free_area:.3f}m2)  "
                 f"occ:{n_occ}cells  "
                 f"sg_objs:{len(sg_objects_on) if sg_objects_on else 0}")
        cv2.putText(full_img, stats, (5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Legend
        cv2.rectangle(full_img, (5, 48), (15, 58), (80, 220, 80), -1)
        cv2.putText(full_img, "FREE", (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 220, 80), 1)
        cv2.rectangle(full_img, (70, 48), (80, 58), (80, 80, 220), -1)
        cv2.putText(full_img, "DEPTH-OCC", (85, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 220), 1)
        cv2.rectangle(full_img, (185, 48), (195, 58), (60, 60, 150), -1)
        cv2.putText(full_img, "SG-OCC", (200, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 150), 1)
        cv2.rectangle(full_img, (265, 48), (275, 58), (0, 220, 220), -1)
        cv2.putText(full_img, "SG-OUTLINE", (280, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1)

        # Coordinate labels
        cv2.putText(full_img, f"x:{x_min:.2f}", (2, panel_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(full_img, f"x:{x_max:.2f}", (grid_w - 55, panel_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(full_img, f"y:{y_min:.2f}", (2, panel_h + grid_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        suffix = f"_{tag}" if tag else ""
        path = os.path.join(DEBUG_DIR, f"hybrid_s{surface_id}{suffix}.png")
        cv2.imwrite(path, full_img)
        latest = os.path.join(DEBUG_DIR, f"hybrid_s{surface_id}_latest.png")
        cv2.imwrite(latest, full_img)
        return path

    def save_camera_overlay(self, rgb_np: np.ndarray,
                            robot_pos: np.ndarray, robot_quat: np.ndarray,
                            sg_objects_on_map: Optional[Dict[int, List[dict]]] = None,
                            tag: str = ""):
        """Save camera RGB image with free/occupied grid cells projected back as overlay.

        This is the most intuitive debug view: you see exactly what the robot
        camera sees, with colored overlays showing what the system detected.

        Colors:
          GREEN semi-transparent = free surface (can place objects)
          RED semi-transparent = occupied (object detected by depth)
          YELLOW outline = SG object footprint
          WHITE outline = surface boundary

        Args:
            rgb_np: Camera RGB image (H, W, 3+), uint8
            robot_pos: Robot position [x, y, z]
            robot_quat: Robot quaternion [w, x, y, z]
            sg_objects_on_map: {surface_id: [sg_objects]} for SG outlines
            tag: suffix for filename
        """
        if rgb_np is None or (len(self.results) == 0 and len(self._accumulated) == 0):
            return

        # Convert to BGR for cv2
        bgr = rgb_np[:, :, :3].astype(np.uint8).copy()
        if bgr.shape[2] == 3:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        img_h, img_w = bgr.shape[:2]

        # Build world→pixel projection
        w_q, qx, qy, qz = robot_quat
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w_q*qz), 2*(qx*qz + w_q*qy)],
            [2*(qx*qy + w_q*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w_q*qx)],
            [2*(qx*qz - w_q*qy), 2*(qy*qz + w_q*qx), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float64)
        R_inv = R.T  # world → robot

        tilt_rad = np.radians(CAMERA_TILT_DEG)
        cos_t, sin_t = np.cos(tilt_rad), np.sin(tilt_rad)
        robot_pos_arr = np.array(robot_pos, dtype=np.float64)

        def world_to_pixel(wx, wy, wz):
            """Project world point to image pixel. Returns (px, py) or None."""
            pt_robot = R_inv @ (np.array([wx, wy, wz]) - robot_pos_arr)
            fwd, left_neg, up_neg_plus_h = pt_robot
            up = -(up_neg_plus_h - CAMERA_HEIGHT)

            z_cam = fwd * cos_t + up * sin_t
            y_cam = -fwd * sin_t + up * cos_t
            x_cam = -left_neg

            if z_cam <= 0.05:
                return None

            px = int(x_cam * self.fx / z_cam + self.ppx)
            py = int(y_cam * self.fy / z_cam + self.ppy)

            if 0 <= px < img_w and 0 <= py < img_h:
                return (px, py)
            return None

        # For each surface, project grid cells onto image
        # Prefer accumulated grids over single-frame results
        all_grids = {}
        for sid in set(list(self._accumulated.keys()) + list(self.results.keys())):
            all_grids[sid] = self._accumulated.get(sid) or self.results.get(sid)
        for sid, result in all_grids.items():
            surf = self.surfaces.get(sid)
            if surf is None:
                continue

            nx, ny = result["nx"], result["ny"]
            free = result["free"]
            occupied = result["occupied"]
            x_min, y_min = result["x_min"], result["y_min"]
            sz = result["surface_z"]
            cs = self.cell_size

            # Project grid cell centers to image (skip every other cell for speed)
            step = max(1, min(nx, ny) // 40)  # ~40 cells per axis max
            for gy in range(0, ny, step):
                for gx in range(0, nx, step):
                    # Cell center in world coords
                    wx = x_min + (gx + 0.5) * cs * step
                    wy = y_min + (gy + 0.5) * cs * step
                    wz = sz + 0.01  # slightly above surface

                    pixel = world_to_pixel(wx, wy, wz)
                    if pixel is None:
                        continue
                    px, py = pixel

                    # Check cells in this block
                    gx_end = min(gx + step, nx)
                    gy_end = min(gy + step, ny)
                    block_free = free[gy:gy_end, gx:gx_end]
                    block_occ = occupied[gy:gy_end, gx:gx_end]

                    n_free = int(block_free.sum())
                    n_occ = int(block_occ.sum())

                    # Determine block state
                    if n_occ > 0:
                        color = (0, 0, 220)  # red = occupied
                    elif n_free > 0:
                        color = (0, 220, 0)  # green = free
                    else:
                        continue  # not visible, skip

                    # Draw a filled circle proportional to cell size
                    # Estimate pixel radius from depth
                    pixel2 = world_to_pixel(wx + cs * step, wy, wz)
                    if pixel2 is not None:
                        radius = max(3, abs(pixel2[0] - px) // 2)
                    else:
                        radius = 5
                    cv2.circle(overlay, (px, py), radius, color, -1)

            # Draw surface grid boundary outline (white = usable area with insets)
            x_max_r, y_max_r = result["x_max"], result["y_max"]
            corners_world = [
                (x_min, y_min, sz + 0.01),
                (x_max_r, y_min, sz + 0.01),
                (x_max_r, y_max_r, sz + 0.01),
                (x_min, y_max_r, sz + 0.01),
            ]
            corner_pixels = [world_to_pixel(*c) for c in corners_world]
            valid_corners = [c for c in corner_pixels if c is not None]
            if len(valid_corners) >= 3:
                pts = np.array(valid_corners, dtype=np.int32)
                # Thick white boundary with cyan fill
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 3)
                # Label with surface tag, id, and dimensions
                sw = x_max_r - x_min
                sd = y_max_r - y_min
                vis_ratio = result.get("visibility_ratio", 0)
                n_free = result.get("n_free_cells", 0)
                label = f"{surf['tag']}(id:{sid}) {sw:.1f}x{sd:.1f}m vis:{vis_ratio:.0%} free:{n_free}"
                cv2.putText(overlay, label,
                            (valid_corners[0][0], valid_corners[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Draw SG object footprints (red fill = occupied by object, yellow outline)
            sg_objects = (sg_objects_on_map or {}).get(sid, [])
            for o in sg_objects:
                m = DEFAULT_SG_MARGIN
                obj_corners = [
                    (o["cx"] - o["w"]/2 - m, o["cy"] - o["d"]/2 - m, sz + 0.02),
                    (o["cx"] + o["w"]/2 + m, o["cy"] - o["d"]/2 - m, sz + 0.02),
                    (o["cx"] + o["w"]/2 + m, o["cy"] + o["d"]/2 + m, sz + 0.02),
                    (o["cx"] - o["w"]/2 - m, o["cy"] + o["d"]/2 + m, sz + 0.02),
                ]
                obj_pixels = [world_to_pixel(*c) for c in obj_corners]
                valid_obj = [c for c in obj_pixels if c is not None]
                if len(valid_obj) >= 3:
                    pts = np.array(valid_obj, dtype=np.int32)
                    # Red filled polygon for object footprint (object is ON the surface)
                    cv2.fillPoly(overlay, [pts], (0, 0, 200))
                    # Yellow outline on top
                    cv2.polylines(overlay, [pts], True, (0, 220, 220), 2)
                    cv2.putText(overlay, o.get("tag", "?"),
                                (valid_obj[0][0] + 2, valid_obj[0][1] + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 220), 1)

        # Draw placement marker (magenta crosshair) if available
        if self.last_placement is not None:
            lp = self.last_placement
            px_place = world_to_pixel(lp["x"], lp["y"], lp["z"])
            if px_place is not None:
                ppx, ppy = px_place
                # Large crosshair
                cv2.drawMarker(overlay, (ppx, ppy), (255, 0, 255),
                               cv2.MARKER_CROSS, 30, 3)
                cv2.circle(overlay, (ppx, ppy), 15, (255, 0, 255), 2)
                cv2.putText(overlay, f"PLACE {lp.get('tag', '?')}",
                            (ppx + 18, ppy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(overlay, f"({lp['x']:.2f},{lp['y']:.2f})",
                            (ppx + 18, ppy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Blend: 60% overlay, 40% original
        result_img = cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0)

        # Add legend bar at top
        bar_h = 30
        result_img[:bar_h, :] = (result_img[:bar_h, :].astype(np.float32) * 0.3).astype(np.uint8)
        cv2.circle(result_img, (15, 15), 6, (0, 220, 0), -1)
        cv2.putText(result_img, "FREE", (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        cv2.circle(result_img, (95, 15), 6, (0, 0, 220), -1)
        cv2.putText(result_img, "OCCUPIED", (108, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
        cv2.rectangle(result_img, (215, 8), (230, 22), (255, 255, 255), 1)
        cv2.putText(result_img, "SURFACE", (235, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(result_img, (325, 8), (340, 22), (0, 220, 220), 1)
        cv2.putText(result_img, "SG OBJ", (345, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
        cv2.drawMarker(result_img, (435, 15), (255, 0, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(result_img, "PLACE", (450, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Numbered frame save (like react_debug) + latest
        if not hasattr(self, '_overlay_frame_num'):
            self._overlay_frame_num = 0
        self._overlay_frame_num += 1
        fnum = self._overlay_frame_num

        cv2.imwrite(os.path.join(DEBUG_DIR, f"overlay_{fnum:04d}.jpg"),
                    result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(DEBUG_DIR, "camera_overlay_latest.jpg"),
                    result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # Also save to react_debug for side-by-side comparison
        react_dbg = os.path.join(os.path.dirname(DEBUG_DIR), "react_debug")
        if os.path.isdir(react_dbg):
            cv2.imwrite(os.path.join(react_dbg, "freespace_overlay_latest.jpg"),
                        result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
