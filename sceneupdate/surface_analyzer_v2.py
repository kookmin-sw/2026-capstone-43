"""
SurfaceAnalyzer V2 - Grid-based free space approach.

Concept: "surface minus objects = free space"
  Instead of computing directional rectangles (N/S/E/W) or MER,
  simply treat the entire surface as available space and subtract
  object footprints. For placement, sample candidate positions on
  a grid and score them by safety (edge distance, object clearance,
  proximity to target).

Differences from V1 (surface_analyzer.py):
  V1: MER X-edge sweep → maximal empty rectangles → pick rect → place inside
  V2: grid sampling → score every valid position → pick safest spot

Both share the same SurfaceAnalyzer class for scene graph integration.
To test V2, change the import in custom_brain.py / pred_brain.py:
  from surface_analyzer_v2 import SurfaceAnalyzer
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Reuse constants & tags from V1 ──────────────────────────────
SURFACE_TAGS = {
    "desk", "table", "counter", "shelf", "cabinet",
    "dresser", "nightstand", "bench", "dining table",
    "coffee table", "side table", "end table", "workbench",
}

MOVABLE_TAGS = {
    "plate", "bowl", "cup", "mug", "glass", "bottle", "book",
    "vase", "remote", "scissors", "laptop", "keyboard", "mouse",
    "cell phone", "knife", "fork", "spoon", "toothbrush",
    "apple", "banana", "orange", "sandwich", "cake", "pizza",
    "potted plant", "teddy bear", "backpack", "handbag", "clock",
}

MIN_FREE_SIZE = 0.05
SURFACE_EDGE_INSET_RATIO = 0.05   # 5% inset from each edge (safety margin)
SURFACE_EDGE_INSET_MIN = 0.03     # minimum 3cm inset

# Extent shrink factor: Isaac Sim bbox is mesh-based (accurate), not point cloud.
# Only minor shrink to account for rounded edges / slight overestimation.
EXTENT_SHRINK_FACTOR = 0.95  # use 95% of raw bbox extent

# Reasonable max dimensions (w, d) per surface type.
# Bbox estimation from point clouds often overshoots; clamp to reality.
SURFACE_MAX_SIZE = {
    "cabinet": (1.0, 1.0),
    "desk": (1.5, 1.0),
    "table": (1.5, 1.2),
    "counter": (2.0, 1.0),
    "shelf": (1.2, 0.5),
    "nightstand": (0.6, 0.6),
}
SURFACE_MAX_SIZE_DEFAULT = (2.0, 1.5)

MOVABLE_MAX_SIZE = {
    "bowl": (0.3, 0.3),
    "cup": (0.15, 0.15),
    "plate": (0.3, 0.3),
    "book": (0.4, 0.3),
    "bottle": (0.15, 0.15),
}
MOVABLE_MAX_SIZE_DEFAULT = (0.5, 0.5)

# Minimum footprint (even if SG extent is small, block at least this much space)
MOVABLE_MIN_SIZE = {
    "book": 0.15,
    "bowl": 0.10,
    "cup": 0.08,
    "plate": 0.12,
    "apple": 0.08,
    "bottle": 0.08,
}
MOVABLE_MIN_SIZE_DEFAULT = 0.08


# ── FreeRect (kept for visualization compatibility) ─────────────
class FreeRect:
    """A rectangular free region on a surface (used for coverage tiles)."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float,
                 direction: str, surface_z: float):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.direction = direction
        self.surface_z = surface_z

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def depth(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.depth

    @property
    def center(self) -> List[float]:
        return [
            round((self.x_min + self.x_max) / 2.0, 3),
            round((self.y_min + self.y_max) / 2.0, 3),
            round(self.surface_z, 3),
        ]

    @property
    def is_valid(self) -> bool:
        return self.width >= MIN_FREE_SIZE and self.depth >= MIN_FREE_SIZE

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "center": self.center,
            "available_size": [round(self.width, 3), round(self.depth, 3)],
            "area_m2": round(self.area, 4),
            "bounds": {
                "x_min": round(self.x_min, 3),
                "y_min": round(self.y_min, 3),
                "x_max": round(self.x_max, 3),
                "y_max": round(self.y_max, 3),
            },
        }


# ── SurfaceInfo V2 ─────────────────────────────────────────────
class SurfaceInfo:
    """A surface with grid-based free space analysis."""

    def __init__(self, obj_id: int, tag: str, center: np.ndarray, extent: np.ndarray):
        self.obj_id = obj_id
        self.tag = tag
        self.center = np.array(center, dtype=float)
        self.extent = np.array(extent, dtype=float)

        # Shrink raw extent (point cloud bbox overestimates)
        shrunk_w = self.extent[0] * EXTENT_SHRINK_FACTOR
        shrunk_d = self.extent[1] * EXTENT_SHRINK_FACTOR

        # Clamp surface size to reasonable max for this tag
        max_w, max_d = SURFACE_MAX_SIZE.get(tag.lower(), SURFACE_MAX_SIZE_DEFAULT)
        # Surface Z: top of bbox, but pull down slightly to avoid overshoot
        self.surface_z = self.center[2] + self.extent[2] / 2.0 - 0.02
        self.surface_width = min(shrunk_w, max_w)
        self.surface_depth = min(shrunk_d, max_d)
        self.surface_area = self.surface_width * self.surface_depth

        if self.extent[0] > max_w or self.extent[1] > max_d:
            pass

        inset_x = max(SURFACE_EDGE_INSET_MIN, self.surface_width * SURFACE_EDGE_INSET_RATIO)
        inset_y = max(SURFACE_EDGE_INSET_MIN, self.surface_depth * SURFACE_EDGE_INSET_RATIO)
        self.x_min = self.center[0] - self.surface_width / 2.0 + inset_x
        self.x_max = self.center[0] + self.surface_width / 2.0 - inset_x
        self.y_min = self.center[1] - self.surface_depth / 2.0 + inset_y
        self.y_max = self.center[1] + self.surface_depth / 2.0 - inset_y

        self.objects_on: List[Dict] = []
        self._custom_free_rects: Optional[List[FreeRect]] = None

    def set_custom_free_spaces(self, rects: List[FreeRect]):
        """Set custom free space rectangles (e.g. from depth analysis).
        If set, to_dict() will use these instead of computing from objects."""
        self._custom_free_rects = rects

    @property
    def occupied_area(self) -> float:
        return sum(o["area"] for o in self.objects_on)

    @property
    def free_area(self) -> float:
        return max(0.0, self.surface_area - self.occupied_area)

    @property
    def occupancy_ratio(self) -> float:
        if self.surface_area <= 0:
            return 1.0
        return self.occupied_area / self.surface_area

    # ── Grid-based coverage tiles (for RViz visualization) ──────
    def compute_coverage_rects(self, cell_size: float = 0.02,
                               margin: float = 0.08) -> List[FreeRect]:
        """
        Non-overlapping tiles that cover ALL free space.
        Grid cells overlapping objects → occupied, rest → free.
        Adjacent free cells merged into rectangles via row-run merge.

        margin: exclusion buffer around each object (8cm default).
                Accounts for bbox estimation error so green tiles
                never visually overlap objects in RViz.
        """
        w = self.x_max - self.x_min
        d = self.y_max - self.y_min
        if w <= 0 or d <= 0:
            return []

        nx = max(1, int(round(w / cell_size)))
        ny = max(1, int(round(d / cell_size)))
        cx = w / nx
        cy = d / ny

        grid = [[True] * nx for _ in range(ny)]

        for o in self.objects_on:
            ox_min = o["cx"] - o["w"] / 2.0 - margin
            ox_max = o["cx"] + o["w"] / 2.0 + margin
            oy_min = o["cy"] - o["d"] / 2.0 - margin
            oy_max = o["cy"] + o["d"] / 2.0 + margin

            gi_x0 = max(0, int((ox_min - self.x_min) / cx))
            gi_x1 = min(nx, int(math.ceil((ox_max - self.x_min) / cx)))
            gi_y0 = max(0, int((oy_min - self.y_min) / cy))
            gi_y1 = min(ny, int(math.ceil((oy_max - self.y_min) / cy)))

            for gy in range(gi_y0, gi_y1):
                for gx in range(gi_x0, gi_x1):
                    grid[gy][gx] = False

        # Row-run merge
        active_runs = []
        result = []

        for gy in range(ny):
            row_runs = []
            run_start = None
            for gx in range(nx):
                if grid[gy][gx]:
                    if run_start is None:
                        run_start = gx
                else:
                    if run_start is not None:
                        row_runs.append((run_start, gx))
                        run_start = None
            if run_start is not None:
                row_runs.append((run_start, nx))

            new_active = []
            matched = set()
            for ar in active_runs:
                ax0, ax1, ay0, ay_cur = ar
                found = False
                for idx, (rx0, rx1) in enumerate(row_runs):
                    if rx0 == ax0 and rx1 == ax1 and idx not in matched:
                        new_active.append((ax0, ax1, ay0, gy))
                        matched.add(idx)
                        found = True
                        break
                if not found:
                    r = self._grid_to_freerect(ax0, ax1, ay0, ay_cur + 1, cx, cy)
                    if r.is_valid:
                        result.append(r)

            for idx, (rx0, rx1) in enumerate(row_runs):
                if idx not in matched:
                    new_active.append((rx0, rx1, gy, gy))

            active_runs = new_active

        for ax0, ax1, ay0, ay_cur in active_runs:
            r = self._grid_to_freerect(ax0, ax1, ay0, ay_cur + 1, cx, cy)
            if r.is_valid:
                result.append(r)

        return result

    def _grid_to_freerect(self, gx0, gx1, gy0, gy1, cx, cy) -> FreeRect:
        return FreeRect(
            x_min=self.x_min + gx0 * cx,
            y_min=self.y_min + gy0 * cy,
            x_max=self.x_min + gx1 * cx,
            y_max=self.y_min + gy1 * cy,
            direction="tile",
            surface_z=self.surface_z,
        )

    # ── Safe placement by grid sampling + scoring ───────────────
    #
    # Three scoring dimensions:
    #   1. edge_safety   — 표면 가장자리에서 떨어지지 않을 안전도 (항상 적용)
    #   2. obj_safety    — 다른 물체와 충돌/접촉 안전도 (물체 있을 때만)
    #   3. efficiency    — 공간 효율 (가장자리에 붙여서 나머지 큰 공간 확보,
    #                      near_obj면 가까이 배치)
    #
    # 비중: 안전(edge+obj) > 효율
    #   W_EDGE = 4.0, W_OBJ = 2.5, W_EFF = 1.5

    W_EDGE = 4.0   # 낙하 안전 (최우선)
    W_OBJ  = 3.5   # 충돌 안전
    W_EFF  = 1.5   # 효율

    def find_safe_placement(self, obj_w: float, obj_d: float,
                            near_obj_id: Optional[int] = None,
                            grid_step: float = 0.02,
                            edge_margin: float = 0.08,
                            obj_clearance: float = 0.10,
                            depth_free_grid: Optional[np.ndarray] = None,
                            depth_grid_bounds: Optional[dict] = None,
                            ) -> Optional[List[float]]:
        """
        Find the best placement position on the surface using 3-score system.

        Scores (all normalized 0~1):
          1. edge_safety:  min distance from object edge to surface edge
                           → 0 at edge, 1 at center. (낙하 방지)
          2. obj_safety:   min distance from nearest existing object
                           → 0 when touching, 1 when far enough. (충돌 방지)
                           Only applies when objects exist on surface.
          3. efficiency:   how well the placement uses space
                           - Without near_obj: push to wall → big remaining space
                           - With near_obj: closer to target = better

        depth_free_grid: If provided, a 2D numpy array where 1=free, 0=not-free.
                         Candidate positions are REJECTED if the object footprint
                         overlaps any non-free cell. This integrates depth camera
                         observations directly into placement.
        depth_grid_bounds: {"x_min", "y_min", "nx", "ny", "cell_size"} for the grid.

        Final = W_EDGE * edge_safety + W_OBJ * obj_safety + W_EFF * efficiency

        Returns [x, y, z] or None.
        """
        sw = self.x_max - self.x_min
        sd = self.y_max - self.y_min
        if sw <= 0 or sd <= 0:
            return None

        hw, hd = obj_w / 2.0, obj_d / 2.0

        # Try both orientations
        orientations = [(hw, hd)]
        if abs(obj_w - obj_d) > 0.01:
            orientations.append((hd, hw))

        # Reference object for "next to" scoring
        ref_xy = None
        if near_obj_id is not None:
            for o in self.objects_on:
                if o["obj_id"] == near_obj_id:
                    ref_xy = (o["cx"], o["cy"])
                    break

        # Raw object footprints (no clearance — for distance measurement)
        raw_footprints = []
        for o in self.objects_on:
            raw_footprints.append((
                o["cx"] - o["w"] / 2.0,
                o["cy"] - o["d"] / 2.0,
                o["cx"] + o["w"] / 2.0,
                o["cy"] + o["d"] / 2.0,
            ))

        # Footprints with clearance buffer (for collision rejection)
        buf_footprints = []
        for o in self.objects_on:
            buf_footprints.append((
                o["cx"] - o["w"] / 2.0 - obj_clearance,
                o["cy"] - o["d"] / 2.0 - obj_clearance,
                o["cx"] + o["w"] / 2.0 + obj_clearance,
                o["cy"] + o["d"] / 2.0 + obj_clearance,
            ))

        has_objects = len(raw_footprints) > 0

        # Max possible edge distance (for normalization)
        max_edge_dist = min(sw, sd) / 2.0
        if max_edge_dist <= 0:
            max_edge_dist = 0.01

        # Max diagonal for efficiency normalization
        max_diag = (sw ** 2 + sd ** 2) ** 0.5
        if max_diag <= 0:
            max_diag = 0.01

        # Safe object distance threshold: beyond this, obj_safety = 1.0
        safe_obj_dist = 0.15  # 15cm

        best_pos = None
        best_score = -float('inf')
        best_detail = None

        for ohw, ohd in orientations:
            cx_min = self.x_min + ohw + edge_margin
            cx_max = self.x_max - ohw - edge_margin
            cy_min = self.y_min + ohd + edge_margin
            cy_max = self.y_max - ohd - edge_margin

            if cx_min >= cx_max or cy_min >= cy_max:
                continue

            nx = max(1, int(round((cx_max - cx_min) / grid_step)))
            ny = max(1, int(round((cy_max - cy_min) / grid_step)))
            step_x = (cx_max - cx_min) / nx
            step_y = (cy_max - cy_min) / ny

            for ix in range(nx + 1):
                px = cx_min + ix * step_x
                for iy in range(ny + 1):
                    py = cy_min + iy * step_y

                    obj_xmin = px - ohw
                    obj_xmax = px + ohw
                    obj_ymin = py - ohd
                    obj_ymax = py + ohd

                    # Hard rejection: collision with buffered footprints
                    collides = False
                    for (fx0, fy0, fx1, fy1) in buf_footprints:
                        if (obj_xmin < fx1 and obj_xmax > fx0 and
                                obj_ymin < fy1 and obj_ymax > fy0):
                            collides = True
                            break
                    if collides:
                        continue

                    # Hard rejection: depth camera says NOT free
                    # Check that ALL grid cells under the object footprint are free
                    if depth_free_grid is not None and depth_grid_bounds is not None:
                        dgb = depth_grid_bounds
                        dcs = dgb["cell_size"]
                        # Object footprint → grid cell range
                        gi_x0 = max(0, int((obj_xmin - dgb["x_min"]) / dcs))
                        gi_x1 = min(dgb["nx"], int(math.ceil((obj_xmax - dgb["x_min"]) / dcs)))
                        gi_y0 = max(0, int((obj_ymin - dgb["y_min"]) / dcs))
                        gi_y1 = min(dgb["ny"], int(math.ceil((obj_ymax - dgb["y_min"]) / dcs)))
                        if gi_x1 > gi_x0 and gi_y1 > gi_y0:
                            footprint_cells = depth_free_grid[gi_y0:gi_y1, gi_x0:gi_x1]
                            # Require at least 70% of footprint cells to be free
                            total_cells = footprint_cells.size
                            free_cells = int(footprint_cells.sum())
                            if total_cells > 0 and free_cells / total_cells < 0.7:
                                continue

                    # ── Score 1: Edge Safety (낙하 안전) ──
                    # Min distance from placed object's edge to surface edge
                    edge_dist = min(
                        obj_xmin - self.x_min,
                        self.x_max - obj_xmax,
                        obj_ymin - self.y_min,
                        self.y_max - obj_ymax,
                    )
                    edge_safety = min(1.0, max(0.0, edge_dist / max_edge_dist))

                    # ── Score 2: Object Safety (충돌 안전) ──
                    obj_safety = 1.0
                    if has_objects:
                        min_obj_dist = float('inf')
                        for (fx0, fy0, fx1, fy1) in raw_footprints:
                            dx = max(0, fx0 - obj_xmax, obj_xmin - fx1)
                            dy = max(0, fy0 - obj_ymax, obj_ymin - fy1)
                            d = (dx * dx + dy * dy) ** 0.5
                            if d < min_obj_dist:
                                min_obj_dist = d
                        obj_safety = min(1.0, max(0.0, min_obj_dist / safe_obj_dist))

                    # ── Score 3: Efficiency (효율) ──
                    if ref_xy is not None:
                        # "next to X" — closer to reference = higher efficiency
                        dist_to_ref = ((px - ref_xy[0]) ** 2 +
                                       (py - ref_xy[1]) ** 2) ** 0.5
                        efficiency = min(1.0, max(0.0, 1.0 - dist_to_ref / max_diag))
                    else:
                        # No reference — prefer center-ish positions (safer).
                        # Avoid both extreme center (wastes space) and extreme edges.
                        # Sweet spot: ~30% from edge = good balance of safety + space.
                        dist_to_wall_x = min(px - self.x_min, self.x_max - px)
                        dist_to_wall_y = min(py - self.y_min, self.y_max - py)
                        wall_ratio = min(dist_to_wall_x / (sw / 2.0),
                                         dist_to_wall_y / (sd / 2.0))
                        # Peak at wall_ratio ~0.3 (30% inward from edge)
                        # Falls off toward both edges and center
                        efficiency = min(1.0, max(0.0, 1.0 - abs(wall_ratio - 0.3) / 0.7))

                    # ── Final Score ──
                    score = (self.W_EDGE * edge_safety +
                             self.W_OBJ * obj_safety +
                             self.W_EFF * efficiency)

                    if score > best_score:
                        best_score = score
                        best_pos = [round(px, 3), round(py, 3),
                                    round(self.surface_z, 3)]
                        best_detail = (edge_safety, obj_safety, efficiency)

        if best_pos and best_detail:
            pass

        return best_pos

    # ── Serialization ───────────────────────────────────────────
    def to_dict(self) -> dict:
        if self._custom_free_rects is not None:
            coverage = self._custom_free_rects
        else:
            coverage = self.compute_coverage_rects()

        total_free = sum(r.area for r in coverage)
        # Largest single tile gives an estimate of max placeable
        max_w, max_d = 0.0, 0.0
        for r in coverage:
            if r.width >= max_w and r.depth >= max_d:
                max_w, max_d = r.width, r.depth

        return {
            "surface_id": self.obj_id,
            "surface_tag": self.tag,
            "surface_z": round(self.surface_z, 3),
            "surface_size": [round(self.surface_width, 3), round(self.surface_depth, 3)],
            "surface_area_m2": round(self.surface_area, 4),
            "objects_on_surface": [
                {
                    "obj_id": o["obj_id"],
                    "tag": o["tag"],
                    "footprint": [round(o["w"], 3), round(o["d"], 3)],
                    "center_xy": [round(o["cx"], 3), round(o["cy"], 3)],
                }
                for o in self.objects_on
            ],
            "occupied_area_m2": round(self.occupied_area, 4),
            "free_area_m2": round(total_free, 4),
            "occupancy_ratio": round(self.occupancy_ratio, 3),
            "max_placeable_size": [round(max_w, 3), round(max_d, 3)],
            # V2: coverage tiles for RViz (no MER free_spaces)
            "free_spaces": [r.to_dict() for r in coverage],
        }


# ── SurfaceAnalyzer V2 ─────────────────────────────────────────
# Same external API as V1 — drop-in replacement.
class SurfaceAnalyzer:
    """
    Scene graph surface analysis using grid-based free space.
    Drop-in replacement for V1: same update() / find_best_placement() API.
    """

    def __init__(self):
        self.surfaces: Dict[int, SurfaceInfo] = {}
        self._obj_data: Dict[int, dict] = {}
        self._edges: List[dict] = []
        # Depth free space grids (updated by sayplan_ui from DepthFreeSpaceAnalyzer)
        # {surface_id: {"free_grid": np.ndarray, "bounds": dict}}
        self._depth_grids: Dict[int, dict] = {}
        # Last placement result (for debug overlay)
        self._last_placement: Optional[dict] = None

    def update(self, obj_data: dict, edges: List[dict],
               react_sg: Optional[dict] = None):
        self.surfaces.clear()
        self._obj_data.clear()
        self._edges = edges

        # Step 1: Merge brain raw_data + react live data
        merged = {}
        for key, info in obj_data.items():
            obj_id = info.get("id")
            merged[obj_id] = dict(info)

        if react_sg:
            for key, info in react_sg.items():
                obj_id = info.get("id")
                if obj_id is not None:
                    if obj_id in merged:
                        merged[obj_id]["bbox_center"] = info.get(
                            "bbox_center", info.get("center",
                                                    merged[obj_id].get("bbox_center")))
                        if "bbox_extent" in info:
                            merged[obj_id]["bbox_extent"] = info["bbox_extent"]
                        if "_status" in info:
                            merged[obj_id]["_status"] = info["_status"]
                    else:
                        merged[obj_id] = dict(info)

        self._obj_data = merged

        # Step 2: Identify surfaces
        for obj_id, info in merged.items():
            tag = info.get("object_tag", "").lower()
            if self._is_surface_tag(tag):
                status = info.get("_status", "present")
                if status == "absent":
                    continue
                center = info.get("bbox_center", info.get("center", [0, 0, 0]))
                extent = info.get("bbox_extent", [0.5, 0.5, 0.5])
                self.surfaces[obj_id] = SurfaceInfo(obj_id, tag, center, extent)

        # Step 3: Assign objects to surfaces (edges first, then proximity)
        self._assign_objects_from_edges()
        self._assign_objects_by_proximity()

        # Step 4: Log summary
        for sid, surf in self.surfaces.items():
            if surf.objects_on:
                obj_strs = [f"{o['tag']}(id:{o['obj_id']}, {o['w']:.2f}x{o['d']:.2f})"
                            for o in surf.objects_on]
                pass

    # ── Tag checks ──────────────────────────────────────────────
    def _is_surface_tag(self, tag: str) -> bool:
        tag_lower = tag.lower().strip()
        for st in SURFACE_TAGS:
            if st in tag_lower or tag_lower in st:
                return True
        return False

    def _is_movable_tag(self, tag: str) -> bool:
        tag_lower = tag.lower().strip()
        for mt in MOVABLE_TAGS:
            if mt in tag_lower or tag_lower in mt:
                return True
        return False

    # ── Object-to-surface assignment (same as V1) ───────────────
    def _assign_objects_from_edges(self):
        for edge in self._edges:
            relation = edge.get("object_relation", "")
            if relation not in ("a on b", "b on a"):
                continue

            src_obj_id = edge.get("src_obj_id")
            dst_obj_id = edge.get("dst_obj_id")

            if src_obj_id is not None and dst_obj_id is not None:
                if relation == "a on b":
                    on_obj_id, surface_id = src_obj_id, dst_obj_id
                else:
                    on_obj_id, surface_id = dst_obj_id, src_obj_id

                if surface_id in self.surfaces and on_obj_id in self._obj_data:
                    self._add_object_to_surface(surface_id, on_obj_id)
                continue

            src_tag = edge.get("src_tag", "")
            dst_tag = edge.get("dst_tag", "")

            if relation == "a on b":
                on_tag, surface_tag = src_tag, dst_tag
            else:
                on_tag, surface_tag = dst_tag, src_tag

            on_candidates = []
            for obj_id, info in self._obj_data.items():
                if obj_id in self.surfaces:
                    continue
                if info.get("object_tag", "").lower() == on_tag.lower():
                    center = info.get("bbox_center", info.get("center"))
                    on_candidates.append((obj_id, np.array(center[:2]) if center else None))

            for on_obj_id, on_xy in on_candidates:
                best_surf_id = None
                best_dist = float('inf')
                for surf_id, surf in self.surfaces.items():
                    if surf.tag.lower() != surface_tag.lower():
                        continue
                    if on_xy is not None:
                        dist = float(np.linalg.norm(on_xy - surf.center[:2]))
                        if dist < best_dist:
                            best_dist = dist
                            best_surf_id = surf_id
                    elif best_surf_id is None:
                        best_surf_id = surf_id

                if best_surf_id is not None:
                    self._add_object_to_surface(best_surf_id, on_obj_id)

    def _assign_objects_by_proximity(self):
        """Assign movable objects to the best matching surface by proximity.

        Checks:
        1. Object must be a movable type (cup, plate, book, etc.)
        2. Object's bottom Z must be near surface Z (within 0.4m)
        3. Object's XY must be within or near the surface bounds
        """
        assigned = set()
        for surf in self.surfaces.values():
            for o in surf.objects_on:
                assigned.add(o["obj_id"])

        for obj_id, info in self._obj_data.items():
            if obj_id in assigned or obj_id in self.surfaces:
                continue

            tag = info.get("object_tag", "")
            if not self._is_movable_tag(tag):
                continue

            status = info.get("_status", "present")
            if status == "absent":
                continue

            center = np.array(info.get("bbox_center", info.get("center", [0, 0, 0])),
                              dtype=float)
            extent = np.array(info.get("bbox_extent", [0.1, 0.1, 0.1]), dtype=float)
            obj_bottom_z = center[2] - extent[2] / 2.0

            # Find the best matching surface (closest in Z, within XY bounds)
            best_surf_id = None
            best_z_diff = float('inf')
            for surf_id, surf in self.surfaces.items():
                # Z check: object bottom should be near surface top
                # Also check object center Z vs surface Z (handles extent errors)
                z_diff_bottom = abs(obj_bottom_z - surf.surface_z)
                z_diff_center = abs(center[2] - surf.surface_z)
                z_diff = min(z_diff_bottom, z_diff_center)
                if z_diff > 0.4:
                    continue

                # XY check: within surface bounds + generous margin
                margin = 0.2
                if (surf.x_min - margin <= center[0] <= surf.x_max + margin and
                        surf.y_min - margin <= center[1] <= surf.y_max + margin):
                    if z_diff < best_z_diff:
                        best_z_diff = z_diff
                        best_surf_id = surf_id

            if best_surf_id is not None:
                self._add_object_to_surface(best_surf_id, obj_id)
                assigned.add(obj_id)

    def _add_object_to_surface(self, surface_id: int, obj_id: int):
        if surface_id not in self.surfaces:
            return
        info = self._obj_data.get(obj_id)
        if info is None:
            return
        for existing in self.surfaces[surface_id].objects_on:
            if existing["obj_id"] == obj_id:
                return

        center = np.array(info.get("bbox_center", info.get("center", [0, 0, 0])),
                          dtype=float)
        extent = np.array(info.get("bbox_extent", [0.1, 0.1, 0.1]), dtype=float)

        tag = info.get("object_tag", "unknown").lower()
        # Shrink movable extent (point cloud bbox overestimates)
        w = extent[0] * EXTENT_SHRINK_FACTOR
        d = extent[1] * EXTENT_SHRINK_FACTOR

        # Clamp movable object size to reasonable max
        max_w, max_d = MOVABLE_MAX_SIZE.get(tag, MOVABLE_MAX_SIZE_DEFAULT)
        if w > max_w or d > max_d:
            old_w, old_d = w, d
            w = min(w, max_w)
            d = min(d, max_d)
            pass

        # Minimum footprint: ensure small objects still block enough space
        min_footprint = MOVABLE_MIN_SIZE.get(tag, MOVABLE_MIN_SIZE_DEFAULT)
        w = max(w, min_footprint)
        d = max(d, min_footprint)

        self.surfaces[surface_id].objects_on.append({
            "obj_id": obj_id,
            "tag": tag,
            "cx": center[0],
            "cy": center[1],
            "w": w,
            "d": d,
            "area": w * d,
        })

    # ================================================================
    # Public query API
    # ================================================================

    def get_surface_info(self, surface_id: int) -> Optional[dict]:
        surf = self.surfaces.get(surface_id)
        if surf is None:
            return None
        return surf.to_dict()

    def get_all_surfaces(self) -> List[dict]:
        return [s.to_dict() for s in self.surfaces.values()]

    def can_place_on(self, surface_id: int, obj_width: float,
                     obj_depth: float) -> Optional[dict]:
        """Check if object fits using grid sampling (returns placement dict or None)."""
        surf = self.surfaces.get(surface_id)
        if surf is None:
            return None
        pos = surf.find_safe_placement(obj_width, obj_depth)
        if pos is None:
            return None
        return {"center": pos, "direction": "safe"}

    def find_best_placement(self, surface_id: int, obj_width: float,
                            obj_depth: float,
                            near_obj_id: Optional[int] = None,
                            edge_margin: Optional[float] = None,
                            obj_clearance: Optional[float] = None,
                            depth_free_grid: Optional[np.ndarray] = None,
                            depth_grid_bounds: Optional[dict] = None,
                            ) -> Optional[dict]:
        """
        V2: grid-sampling + scoring.
        "Surface minus objects = free space" → score by safety.

        If depth_free_grid is provided, candidates are rejected if their
        footprint overlaps non-free cells from depth camera observation.
        """
        surf = self.surfaces.get(surface_id)
        if surf is None:
            pass
            return None

        pass

        kwargs = {"near_obj_id": near_obj_id}
        if edge_margin is not None:
            kwargs["edge_margin"] = edge_margin
        if obj_clearance is not None:
            kwargs["obj_clearance"] = obj_clearance
        # Use depth grid: explicit param > stored grid
        d_grid = depth_free_grid
        d_bounds = depth_grid_bounds
        if d_grid is None and surface_id in self._depth_grids:
            dg = self._depth_grids[surface_id]
            d_grid = dg.get("free_grid")
            d_bounds = dg.get("bounds")
        if d_grid is not None:
            kwargs["depth_free_grid"] = d_grid
        if d_bounds is not None:
            kwargs["depth_grid_bounds"] = d_bounds
        pos = surf.find_safe_placement(obj_width, obj_depth, **kwargs)
        if pos is None:
            pass  # print(f"[SurfaceV2] No valid placement on surface {surface_id}")
            return None

        pass  # print(f"[SurfaceV2] Placement: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        return {
            "direction": "safe",
            "center": pos,
            "available_size": [round(surf.x_max - surf.x_min, 3),
                               round(surf.y_max - surf.y_min, 3)],
            "area_m2": round((surf.x_max - surf.x_min) *
                             (surf.y_max - surf.y_min), 4),
        }

    def find_surfaces_for_object(self, obj_width: float,
                                  obj_depth: float) -> List[dict]:
        results = []
        for surf_id, surf in self.surfaces.items():
            placement = self.can_place_on(surf_id, obj_width, obj_depth)
            if placement is not None:
                results.append({
                    "surface_id": surf_id,
                    "surface_tag": surf.tag,
                    "free_area_m2": round(surf.free_area, 4),
                    "placement": placement,
                })
        results.sort(key=lambda r: r["free_area_m2"], reverse=True)
        return results

    def get_graph_summary(self) -> str:
        """Simple: surface → objects → free % + max placeable size."""
        if not self.surfaces:
            return ""

        lines = ["Surface Placement Info:"]
        for surf in self.surfaces.values():
            info = surf.to_dict()
            objs_str = ""
            if info["objects_on_surface"]:
                obj_names = [f"{o['tag']}(id:{o['obj_id']})" for o in info["objects_on_surface"]]
                objs_str = f", objects_on=[{', '.join(obj_names)}]"

            free_pct = round((1.0 - info["occupancy_ratio"]) * 100)
            max_sz = info["max_placeable_size"]
            lines.append(
                f"  - {info['surface_tag']}(id:{info['surface_id']}): "
                f"size={info['surface_size'][0]}x{info['surface_size'][1]}m, "
                f"free={free_pct}%, max_placeable={max_sz[0]}x{max_sz[1]}m"
                f"{objs_str}"
            )

        return "\n".join(lines)

    # ================================================================
    # JSON 파일 기반 저장 / 로드
    # ================================================================

    def save_to_json(self, path: str):
        """현재 surface 분석 결과를 JSON 파일로 저장."""
        surfaces = []
        for surf in self.surfaces.values():
            surfaces.append(surf.to_dict())
        surfaces.sort(key=lambda s: s["surface_id"])

        total_free_spaces = sum(len(s["free_spaces"]) for s in surfaces)
        all_on_objs = []
        for s in surfaces:
            all_on_objs.extend(s["objects_on_surface"])

        output = {
            "surfaces": surfaces,
            "metadata": {
                "n_surfaces": len(surfaces),
                "n_surfaces_with_objects": sum(1 for s in surfaces if s["objects_on_surface"]),
                "n_objects_on_surfaces": len(all_on_objs),
                "n_free_spaces": total_free_spaces,
                "analyzer": "v2_grid",
            }
        }

        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        pass

    def load_from_json(self, path: str) -> bool:
        """JSON 파일에서 surface 분석 결과를 로드.
        Returns True if loaded successfully."""
        import os
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            pass  # print(f"[SurfaceV2] Failed to load {path}: {e}")
            return False

        self.surfaces.clear()
        for s in data.get("surfaces", []):
            sid = s["surface_id"]
            tag = s["surface_tag"]
            sz = s.get("surface_size", [0.5, 0.5])
            surface_z = s.get("surface_z", 1.0)
            # Reconstruct SurfaceInfo from saved data
            # We create a minimal one and override computed fields
            center = np.array([
                (s.get("free_spaces", [{}])[0].get("bounds", {}).get("x_min", 0) +
                 s.get("free_spaces", [{}])[0].get("bounds", {}).get("x_max", 0)) / 2.0
                if s.get("free_spaces") else 0.0,
                (s.get("free_spaces", [{}])[0].get("bounds", {}).get("y_min", 0) +
                 s.get("free_spaces", [{}])[0].get("bounds", {}).get("y_max", 0)) / 2.0
                if s.get("free_spaces") else 0.0,
                surface_z
            ])
            extent = np.array([sz[0], sz[1], 0.5])
            surf = SurfaceInfo(sid, tag, center, extent)
            # Override with saved values
            surf.surface_z = surface_z
            surf.surface_width = sz[0]
            surf.surface_depth = sz[1]
            surf.surface_area = sz[0] * sz[1]

            # Load objects_on
            for o in s.get("objects_on_surface", []):
                fp = o.get("footprint", [0.1, 0.1])
                cxy = o.get("center_xy", [0, 0])
                surf.objects_on.append({
                    "obj_id": o["obj_id"],
                    "tag": o.get("tag", "unknown"),
                    "cx": cxy[0],
                    "cy": cxy[1],
                    "w": fp[0],
                    "d": fp[1],
                    "area": fp[0] * fp[1],
                })

            self.surfaces[sid] = surf

        # Loaded silently
        return True
