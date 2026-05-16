"""
ScanNet++ indoor support-surface QA generator.

Finds placeable surfaces (table, desk, counter, shelf, ...) in ScanNet++ scenes
using OBB annotations + mesh data. No dependency on RoboSpatial library.

Pipeline per frame:
  1. Load segments_anno.json  → OBBs + labels
  2. Load mesh + segments.json → per-instance mesh vertices (optional, for boundary refinement)
  3. For each support-surface object:
       a. Compute 3D top-face corners from OBB
       b. Optionally refine boundary with mesh top vertices (convex hull)
       c. Project to 2D with fisheye model
       d. Check visibility (min visible area)
       e. Detect objects sitting ON the surface
       f. Compute free polygon (surface minus on-surface objects)
  4. Output QA pairs + debug image

Usage:
    python scripts/scannetpp_indoor_surface_qa.py \
        --data_root /home/sungbin/Downloads/ScanNetPP/data \
        --out_json  annotations/scannetpp/scannetpp_indoor_qa.json \
        --debug_dir debug/scannetpp_indoor_qa \
        [--scenes scene_id1 scene_id2 ...] \
        [--max_frames 20] \
        [--workers 4] \
        [--no_mesh_refine]
"""

import sys
import json
import argparse
import random
import math
import numpy as np
import cv2
import open3d as o3d
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from scipy.spatial import ConvexHull
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

# ── support-surface categories ─────────────────────────────────────────────────
SUPPORT_SURFACE_LABELS = frozenset({
    'table', 'desk', 'kitchen counter', 'counter',
    'coffee table', 'dining table', 'end table', 'side table',
    'nightstand', 'tv stand', 'bench',
})

# Minimum visible polygon area in pixels to process a surface
MIN_SURFACE_AREA_PX = 5000
# Z-tolerance for "on surface": how much above the surface top face
ON_SURFACE_Z_TOLERANCE_BELOW = 0.08   # 8 cm below (annotation noise)
ON_SURFACE_Z_MAX_HEIGHT      = 1.50   # 150 cm max object height on surface
# Mesh top-face: vertices within this delta of max-Z are included
MESH_TOP_Z_DELTA = 0.04               # 4 cm
# Output polygon vertex limit
MAX_POLYGON_CORNERS = 8
# Empty-space sample points per surface
EMPTY_SAMPLE_N = 6
# Colors for debug (BGR)
SURFACE_COLORS = [(0, 210, 0), (0, 160, 255), (220, 0, 220), (0, 200, 200),
                  (255, 160, 0), (0, 255, 160)]


# ── OBB helpers ────────────────────────────────────────────────────────────────

def obb_corners(obb_dict) -> np.ndarray:
    """Return (8, 3) world-space corners of the OBB."""
    centroid = np.array(obb_dict['centroid'], dtype=np.float64)
    lengths  = np.array(obb_dict['axesLengths'], dtype=np.float64)
    axes     = np.array(obb_dict['normalizedAxes'], dtype=np.float64).reshape(3, 3)
    # axes[i] is the i-th axis unit vector; lengths[i] is the full extent
    half = lengths / 2.0
    signs = np.array([[s0, s1, s2]
                       for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)])
    # corner = centroid + sum_i( sign_i * half_i * axis_i )
    offsets = (signs[:, :, None] * half[None, :, None]) * axes[None, :, :]
    return centroid + offsets.sum(axis=1)   # (8, 3)


def obb_top_face(obb_dict) -> np.ndarray:
    """
    Return (4, 3) corners of the top face of the OBB.

    'Top' = the face whose outward normal is most aligned with +Z (world up).
    For a horizontal table this is the actual tabletop.
    """
    centroid = np.array(obb_dict['centroid'], dtype=np.float64)
    lengths  = np.array(obb_dict['axesLengths'], dtype=np.float64)
    axes     = np.array(obb_dict['normalizedAxes'], dtype=np.float64).reshape(3, 3)
    half     = lengths / 2.0

    up = np.array([0.0, 0.0, 1.0])
    # Find which axis is most vertical
    dots      = [abs(float(np.dot(axes[i], up))) for i in range(3)]
    vi        = int(np.argmax(dots))          # vertical axis index
    other     = [i for i in range(3) if i != vi]

    v_axis    = axes[vi]
    # The face in the +Z direction from centroid
    if np.dot(v_axis, up) > 0:
        face_center = centroid + half[vi] * v_axis
    else:
        face_center = centroid - half[vi] * v_axis

    a1, a2   = axes[other[0]], axes[other[1]]
    h1, h2   = half[other[0]], half[other[1]]

    corners = np.array([
        face_center + h1 * a1 + h2 * a2,
        face_center + h1 * a1 - h2 * a2,
        face_center - h1 * a1 - h2 * a2,
        face_center - h1 * a1 + h2 * a2,
    ])
    return corners   # (4, 3) in order for a convex quad


def obb_bottom_z(obb_dict) -> float:
    """Return the minimum Z of all 8 OBB corners (bottom of object)."""
    return float(obb_corners(obb_dict)[:, 2].min())


def obb_top_z(obb_dict) -> float:
    """Return the maximum Z of all 8 OBB corners (top of object)."""
    return float(obb_corners(obb_dict)[:, 2].max())


# ── COLMAP camera / pose parsing ───────────────────────────────────────────────

def parse_colmap_cameras(cameras_txt: Path):
    """Return K (3x3) and D (4,) for the OPENCV_FISHEYE camera model."""
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if parts[1] == 'OPENCV_FISHEYE':
                fx, fy, cx, cy = map(float, parts[4:8])
                k1, k2, k3, k4 = map(float, parts[8:12])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                D = np.array([k1, k2, k3, k4], dtype=np.float64)
                return K, D
            elif parts[1] in ('PINHOLE', 'OPENCV'):
                fx, fy, cx, cy = map(float, parts[4:8])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                D = np.zeros(4, dtype=np.float64)
                return K, D
    return None, None


def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    n = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def parse_colmap_images(images_txt: Path) -> dict:
    """Return dict[image_name → c2w (4x4)] camera-to-world transforms."""
    poses = {}
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz      = map(float, parts[5:8])
            name            = parts[9]
            R_w2c = _quat_to_R(qw, qx, qy, qz)
            w2c   = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R_w2c
            w2c[:3,  3] = [tx, ty, tz]
            poses[name] = np.linalg.inv(w2c)   # c2w
        i += 2
    return poses


# ── fisheye projection ─────────────────────────────────────────────────────────

def project_fisheye(pts_3d: np.ndarray, c2w: np.ndarray,
                    K: np.ndarray, D: np.ndarray) -> tuple:
    """
    Project world-space points through a fisheye camera.

    Returns:
        uv   (N, 2) float  – pixel coordinates (may be outside image bounds)
        mask (N,)   bool   – True if point is in front of camera (z > 0)
    """
    w2c = np.linalg.inv(c2w)
    ph  = np.hstack([pts_3d, np.ones((len(pts_3d), 1), dtype=np.float64)])
    cam = (w2c @ ph.T).T                       # (N, 4)
    z   = cam[:, 2]
    mask = z > 0.02
    uv   = np.full((len(pts_3d), 2), np.nan, dtype=np.float64)

    if mask.sum() == 0:
        return uv, mask

    pts_cam = cam[mask, :3].reshape(-1, 1, 3).astype(np.float64)
    proj, _ = cv2.fisheye.projectPoints(
        pts_cam, np.zeros(3), np.zeros(3), K, D.reshape(4, 1))
    uv[mask] = proj.reshape(-1, 2)
    return uv, mask


# ── mesh loading (instances + raycast scene) ───────────────────────────────────

def load_scene_data(scan_dir: Path, use_mesh: bool = True) -> dict:
    """
    Load scene mesh once and return:
      instance_verts  – dict[object_id → ndarray(N,3)] (empty if use_mesh=False)
      raycast_scene   – o3d.t.geometry.RaycastingScene (always built if mesh exists)
    """
    result = {'instance_verts': {}, 'raycast_scene': None}
    ply_path  = scan_dir / 'mesh_aligned_0.05.ply'
    if not ply_path.exists():
        return result

    # Load legacy mesh (needed for per-vertex data and triangle mesh)
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    # Build raycasting scene (always, regardless of use_mesh)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(mesh_t)
    result['raycast_scene'] = rc_scene

    if not use_mesh:
        return result

    segs_path = scan_dir / 'segments.json'
    anno_path = scan_dir / 'segments_anno.json'
    if not (segs_path.exists() and anno_path.exists()):
        return result

    seg_idx = np.array(json.load(open(segs_path))['segIndices'], dtype=np.int32)
    seg2v: dict = defaultdict(list)
    for vi, si in enumerate(seg_idx):
        seg2v[int(si)].append(vi)

    for g in json.load(open(anno_path))['segGroups']:
        oid = g['objectId']
        vis = []
        for s in g['segments']:
            vis.extend(seg2v.get(int(s), []))
        if len(vis) >= 4:
            result['instance_verts'][oid] = verts[vis]
    return result


def surface_visibility_ratio(top_face: np.ndarray,
                              cam_pos: np.ndarray,
                              raycast_scene,
                              n_samples: int = 25,
                              tolerance: float = 0.15) -> float:
    """
    Fraction of sample points on the surface top face that are directly
    visible from the camera (ray reaches surface without being blocked).

    A ray is 'visible' if the first mesh hit is within `tolerance` metres
    of the expected camera-to-surface distance.
    """
    pts      = _dense_sample_face(top_face, n=n_samples)
    cam_f32  = cam_pos.astype(np.float32)
    origins  = np.tile(cam_f32, (len(pts), 1))
    diff     = pts.astype(np.float32) - cam_f32
    dists    = np.linalg.norm(diff, axis=1)          # expected hit distance
    dirs     = diff / (dists[:, None] + 1e-8)

    rays_t   = o3d.core.Tensor(
        np.concatenate([origins, dirs], axis=1),
        dtype=o3d.core.Dtype.Float32)
    hit_dist = raycast_scene.cast_rays(rays_t)['t_hit'].numpy()

    visible  = np.sum(np.abs(hit_dist - dists) < tolerance)
    return float(visible) / len(pts)


def mesh_top_boundary(vertices: np.ndarray, z_delta: float = MESH_TOP_Z_DELTA
                      ) -> np.ndarray | None:
    """
    Extract the 2D convex hull of the top-surface vertices.

    Returns (K, 3) points in world space (Z = max_z), or None if too few pts.
    """
    max_z = float(vertices[:, 2].max())
    top   = vertices[vertices[:, 2] >= max_z - z_delta]
    if len(top) < 4:
        return None
    try:
        hull = ConvexHull(top[:, :2])
        return top[hull.vertices]   # (K, 3)
    except Exception:
        return None


# ── polygon helpers ────────────────────────────────────────────────────────────

def _largest(poly):
    if isinstance(poly, MultiPolygon):
        return max(poly.geoms, key=lambda p: p.area)
    return poly


def _clip_poly_to_image(poly: Polygon, img_w: int, img_h: int) -> Polygon:
    img_rect = Polygon([(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)])
    clipped  = poly.intersection(img_rect)
    return _largest(clipped) if not clipped.is_empty else Polygon()


def projected_polygon(pts_3d: np.ndarray, c2w: np.ndarray,
                      K: np.ndarray, D: np.ndarray,
                      img_w: int, img_h: int) -> Polygon:
    """
    Project 3D boundary points, build convex hull in 2D, clip to image.
    """
    uv, mask = project_fisheye(pts_3d, c2w, K, D)
    uv_valid  = uv[mask & np.all(np.isfinite(uv), axis=1)]
    if len(uv_valid) < 3:
        return Polygon()
    try:
        hull = ConvexHull(uv_valid)
        coords = uv_valid[hull.vertices].tolist()
        poly   = Polygon(coords)
        return _clip_poly_to_image(poly, img_w, img_h)
    except Exception:
        return Polygon()


def poly_to_norm_str(poly: Polygon, img_w: int, img_h: int,
                     max_corners: int = MAX_POLYGON_CORNERS) -> str | None:
    """Simplify polygon and return as [(x,y)...] string in [0,1000] scale."""
    poly = _largest(poly)
    if poly is None or poly.is_empty:
        return None
    coords = list(poly.exterior.coords[:-1])
    tol = 5.0
    while len(coords) > max_corners and tol < 300:
        s = _largest(poly.simplify(tol, preserve_topology=True))
        if s and not s.is_empty:
            coords = list(s.exterior.coords[:-1])
        tol *= 1.5
    pts = [(int(x / img_w * 1000), int(y / img_h * 1000)) for x, y in coords]
    return '[' + ', '.join(f'({x},{y})' for x, y in pts) + ']'


def sample_points_in_poly(poly: Polygon, n: int,
                           seed: int = 42) -> list[tuple[int, int]]:
    """Random sample n points inside polygon."""
    poly = _largest(poly)
    if poly is None or poly.is_empty:
        return []
    rng  = random.Random(seed)
    b    = poly.bounds
    pts, tries = [], 0
    while len(pts) < n and tries < n * 300:
        tries += 1
        x = rng.uniform(b[0], b[2])
        y = rng.uniform(b[1], b[3])
        if poly.contains(Point(x, y)):
            pts.append((int(x), int(y)))
    return pts


# ── on-surface object detection ───────────────────────────────────────────────

def objects_on_surface(surface_obb: dict, surface_top_z: float,
                       surface_xy_poly: Polygon, all_objects: list,
                       surface_id: int) -> list:
    """
    Find objects that are physically sitting on this surface.

    An object is considered 'on the surface' if:
      - It is not the surface object itself
      - Its bottom Z is within [surface_top_z - tolerance, surface_top_z + 20 cm]
      - Its centroid XY (in 2D) is within the surface XY footprint
    """
    z_lo = surface_top_z - ON_SURFACE_Z_TOLERANCE_BELOW
    z_hi = surface_top_z + ON_SURFACE_Z_MAX_HEIGHT
    on   = []
    for obj in all_objects:
        if obj['object_id'] == surface_id:
            continue
        b_z = obb_bottom_z(obj['obb_dict'])
        if not (z_lo <= b_z <= z_hi):
            continue
        cx, cy = obj['obb_dict']['centroid'][:2]
        if surface_xy_poly.contains(Point(cx, cy)):
            on.append(obj)
    return on


def surface_xy_polygon(top_face: np.ndarray) -> Polygon:
    """Horizontal (XY) polygon of the top face (used for on-surface detection)."""
    return Polygon(top_face[:, :2])


# ── free-space computation ─────────────────────────────────────────────────────

def compute_free_polygon(surface_poly: Polygon,
                          on_surface_objects: list,
                          c2w: np.ndarray,
                          K: np.ndarray,
                          D: np.ndarray,
                          img_w: int,
                          img_h: int,
                          safety_margin_px: int = 15) -> Polygon:
    """
    Subtract projected on-surface objects from the surface polygon.
    Adds a pixel-space safety margin around each object.
    """
    occupied = []
    for obj in on_surface_objects:
        corners = obb_corners(obj['obb_dict'])
        obj_poly = projected_polygon(corners, c2w, K, D, img_w, img_h)
        if not obj_poly.is_empty and obj_poly.area > 100:
            buffered = obj_poly.buffer(safety_margin_px)
            occupied.append(buffered)

    if not occupied:
        return surface_poly

    union = unary_union(occupied)
    free  = surface_poly.difference(union)
    free  = _largest(free)
    return free if (free and not free.is_empty) else Polygon()


# ── QA generation ─────────────────────────────────────────────────────────────

def make_qa_pairs(surface_obj: dict,
                  surface_poly: Polygon,
                  free_poly: Polygon,
                  on_objects: list,
                  img_w: int,
                  img_h: int) -> list[dict]:
    """Generate QA pairs for one surface in one frame."""
    qa = []
    label = surface_obj['label']
    name  = surface_obj.get('display_name', label)

    # 1. surface_layout – where is the surface?
    surf_str = poly_to_norm_str(surface_poly, img_w, img_h)
    if surf_str:
        area_px = int(surface_poly.area)
        qa.append({
            'qa_type':  'surface_layout',
            'question': f'<image>What is the visible area and boundary of the {name} surface? '
                        f'Output coordinates in [0, 1000] scale.',
            'answer':   f'The visible {name} surface occupies {area_px} pixels. '
                        f'Its boundary polygon is: {surf_str}.',
        })

    # 2. objects_on_surface – what is on it?
    if on_objects:
        names = [o.get('display_name', o['label']) for o in on_objects]
        names_str = ', '.join(names)
        qa.append({
            'qa_type':  'objects_on_surface',
            'question': f'<image>What objects are on the {name}?',
            'answer':   f'The following objects are on the {name}: {names_str}.',
        })
    elif surf_str:
        qa.append({
            'qa_type':  'objects_on_surface',
            'question': f'<image>What objects are on the {name}?',
            'answer':   f'The {name} is empty — no objects are placed on it.',
        })

    return qa


# ── debug visualization ────────────────────────────────────────────────────────

def draw_debug(img: np.ndarray,
               results: list,
               img_w: int,
               img_h: int) -> np.ndarray:
    vis     = img.copy()
    overlay = vis.copy()

    for idx, res in enumerate(results):
        col  = SURFACE_COLORS[idx % len(SURFACE_COLORS)]
        name = res['label']

        # Surface polygon
        sp = res['surface_poly']
        if sp and not sp.is_empty:
            pts = np.array(sp.exterior.coords[:-1], dtype=np.int32)
            cv2.polylines(vis, [pts], True, col, 2)
            cx, cy = int(sp.centroid.x), int(sp.centroid.y)
            cv2.putText(vis, f'{name}', (cx - 40, max(cy - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # Free space (filled overlay)
        fp = res['free_poly']
        if fp and not fp.is_empty:
            mask = np.zeros(vis.shape[:2], np.uint8)
            fpts = np.array(fp.exterior.coords[:-1], dtype=np.int32)
            cv2.fillPoly(mask, [fpts], 255)
            tint = np.full_like(overlay, (40, 80, 180))
            overlay[mask > 0] = (overlay[mask > 0] * 0.4 + tint[mask > 0] * 0.6
                                  ).astype(np.uint8)
            for px, py in res['free_sample_pts']:
                cv2.circle(vis, (px, py), 6, (0, 220, 255), -1)
                cv2.circle(vis, (px, py), 6, (0, 0, 0), 1)

        # On-surface objects (light outline)
        for obj_poly in res.get('on_obj_polys', []):
            if obj_poly and not obj_poly.is_empty:
                opts = np.array(obj_poly.exterior.coords[:-1], dtype=np.int32)
                cv2.polylines(vis, [opts], True, (80, 80, 255), 1)

    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # Legend
    ly = 18
    for idx, res in enumerate(results):
        col = SURFACE_COLORS[idx % len(SURFACE_COLORS)]
        cv2.rectangle(vis, (8, ly - 10), (20, ly + 2), col, -1)
        n_on = len(res.get('on_objects', []))
        cv2.putText(vis,
                    f"{res['label']}  surf={int(res['surface_poly'].area if res['surface_poly'] else 0)}px"
                    f"  free={int(res['free_poly'].area if res['free_poly'] else 0)}px"
                    f"  on={n_on}",
                    (26, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        ly += 16

    return vis


# ── per-frame processing ───────────────────────────────────────────────────────

def process_frame(image_name: str,
                  img_path: Path,
                  c2w: np.ndarray,
                  K: np.ndarray,
                  D: np.ndarray,
                  objects: list,
                  instance_verts: dict,
                  use_mesh_refine: bool = True,
                  raycast_scene=None,
                  ) -> tuple[list, dict | None]:
    """
    Process one frame. Returns (qa_list, debug_result_dict).
    debug_result_dict is None if image cannot be loaded.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return [], None
    img_h, img_w = img.shape[:2]

    surface_results = []
    all_qa = []

    # Gather support surfaces
    surfaces = [o for o in objects if o['is_support_surface']]

    for obj in surfaces:
        obb   = obj['obb_dict']
        top4  = obb_top_face(obb)            # (4, 3) top face corners in world

        # Get 3D boundary points for projection
        if use_mesh_refine and obj['object_id'] in instance_verts:
            verts = instance_verts[obj['object_id']]
            mesh_top = mesh_top_boundary(verts)
            if mesh_top is not None and len(mesh_top) >= 3:
                boundary_3d = mesh_top
            else:
                boundary_3d = top4
        else:
            # Dense sample the top face for better projection
            boundary_3d = _dense_sample_face(top4, n=40)

        # Visibility check 1: camera must be above the surface top face
        cam_pos = c2w[:3, 3]
        cam_z   = float(cam_pos[2])
        top_z   = float(top4[:, 2].mean())
        if cam_z < top_z - 0.3:   # camera more than 30 cm below surface → skip
            continue

        # Visibility check 2: surface centroid must project inside the image
        centroid_3d = np.array(obb['centroid'], dtype=np.float64).reshape(1, 3)
        uv_c, mask_c = project_fisheye(centroid_3d, c2w, K, D)
        if not mask_c[0]:
            continue
        cx_px, cy_px = float(uv_c[0, 0]), float(uv_c[0, 1])
        margin = -50   # allow 50px outside for partially visible surfaces
        if not (margin < cx_px < img_w - margin and margin < cy_px < img_h - margin):
            continue

        # Visibility check 3: ray casting – at least 20% of surface must be
        # directly visible (not blocked by walls / other objects)
        if raycast_scene is not None:
            vis_ratio = surface_visibility_ratio(top4, cam_pos, raycast_scene)
            if vis_ratio < 0.20:
                continue

        # Project to 2D
        surf_poly = projected_polygon(boundary_3d, c2w, K, D, img_w, img_h)
        if surf_poly.is_empty or surf_poly.area < MIN_SURFACE_AREA_PX:
            continue

        # Surface XY footprint for on-surface detection
        surf_xy_2d  = surface_xy_polygon(top4)

        # On-surface objects
        on_objs = objects_on_surface(obb, top_z, surf_xy_2d, objects, obj['object_id'])

        # Free polygon
        free_poly = compute_free_polygon(surf_poly, on_objs, c2w, K, D, img_w, img_h)

        # Sample free points
        free_pts = sample_points_in_poly(free_poly, EMPTY_SAMPLE_N)

        # Build per-object projected polys for debug
        on_polys = []
        for on_obj in on_objs:
            corners = obb_corners(on_obj['obb_dict'])
            p = projected_polygon(corners, c2w, K, D, img_w, img_h)
            on_polys.append(p)

        surface_results.append({
            'label':           obj['label'],
            'display_name':    obj.get('display_name', obj['label']),
            'surface_poly':    surf_poly,
            'free_poly':       free_poly,
            'free_sample_pts': free_pts,
            'on_objects':      on_objs,
            'on_obj_polys':    on_polys,
        })

        # QA
        qa = make_qa_pairs(
            {'label': obj['label'], 'display_name': obj.get('display_name', obj['label'])},
            surf_poly, free_poly, on_objs, img_w, img_h)
        all_qa.extend(qa)

    if not surface_results:
        return [], None

    debug = {
        'img':     img,
        'results': surface_results,
        'img_w':   img_w,
        'img_h':   img_h,
    }
    return all_qa, debug


def _dense_sample_face(corners: np.ndarray, n: int = 40) -> np.ndarray:
    """Densely sample points on a quad face (for better convex hull projection)."""
    rng = np.random.default_rng(0)
    u   = rng.random(n)
    v   = rng.random(n)
    # Bilinear interpolation on the quad
    c0, c1, c2, c3 = corners
    pts = (np.outer(1-u, 1-v)[:, :, None] * c0 +
           np.outer(u,   1-v)[:, :, None] * c1 +
           np.outer(u,   v  )[:, :, None] * c2 +
           np.outer(1-u, v  )[:, :, None] * c3)
    return pts.reshape(-1, 3)


# ── per-scene processing ───────────────────────────────────────────────────────

def load_objects(segments_anno_path: Path) -> list:
    """Parse segments_anno.json into a list of object dicts."""
    with open(segments_anno_path) as f:
        d = json.load(f)

    label_count: dict = defaultdict(int)
    for g in d['segGroups']:
        label_count[g.get('label', 'unknown').lower()] += 1

    label_seen: dict = defaultdict(int)
    objects = []
    for g in d['segGroups']:
        raw   = g.get('label', 'unknown').lower()
        count = label_count[raw]
        disp  = f'{raw}_{label_seen[raw]}' if count > 1 else raw
        label_seen[raw] += 1

        objects.append({
            'object_id':        g['objectId'],
            'label':            raw,
            'display_name':     disp,
            'obb_dict':         g['obb'],
            'is_support_surface': raw in SUPPORT_SURFACE_LABELS,
        })
    return objects


def process_scene(args: tuple) -> list:
    (scene_id, data_root, debug_dir, max_frames,
     use_mesh_refine, seed) = args

    scene_dir   = Path(data_root) / scene_id
    scan_dir    = scene_dir / 'scans'
    dslr_dir    = scene_dir / 'dslr'
    colmap_dir  = dslr_dir / 'colmap'
    images_dir  = dslr_dir / 'resized_images'

    anno_path   = scan_dir / 'segments_anno.json'
    cameras_txt = colmap_dir / 'cameras.txt'
    images_txt  = colmap_dir / 'images.txt'

    for p in (anno_path, cameras_txt, images_txt, images_dir):
        if not p.exists():
            print(f'[{scene_id}] missing: {p}', flush=True)
            return []

    # Load camera model
    K, D = parse_colmap_cameras(cameras_txt)
    if K is None:
        print(f'[{scene_id}] unsupported camera model', flush=True)
        return []

    # Load camera poses
    poses = parse_colmap_images(images_txt)

    # Load objects
    objects = load_objects(anno_path)
    n_surfaces = sum(1 for o in objects if o['is_support_surface'])
    if n_surfaces == 0:
        print(f'[{scene_id}] no support surfaces found', flush=True)
        return []

    # Load mesh data: instance vertices (for boundary) + raycast scene (for visibility)
    instance_verts = {}
    raycast_scene  = None
    try:
        scene_data     = load_scene_data(scan_dir, use_mesh=use_mesh_refine)
        instance_verts = scene_data['instance_verts']
        raycast_scene  = scene_data['raycast_scene']
    except Exception as e:
        print(f'[{scene_id}] mesh load error: {e}', flush=True)

    # Select frames
    image_names = sorted(poses.keys())
    rng = random.Random(seed)
    rng.shuffle(image_names)
    image_names = image_names[:max_frames]

    if debug_dir:
        out_dbg = Path(debug_dir) / scene_id
        out_dbg.mkdir(parents=True, exist_ok=True)

    scene_qa = []
    for img_name in image_names:
        img_path = images_dir / img_name
        if not img_path.exists():
            # Try jpeg/jpg fallback
            stem = Path(img_name).stem
            for ext in ('.jpg', '.JPG', '.jpeg', '.png'):
                candidate = images_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                continue

        c2w = poses[img_name]

        try:
            qa_list, debug = process_frame(
                img_name, img_path, c2w, K, D,
                objects, instance_verts, use_mesh_refine,
                raycast_scene=raycast_scene)
        except Exception as e:
            import traceback
            print(f'[{scene_id}/{img_name}] error: {e}', flush=True)
            traceback.print_exc()
            continue

        if not qa_list:
            continue

        scene_qa.append({
            'scene_id':   scene_id,
            'image_name': img_name,
            'image_path': str(img_path),
            'qa':         qa_list,
        })

        if debug and debug_dir:
            vis = draw_debug(debug['img'], debug['results'],
                             debug['img_w'], debug['img_h'])
            stem = Path(img_name).stem
            cv2.imwrite(str(out_dbg / f'{stem}_surface.jpg'), vis)

    print(f'[{scene_id}] {len(scene_qa)} frames with surfaces '
          f'(out of {len(image_names)} sampled)', flush=True)
    return scene_qa


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='ScanNet++ indoor surface QA generator')
    ap.add_argument('--data_root', default='/home/sungbin/Downloads/ScanNetPP/data',
                    help='ScanNet++ data root (contains scene dirs)')
    ap.add_argument('--out_json',  default='annotations/scannetpp/scannetpp_indoor_qa.json')
    ap.add_argument('--debug_dir', default='debug/scannetpp_indoor_qa')
    ap.add_argument('--scenes',    nargs='*', default=None,
                    help='specific scene IDs to process (default: all)')
    ap.add_argument('--max_frames', type=int, default=20,
                    help='max frames to sample per scene')
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--no_mesh_refine', action='store_true',
                    help='skip mesh-based boundary refinement (faster)')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if args.scenes:
        scene_ids = args.scenes
    else:
        scene_ids = sorted(
            d.name for d in data_root.iterdir()
            if d.is_dir() and (d / 'scans' / 'segments_anno.json').exists()
        )

    print(f'Processing {len(scene_ids)} scenes, max {args.max_frames} frames each')

    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    task_args = [
        (sid, str(data_root), args.debug_dir, args.max_frames,
         not args.no_mesh_refine, args.seed)
        for sid in scene_ids
    ]

    all_qa = []
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for result in tqdm(
                    ex.map(process_scene, task_args),
                    total=len(task_args), desc='scenes'):
                all_qa.extend(result)
    else:
        for ta in tqdm(task_args, desc='scenes'):
            all_qa.extend(process_scene(ta))

    with open(args.out_json, 'w') as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    total_qa = sum(len(r['qa']) for r in all_qa)
    print(f'\nDone. {len(all_qa)} frames, {total_qa} QA pairs → {args.out_json}')


if __name__ == '__main__':
    main()
