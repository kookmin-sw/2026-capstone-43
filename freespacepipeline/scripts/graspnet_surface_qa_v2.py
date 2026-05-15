"""
GraspNet table-surface freespace QA generator v2.

기존 v1 문제:
  - depth-only로 추정한 table_z + convex_hull → 책상 밖까지 freespace로 잘못 표시.
  - 물체는 OBB convex hull로만 차감 → 부정확.

v2 개선:
  - cam0_wrt_table.npy (camera→table frame) 사용 → table z=0 평면이 정확히 정의됨.
  - depth를 table frame으로 변환 후 |z|<3cm 픽셀만 keep → table 표면 정확히 추출.
  - label/{frame}.png (per-pixel object id) 사용 → 물체 픽셀 100% 정확. mesh/OBB 불필요.
  - convex_hull 제거 → 모서리 둥근 책상 / 잘려보이는 책상 경계 보존.
  - image edge에 닿은 픽셀 freespace에서 제외.

Usage:
  python scripts/graspnet_surface_qa_v2.py \
      --graspnet_root /home/sungbin/Downloads/train_4 \
      --scene scene_0093 \
      --debug_dir /home/sungbin/Robospatial/debug/graspnet_qa_v2
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

CAMERA = "realsense"
SAFETY_MARGIN_PX = 15
TABLE_Z_TOL = 0.03           # ±3 cm
EDGE_MARGIN_PX = 5           # image edge에 닿은 부분 freespace에서 제외
MIN_TABLE_AREA = 10000
MIN_FREE_AREA = 1000
DEPTH_SCALE = 1000.0         # mm → m


def _largest_poly(p):
    # Handle Polygon, MultiPolygon, GeometryCollection — return largest Polygon piece
    if hasattr(p, "geoms"):
        polys = [g for g in p.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    if isinstance(p, Polygon):
        return p
    return None


def _polygon_to_norm_str(poly, img_w, img_h, simplify_tol=3.0):
    p = poly.simplify(simplify_tol)
    p = _largest_poly(p)
    if p is None or p.is_empty:
        return "[]"
    coords = list(p.exterior.coords)
    return "[" + ", ".join(
        f"({int(x / img_w * 1000)},{int(y / img_h * 1000)})" for x, y in coords
    ) + "]"


def _vertices_to_norm_str(vertices, img_w, img_h):
    """Vertex list (with self-intersecting hole-traversal) → normalized polygon string."""
    return "[" + ", ".join(
        f"({int(x / img_w * 1000)},{int(y / img_h * 1000)})" for x, y in vertices
    ) + "]"


def _ring_polygon_from_mask(mask, simplify_epsilon=3.0):
    """
    mask에서 가장 큰 connected component(outer)를 찾고, 그 안의 holes를
    self-intersecting single vertex list로 표현 (cv2 fillPoly even-odd rule 활용).
    Returns: (vertices, area_pixels, comp_mask) or None.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]  # (N, 4): [next, prev, first_child, parent]

    outer_idxs = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    if not outer_idxs:
        return None
    largest_outer_idx = max(outer_idxs, key=lambda i: cv2.contourArea(contours[i]))

    outer = cv2.approxPolyDP(contours[largest_outer_idx], simplify_epsilon, True).squeeze()
    if outer.ndim != 2 or len(outer) < 3:
        return None

    hole_idxs = [i for i in range(len(contours)) if hierarchy[i][3] == largest_outer_idx]

    vertices = outer.tolist()
    outer_start = outer[0].tolist()
    for hi in hole_idxs:
        hole = cv2.approxPolyDP(contours[hi], simplify_epsilon, True).squeeze()
        if hole.ndim != 2 or len(hole) < 3:
            continue
        vertices.append(outer_start)
        vertices.extend(hole.tolist())
        vertices.append(hole[0].tolist())
    if hole_idxs:
        vertices.append(outer_start)

    # Actual freespace pixels (largest CC만 포함)
    comp_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(comp_mask, [contours[largest_outer_idx]], -1, (255,), cv2.FILLED)
    # Drill holes back
    for hi in hole_idxs:
        cv2.drawContours(comp_mask, [contours[hi]], -1, (0,), cv2.FILLED)
    area = int(cv2.countNonZero(comp_mask))
    return vertices, area, comp_mask


def _mask_to_polys(mask, simplify_tol=2.0, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        pts = c.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue
        try:
            poly = Polygon(pts).simplify(simplify_tol)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                polys.extend([g for g in poly.geoms if g.area >= min_area])
            else:
                polys.append(poly)
        except Exception:
            continue
    return polys


def process_frame(scene_dir, frame_id):
    cam_dir = scene_dir / CAMERA
    rgb_path   = cam_dir / "rgb"   / f"{frame_id:04d}.png"
    depth_path = cam_dir / "depth" / f"{frame_id:04d}.png"
    label_path = cam_dir / "label" / f"{frame_id:04d}.png"

    if not (rgb_path.exists() and depth_path.exists() and label_path.exists()):
        return None

    cam_K          = np.load(cam_dir / "camK.npy")
    cam0_wrt_table = np.load(cam_dir / "cam0_wrt_table.npy")
    camera_poses   = np.load(cam_dir / "camera_poses.npy")  # (N, 4, 4): cam_i wrt cam0
    if frame_id >= len(camera_poses):
        return None

    cam_wrt_table = cam0_wrt_table @ camera_poses[frame_id]

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    label     = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or label is None:
        return None

    depth = depth_raw.astype(np.float32) / DEPTH_SCALE
    H, W  = depth.shape
    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])

    # depth → table frame
    ys, xs = np.where(depth > 0.05)
    zs = depth[ys, xs]
    Xc = (xs - cx) * zs / fx
    Yc = (ys - cy) * zs / fy
    pts_cam = np.stack([Xc, Yc, zs, np.ones_like(zs)], axis=1)
    pts_table = (cam_wrt_table @ pts_cam.T).T
    z_table = pts_table[:, 2]

    # 1) visible table pixels: |z| < tolerance
    table_visible = np.zeros((H, W), np.uint8)
    keep = np.abs(z_table) < TABLE_Z_TOL
    table_visible[ys[keep], xs[keep]] = 255

    # 2) object pixels (segmentation: 0=background, >0=object id)
    object_mask = (label > 0).astype(np.uint8) * 255

    # 3) full table = visible surface ∪ objects on it (objects occlude part of the table)
    full_table = cv2.bitwise_or(table_visible, object_mask)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    full_table = cv2.morphologyEx(full_table, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(full_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_TABLE_AREA:
        return None
    pts = largest.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return None

    try:
        table_poly = Polygon(pts).simplify(3.0)
        if not table_poly.is_valid:
            table_poly = table_poly.buffer(0)
        if isinstance(table_poly, MultiPolygon):
            table_poly = _largest_poly(table_poly)
    except Exception:
        return None
    if table_poly.is_empty:
        return None

    # 4) per-object polygons (시각화/QA용 — buffer는 shapely로)
    obj_polys = []
    obj_records = []
    for obj_id in np.unique(label):
        if obj_id == 0:
            continue
        single_mask = (label == obj_id).astype(np.uint8) * 255
        polys = _mask_to_polys(single_mask, simplify_tol=2.0, min_area=50)
        if not polys:
            continue
        merged = unary_union(polys)
        merged_buffered = merged.buffer(SAFETY_MARGIN_PX)
        if not merged_buffered.is_valid:
            merged_buffered = merged_buffered.buffer(0)
        if merged_buffered.is_empty:
            continue
        obj_polys.append(merged_buffered)
        obj_records.append({
            "id": int(obj_id),
            "poly": merged_buffered,
        })

    # 5) freespace는 OpenCV mask 연산으로 (shapely 정밀도 문제 회피)
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * SAFETY_MARGIN_PX + 1, 2 * SAFETY_MARGIN_PX + 1)
    )
    object_dilated = cv2.dilate(object_mask, kernel_dilate)

    freespace_mask = full_table.copy()
    freespace_mask[object_dilated > 0] = 0
    # Image edge band 제외
    freespace_mask[:EDGE_MARGIN_PX, :] = 0
    freespace_mask[-EDGE_MARGIN_PX:, :] = 0
    freespace_mask[:, :EDGE_MARGIN_PX] = 0
    freespace_mask[:, -EDGE_MARGIN_PX:] = 0

    ring = _ring_polygon_from_mask(freespace_mask, simplify_epsilon=3.0)
    if ring is None:
        return None
    free_vertices, free_area, free_comp_mask = ring
    if free_area < MIN_FREE_AREA:
        return None

    return {
        "frame_id": frame_id,
        "rgb_path": rgb_path,
        "img_w": W,
        "img_h": H,
        "table_poly": table_poly,
        "free_vertices": free_vertices,
        "free_area": free_area,
        "free_comp_mask": free_comp_mask,
        "obj_records": obj_records,
        "table_visible_mask": table_visible,
        "object_mask": object_mask,
    }


def build_qa(res):
    W, H = res["img_w"], res["img_h"]
    table_poly = res["table_poly"]
    obj_records = res["obj_records"]

    qa = []

    # Table layout
    table_area = int(table_poly.area)
    qa.append({
        "qa_type": "table_layout",
        "question": "<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.",
        "answer": (
            f"The total table surface occupies {table_area} pixels. "
            f"Its shape is bounded by the polygon: {_polygon_to_norm_str(table_poly, W, H, 5.0)}."
        ),
    })

    # Per-object layout (id만 - 이름 매핑은 v2에서 생략, 필요시 object_id_list.txt 매핑)
    total_obj_area = 0
    for rec in obj_records:
        poly = rec["poly"]
        obj_area = int(poly.area)
        total_obj_area += obj_area
        qa.append({
            "qa_type": "object_layout",
            "question": (
                f"<image>Where is object {rec['id']} and how much space does it occupy on the table "
                f"assuming a {SAFETY_MARGIN_PX} pixel safety margin? Output coordinates in [0, 1000] scale."
            ),
            "answer": (
                f"Including the {SAFETY_MARGIN_PX}px safety margin, object {rec['id']} occupies {obj_area} pixels. "
                f"Its location is defined by the polygon: {_polygon_to_norm_str(poly, W, H, 3.0)}."
            ),
        })

    # Freespace (self-intersecting polygon with hole-traversal)
    free_vertices = res["free_vertices"]
    free_area = res["free_area"]
    actual_obj_area = int(table_poly.intersection(unary_union([r['poly'] for r in obj_records])).area) if obj_records else 0
    
    reason_str = ""
    if obj_records:
        obj_ids = [str(r['id']) for r in obj_records]
        reason_str = f" Because objects (id: {', '.join(obj_ids)}) occupy the space, "
    else:
        reason_str = " "

    qa.append({
        "qa_type": "freespace_layout",
        "question": (
            f"<image>How much empty space is available on the table and where is the largest empty area, "
            f"assuming a {SAFETY_MARGIN_PX} pixel safety margin around all objects? Output coordinates in [0, 1000] scale."
        ),
        "answer": (
            f"With a {SAFETY_MARGIN_PX}px safety margin,{reason_str.lower()}the objects occupy {actual_obj_area} pixels of the table. "
            f"Subtracting this from the table area ({table_area} pixels) leaves {free_area} pixels of free space. "
            f"The largest continuous empty space is defined by the polygon: "
            f"{_vertices_to_norm_str(free_vertices, W, H)}."
        ),
    })

    return qa


def draw_debug(rgb_bgr, res):
    vis = rgb_bgr.copy()
    overlay = vis.copy()

    # table polygon outline (green)
    geoms = res["table_poly"].geoms if isinstance(res["table_poly"], MultiPolygon) else [res["table_poly"]]
    for g in geoms:
        pts = np.array(g.exterior.coords, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    # freespace fill (blue) — self-intersecting polygon, even-odd rule가 hole 자동 처리
    mask = np.zeros(vis.shape[:2], np.uint8)
    free_pts = np.array(res["free_vertices"], dtype=np.int32)
    if len(free_pts) >= 3:
        cv2.fillPoly(mask, [free_pts], (255,))
    overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([180, 80, 0]) * 0.6).astype(np.uint8)

    # object polygons outline (red)
    for rec in res["obj_records"]:
        p = rec["poly"]
        gs = p.geoms if isinstance(p, MultiPolygon) else [p]
        for g in gs:
            pts = np.array(g.exterior.coords, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 0, 255), 1)

    out = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
    label = f"frame {res['frame_id']}: table={int(res['table_poly'].area)}px, free={res['free_area']}px"
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graspnet_root", required=True)
    ap.add_argument("--scene", required=True, help="e.g. scene_0093")
    ap.add_argument("--out_dir",   default=None, help="QA json 저장 위치")
    ap.add_argument("--debug_dir", default=None, help="디버그 시각화 저장 위치")
    ap.add_argument("--frame_step", type=int, default=5)
    args = ap.parse_args()

    scene_dir = Path(args.graspnet_root) / args.scene
    if not scene_dir.exists():
        raise SystemExit(f"scene 없음: {scene_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    dbg_dir = Path(args.debug_dir) if args.debug_dir else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    if dbg_dir: dbg_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(np.load(scene_dir / CAMERA / "camera_poses.npy"))
    success = 0
    for fid in range(0, n_frames, args.frame_step):
        res = process_frame(scene_dir, fid)
        if res is None:
            print(f"  frame {fid:04d}: skip")
            continue
        success += 1
        qa = build_qa(res)

        if out_dir:
            with open(out_dir / f"{args.scene}_{fid:04d}.json", "w") as f:
                json.dump({"qa": qa, "scene": args.scene, "frame_id": fid}, f, indent=2)

        if dbg_dir:
            rgb = cv2.imread(str(res["rgb_path"]))
            if rgb is not None:
                vis = draw_debug(rgb, res)
                cv2.imwrite(str(dbg_dir / f"{args.scene}_{fid:04d}.jpg"), vis)

        free_area = res["free_area"]
        print(f"  frame {fid:04d}: ok (free={free_area}px, n_obj={len(res['obj_records'])})")

    print(f"\n총 {success} 프레임 처리됨")


if __name__ == "__main__":
    main()
