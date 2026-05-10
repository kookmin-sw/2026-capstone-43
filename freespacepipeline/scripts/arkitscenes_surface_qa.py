"""
ARKitScenes surface QA generator — RANSAC horizontal plane fitting.

Pipeline per scene:
  1. annotation JSON → table/desk/counter OBB (obbAligned, meters, Z-up)
  2. Compute 8 OBB corners for 3D search region
  3. traj → per-frame camera poses (axis-angle Rodrigues)
  4. Per frame:
     a. depth → full 3D world point cloud
     b. 카메라 기울기 필터 (>55° tilt → skip)
     c. OBB XY range 내 depth 포인트 수집
     d. RANSAC 수평 평면 피팅 → 실제 table surface Z
     e. inlier 포인트 → image 투영 → table_poly (OBB 의존 없음)
     f. table_poly 내 Z > surface+5cm 포인트 → objects
     g. freespace = table_poly - objects
     h. Generate table_layout / object_layout / freespace_layout QA

Usage:
  python scripts/arkitscenes_surface_qa.py \
      --root /home/sungbin/Downloads/ARKitScenes/3dod/Training \
      --out  /home/sungbin/Robospatial/annotations/arkitscenes_qa \
      --safety 15
"""

import os, json, argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from shapely.geometry import Polygon
from shapely.ops import unary_union

SAFETY_MARGIN  = 15
TABLE_LABELS   = {'table', 'desk', 'counter'}
BLUR_THRESHOLD = 80

# ── 좌표계 변환 ───────────────────────────────────────────────────────────────

def traj_line_to_pose(line: str):
    tokens = line.strip().split()
    ts = float(tokens[0])
    aa = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    t  = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    R, _ = cv2.Rodrigues(aa)
    ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
    return ts, np.linalg.inv(ext)   # cam-to-world


def load_traj(traj_path: Path) -> dict:
    poses = {}
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ts, pose = traj_line_to_pose(line)
            poses[round(ts, 3)] = pose
    return poses


def get_pose(poses: dict, ts: float):
    key = round(ts, 3)
    if key in poses:
        return poses[key]
    nearest = min(poses.keys(), key=lambda k: abs(k - ts))
    return poses[nearest] if abs(nearest - ts) < 0.1 else None


def parse_pincam(path: Path):
    vals = list(map(float, path.read_text().split()))
    return int(vals[0]), int(vals[1]), vals[2], vals[3], vals[4], vals[5]


# ── OBB 처리 ──────────────────────────────────────────────────────────────────

def obb_corners_8(centroid, axes_lengths, norm_axes) -> np.ndarray:
    """(8, 3) world-space corners. ARKit Z-up."""
    l, h, w = np.array(axes_lengths) / 2.0
    x_c = np.array([ l,  l, -l, -l,  l,  l, -l, -l])
    y_c = np.array([ h, -h, -h,  h,  h, -h, -h,  h])
    z_c = np.array([ w,  w,  w,  w, -w, -w, -w, -w])
    R = np.array(norm_axes).reshape(3, 3)
    return (R.T @ np.vstack([x_c, y_c, z_c])).T + np.array(centroid)


# ── Depth → World 3D ─────────────────────────────────────────────────────────

def depth_to_world3d(depth: np.ndarray, pose: np.ndarray,
                     fx, fy, cx, cy):
    """depth (H,W) uint16 mm → (N,3) world pts, (N,) flat pixel indices."""
    H, W = depth.shape
    d = depth.astype(np.float32) / 1000.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    valid = (d > 0.1) & (d < 8.0)
    Z  = d[valid]
    Xc = (u[valid] - cx) * Z / fx
    Yc = (v[valid] - cy) * Z / fy
    pts_c = np.stack([Xc, Yc, Z, np.ones_like(Z)], axis=1)
    pts_w = (pose @ pts_c.T).T[:, :3]
    return pts_w, np.where(valid.ravel())[0], H, W


# ── RANSAC 수평 평면 피팅 ──────────────────────────────────────────────────────

def ransac_horizontal_plane(world_pts: np.ndarray,
                             z_hint: float,
                             in_xy_mask: np.ndarray,
                             z_search_down: float = 0.60,
                             z_search_up: float = 0.10,
                             z_inlier: float = 0.02,
                             min_pts:  int   = 80):
    """
    OBB XY 범위 내부 포인트 중에서, 여러 평면(Peak)을 찾고
    "가장 낮은 높이(최하단)"에 있는 유의미한 평면을 진짜 책상면으로 선택.
    """
    in_z  = ((world_pts[:, 2] >= z_hint - z_search_down) &
             (world_pts[:, 2] <= z_hint + z_search_up))
    cand_mask = in_xy_mask & in_z

    if cand_mask.sum() < min_pts:
        return None, None

    z_vals = world_pts[cand_mask, 2]
    hist, edges = np.histogram(z_vals, bins=100)
    
    # 지역 최댓값(Peak)들 찾기
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] >= 50:
            peaks.append((edges[i], hist[i]))
            
    # Peak가 없으면 그냥 최댓값 선택
    if not peaks:
        peak_idx = int(np.argmax(hist))
        actual_z = float((edges[peak_idx] + edges[peak_idx + 1]) / 2)
    else:
        # Peak들을 높이(Z) 오름차순으로 정렬하여 "가장 낮은 평면"을 선택!
        peaks.sort(key=lambda x: x[0])
        actual_z = float(peaks[0][0] + (edges[1] - edges[0]) / 2)

    inlier_mask = (cand_mask &
                   (world_pts[:, 2] >= actual_z - z_inlier) &
                   (world_pts[:, 2] <= actual_z + z_inlier))

    if inlier_mask.sum() < min_pts:
        return None, None

    return actual_z, inlier_mask


# ── 투영 (object 마스킹용) ────────────────────────────────────────────────────

def world_to_image(pts_w: np.ndarray, pose: np.ndarray,
                   fx, fy, cx, cy, W, H) -> np.ndarray:
    """World (N,3) → (N,2) image coords, out-of-frame set to nan."""
    pose_inv = np.linalg.inv(pose)
    pts_c = (pose_inv @ np.hstack([pts_w, np.ones((len(pts_w), 1))]).T).T
    uv = np.full((len(pts_w), 2), np.nan)
    valid = pts_c[:, 2] > 0.05
    Z = pts_c[valid, 2]
    uv[valid, 0] = fx * pts_c[valid, 0] / Z + cx
    uv[valid, 1] = fy * pts_c[valid, 1] / Z + cy
    uv[~valid] = np.nan
    return uv


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def is_blurry(img, thr=BLUR_THRESHOLD) -> bool:
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                         cv2.CV_64F).var() < thr


def is_camera_too_tilted(pose: np.ndarray, max_deg: float = 55.0) -> bool:
    """ARKit Z-up. Camera 'up' in world = pose[:3,:3] @ [0,-1,0]."""
    cam_up_z = float((pose[:3, :3] @ np.array([0., -1., 0.]))[2])
    return cam_up_z < np.cos(np.radians(max_deg))


def largest_poly(poly):
    if poly is None or poly.is_empty: return None
    if poly.geom_type == 'Polygon':   return poly
    polys = [g for g in poly.geoms
             if g.geom_type == 'Polygon' and not g.is_empty]
    return max(polys, key=lambda p: p.area) if polys else None


def norm_coords(poly, W, H):
    return [(int(x / W * 1000), int(y / H * 1000))
            for x, y in poly.exterior.coords]


def contour_to_poly(cnt, min_pts=4):
    eps = 0.012 * cv2.arcLength(cnt, True)
    app = cv2.approxPolyDP(cnt, eps, True)
    if len(app) < min_pts:
        app = cv2.approxPolyDP(cnt, eps * 0.3, True)
    if len(app) < min_pts:
        app = cnt
    try:
        p = Polygon(app.reshape(-1, 2).astype(np.float64))
        if not p.is_valid: p = p.buffer(0)
        return p.simplify(2.0, preserve_topology=True)
    except Exception:
        return None


# ── 단일 프레임 처리 ──────────────────────────────────────────────────────────

def process_frame(rgb_path: Path, depth_path: Path, pincam_path: Path,
                  pose: np.ndarray, table_infos: list,
                  safety_px: int) -> list[dict]:

    if is_camera_too_tilted(pose):
        return []

    img = cv2.imread(str(rgb_path))
    if img is None or is_blurry(img):
        return []

    H, W = img.shape[:2]
    img_boundary = Polygon([(0, 0), (W, 0), (W, H), (0, H)])

    _, _, fx, fy, cx, cy = parse_pincam(pincam_path)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return []

    # ── 전체 depth → world 3D point cloud (1회 계산) ─────────────────────────
    world_pts, flat_idx, H_dep, W_dep = depth_to_world3d(
        depth_raw, pose, fx, fy, cx, cy)

    results = []

    for tbl in table_infos:
        corners8   = tbl['corners8']          # (8, 3) world
        z_hint     = tbl['table_top_z']

        # ── RANSAC 수평 평면 피팅 ──
        # 1. OBB 내부(기둥) 좌표계로 정확히 제한하여 의자/바닥 등 배제
        centroid = tbl['centroid']
        R_axes   = tbl['R_axes']
        l, h, w  = tbl['axes_lens'] / 2.0
        
        pts_local = (world_pts - centroid) @ R_axes
        in_obb_xy = (np.abs(pts_local[:, 0]) <= l) & (np.abs(pts_local[:, 1]) <= h)

        # 2. OBB 기둥 안에서 "최하단 평면(가장 낮은 Peak)"을 책상면으로 선택
        actual_z, inlier_mask = ransac_horizontal_plane(
            world_pts, z_hint, in_obb_xy, z_search_down=0.60, z_search_up=0.10)

        if actual_z is None:
            continue   # 이 프레임에서 테이블 표면 안 보임

        # ── 1. 빈 공간(Freespace) 표면 마스크 추출 ──
        inlier_flat = flat_idx[inlier_mask]
        surf_mask = np.zeros(H_dep * W_dep, np.uint8)
        surf_mask[inlier_flat] = 255
        surf_mask = surf_mask.reshape(H_dep, W_dep)

        kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        surf_closed = cv2.morphologyEx(surf_mask, cv2.MORPH_CLOSE, kern_close)
        surf_rgb = cv2.resize(surf_closed, (W, H), interpolation=cv2.INTER_NEAREST)

        # === NEW: Visual Color Refinement to handle distant desks cleanly ===
        # Use OBB projection to find the "core" of the table
        z_sorted_indices = np.argsort(corners8[:, 2])
        top4_corners = corners8[z_sorted_indices[-4:]]
        uv4 = world_to_image(top4_corners, pose, fx, fy, cx, cy, W, H)
        valid_uv = uv4[~np.isnan(uv4[:, 0])]
        obb_poly_2d = None
        if len(valid_uv) >= 3:
            try:
                hull_pts = cv2.convexHull(valid_uv.astype(np.float32)).reshape(-1, 2)
                obb_poly_2d = Polygon(hull_pts)
            except: pass

        if obb_poly_2d is not None and obb_poly_2d.is_valid:
            obb_mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(obb_mask, [np.array(list(obb_poly_2d.exterior.coords), dtype=np.int32)], 255)
            
            # Intersection of RANSAC depth and OBB is extremely reliable
            core_mask = cv2.bitwise_and(surf_rgb, obb_mask)
            
            p_ys, p_xs = np.where(core_mask > 0)
            if len(p_ys) > 50:
                lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                pure_pixels = lab_img[p_ys, p_xs]
                median_color = np.median(pure_pixels, axis=0)
                
                # Check pixels in the broader RANSAC depth mask
                m_ys, m_xs = np.where(surf_rgb > 0)
                pixels = lab_img[m_ys, m_xs]
                diff = pixels - median_color
                diff[:, 0] *= 0.2 # Ignore brightness variations
                dist = np.linalg.norm(diff, axis=1)
                
                # Keep visually similar pixels
                color_keep = dist < 30.0
                refined_mask = np.zeros((H, W), np.uint8)
                refined_mask[m_ys[color_keep], m_xs[color_keep]] = 255
                
                # Fill holes dynamically based on distance/area
                radius = int(np.sqrt(len(p_ys)) * 0.1)
                close_ksize = max(5, radius + (1 if radius % 2 == 0 else 0))
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)))
                surf_rgb = refined_mask

        # ── 2. 전체 책상 모양(Table) 추출 (GraspNet / HOPE 방식) ──
        cnts, _ = cv2.findContours(surf_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        
        largest_c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest_c) < max(500, W * H * 0.01):
            continue
            
        c_pts = largest_c.squeeze()
        if c_pts.ndim != 2 or len(c_pts) < 3:
            continue
            
        try:
            # Snap polygon to the visual color boundary + Convex Hull
            table_poly = Polygon(c_pts).simplify(2.0).convex_hull
        except:
            continue
            
        if table_poly is None or table_poly.is_empty:
            continue

        # 이미지 바운더리 클램핑
        try:
            table_poly = largest_poly(table_poly.intersection(img_boundary))
        except Exception:
            pass
        if table_poly is None or table_poly.is_empty:
            continue

        # ── 3. 물체(Object) 마스크 및 빈공간(Freespace) 계산 ──
        tbl_raster = np.zeros((H, W), np.uint8)
        cv2.fillPoly(tbl_raster,
                     [np.array(list(table_poly.exterior.coords),
                                dtype=np.float32).astype(np.int32)], 255)
                                
        # 물체: 테이블 OBB 기둥 내부이면서 actual_z + 0.03 이상인 점
        obj_above = (world_pts[:, 2] > actual_z + 0.03)
        obj_mask_bool = obj_above & in_obb_xy
        obj_flat = flat_idx[obj_mask_bool]
        
        obj_mask_dep = np.zeros(H_dep * W_dep, np.uint8)
        obj_mask_dep[obj_flat] = 255
        obj_mask_dep = obj_mask_dep.reshape(H_dep, W_dep)
        
        obj_rgb = cv2.resize(obj_mask_dep, (W, H), interpolation=cv2.INTER_NEAREST)
        obj_rgb = cv2.bitwise_and(obj_rgb, tbl_raster)

        # Safety Margin 팽창
        if safety_px > 0:
            ks = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * safety_px + 1, 2 * safety_px + 1))
            obj_rgb = cv2.dilate(obj_rgb, ks)
            obj_rgb = cv2.bitwise_and(obj_rgb, tbl_raster)

        kern_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        obj_closed = cv2.morphologyEx(obj_rgb, cv2.MORPH_CLOSE, kern_s)
        free_mask  = cv2.bitwise_and(tbl_raster, cv2.bitwise_not(obj_closed))
        free_mask  = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, kern_s)

        # object polygons
        cobj, _ = cv2.findContours(obj_rgb, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        obj_polys = []
        min_obj_px = max(500, int(table_poly.area * 0.02))
        for cnt in cobj:
            if cv2.contourArea(cnt) < min_obj_px: continue
            p = contour_to_poly(cnt)
            if p is None: continue
            try:
                cl = largest_poly(p.intersection(table_poly))
            except Exception:
                continue
            if cl and cl.area > min_obj_px:
                obj_polys.append(cl)

        # freespace polygon
        cfree, _ = cv2.findContours(free_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        free_poly = None
        cands = []
        for cnt in cfree:
            if cv2.contourArea(cnt) < 200: continue
            p = contour_to_poly(cnt)
            if p is None: continue
            try:
                cl = largest_poly(p.intersection(table_poly))
                if cl and cl.area > 200: cands.append(cl)
            except Exception:
                pass
        if cands:
            free_poly = largest_poly(unary_union(cands))

        # ── QA 생성 ──────────────────────────────────────────────────────────
        rgb_str  = str(rgb_path)
        tbl_norm = norm_coords(table_poly, W, H)
        tbl_area = int(table_poly.area)

        results.append({
            'category': 'table_layout',
            'image':    rgb_str,
            'question': ('<image>What is the total area of the table surface '
                         'and what is its polygon shape? '
                         'Output coordinates in [0, 1000] scale.'),
            'answer':   (f'The table surface occupies {tbl_area} pixels. '
                         f'Its shape is bounded by the polygon: {tbl_norm}.'),
        })

        for i, op in enumerate(obj_polys):
            results.append({
                'category': 'object_layout',
                'image':    rgb_str,
                'question': (f'<image>Where is object {i+1} on the table and how much '
                             f'space does it occupy assuming a {safety_px} pixel safety margin? '
                             'Output coordinates in [0, 1000] scale.'),
                'answer':   (f'Including the {safety_px}px safety margin, object {i+1} '
                             f'occupies {int(op.area)} pixels. '
                             f'Its location is defined by the polygon: {norm_coords(op, W, H)}.'),
            })

        if free_poly and not free_poly.is_empty and free_poly.area > 200:
            free_area = int(free_poly.area)
            obj_area  = max(0, tbl_area - free_area)
            results.append({
                'category': 'freespace_layout',
                'image':    rgb_str,
                'question': (f'<image>How much empty space is available on the table '
                             f'and where is the largest empty area, assuming a {safety_px} pixel '
                             'safety margin around all objects? '
                             'Output coordinates in [0, 1000] scale.'),
                'answer':   (f'With a {safety_px}px safety margin, the objects occupy '
                             f'{obj_area} pixels of the table. '
                             f'Subtracting this from the table area ({tbl_area} pixels) '
                             f'leaves {free_area} pixels of free space. '
                             f'The largest continuous empty space is defined by the polygon: '
                             f'{norm_coords(free_poly, W, H)}.'),
            })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',         default='/home/sungbin/Downloads/ARKitScenes/3dod/Training')
    p.add_argument('--out',          default='/home/sungbin/Robospatial/annotations/arkitscenes_qa')
    p.add_argument('--safety',       type=int, default=SAFETY_MARGIN)
    p.add_argument('--max_scenes',   type=int, default=None)
    p.add_argument('--frame_stride', type=int, default=5)
    return p.parse_args()


def main():
    args  = parse_args()
    root  = Path(args.root)
    out   = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenes = sorted(root.iterdir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    print(f'[INFO] 씬 수: {len(scenes)}')

    all_qas = []
    n_ok = n_skip = n_no_table = 0

    for scene_dir in tqdm(scenes, desc='처리 중'):
        out_json = out / f'{scene_dir.name}.json'
        if out_json.exists():
            all_qas.extend(json.loads(out_json.read_text()))
            n_ok += 1
            continue

        ann_path   = scene_dir / f'{scene_dir.name}_3dod_annotation.json'
        frames_dir = scene_dir / f'{scene_dir.name}_frames'
        if not ann_path.exists() or not frames_dir.exists():
            n_skip += 1; continue

        ann = json.load(open(ann_path))
        if ann.get('skipped', False):
            n_skip += 1; continue

        table_infos = []
        for obj in ann['data']:
            if obj['label'] not in TABLE_LABELS: continue
            if 'obbAligned' not in obj['segments']: continue
            obb = obj['segments']['obbAligned']
            c8  = obb_corners_8(obb['centroid'], obb['axesLengths'],
                                 obb['normalizedAxes'])
            top_z = float(c8[np.argsort(c8[:, 2])[-4:], 2].mean())
            table_infos.append({
                'corners8':    c8,
                'table_top_z': top_z,
                'label':       obj['label'],
                'centroid':    np.array(obb['centroid']),
                'axes_lens':   np.array(obb['axesLengths']),
                'R_axes':      np.array(obb['normalizedAxes']).reshape(3, 3)
            })

        if not table_infos:
            n_no_table += 1; continue

        traj_path = frames_dir / 'lowres_wide.traj'
        if not traj_path.exists():
            n_skip += 1; continue
        poses = load_traj(traj_path)

        rgb_files = sorted((frames_dir / 'lowres_wide').glob('*.png'))
        scene_qas = []

        for i, rgb_path in enumerate(rgb_files):
            if i % args.frame_stride != 0:
                continue
            ts_str = rgb_path.stem.split('_', 1)[1]
            pose   = get_pose(poses, float(ts_str))
            if pose is None: continue

            depth_path  = frames_dir / 'lowres_depth'             / rgb_path.name
            pincam_path = frames_dir / 'lowres_wide_intrinsics' / (rgb_path.stem + '.pincam')
            if not depth_path.exists() or not pincam_path.exists(): continue

            try:
                scene_qas.extend(
                    process_frame(rgb_path, depth_path, pincam_path,
                                  pose, table_infos, args.safety))
            except Exception:
                pass

        if scene_qas:
            out_json.write_text(json.dumps(scene_qas))
            all_qas.extend(scene_qas)
            n_ok += 1
        else:
            n_skip += 1

    combined = out / 'arkitscenes_qa.json'
    combined.write_text(json.dumps(all_qas, indent=2))
    print(f'\n[DONE] 성공: {n_ok}  스킵: {n_skip}  테이블없음: {n_no_table}')
    print(f'[DONE] 총 QA: {len(all_qas)}')
    by_cat = {}
    for q in all_qas:
        by_cat[q['category']] = by_cat.get(q['category'], 0) + 1
    for k, v in sorted(by_cat.items()):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
