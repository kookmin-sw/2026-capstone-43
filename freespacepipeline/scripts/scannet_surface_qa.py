"""
ScanNet surface QA generator - per-instance table processing via labeled PLY.

Pipeline:
  0. _vh_clean_2.labels.ply → table vertices (label 7/12/14) → DBSCAN clusters
     → per-instance 3D AABB → projected ROI mask per frame
  1. Within each ROI: label.png (NYU20) → semantic_mask
  2. depth.png + pose.txt → world Z height filter → top surface pixels
  3. semantic AND height → top_mask
  4. top_mask + large closing → table_layout
  5. top_mask + small closing → freespace_layout
  6. large - small → object_layout
  7. Multi-table QA when multiple instances visible

Usage:
  python scripts/scannet_surface_qa.py \\
      --root /home/sungbin/Downloads/ScanNet/scannet_frames_25k \\
      --ply  /home/sungbin/Downloads/ScanNet/ply/scans \\
      --out  /home/sungbin/Robospatial/annotations/scannet_qa \\
      --max_scenes 30
"""

import os, json, argparse, random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from plyfile import PlyData

SAFETY_MARGIN    = 15
TABLE_LABELS_PLY = {7, 12, 14}   # NYU40 in PLY: 7=table, 12=counter, 14=desk
TABLE_LABELS     = {7, 12, 13}   # NYU20 in frame label PNGs
BLUR_THRESHOLD   = 80            # Laplacian variance 이하면 모션블러로 스킵


# ── 데이터 로더 ───────────────────────────────────────────────────────────────
def load_intrinsics(path):
    M = np.loadtxt(path)
    return float(M[0,0]), float(M[1,1]), float(M[0,2]), float(M[1,2])

def load_depth(path):
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32) / 1000.0

def load_label(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

def load_pose(path):
    return np.loadtxt(str(path))


# ── PLY 기반 테이블 인스턴스 추출 ─────────────────────────────────────────────
def load_ply_table_instances(ply_path, min_vertices=100):
    """
    labeled PLY → table 레이블 vertices → DBSCAN cluster → 인스턴스별 3D AABB 리스트.
    Returns: list of dict {pts, aabb_min (xyz), aabb_max (xyz), center_z}
    """
    try:
        ply = PlyData.read(str(ply_path))
    except Exception:
        return []

    v = ply['vertex'].data
    labels = v['label'].astype(np.int32)
    mask = np.zeros(len(labels), bool)
    for lbl in TABLE_LABELS_PLY:
        mask |= (labels == lbl)

    if mask.sum() < min_vertices:
        return []

    xyz = np.stack([v['x'][mask], v['y'][mask], v['z'][mask]], axis=1).astype(np.float64)

    # DBSCAN: 0.3m 이내 점들을 같은 인스턴스로 묶음
    db = DBSCAN(eps=0.3, min_samples=30).fit(xyz)
    instance_ids = db.labels_

    instances = []
    for iid in np.unique(instance_ids):
        if iid == -1:
            continue
        pts = xyz[instance_ids == iid]
        if len(pts) < min_vertices:
            continue
        aabb_min = pts.min(axis=0)
        aabb_max = pts.max(axis=0)
        # 상단 20% 높이 중간값을 테이블 top z로 사용
        z_sorted = np.sort(pts[:, 2])
        top_z = float(np.median(z_sorted[int(len(z_sorted) * 0.8):]))
        instances.append({
            'pts':      pts,
            'aabb_min': aabb_min,
            'aabb_max': aabb_max,
            'top_z':    top_z,
        })

    return instances


def project_world_to_image(pts_w, pose_inv, fx, fy, cx, cy):
    """World 3D points → image (u, v) coords. Returns (N,2) float array."""
    ones = np.ones((len(pts_w), 1))
    pts_h = np.hstack([pts_w, ones])
    pts_c = (pose_inv @ pts_h.T).T
    valid = pts_c[:, 2] > 0.1
    uv = np.full((len(pts_w), 2), np.nan)
    Z = pts_c[valid, 2]
    uv[valid, 0] = fx * pts_c[valid, 0] / Z + cx
    uv[valid, 1] = fy * pts_c[valid, 1] / Z + cy
    return uv


def aabb_roi_mask(aabb_min, aabb_max, pose_inv, fx, fy, cx, cy, H, W, pad=25):
    """
    3D AABB 8꼭짓점을 이미지에 투영 → convex hull 영역을 ROI 마스크로 반환.
    """
    mn, mx = aabb_min, aabb_max
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ])
    uv = project_world_to_image(corners, pose_inv, fx, fy, cx, cy)
    valid = ~np.isnan(uv[:, 0])
    if valid.sum() < 3:
        return None

    pts_img = uv[valid].astype(np.float32)
    pts_img[:, 0] = np.clip(pts_img[:, 0], -pad, W + pad)
    pts_img[:, 1] = np.clip(pts_img[:, 1], -pad, H + pad)

    hull = cv2.convexHull(pts_img.reshape(-1, 1, 2).astype(np.float32))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [hull.astype(np.int32)], 255)

    # 패딩 팽창
    if pad > 0:
        ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad+1, 2*pad+1))
        mask = cv2.dilate(mask, ks)
    return mask


# ── 모션블러 감지 ─────────────────────────────────────────────────────────────
def is_blurry(img, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# ── 수평면 법선 마스크 (side face 제거 핵심) ──────────────────────────────────
def compute_horizontal_mask(depth, pose, fx, fy, cx, cy, min_normal_z=0.5):
    """
    depth → world 3D points → cross product 법선 추정.
    법선의 world Z 성분 > min_normal_z 인 픽셀만 255 (수평면).
    수직면(side face)은 normal Z ≈ 0 이므로 자동 제거.
    노이즈 대응: depth smoothing + 넓은 차분 커널 + 결과 dilation.
    """
    H, W = depth.shape

    # 깊이 스무딩 (노이즈 억제)
    depth_s = cv2.GaussianBlur(depth, (5, 5), 1.0)
    depth_s[depth < 0.1] = 0.0

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    valid = (depth_s > 0.1) & (depth_s < 8.0)

    pts_w = np.full((H, W, 3), np.nan, dtype=np.float32)
    Z  = depth_s[valid]
    Xc = (u[valid] - cx) * Z / fx
    Yc = (v[valid] - cy) * Z / fy
    pts_c = np.stack([Xc, Yc, Z, np.ones_like(Z)], axis=1)
    pw = (pose @ pts_c.T).T
    pts_w[valid] = pw[:, :3]

    # 4픽셀 간격 중심차분 → 노이즈에 강한 법선 추정
    s = 4
    dx = pts_w[:, 2*s:, :] - pts_w[:, :-2*s, :]   # (H, W-2s, 3)
    dy = pts_w[2*s:, :, :] - pts_w[:-2*s, :, :]   # (H-2s, W, 3)
    dx_c = dx[s:-s, :, :]
    dy_c = dy[:, s:-s, :]

    normals = np.cross(dx_c, dy_c)
    norms   = np.linalg.norm(normals, axis=2, keepdims=True)
    ok      = (norms[:, :, 0] > 1e-4) & ~np.isnan(normals[:, :, 0])
    normal_z = np.zeros(normals.shape[:2], np.float32)
    normal_z[ok] = np.abs(normals[ok] / norms[ok])[:, 2]

    inner = np.zeros((H, W), np.uint8)
    inner[s:-s, s:-s] = ((normal_z > min_normal_z) & ok).astype(np.uint8) * 255

    # 법선 불확실 영역(깊이 노이즈)을 채우기 위해 dilation 적용
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.dilate(inner, ks)


# ── depth → world Z 높이 ──────────────────────────────────────────────────────
def depth_to_world_z(depth, pose, fx, fy, cx, cy):
    H, W   = depth.shape
    u, v   = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    valid  = (depth > 0.1) & (depth < 8.0)
    Z      = depth[valid]
    Xc     = (u[valid] - cx) * Z / fx
    Yc     = (v[valid] - cy) * Z / fy
    pts_c  = np.stack([Xc, Yc, Z, np.ones_like(Z)], axis=1)
    pts_w  = (pose @ pts_c.T).T
    flat_idx = np.where(valid.ravel())[0]
    return pts_w[:, 2], flat_idx, H, W


# ── Shapely 유틸 ──────────────────────────────────────────────────────────────
def largest_poly(poly):
    if poly.geom_type == 'Polygon':
        return poly
    polys = [g for g in (poly.geoms if hasattr(poly, 'geoms') else [])
             if g.geom_type == 'Polygon' and not g.is_empty]
    return max(polys, key=lambda p: p.area) if polys else None

def norm_coords(poly, W, H):
    return [(int(x/W*1000), int(y/H*1000)) for x, y in poly.exterior.coords]

def contour_to_poly(cnt, min_pts=4):
    eps = 0.012 * cv2.arcLength(cnt, True)
    app = cv2.approxPolyDP(cnt, eps, True)
    if len(app) < min_pts:
        app = cv2.approxPolyDP(cnt, eps*0.3, True)
    if len(app) < min_pts:
        app = cnt
    try:
        p = Polygon(app.reshape(-1, 2).astype(np.float64))
        if not p.is_valid:
            p = p.buffer(0)
        return p.simplify(2.0, preserve_topology=True)
    except Exception:
        return None


# ── 단일 테이블 인스턴스 처리 ─────────────────────────────────────────────────
def process_table_instance(
    img, sem_mask, depth, pose, fx, fy, cx, cy,
    roi_mask, table_z_top, safety_px, horiz_normal_mask=None
):
    """
    단일 인스턴스 ROI 내에서 table/free/object 폴리곤을 추출.
    Returns: (table_poly, free_poly, obj_polys) or None
    """
    H, W = img.shape[:2]

    if roi_mask is not None:
        sem_roi = cv2.bitwise_and(sem_mask, roi_mask)
    else:
        sem_roi = sem_mask

    if sem_roi.sum() < 500:
        return None

    # height filter
    world_z, flat_idx, H_d, W_d = depth_to_world_z(depth, pose, fx, fy, cx, cy)
    keep = (world_z >= table_z_top - 0.04) & (world_z <= table_z_top + 0.04)
    height_mask = np.zeros(H_d * W_d, np.uint8)
    height_mask[flat_idx[keep]] = 255
    height_mask = height_mask.reshape(H_d, W_d)
    if height_mask.shape != (H, W):
        height_mask = cv2.resize(height_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 수평면 법선 필터: side face 제거
    horiz_mask = horiz_normal_mask
    if horiz_mask is not None and horiz_mask.shape != (H, W):
        horiz_mask = cv2.resize(horiz_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    top_mask = cv2.bitwise_and(sem_roi, height_mask)
    if horiz_mask is not None:
        top_mask = cv2.bitwise_and(top_mask, horiz_mask)

    sem_px = int(sem_roi.sum()) // 255
    top_px = int(top_mask.sum()) // 255
    if sem_px > 0 and top_px / sem_px < 0.10:
        return None  # glass / transparent surface
    if top_mask.sum() < 300:
        return None

    # table_layout: large closing
    kern_l  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_l  = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kern_l)
    if roi_mask is not None:
        mask_l = cv2.bitwise_and(mask_l, roi_mask)

    cnts, _ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt_tbl = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt_tbl) < 400:
        return None
    table_poly = contour_to_poly(cnt_tbl)
    if table_poly is None or table_poly.is_empty:
        return None
    table_poly = largest_poly(table_poly)
    if table_poly is None:
        return None

    # freespace: small closing
    kern_s  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_s  = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kern_s)
    mask_s  = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN, kern_s)
    cfree, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    free_poly = None
    if cfree:
        cands = []
        for cnt in cfree:
            if cv2.contourArea(cnt) < 200:
                continue
            p = contour_to_poly(cnt)
            if p is None:
                continue
            try:
                cl = p.intersection(table_poly)
                if not cl.is_empty and cl.area > 200:
                    cands.append(cl)
            except Exception:
                pass
        if cands:
            m = unary_union(cands)
            free_poly = largest_poly(m)

    # object_layout: large - small
    obj_m = cv2.bitwise_and(mask_l, cv2.bitwise_not(mask_s)) # Placeholder for object mask
    if safety_px > 0:
        ks = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*safety_px+1, 2*safety_px+1))
        obj_m = cv2.dilate(obj_m, ks)
    tbl_r = np.zeros((H, W), np.uint8)
    cv2.fillPoly(tbl_r, [cnt_tbl], 255)
    obj_m = cv2.bitwise_and(obj_m, tbl_r)

    try:
        inner_poly = table_poly.buffer(-30)
    except Exception:
        inner_poly = table_poly

    cobj, _ = cv2.findContours(obj_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obj_polys = []
    min_obj_px = max(800, int(table_poly.area * 0.03))
    for cnt in cobj:
        if cv2.contourArea(cnt) < min_obj_px:
            continue
        p = contour_to_poly(cnt)
        if p is None:
            continue
        try:
            clip_poly = inner_poly if not inner_poly.is_empty else table_poly
            cl = p.intersection(clip_poly)
        except Exception:
            continue
        lp = largest_poly(cl)
        if lp and lp.area > min_obj_px:
            obj_polys.append(lp)

    return table_poly, free_poly, obj_polys


# ── 프레임 처리 ───────────────────────────────────────────────────────────────
def process_frame(scene_dir: Path, frame_id: str, safety_px: int,
                  table_instances: list) -> list[dict]:
    rgb_path   = scene_dir / 'color' / f'{frame_id}.jpg'
    depth_path = scene_dir / 'depth' / f'{frame_id}.png'
    label_path = scene_dir / 'label' / f'{frame_id}.png'
    pose_path  = scene_dir / 'pose'  / f'{frame_id}.txt'
    intr_path  = scene_dir / 'intrinsics_color.txt'

    img = cv2.imread(str(rgb_path))
    if img is None:
        return []
    H, W = img.shape[:2]

    # 모션블러 프레임 스킵
    if is_blurry(img):
        return []

    label = load_label(label_path)
    depth = load_depth(depth_path)
    if depth is None:
        return []
    pose             = load_pose(pose_path)
    fx, fy, cx, cy   = load_intrinsics(intr_path)
    pose_inv         = np.linalg.inv(pose)

    # 수평면 법선 마스크 (현재 비활성화 - depth noise로 인해 과잉 필터링 발생)
    horiz_mask = None  # compute_horizontal_mask(depth, pose, fx, fy, cx, cy)

    # 전체 semantic mask (ROI 결정에 사용)
    lbl_r = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST) \
            if label.shape != (H, W) else label
    sem_mask_full = np.zeros((H, W), np.uint8)
    for lid in TABLE_LABELS:
        sem_mask_full[lbl_r == lid] = 255

    rgb_path_str = str(rgb_path)
    results = []

    # PLY 인스턴스가 없으면 전체 semantic mask로 단일 처리 (fallback)
    if not table_instances:
        table_instances_to_use = [{'roi_mask': None, 'top_z': None, 'aabb_min': None, 'aabb_max': None}]
        use_fallback = True
    else:
        table_instances_to_use = table_instances
        use_fallback = False

    visible_instances = []

    for inst in table_instances_to_use:
        if use_fallback:
            roi_mask  = None
            top_z_ply = None
        else:
            roi_mask = aabb_roi_mask(
                inst['aabb_min'], inst['aabb_max'],
                pose_inv, fx, fy, cx, cy, H, W, pad=25)
            if roi_mask is None:
                continue
            # ROI가 이미지에 실제로 겹치는지 확인
            if roi_mask.sum() < 500:
                continue
            top_z_ply = inst['top_z']

        # semantic과 ROI가 겹치는지 확인
        if roi_mask is not None:
            overlap = cv2.bitwise_and(sem_mask_full, roi_mask).sum()
            if overlap < 500:
                continue

        # table top z 추정: PLY에서 얻은 top_z 사용 또는 depth에서 추정
        if top_z_ply is not None:
            table_z_top = top_z_ply
        else:
            # fallback: depth에서 90th percentile 추정
            sem_for_z = cv2.bitwise_and(sem_mask_full, roi_mask) if roi_mask is not None else sem_mask_full
            H_d, W_d = depth.shape
            sem_r = cv2.resize(sem_for_z, (W_d, H_d), interpolation=cv2.INTER_NEAREST)
            tbl_bin = np.zeros(H_d * W_d, bool)
            tbl_bin |= (sem_r.ravel() > 0)
            world_z, flat_idx, _, _ = depth_to_world_z(depth, pose, fx, fy, cx, cy)
            sem_keep = tbl_bin[flat_idx]
            heights = world_z[sem_keep]
            if len(heights) < 50:
                continue
            table_z_top = float(np.percentile(heights, 90))

        result = process_table_instance(
            img, sem_mask_full, depth, pose, fx, fy, cx, cy,
            roi_mask, table_z_top, safety_px, horiz_normal_mask=horiz_mask)

        if result is None:
            continue

        table_poly, free_poly, obj_polys = result
        visible_instances.append({
            'table_poly': table_poly,
            'free_poly':  free_poly,
            'obj_polys':  obj_polys,
        })

    if not visible_instances:
        return []

    # ── QA 생성 ───────────────────────────────────────────────────────────────
    n_tables = len(visible_instances)

    # 멀티 테이블 QA
    if n_tables > 1:
        areas = [int(vi['table_poly'].area) for vi in visible_instances]
        results.append({
            'category': 'multi_table',
            'image':    rgb_path_str,
            'question': '<image>How many table surfaces are visible in this image?',
            'answer':   (f'There are {n_tables} table surfaces visible in this image. '
                         f'Their approximate areas are: '
                         + ', '.join(f'table {i+1}: {a} pixels'
                                     for i, a in enumerate(areas)) + '.'),
        })

    for t_idx, vi in enumerate(visible_instances):
        table_poly = vi['table_poly']
        free_poly  = vi['free_poly']
        obj_polys  = vi['obj_polys']
        tbl_norm   = norm_coords(table_poly, W, H)

        prefix = f'table {t_idx+1} ' if n_tables > 1 else ''

        results.append({
            'category': 'table_layout',
            'image':    rgb_path_str,
            'question': (f'<image>What is the total area of the {prefix}table surface '
                         f'and what is its polygon shape? Output coordinates in [0, 1000] scale.'),
            'answer':   (f'The {prefix}table surface occupies {int(table_poly.area)} pixels. '
                         f'Its shape is bounded by the polygon: {tbl_norm}.'),
        })

        if free_poly and not free_poly.is_empty and free_poly.area > 100:
            free_norm  = norm_coords(free_poly, W, H)
            table_area = int(table_poly.area)
            free_area  = int(free_poly.area)
            results.append({
                'category': 'freespace_layout',
                'image':    rgb_path_str,
                'question': (f'<image>How much empty space is available on the {prefix}table '
                             f'and where is the largest empty area, assuming a {safety_px} pixel '
                             f'safety margin around all objects? Output coordinates in [0, 1000] scale.'),
                'answer':   (f'With a {safety_px}px safety margin, the objects occupy '
                             f'{max(0, table_area - free_area)} pixels of the {prefix}table. '
                             f'Subtracting this from the table area ({table_area} pixels) '
                             f'leaves {free_area} pixels of free space. '
                             f'The largest continuous empty space is defined by the polygon: {free_norm}.'),
            })

        for i, op in enumerate(obj_polys):
            results.append({
                'category': 'object_layout',
                'image':    rgb_path_str,
                'question': (f'<image>Where is object {i+1} on the {prefix}table and how much '
                             f'space does it occupy assuming a {safety_px} pixel safety margin? '
                             f'Output coordinates in [0, 1000] scale.'),
                'answer':   (f'Including the {safety_px}px safety margin, object {i+1} '
                             f'occupies {int(op.area)} pixels. '
                             f'Its location is defined by the polygon: {norm_coords(op, W, H)}.'),
            })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',       default='/home/sungbin/Downloads/ScanNet/scannet_frames_25k')
    p.add_argument('--ply',        default='/home/sungbin/Downloads/ScanNet/ply/scans')
    p.add_argument('--out',        default='/home/sungbin/Robospatial/annotations/scannet_qa')
    p.add_argument('--safety',     type=int, default=SAFETY_MARGIN)
    p.add_argument('--max_scenes', type=int, default=None)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed)
    root   = Path(args.root)
    ply_root = Path(args.ply)
    out    = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenes = sorted(root.iterdir())
    print(f'[INFO] 전체 씬: {len(scenes)}')
    if args.max_scenes:
        random.shuffle(scenes)
        scenes = sorted(scenes[:args.max_scenes])
        print(f'[INFO] 선택: {len(scenes)}씬')

    all_qas = []
    n_ok = n_skip = n_no_ply = 0

    for scene_dir in tqdm(scenes, desc='처리 중'):
        out_json = out / f'{scene_dir.name}.json'
        if out_json.exists():
            all_qas.extend(json.loads(out_json.read_text()))
            n_ok += 1
            continue

        # PLY 로드
        ply_path = ply_root / scene_dir.name / f'{scene_dir.name}_vh_clean_2.labels.ply'
        if ply_path.exists():
            table_instances = load_ply_table_instances(ply_path)
        else:
            table_instances = []
            n_no_ply += 1

        frames = sorted({p.stem for p in (scene_dir/'color').glob('*.jpg')})
        scene_qas = []
        for fid in frames:
            try:
                scene_qas.extend(
                    process_frame(scene_dir, fid, args.safety, table_instances))
            except Exception:
                pass

        if scene_qas:
            out_json.write_text(json.dumps(scene_qas))
            all_qas.extend(scene_qas)
            n_ok += 1
        else:
            n_skip += 1

    combined = out / 'scannet_qa.json'
    combined.write_text(json.dumps(all_qas, indent=2))

    print(f'\n[DONE] 성공: {n_ok}  스킵: {n_skip}  PLY없음: {n_no_ply}')
    print(f'[DONE] 총 QA: {len(all_qas)}')
    print(f'[DONE] 저장: {combined}')
    by_cat = {}
    for q in all_qas:
        by_cat[q['category']] = by_cat.get(q['category'], 0) + 1
    for k, v in sorted(by_cat.items()):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
