import os, json, argparse, random
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import scipy.io
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

SAFETY_MARGIN  = 15
TABLE_KEYWORDS = frozenset({
    'table', 'desk', 'counter', 'kitchen counter', 'coffee table',
    'end table', 'night stand', 'dining table', 'work table',
    'kitchen_counter', 'coffee_table', 'endtable', 'night_stand',
    'dining_table', 'work_table',
})

def largest_poly(poly):
    if poly.geom_type == 'Polygon':
        return poly
    polys = [g for g in (poly.geoms if hasattr(poly, 'geoms') else [poly])
             if g.geom_type == 'Polygon' and not g.is_empty]
    if not polys:
        return None
    return max(polys, key=lambda p: p.area)

def norm_coords(poly, W, H):
    return [(int(np.clip(x / W * 1000, 0, 1000)), int(np.clip(y / H * 1000, 0, 1000))) for x, y in poly.exterior.coords]

def project_pts(pts_upright, Rtilt, fx, fy, cx, cy, H, W):
    """
    Project 3D points in Upright coordinate system to 2D pixels.
    """
    uvs = []
    for pt in pts_upright:
        q = Rtilt.T @ pt
        x3 = q[0]
        y3 = -q[2]
        z3 = q[1]
        
        if z3 <= 0.1: # behind camera
            continue
            
        u = fx * x3 / z3 + cx
        v = fy * y3 / z3 + cy
        uvs.append((u, v))
        
    if len(uvs) < 3:
        return None
        
    pts = np.array(uvs, dtype=np.float32)
    # Clip and convex hull
    hull = cv2.convexHull(pts)
    if hull is None or len(hull) < 3:
        return None
        
    poly = Polygon(hull.squeeze(1))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def get_box_corners(centroid, basis, coeffs):
    """
    Returns 8 corners of the 3D bounding box
    """
    corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corner = centroid + i * coeffs[0] * basis[:, 0] + j * coeffs[1] * basis[:, 1] + k * coeffs[2] * basis[:, 2]
                corners.append(corner)
    return np.array(corners)

def get_top_face_corners(centroid, basis, coeffs):
    """
    Returns 4 corners of the top face of the 3D bounding box
    basis[:, 2] is assumed to be the up direction.
    """
    corners = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            # top face: k = 1 for the up vector (basis[:, 2])
            corner = centroid + i * coeffs[0] * basis[:, 0] + j * coeffs[1] * basis[:, 1] + 1.0 * coeffs[2] * basis[:, 2]
            corners.append(corner)
    return np.array(corners)

def process_scene(item, sunrgbd_root: str, safety_px: int) -> list[dict]:
    seq    = str(item['sequenceName'][0])
    K      = item['K']
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    Rtilt  = item['Rtilt'].astype(np.float64)

    local_dir  = os.path.join(sunrgbd_root, seq)
    rgb_path   = os.path.join(local_dir, 'image', str(item['rgbname'][0]))
    
    # We just need to read image to get width and height, if possible. But wait, SUNRGBD images might be in sunrgbd_root
    # Let's try to read it to get dimensions. If it doesn't exist, we skip or use default.
    img = cv2.imread(rgb_path)
    if img is None:
        # Fallback dimensions or skip? Many times images might not be extracted in the root directly
        # Let's just try to get dimensions if image is accessible
        # If not, skip
        return []
        
    H, W = img.shape[:2]
    image_boundary = Polygon([(0, 0), (W, 0), (W, H), (0, H)])

    if 'groundtruth3DBB' not in item.dtype.names:
        return []
        
    bbs = item['groundtruth3DBB']
    if bbs is None or len(bbs) == 0:
        return []
        
    tables = []
    objects = []
    
    # Parse all bounding boxes
    for bb in bbs.flat:
        try:
            classname = str(bb['classname'][0]).lower()
        except:
            continue
            
        basis = bb['basis']
        coeffs = bb['coeffs'].flatten()
        centroid = bb['centroid'].flatten()
        
        bb_data = {'classname': classname, 'basis': basis, 'coeffs': coeffs, 'centroid': centroid}
        
        is_table = any(kw in classname for kw in TABLE_KEYWORDS)
        if is_table:
            tables.append(bb_data)
        else:
            # We treat all non-table objects as potential obstacles
            objects.append(bb_data)
            
    if not tables:
        return []
        
    results = []
    
    for t_idx, table in enumerate(tables):
        top_corners = get_top_face_corners(table['centroid'], table['basis'], table['coeffs'])
        table_poly = project_pts(top_corners, Rtilt, fx, fy, cx, cy, H, W)
        
        if table_poly is None or table_poly.is_empty:
            continue
            
        table_poly = table_poly.intersection(image_boundary)
        table_poly = largest_poly(table_poly)
        
        if table_poly is None or table_poly.area < 1000:
            continue
            
        # === NEW: Visual Color Refinement to snap OBB to actual table pixels ===
        if img is not None:
            # Render the sloppy OBB projection
            obb_mask = np.zeros((H, W), np.uint8)
            pts = np.array(list(table_poly.exterior.coords), dtype=np.int32)
            cv2.fillPoly(obb_mask, [pts], 255)
            
            # Erode to safely sample only the core of the table (ignoring background bleed)
            # Scale erosion dynamically based on table size (e.g. 5% of width)
            table_radius = np.sqrt(table_poly.area)
            erode_ksize = max(3, int(table_radius * 0.05))
            if erode_ksize % 2 == 0: erode_ksize += 1
            
            core_mask = cv2.erode(obb_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize)))
            p_ys, p_xs = np.where(core_mask > 0)
            
            if len(p_ys) > 50:
                lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                pure_pixels = lab_img[p_ys, p_xs]
                median_color = np.median(pure_pixels, axis=0)
                
                # Check all pixels in the original OBB mask
                m_ys, m_xs = np.where(obb_mask > 0)
                pixels = lab_img[m_ys, m_xs]
                diff = pixels - median_color
                diff[:, 0] *= 0.2 # Ignore brightness variations (shadows, highlights)
                dist = np.linalg.norm(diff, axis=1)
                
                # Keep visually similar pixels
                color_keep = dist < 30.0
                refined_mask = np.zeros((H, W), np.uint8)
                refined_mask[m_ys[color_keep], m_xs[color_keep]] = 255
                
                # Fill internal holes (e.g. from objects or textures)
                close_ksize = max(5, int(table_radius * 0.1))
                if close_ksize % 2 == 0: close_ksize += 1
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close)
                
                cnts, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    largest_c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(largest_c) > 500:
                        c_pts = largest_c.squeeze()
                        if c_pts.ndim == 2 and len(c_pts) >= 3:
                            try:
                                # Snap polygon to the visual color boundary + Convex Hull
                                refined_poly = Polygon(c_pts).simplify(2.0).convex_hull
                                if refined_poly.is_valid and not refined_poly.is_empty:
                                    table_poly = refined_poly.intersection(image_boundary)
                            except:
                                pass
            
        # Objects on this table
        table_z_top = (table['centroid'] + table['coeffs'][2] * table['basis'][:, 2])[2] # approximate top Z in upright
        table_z_bottom = (table['centroid'] - table['coeffs'][2] * table['basis'][:, 2])[2]
        
        obj_polys = []
        for obj in objects:
            # check if object is on or intersecting the table
            obj_z_bottom = (obj['centroid'] - obj['coeffs'][2] * obj['basis'][:, 2])[2]
            
            # Simple heuristic: object bottom should be near table top (or intersecting)
            # If object is completely below the table top, or way above it, skip
            if obj_z_bottom < table_z_bottom - 0.2 or obj_z_bottom > table_z_top + 0.5:
                continue
                
            obj_corners = get_box_corners(obj['centroid'], obj['basis'], obj['coeffs'])
            obj_poly = project_pts(obj_corners, Rtilt, fx, fy, cx, cy, H, W)
            
            if obj_poly is None or obj_poly.is_empty:
                continue
                
            # Intersect with table
            clipped = obj_poly.intersection(table_poly)
            if clipped.is_empty or clipped.area < 200:
                continue
                
            lp = largest_poly(clipped)
            if lp is not None:
                if safety_px > 0:
                    lp = lp.buffer(safety_px)
                obj_polys.append(lp)
                
        # Calculate free space
        if obj_polys:
            free_poly = table_poly.difference(unary_union(obj_polys))
            free_poly = largest_poly(free_poly)
        else:
            free_poly = table_poly
            
        tbl_norm = norm_coords(table_poly, W, H)
        results.append({
            'category': 'table_layout',
            'image': rgb_path,
            'question': "<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.",
            'answer': f"The total table surface occupies {int(table_poly.area)} pixels. Its shape is bounded by the polygon: {tbl_norm}."
        })
        
        if free_poly is not None and not free_poly.is_empty and free_poly.area > 500:
            free_norm = norm_coords(free_poly, W, H)
            table_area = int(table_poly.area)
            free_area = int(free_poly.area)
            obj_area = max(0, table_area - free_area)
            
            results.append({
                'category': 'freespace_layout',
                'image': rgb_path,
                'question': f"<image>How much empty space is available on the table and where is the largest empty area, assuming a {safety_px} pixel safety margin around all objects? Output coordinates in [0, 1000] scale.",
                'answer': f"With a {safety_px}px safety margin, the objects occupy {obj_area} pixels of the table. Subtracting this from the table area ({table_area} pixels) leaves {free_area} pixels of free space. The largest continuous empty space is defined by the polygon: {free_norm}."
            })
            
        for i, op in enumerate(obj_polys):
            # Clip object buffer to table boundary for final answer output
            op_clipped = op.intersection(table_poly)
            if op_clipped.is_empty:
                continue
            op_lp = largest_poly(op_clipped)
            if op_lp is None:
                continue
                
            obj_norm = norm_coords(op_lp, W, H)
            results.append({
                'category': 'object_layout',
                'image': rgb_path,
                'question': f"<image>Where is object {i+1} on the table and how much space does it occupy assuming a {safety_px} pixel safety margin? Output coordinates in [0, 1000] scale.",
                'answer': f"Including the {safety_px}px safety margin, object {i+1} occupies {int(op_lp.area)} pixels. Its location is defined by the polygon: {obj_norm}."
            })
            
    return results

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mat',        default='/home/sungbin/Downloads/SUNRGBDMeta3DBB_v2.mat')
    p.add_argument('--root',       default='/home/sungbin/Downloads')
    p.add_argument('--out',        default='/home/sungbin/Robospatial/annotations/sunrgbd')
    p.add_argument('--safety',     type=int, default=SAFETY_MARGIN)
    p.add_argument('--max_scenes', type=int, default=None)
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"[INFO] {args.mat} 로드 중...")
    mat  = scipy.io.loadmat(args.mat)
    data = mat['SUNRGBDMeta']
    print(f"[INFO] 전체 씬: {len(data[0])}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_indices = list(range(len(data[0])))
    if args.max_scenes:
        random.shuffle(scene_indices)
        scene_indices = sorted(scene_indices[:args.max_scenes])

    all_qas   = []
    n_success = 0
    n_skip    = 0

    for idx in tqdm(scene_indices, desc='처리 중'):
        item     = data[0, idx]
        seq      = str(item['sequenceName'][0])
        out_json = out_dir / (seq.replace('/', '_') + '.json')

        if out_json.exists():
            with open(out_json) as f:
                qas = json.load(f)
            all_qas.extend(qas)
            n_success += 1
            continue

        qas = process_scene(item, args.root, args.safety)
        if qas:
            with open(out_json, 'w') as f:
                json.dump(qas, f)
            all_qas.extend(qas)
            n_success += 1
        else:
            n_skip += 1

    combined = out_dir / 'sunrgbd_qa.json'
    with open(combined, 'w') as f:
        json.dump(all_qas, f, indent=2)

    print(f"\n[DONE] 성공: {n_success}  스킵: {n_skip}")
    print(f"[DONE] 총 QA: {len(all_qas)}")
    print(f"[DONE] 저장: {combined}")
    by_cat = {}
    for q in all_qas:
        by_cat[q['category']] = by_cat.get(q['category'], 0) + 1
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt}")

if __name__ == '__main__':
    main()
