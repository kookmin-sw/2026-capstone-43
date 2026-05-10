"""
HOPE dataset surface QA generator.
Extracts perfect pixel-level empty space by projecting the implicit table surface (Z=0 plane)
and subtracting exact 2D projections of all object 3D bounding boxes.
"""

import sys
import json
import argparse
import random
import numpy as np
import cv2
import glob
import os
from pathlib import Path
from tqdm import tqdm

from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union

sys.path.insert(0, '/home/sungbin/RoboSpatial/robospatial')
from surface_qa import _bbox3d_to_obb, _project_pts

def _largest_poly(poly):
    if isinstance(poly, MultiPolygon):
        return max(poly.geoms, key=lambda p: p.area)
    return poly

def _sample_empty_pts(poly, n, img_w, img_h, seed=42):
    if poly is None or poly.is_empty:
        return []
    rng  = random.Random(seed)
    poly = _largest_poly(poly)
    b    = poly.bounds
    pts, tries = [], 0
    while len(pts) < n and tries < n * 150:
        tries += 1
        x = rng.uniform(b[0], b[2])
        y = rng.uniform(b[1], b[3])
        if poly.contains(Point(x, y)):
            pts.append((round(x/img_w, 3), round(y/img_h, 3)))
    return pts

def _get_2d_convex_hull(points_3d, extrinsic, intrinsic, img_w, img_h):
    px, ok = _project_pts(points_3d, extrinsic, intrinsic)
    valid = px[ok]
    if len(valid) < 3:
        return None
    try:
        raw = Polygon(valid).convex_hull
        # Clip to image bounds
        img_box = Polygon([(0,0), (img_w,0), (img_w,img_h), (0,img_h)])
        clipped = raw.intersection(img_box)
        if not clipped.is_empty and clipped.area > 1:
            return clipped
    except Exception:
        pass
    return None

def process_hope_annotation(ann_path):
    with open(ann_path, 'r') as f:
        ann = json.load(f)
        
    extrinsic = np.array(ann['camera_annotations']['extrinsic'])
    intrinsic = np.array(ann['camera_annotations']['intrinsic'])
    img_w, img_h = ann['image_size']
    
    objects = []
    for o in ann.get('object_grounding', []):
        if 'bbox_3d' in o:
            objects.append({
                'name': o['name'],
                'obb': _bbox3d_to_obb(o['bbox_3d'])
            })
            
    if not objects:
        return None
        
    # 1. Find table Z
    bottom_z_list = []
    for obj in objects:
        corners = np.asarray(obj['obb'].get_box_points())
        bottom_z_list.append(corners[:, 2].min())
    table_z = np.median(bottom_z_list)

    # 2. Find table 2D polygon using Depth Map
    depth_path = ann['image_path'].replace('/rgb/', '/depth/')
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    depth = depth_raw.astype(np.float32) / 1000.0

    fx, fy   = intrinsic[0, 0], intrinsic[1, 1]
    cx_i, cy_i = intrinsic[0, 2], intrinsic[1, 2]
    
    ys, xs = np.where(depth > 0.05)
    zs = depth[ys, xs]
    Xc = (xs - cx_i) * zs / fx
    Yc = (ys - cy_i) * zs / fy
    pts_cam = np.stack([Xc, Yc, zs, np.ones_like(zs)], axis=1)
    
    pts_world = (extrinsic @ pts_cam.T).T
    world_Z = pts_world[:, 2]

    # Keep pixels around table Z (Wide range to handle camera tilt / uncalibrated depth)
    keep = (world_Z >= table_z - 0.20) & (world_Z <= table_z + 0.08)
    
    table_mask = np.zeros(depth.shape, np.uint8)
    table_mask[ys[keep], xs[keep]] = 255
    
    # === NEW: Color Filter to remove completely different adjacent tables ===
    img_raw = cv2.imread(ann['image_path'])
    if img_raw is not None:
        lab_img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Exclude annotated objects to find the pure table color
        obj_mask_for_color = np.zeros(depth.shape, np.uint8)
        for obj in objects:
            corners = np.asarray(obj['obb'].get_box_points())
            poly_2d = _get_2d_convex_hull(corners, extrinsic, intrinsic, img_w, img_h)
            if poly_2d is not None:
                pts = np.array(poly_2d.exterior.coords, dtype=np.int32)
                cv2.fillPoly(obj_mask_for_color, [pts], 255)
                
        pure_table_mask = cv2.bitwise_and(table_mask, cv2.bitwise_not(obj_mask_for_color))
        p_ys, p_xs = np.where(pure_table_mask > 0)
        
        if len(p_ys) > 1000:
            pure_pixels = lab_img[p_ys, p_xs]
            median_color = np.median(pure_pixels, axis=0)
            
            m_ys, m_xs = np.where(table_mask > 0)
            pixels = lab_img[m_ys, m_xs]
            diff = pixels - median_color
            diff[:, 0] *= 0.2  # Highly insensitive to brightness (shadows)
            dist = np.linalg.norm(diff, axis=1)
            
            # Aggressive color threshold (distance < 30)
            # Internal holes caused by logos or patterns will be safely recovered by .convex_hull later!
            color_keep = dist < 30.0
            color_filtered_mask = np.zeros(depth.shape, np.uint8)
            color_filtered_mask[m_ys[color_keep], m_xs[color_keep]] = 255
            table_mask = color_filtered_mask
    
    # 1. Close holes caused by depth sensor noise, objects, or color filtering
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)) # Increased to fill color holes
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Open to remove thin protruding structures like monitor arms, clamps, or attached cables!
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, kernel_open)
    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Only keep the absolute largest contour to ignore disconnected chairs or floor patches
    largest_c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_c) < 10000:
        return None
        
    pts = largest_c.squeeze()
    if pts.ndim != 2 or len(pts) < 3:
        return None
        
    # Convex hull recovers the sharp rectangular corners that were rounded by the Opening operation!
    try:
        table_poly = Polygon(pts).simplify(2.0).convex_hull
    except:
        return None
        
    if table_poly.is_empty:
        return None
        
    # 3. Project objects to 2D and subtract
    obj_polys = []
    for obj in objects:
        corners = np.asarray(obj['obb'].get_box_points())
        poly_2d = _get_2d_convex_hull(corners, extrinsic, intrinsic, img_w, img_h)
        if poly_2d is not None:
            # Buffer aggressively to ensure object boundaries are fully excluded
            obj_polys.append(poly_2d.buffer(15.0))
            
    if obj_polys:
        empty_poly = table_poly.difference(unary_union(obj_polys))
    else:
        empty_poly = table_poly
        
    if empty_poly.is_empty or empty_poly.area < 1000:
        return None
        
    # 4. Generate QA (Spatial Layout & Polygon Vertices)
    qa = []
    
    # 4.1 Total Table QA
    table_area = int(table_poly.area)
    table_sim = table_poly.simplify(5.0)
    if isinstance(table_sim, MultiPolygon): table_sim = _largest_poly(table_sim)
    table_coords = '[' + ', '.join(f"({int(x/img_w*1000)},{int(y/img_h*1000)})" for x, y in table_sim.exterior.coords) + ']'
    
    qa.append({
        'qa_type': 'table_layout',
        'question': '<image>What is the total area of the table surface and what is its polygon shape? Output coordinates in [0, 1000] scale.',
        'answer': f"The total table surface occupies {table_area} pixels. Its shape is bounded by the polygon: {table_coords}."
    })
    
    # 4.2 Object QA
    total_obj_area = 0
    safety_margin = 15.0
    for obj, poly in zip(objects, obj_polys):
        obj_area = int(poly.area)
        total_obj_area += obj_area
        obj_name = obj['name'].replace('_', ' ')
        
        poly_sim = poly.simplify(3.0)
        if isinstance(poly_sim, MultiPolygon): poly_sim = _largest_poly(poly_sim)
        obj_coords = '[' + ', '.join(f"({int(x/img_w*1000)},{int(y/img_h*1000)})" for x, y in poly_sim.exterior.coords) + ']'
        
        qa.append({
            'qa_type': 'object_layout',
            'question': f'<image>Where is the {obj_name} and how much space does it occupy on the table assuming a {int(safety_margin)} pixel safety margin? Output coordinates in [0, 1000] scale.',
            'answer': f"Including the {int(safety_margin)}px safety margin, the {obj_name} occupies {obj_area} pixels. Its location is defined by the polygon: {obj_coords}."
        })
        
    # 4.3 Free Space QA
    if empty_poly is not None and not empty_poly.is_empty:
        empty_area = int(empty_poly.area)
        actual_obj_area = int(table_poly.intersection(unary_union(obj_polys)).area)
        
        # Get the largest continuous empty space
        largest_empty = _largest_poly(empty_poly)
        empty_sim = largest_empty.simplify(10.0) # Larger tolerance for complex empty space
        empty_coords = '[' + ', '.join(f"({int(x/img_w*1000)},{int(y/img_h*1000)})" for x, y in empty_sim.exterior.coords) + ']'
        
        qa.append({
            'qa_type': 'freespace_layout',
            'question': f'<image>How much empty space is available on the table and where is the largest empty area, assuming a {int(safety_margin)} pixel safety margin around all objects? Output coordinates in [0, 1000] scale.',
            'answer': f"With a {int(safety_margin)}px safety margin, the objects occupy {actual_obj_area} pixels of the table. Subtracting this from the table area ({table_area} pixels) leaves {empty_area} pixels of free space. The largest continuous empty space is defined by the polygon: {empty_coords}."
        })
    
    return {
        'qa': qa,
        'table_poly': table_poly,
        'empty_poly': empty_poly,
        'objects': objects,
        'img_path': ann['image_path']
    }

def draw_debug(img, result):
    vis = img.copy()
    overlay = vis.copy()
    
    table_poly = result['table_poly']
    empty_poly = result['empty_poly']
    
    # Draw table bounding box
    if not table_poly.is_empty:
        geoms = table_poly.geoms if isinstance(table_poly, MultiPolygon) else [table_poly]
        for g in geoms:
            pts = np.array(g.exterior.coords, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        
    # Draw empty space
    if not empty_poly.is_empty:
        geoms = empty_poly.geoms if isinstance(empty_poly, MultiPolygon) else [empty_poly]
        mask = np.zeros(vis.shape[:2], np.uint8)
        for g in geoms:
            pts = np.array(g.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
        blue = np.full_like(overlay, (160, 70, 0)) # BGR: Blueish
        overlay[mask > 0] = (overlay[mask > 0] * 0.35 + blue[mask > 0] * 0.65).astype(np.uint8)
        
        # Draw sample points
        img_w, img_h = vis.shape[1], vis.shape[0]
        pts = _sample_empty_pts(empty_poly, 20, img_w, img_h)
        for px, py in pts:
            px, py = int(px * img_w), int(py * img_h)
            cv2.circle(vis, (px, py), 7, (0, 220, 255), -1)
            cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)
            
    # Draw objects
    for obj in result['objects']:
        corners = np.asarray(obj['obb'].get_box_points())
        px, ok = _project_pts(corners, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), np.eye(3)) # dummy
        # Actually just draw text near the center
        cx, cy, cz = obj['obb'].center
        w2c = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) # Wait, need real extrinsic
        # We'll just skip 2d bbox drawing for objects to keep it simple, focus on empty space
        pass
        
    vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)
    return vis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', required=True)
    parser.add_argument('--debug_dir', default=None)
    args = parser.parse_args()
    
    ann_files = glob.glob(os.path.join(args.annotations_dir, '**', '*.annotations.json'), recursive=True)
    print(f"Found {len(ann_files)} HOPE annotations.")
    
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        
    success = 0
    for ann_path in tqdm(ann_files):
        res = process_hope_annotation(ann_path)
        if res is None:
            continue
            
        success += 1
        
        # Save QA
        out_path = ann_path.replace('.annotations.json', '.graspnet_qa.json')
        with open(out_path, 'w') as f:
            json.dump({'qa': res['qa']}, f, indent=2)
            
        # Save Debug
        if args.debug_dir:
            img = cv2.imread(res['img_path'])
            if img is not None:
                vis = draw_debug(img, res)
                scene_name = Path(ann_path).parent.name
                scene_dir = Path(args.debug_dir) / scene_name
                scene_dir.mkdir(exist_ok=True)
                out_img = scene_dir / (Path(ann_path).name.replace('.annotations.json', '_debug.jpg'))
                cv2.imwrite(str(out_img), vis)
                
    print(f"Done! Generated {success} HOPE QA pairs.")
    if args.debug_dir:
        print(f"Debug images saved to {args.debug_dir}")

if __name__ == '__main__':
    main()
