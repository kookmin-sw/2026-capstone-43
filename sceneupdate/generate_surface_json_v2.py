#!/usr/bin/env python3
"""
Generate surface placement JSON using V2 (grid-based) analyzer.

Same as generate_surface_json.py but uses surface_analyzer_v2.
Output format is identical — free_spaces are coverage tiles instead of MER rects.

Usage:
  python generate_surface_json_v2.py
  python generate_surface_json_v2.py --obj data/obj_json_r_mapping_stride10.json \
                                      --edge data/edge_json_r_mapping_stride10_gemini_v2.json \
                                      --out data/surface_json_v2_r_mapping_stride10.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from surface_analyzer_v2 import SurfaceAnalyzer


def generate(obj_path: str, edge_path: str, output_path: str):
    with open(obj_path, 'r') as f:
        obj_data = json.load(f)
    print(f"[SurfaceV2] Loaded {len(obj_data)} objects from {obj_path}")

    edges = []
    pcd_objects = []
    if os.path.exists(edge_path):
        with open(edge_path, 'r') as f:
            edge_data = json.load(f)
        edges = edge_data.get("edges", edge_data if isinstance(edge_data, list) else [])
        pcd_objects = edge_data.get("objects", [])
        print(f"[SurfaceV2] Loaded {len(edges)} edges from {edge_path}"
              f" ({len(pcd_objects)} pcd objects)")

    if pcd_objects:
        import numpy as np
        pcd_to_objid = {}
        for pcd_obj in pcd_objects:
            pcd_idx = pcd_obj.get("pcd_index")
            pcd_center = pcd_obj.get("bbox_center")
            if pcd_idx is None or pcd_center is None:
                continue
            pcd_pos = np.array(pcd_center)
            best_id, best_dist = None, float('inf')
            for info in obj_data.values():
                obj_center = info.get("bbox_center", info.get("center"))
                if obj_center is None:
                    continue
                dist = float(np.linalg.norm(np.array(obj_center) - pcd_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_id = info.get("id")
            if best_id is not None and best_dist < 0.5:
                pcd_to_objid[pcd_idx] = best_id
        for edge in edges:
            src_idx = edge.get("src_index")
            dst_idx = edge.get("dst_index")
            if src_idx in pcd_to_objid:
                edge["src_obj_id"] = pcd_to_objid[src_idx]
            if dst_idx in pcd_to_objid:
                edge["dst_obj_id"] = pcd_to_objid[dst_idx]
        print(f"[SurfaceV2] Resolved {len(pcd_to_objid)} pcd->objid mappings")

    analyzer = SurfaceAnalyzer()
    analyzer.update(obj_data, edges)

    surfaces = []
    for surf_info in analyzer.surfaces.values():
        surfaces.append(surf_info.to_dict())

    surfaces.sort(key=lambda s: s["surface_id"])

    total_surfaces = len(surfaces)
    surfaces_with_objects = sum(1 for s in surfaces if s["objects_on_surface"])
    total_free_spaces = sum(len(s["free_spaces"]) for s in surfaces)
    all_on_objs = []
    for s in surfaces:
        all_on_objs.extend(s["objects_on_surface"])

    output = {
        "surfaces": surfaces,
        "metadata": {
            "n_surfaces": total_surfaces,
            "n_surfaces_with_objects": surfaces_with_objects,
            "n_objects_on_surfaces": len(all_on_objs),
            "n_free_spaces": total_free_spaces,
            "source_obj_json": os.path.basename(obj_path),
            "source_edge_json": os.path.basename(edge_path),
            "analyzer": "v2_grid",
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SurfaceV2] Generated: {output_path}")
    print(f"  Surfaces: {total_surfaces}")
    print(f"  With objects: {surfaces_with_objects}")
    print(f"  Total objects on surfaces: {len(all_on_objs)}")
    print(f"  Total free space tiles: {total_free_spaces}")

    print(f"\n{'='*60}")
    print(analyzer.get_graph_summary())

    return output


def main():
    from config import SG_JSON_FILE, EDGE_JSON_FILE
    SURFACE_JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "surface_json_v2_r_mapping_stride10.json")

    parser = argparse.ArgumentParser(description="Generate surface JSON (V2 grid-based)")
    parser.add_argument("--obj", default=SG_JSON_FILE, help="Path to object JSON")
    parser.add_argument("--edge", default=EDGE_JSON_FILE, help="Path to edge JSON")
    parser.add_argument("--out", default=SURFACE_JSON_FILE, help="Output path")
    args = parser.parse_args()

    generate(args.obj, args.edge, args.out)


if __name__ == "__main__":
    main()
