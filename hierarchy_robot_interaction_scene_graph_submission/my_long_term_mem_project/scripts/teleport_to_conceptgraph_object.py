#!/usr/bin/env python3

"""
Teleport the Fetch robot near a ConceptGraph object using its exported bbox_center.

This script reuses the same Habitat scene/robot setup as execute_LLM_plan.py.
It loads a ConceptGraph obj_json file, selects one object, snaps its bbox_center
to the nearest reachable navmesh point, teleports the robot there, and saves
debug camera views plus a small JSON report.
"""

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_OBJ_JSON_PATH = (
    "/home/yuchaehee/long_term_memory_project/my_local_data/hssd/"
    "108736872_177263607/exps/r_mapping_stride10/obj_json_r_mapping_stride10.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/yuchaehee/long_term_memory_project/"
    "my_long_term_mem_project/logs/conceptgraph_bbox_teleport"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a ConceptGraph obj_json entry, snap its bbox_center to the "
            "nearest reachable point, and teleport the Fetch robot there."
        )
    )
    parser.add_argument(
        "--obj-json",
        default=DEFAULT_OBJ_JSON_PATH,
        help="Path to ConceptGraph obj_json_*.json",
    )
    parser.add_argument(
        "--object-key",
        default=None,
        help="Exact obj_json key to use, for example object_1",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="Exact numeric object id from obj_json",
    )
    parser.add_argument(
        "--object-tag",
        default=None,
        help="Case-insensitive regex for object_tag",
    )
    parser.add_argument(
        "--object-caption",
        default=None,
        help="Case-insensitive regex for object_caption",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available objects and exit",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for debug images and JSON report",
    )
    return parser.parse_args()


def _load_obj_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected obj_json to be a dict, got {type(data).__name__}")
    return data


def _object_summary(obj_key: str, obj: dict) -> str:
    return (
        f"{obj_key}: id={obj.get('id')} | "
        f"tag={obj.get('object_tag', '')} | "
        f"center={obj.get('bbox_center')} | "
        f"extent={obj.get('bbox_extent')}"
    )


def _list_objects(obj_data: Dict[str, dict]):
    for obj_key, obj in obj_data.items():
        print(_object_summary(obj_key, obj))


def _compile_optional_regex(pattern: str):
    if pattern is None:
        return None
    return re.compile(pattern, flags=re.IGNORECASE)


def _select_object(args, obj_data: Dict[str, dict]) -> Tuple[str, dict]:
    if (
        args.object_key is None
        and args.object_id is None
        and args.object_tag is None
        and args.object_caption is None
    ):
        if not obj_data:
            raise ValueError("obj_json is empty.")
        first_key = next(iter(obj_data))
        print(
            "[conceptgraph] no selector provided, defaulting to "
            f"{first_key}. Use --list to inspect all objects."
        )
        return first_key, obj_data[first_key]

    tag_re = _compile_optional_regex(args.object_tag)
    caption_re = _compile_optional_regex(args.object_caption)

    matches: List[Tuple[str, dict]] = []
    for obj_key, obj in obj_data.items():
        if args.object_key is not None and obj_key != args.object_key:
            continue
        if args.object_id is not None and int(obj.get("id", -1)) != int(args.object_id):
            continue
        if tag_re is not None and not tag_re.search(str(obj.get("object_tag", ""))):
            continue
        if caption_re is not None and not caption_re.search(str(obj.get("object_caption", ""))):
            continue
        matches.append((obj_key, obj))

    if not matches:
        raise ValueError("No obj_json entries matched the requested selector.")

    if len(matches) > 1:
        preview = "\n".join(_object_summary(k, o) for k, o in matches[:10])
        raise ValueError(
            "Multiple objects matched. Please narrow it down with --object-key or --object-id.\n"
            f"{preview}"
        )

    return matches[0]


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return safe.strip("._") or "object"


def _is_finite_point(point) -> bool:
    arr = np.asarray(point, dtype=np.float32).reshape(-1)
    return arr.size == 3 and bool(np.all(np.isfinite(arr)))


def _distance_xz(a_xyz, b_xyz) -> float:
    a = np.asarray(a_xyz, dtype=np.float32)
    b = np.asarray(b_xyz, dtype=np.float32)
    return float(np.linalg.norm((a - b)[[0, 2]]))


def _snap_bbox_center_to_navmesh(karma_env, raw_target: np.ndarray, agent_idx: int = 0):
    pf = karma_env.sim.pathfinder
    if not pf.is_loaded:
        raise RuntimeError("NavMesh is not loaded in the simulator.")

    cur_pos, _ = karma_env.get_fetch_base_pose(karma_env.sim, agent_idx=agent_idx)
    cur_pos = np.asarray(cur_pos, dtype=np.float32)
    raw_target = np.asarray(raw_target, dtype=np.float32)

    candidates = []

    # Best-effort snap on the agent's current floor height.
    same_floor_target = np.array([raw_target[0], cur_pos[1], raw_target[2]], dtype=np.float32)
    try:
        snap_same_floor = np.array(pf.snap_point(same_floor_target), dtype=np.float32)
        if _is_finite_point(snap_same_floor):
            candidates.append(
                {
                    "source": "pathfinder.snap_point_same_floor",
                    "point": snap_same_floor,
                    "target_gap_xz": _distance_xz(raw_target, snap_same_floor),
                }
            )
    except Exception:
        pass

    # Also try snapping the raw 3D center directly.
    try:
        snap_raw = np.array(pf.snap_point(raw_target), dtype=np.float32)
        if _is_finite_point(snap_raw):
            candidates.append(
                {
                    "source": "pathfinder.snap_point_raw_center",
                    "point": snap_raw,
                    "target_gap_xz": _distance_xz(raw_target, snap_raw),
                }
            )
    except Exception:
        pass

    # Reuse the existing KARMA helper as a robust fallback.
    try:
        proj_goal, proj_source, geo_dist, obj_gap = karma_env.project_goal_to_navmesh(
            karma_env.sim,
            raw_target,
            agent_idx=agent_idx,
        )
        proj_goal = np.asarray(proj_goal, dtype=np.float32)
        if _is_finite_point(proj_goal):
            candidates.append(
                {
                    "source": f"project_goal_to_navmesh:{proj_source}",
                    "point": proj_goal,
                    "target_gap_xz": float(obj_gap),
                    "geodesic_distance": float(geo_dist),
                }
            )
    except Exception:
        pass

    if not candidates:
        raise RuntimeError(
            f"Could not snap bbox_center to navmesh. raw_target={raw_target.tolist()}"
        )

    candidates.sort(key=lambda c: (c["target_gap_xz"], c.get("geodesic_distance", float("inf"))))
    return candidates[0], candidates


def _save_debug_views(karma_env, obs, output_dir: str, stem: str):
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    for suffix in ("head_rgb", "head_down_rgb", "third_rgb", "scene_camera_rgb"):
        rgb_obs = karma_env._obs_by_suffix(obs, suffix)
        if rgb_obs is None:
            continue
        filename = f"{stem}_{suffix}.png"
        karma_env.save_rgb_observation_png(rgb_obs, output_dir, filename)
        saved[suffix] = os.path.join(output_dir, filename)
    return saved


def main():
    args = _parse_args()

    if args.list:
        obj_data = _load_obj_json(args.obj_json)
        _list_objects(obj_data)
        return

    os.environ.setdefault("KARMA_INTERACTIVE_VIEW", "0")
    os.environ.setdefault("KARMA_START_TASK_EXECUTOR", "0")
    os.environ.setdefault("KARMA_ENABLE_HEAD_PANOPTIC", "0")

    import execute_LLM_plan as karma_env

    obj_data = _load_obj_json(args.obj_json)
    obj_key, obj = _select_object(args, obj_data)

    raw_target = np.asarray(obj["bbox_center"], dtype=np.float32)
    if raw_target.shape != (3,):
        raise ValueError(f"bbox_center must have length 3, got {obj['bbox_center']}")

    start_pos, start_yaw = karma_env.get_fetch_base_pose(karma_env.sim, agent_idx=0)
    best_snap, all_candidates = _snap_bbox_center_to_navmesh(karma_env, raw_target, agent_idx=0)

    nav_goal = np.asarray(best_snap["point"], dtype=np.float32)
    yaw_to_target = float(np.arctan2(raw_target[2] - nav_goal[2], raw_target[0] - nav_goal[0]))

    karma_env.obs = karma_env.teleport_fetch_base(
        karma_env.sim,
        karma_env.env,
        karma_env.EMPTY_ACTION,
        nav_goal,
        yaw_rad=yaw_to_target,
        agent_idx=0,
    )

    final_pos, final_yaw = karma_env.get_fetch_base_pose(karma_env.sim, agent_idx=0)

    stem = _sanitize_filename(f"{obj_key}_{obj.get('object_tag', 'object')}")
    image_paths = _save_debug_views(karma_env, karma_env.obs, args.output_dir, stem)

    report = {
        "obj_json_path": os.path.abspath(args.obj_json),
        "selected_object_key": obj_key,
        "selected_object": obj,
        "raw_bbox_center": raw_target.tolist(),
        "start_base_pos": np.asarray(start_pos, dtype=np.float32).tolist(),
        "start_base_yaw_rad": float(start_yaw),
        "teleport_goal": nav_goal.tolist(),
        "teleport_yaw_rad": float(yaw_to_target),
        "final_base_pos": np.asarray(final_pos, dtype=np.float32).tolist(),
        "final_base_yaw_rad": float(final_yaw),
        "target_gap_xz_m": _distance_xz(raw_target, nav_goal),
        "snap_choice": {
            "source": best_snap["source"],
            "target_gap_xz": float(best_snap["target_gap_xz"]),
            "geodesic_distance": best_snap.get("geodesic_distance"),
        },
        "all_snap_candidates": [
            {
                "source": cand["source"],
                "point": np.asarray(cand["point"], dtype=np.float32).tolist(),
                "target_gap_xz": float(cand["target_gap_xz"]),
                "geodesic_distance": cand.get("geodesic_distance"),
            }
            for cand in all_candidates
        ],
        "saved_images": image_paths,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{stem}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[conceptgraph] selected: {_object_summary(obj_key, obj)}")
    print(
        "[conceptgraph] snap: "
        f"source={best_snap['source']}, "
        f"raw_center={np.round(raw_target, 3).tolist()}, "
        f"goal={np.round(nav_goal, 3).tolist()}, "
        f"target_gap_xz={_distance_xz(raw_target, nav_goal):.3f}m"
    )
    print(
        "[conceptgraph] base: "
        f"start={np.round(np.asarray(start_pos, dtype=np.float32), 3).tolist()}, "
        f"final={np.round(np.asarray(final_pos, dtype=np.float32), 3).tolist()}, "
        f"yaw_deg={math.degrees(yaw_to_target):.1f}"
    )
    if image_paths:
        print(f"[conceptgraph] saved_images={image_paths}")
    print(f"[conceptgraph] report={report_path}")


if __name__ == "__main__":
    main()
