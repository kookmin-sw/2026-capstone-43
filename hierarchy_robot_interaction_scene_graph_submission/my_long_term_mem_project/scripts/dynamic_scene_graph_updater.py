import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except Exception:  # pragma: no cover - graceful fallback if Open3D is unavailable
    o3d = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from actions_utils import (
        DEFAULT_CONCEPTGRAPH_EXPERIMENT,
        DEFAULT_CONCEPTGRAPH_ROOT,
        DEFAULT_HSG_DATASET,
        DEFAULT_HSG_ROOT,
        load_conceptgraph_index,
        load_hierarchical_graph_index,
    )
except ModuleNotFoundError:
    # 실행 위치가 repo root일 때는 scripts 디렉토리가 import path에 없을 수 있다.
    # 이 fallback 덕분에 standalone 테스트와 실제 파이프라인 둘 다 같은 모듈을 쓸 수 있다.
    from my_long_term_mem_project.scripts.actions_utils import (
        DEFAULT_CONCEPTGRAPH_EXPERIMENT,
        DEFAULT_CONCEPTGRAPH_ROOT,
        DEFAULT_HSG_DATASET,
        DEFAULT_HSG_ROOT,
        load_conceptgraph_index,
        load_hierarchical_graph_index,
    )


# 왜 별도 상태 디렉토리를 두는가:
# - ConceptGraph / HSG 결과물은 "초기 정적 장면"을 표현하는 base memory이다.
# - dynamic adaptation은 그 위에 덧입히는 overlay이므로, 원본 파일을 직접 덮어쓰기보다
#   현재 상태를 별도 JSON + current_objects 폴더에 저장하는 편이 안전하고 디버깅이 쉽다.
DYNAMIC_SCENE_GRAPH_ROOT = Path(
    "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/dynamic_scene_graph"
)


# 이동량 기반 change type 분류 기준
# - action context가 있으므로 삭제/신규 생성으로 오판하지 않도록 하되,
#   "약간 움직였는가"와 "관계가 바뀔 정도로 크게 움직였는가"는 분리해서 남긴다.
MINOR_ADJUSTMENT_THRESH_M = 0.35
WORLD_VISIBLE_OBJECT_STATUSES = {"active"}
# frame-wide direct update는 현재 프레임에서 "지금 존재한다고 가정하는" active 객체에만 적용한다.
# missing/deleted는 residual cluster 재식별 단계에서 복구를 시도한다.
PERCEPTION_CANDIDATE_OBJECT_STATUSES = {"active"}
LOCAL_RGBD_RECONSTRUCTION_VOXEL_SIZE_M = 0.01
LOCAL_RGBD_RECONSTRUCTION_MIN_POINTS = 40
LOCAL_RGBD_RECONSTRUCTION_DBSCAN_EPS_M = 0.035
LOCAL_RGBD_RECONSTRUCTION_DBSCAN_MIN_POINTS = 20
LOCAL_RGBD_RECONSTRUCTION_MARGIN_M = 0.04
LOCAL_RGBD_RECONSTRUCTION_MIN_HALF_EXTENT_M = np.array([0.04, 0.03, 0.04], dtype=np.float32)
LOCAL_RGBD_RECONSTRUCTION_CROP_SCALE = np.array([1.8, 1.4, 1.8], dtype=np.float32)
LOCAL_RGBD_RECONSTRUCTION_ROI_PAD_PX = 24
LOCAL_RGBD_RECONSTRUCTION_MIN_ROI_RADIUS_PX = 48
LOCAL_RGBD_RECONSTRUCTION_MIN_ROI_AREA_PX = 64

# "보였어야 했는데 안 보임"을 누적해서 missing/deleted를 구분한다.
# - 1회 miss: missing
# - N회 연속 miss: deleted
MISSING_TO_DELETED_COUNT = 3
MISSING_STATUS_SET = {"missing", "deleted"}

# -----------------------------------------------------------------------------
# Depth clip 설정
# -----------------------------------------------------------------------------
# ConceptGraph 파이프라인과 맞추기 위해, dynamic update에서도 동일하게 원거리 depth를 무효화한다.
# 중요:
# - "projection(시야 판단)" 자체는 그대로 유지한다.
# - depth clip은 3D backprojection 직전에 적용한다.
#   즉, 카메라 기하 판단은 유지하면서 point cloud 생성에만 제한을 둔다.
DEPTH_CLIP_ENABLED = True
DEPTH_CLIP_MAX_M = 4.0

# -----------------------------------------------------------------------------
# 부분 관측(partial observation) 보호 로직
# -----------------------------------------------------------------------------
# 문 너머/원거리 물체는 depth clip 후 앞부분만 남을 수 있다.
# 이때 그대로 center/bbox를 갱신하면 위치가 카메라 쪽으로 당겨지는 문제가 생긴다.
# 따라서 "원거리 + 축소 관측" 패턴이면 geometry 덮어쓰기를 건너뛴다.
PARTIAL_OBSERVATION_NEAR_CLIP_RATIO = 0.85
PARTIAL_OBSERVATION_MIN_VOLUME_RATIO = 0.35
PARTIAL_OBSERVATION_MIN_EXTENT_RATIO = 0.50
PARTIAL_OBSERVATION_MIN_POINTS = 140

# 투영 깊이 대비 현재 depth가 훨씬 앞쪽이면(작은 z) 다른 물체가 가린 것으로 본다.
# 이런 경우 missing 처리하지 않고 skip한다.
OCCLUSION_DEPTH_MARGIN_M = 0.18
OCCLUSION_DEPTH_SAMPLING_RADIUS_PX = 2

# DovSG-style point-wise depth/color evidence.
# - reconstructed 실패를 곧바로 missing으로 보내지 않고,
#   "실제로 없어졌다는 깊이 증거"가 충분할 때만 missing 전이한다.
DOVSG_OBS_MIN_PROJECTED_PIXELS = 40
DOVSG_OBS_MIN_VALID_DEPTH_PIXELS = 30
DOVSG_OBS_DEPTH_DELETE_THRESH_M = 0.08
DOVSG_OBS_DEPTH_COLOR_THRESH_M = 0.04
DOVSG_OBS_COLOR_DIFF_THRESH = 0.18
DOVSG_OBS_SUPPORT_DEPTH_TOL_M = 0.05
DOVSG_OBS_MISSING_RATIO_MARK = 0.45
# 지원 비율 상한을 소폭 완화(0.15 -> 0.20):
# - 실제로 제거된 물체가 일부 픽셀 support를 남기는 경우(노이즈/부분 관측)도
#   missing_evidence로 넘어갈 수 있게 한다.
# - 너무 크게 올리면 false missing 위험이 커지므로 우선 0.20으로 제한한다.
DOVSG_OBS_SUPPORT_RATIO_MAX_FOR_MISS = 0.20


def _scene_key_from_scene_id(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(str(scene_id)))[0]


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(str(value).strip())
        except Exception:
            return None


def _source_key(source_id: int) -> str:
    return str(int(source_id))


def _jsonify(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def _as_point3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    arr = arr[:3]
    if not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32)


def _as_feature_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0:
        return None
    valid = np.isfinite(arr)
    if not np.any(valid):
        return None
    arr = np.where(valid, arr, 0.0).astype(np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def _relationship_label(raw: Optional[str]) -> str:
    if not raw:
        return "related_to"
    rel = str(raw).strip().lower()
    if rel == "on top of":
        return "on_top_of"
    return rel.replace(" ", "_")


def _conceptgraph_edge_json_path(
    scene_id: str,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Path:
    scene_key = _scene_key_from_scene_id(scene_id)
    return (
        Path(conceptgraph_root)
        / scene_key
        / "exps"
        / experiment_name
        / f"edge_json_{experiment_name}.json"
    )


def _load_conceptgraph_edges(
    scene_id: str,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Dict[str, Dict[str, Any]]:
    path = _conceptgraph_edge_json_path(
        scene_id=scene_id,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )
    if not path.exists():
        return {}
    raw = _read_json(path)
    if not isinstance(raw, dict):
        return {}
    return raw


def _polygon_centroid(vertices: List[List[float]]) -> Optional[np.ndarray]:
    if not vertices:
        return None
    pts = np.asarray(vertices, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    return pts[:, :2].mean(axis=0)


def _point_in_polygon_2d(point_xz: np.ndarray, polygon_xz: List[List[float]]) -> bool:
    # Ray casting algorithm.
    if len(polygon_xz) < 3:
        return False
    x = float(point_xz[0])
    z = float(point_xz[1])
    inside = False
    n = len(polygon_xz)
    for i in range(n):
        x1, z1 = polygon_xz[i][0], polygon_xz[i][1]
        x2, z2 = polygon_xz[(i + 1) % n][0], polygon_xz[(i + 1) % n][1]
        intersects = ((z1 > z) != (z2 > z)) and (
            x < (x2 - x1) * (z - z1) / ((z2 - z1) + 1e-8) + x1
        )
        if intersects:
            inside = not inside
    return inside


def _choose_room_from_center(
    center_xyz: Optional[np.ndarray],
    graph_index: Dict[str, Any],
    preferred_room_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    rooms_by_id = graph_index.get("rooms_by_id", {})
    if center_xyz is None:
        room = rooms_by_id.get(str(preferred_room_id)) if preferred_room_id is not None else None
        if room is None:
            return None, None, None
        return room.get("room_id"), room.get("name"), room.get("floor_id")

    point_xz = np.asarray([center_xyz[0], center_xyz[2]], dtype=np.float32)

    if preferred_room_id is not None:
        preferred_room = rooms_by_id.get(str(preferred_room_id))
        if preferred_room and _point_in_polygon_2d(point_xz, preferred_room.get("vertices", [])):
            return (
                preferred_room.get("room_id"),
                preferred_room.get("name"),
                preferred_room.get("floor_id"),
            )

    containing = []
    nearest = []
    for room in rooms_by_id.values():
        vertices = room.get("vertices", [])
        if _point_in_polygon_2d(point_xz, vertices):
            containing.append(room)
            continue
        centroid = _polygon_centroid(vertices)
        if centroid is None:
            continue
        dist = float(np.linalg.norm(point_xz - centroid))
        nearest.append((dist, room))

    if containing:
        room = containing[0]
        return room.get("room_id"), room.get("name"), room.get("floor_id")

    if nearest:
        nearest.sort(key=lambda item: item[0])
        room = nearest[0][1]
        return room.get("room_id"), room.get("name"), room.get("floor_id")

    return None, None, None


def _movement_change_type(
    old_center: Optional[np.ndarray],
    new_center: Optional[np.ndarray],
    *,
    default: str = "reobserved",
) -> str:
    if old_center is None or new_center is None:
        return default
    dist = float(np.linalg.norm(new_center - old_center))
    if dist < 1e-6:
        return "reobserved"
    if dist < MINOR_ADJUSTMENT_THRESH_M:
        return "minor_adjustment"
    return "positional_shift"


def _as_matrix4(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        mat = np.asarray(value, dtype=np.float32).reshape(4, 4)
    except Exception:
        return None
    if not np.all(np.isfinite(mat)):
        return None
    return mat


def _load_event_rgbd_payload(event_dir: Path, event_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    meta = event_meta
    if meta is None:
        meta_path = event_dir / "meta.json"
        if not meta_path.exists():
            return None
        meta = _read_json(meta_path)

    depth_path = meta.get("depth_npy_path")
    rgb_path = meta.get("rgb_path")
    intrinsics = meta.get("intrinsics")
    raw_camera_to_world = _as_matrix4(meta.get("camera_transform_world_to_opencv_camera"))

    if not depth_path or not Path(depth_path).exists():
        return None
    if not isinstance(intrinsics, dict):
        return None
    if raw_camera_to_world is None:
        return None

    depth_m = np.asarray(np.load(depth_path), dtype=np.float32)
    if depth_m.ndim != 2:
        depth_m = np.squeeze(depth_m)
    if depth_m.ndim != 2:
        return None

    # -------------------------------------------------------------------------
    # Depth clip 적용:
    # - 원거리(depth > DEPTH_CLIP_MAX_M) 픽셀을 invalid(0)로 처리한다.
    # - pose 오차에 민감한 원거리 포인트를 줄여서 false missing/new_object를 완화한다.
    # -------------------------------------------------------------------------
    valid_before_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped_count = 0
    if DEPTH_CLIP_ENABLED:
        clip_mask = valid_before_mask & (depth_m > float(DEPTH_CLIP_MAX_M))
        clipped_count = int(np.count_nonzero(clip_mask))
        if clipped_count > 0:
            depth_m = depth_m.copy()
            depth_m[clip_mask] = 0.0

    valid_after_mask = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_before_count = int(np.count_nonzero(valid_before_mask))
    valid_after_count = int(np.count_nonzero(valid_after_mask))

    rgb_bgr = None
    if rgb_path and Path(rgb_path).exists():
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    depth_clip_info = {
        "enabled": bool(DEPTH_CLIP_ENABLED),
        "max_depth_m": float(DEPTH_CLIP_MAX_M),
        "valid_before_count": valid_before_count,
        "valid_after_count": valid_after_count,
        "clipped_count": int(clipped_count),
        "clipped_ratio": (
            float(clipped_count) / float(max(valid_before_count, 1))
            if DEPTH_CLIP_ENABLED
            else 0.0
        ),
    }

    return {
        "meta": meta,
        "depth_m": depth_m,
        "rgb_bgr": rgb_bgr,
        "intrinsics": intrinsics,
        "raw_camera_to_world": raw_camera_to_world,
        "depth_clip_info": depth_clip_info,
    }


def _project_world_point(
    point_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: Dict[str, Any],
) -> Optional[np.ndarray]:
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError:
        return None

    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = np.asarray(point_world, dtype=np.float32)
    point_camera = world_to_camera @ point_h
    z = float(point_camera[2])
    if z <= 1e-5:
        return None

    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        return None

    u = fx * float(point_camera[0]) / z + cx
    v = fy * float(point_camera[1]) / z + cy
    return np.asarray([u, v, z], dtype=np.float32)


def _projection_fit_score(
    projected_uvz: Optional[np.ndarray],
    *,
    width: int,
    height: int,
) -> float:
    if projected_uvz is None:
        return -1e9
    u, v, z = [float(x) for x in projected_uvz]
    if z <= 1e-5:
        return -1e9

    inside = 0.0 <= u < float(width) and 0.0 <= v < float(height)
    overflow = (
        max(0.0, -u)
        + max(0.0, u - float(width - 1))
        + max(0.0, -v)
        + max(0.0, v - float(height - 1))
    )
    center_penalty = abs(u - float(width) * 0.5) + abs(v - float(height) * 0.5)
    return (1000.0 if inside else 0.0) - overflow - 0.001 * center_penalty


def _resolve_camera_to_world_matrix(
    *,
    raw_camera_to_world: np.ndarray,
    intrinsics: Dict[str, Any],
    image_shape: Tuple[int, int],
    reference_center_world: Optional[np.ndarray],
) -> Tuple[np.ndarray, str]:
    if reference_center_world is None:
        return raw_camera_to_world, "camera_to_world_as_recorded"

    height, width = image_shape
    candidates: List[Tuple[str, np.ndarray]] = [("camera_to_world_as_recorded", raw_camera_to_world)]
    try:
        candidates.append(("camera_to_world_from_inverse", np.linalg.inv(raw_camera_to_world)))
    except np.linalg.LinAlgError:
        pass

    best_label = candidates[0][0]
    best_matrix = candidates[0][1]
    best_score = -1e9
    for label, candidate in candidates:
        projected = _project_world_point(reference_center_world, candidate, intrinsics)
        score = _projection_fit_score(projected, width=width, height=height)
        if score > best_score:
            best_score = score
            best_label = label
            best_matrix = candidate

    return best_matrix, best_label


def _estimate_object_extent(obj_state: Dict[str, Any]) -> np.ndarray:
    extent = _as_point3(obj_state.get("bbox_extent"))
    if extent is not None and float(np.min(extent)) > 1e-4:
        return extent.astype(np.float32)

    source_path = _current_object_pcd_path(obj_state)
    if source_path and o3d is not None:
        pcd = o3d.io.read_point_cloud(str(source_path))
        if pcd is not None and len(pcd.points) > 0:
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = np.asarray(bbox.get_extent(), dtype=np.float32)
            if float(np.min(extent)) > 1e-4:
                return extent

    return np.asarray([0.12, 0.12, 0.12], dtype=np.float32)


def _fuse_bbox_extent(
    prior_extent: Optional[Any],
    observed_extent: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    observed = _as_point3(observed_extent)
    if observed is None:
        return _as_point3(prior_extent)
    prior = _as_point3(prior_extent)
    if prior is None:
        return observed
    return np.maximum(prior, observed).astype(np.float32)


def _reconstruction_half_extent(extent: np.ndarray) -> np.ndarray:
    half_extent = 0.5 * np.asarray(extent, dtype=np.float32)
    half_extent = np.maximum(
        half_extent * LOCAL_RGBD_RECONSTRUCTION_CROP_SCALE + LOCAL_RGBD_RECONSTRUCTION_MARGIN_M,
        LOCAL_RGBD_RECONSTRUCTION_MIN_HALF_EXTENT_M,
    )
    return half_extent.astype(np.float32)


def _compute_reconstruction_roi(
    *,
    center_world: np.ndarray,
    half_extent_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: Dict[str, Any],
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    projected = _project_world_point(center_world, camera_to_world, intrinsics)
    if projected is None:
        return None

    u, v, z = [float(x) for x in projected]
    height, width = image_shape
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        return None

    half_extent_max = float(np.max(half_extent_world))
    radius_px = int(
        np.ceil(max(fx, fy) * half_extent_max / max(z, 1e-3))
    ) + LOCAL_RGBD_RECONSTRUCTION_ROI_PAD_PX
    radius_px = max(radius_px, LOCAL_RGBD_RECONSTRUCTION_MIN_ROI_RADIUS_PX)

    x0 = max(0, int(np.floor(u)) - radius_px)
    x1 = min(width, int(np.ceil(u)) + radius_px + 1)
    y0 = max(0, int(np.floor(v)) - radius_px)
    y1 = min(height, int(np.ceil(v)) + radius_px + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _backproject_depth_roi_to_world(
    *,
    depth_m: np.ndarray,
    rgb_bgr: Optional[np.ndarray],
    intrinsics: Dict[str, Any],
    camera_to_world: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    x0, y0, x1, y1 = roi
    depth_roi = depth_m[y0:y1, x0:x1]
    if depth_roi.size <= 0:
        return np.empty((0, 3), dtype=np.float32), None

    valid = np.isfinite(depth_roi) & (depth_roi > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), None

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys, indexing="xy")

    z = depth_roi[valid].astype(np.float32)
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=-1)

    rotation = camera_to_world[:3, :3].astype(np.float32)
    translation = camera_to_world[:3, 3].astype(np.float32)
    points_world = points_camera @ rotation.T + translation

    colors = None
    if rgb_bgr is not None and rgb_bgr.shape[:2] == depth_m.shape[:2]:
        rgb_roi = rgb_bgr[y0:y1, x0:x1, ::-1].astype(np.float32) / 255.0
        colors = rgb_roi[valid]

    return points_world.astype(np.float32), colors


def _sample_valid_depth_near_pixel(
    depth_m: np.ndarray,
    *,
    u: float,
    v: float,
    radius_px: int = OCCLUSION_DEPTH_SAMPLING_RADIUS_PX,
) -> Optional[float]:
    """
    (u, v) 주변의 유효 depth를 robust하게 추정한다.

    단일 픽셀 값은 노이즈/홀에 취약하므로 주변 window의 median을 사용한다.
    """
    if depth_m.ndim != 2:
        return None
    h, w = depth_m.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    x0 = max(0, ui - radius_px)
    x1 = min(w, ui + radius_px + 1)
    y0 = max(0, vi - radius_px)
    y1 = min(h, vi + radius_px + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size <= 0:
        return None
    return float(np.median(valid))


def _project_object_points_to_pixel_min_depth(
    *,
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: Dict[str, Any],
    image_shape: Tuple[int, int],
) -> Optional[Dict[str, np.ndarray]]:
    """
    객체 point cloud를 현재 카메라로 투영하고, pixel별 최소 depth를 만든다.

    DovSG의 obsolete index 추출 단계와 같은 핵심 아이디어:
    - 3D point를 카메라 좌표로 변환
    - image plane으로 투영
    - 같은 픽셀에 여러 점이 걸리면 가장 가까운 depth만 사용
    """
    if points_world.size <= 0:
        return None

    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        return None

    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError:
        return None

    h, w = image_shape
    rotation = world_to_camera[:3, :3].astype(np.float32)
    translation = world_to_camera[:3, 3].astype(np.float32)
    points_camera = points_world @ rotation.T + translation
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    front = (z > 1e-5) & finite
    if not np.any(front):
        return None

    u = fx * x / np.maximum(z, 1e-6) + cx
    v = fy * y / np.maximum(z, 1e-6) + cy
    finite_uv = np.isfinite(u) & np.isfinite(v)
    front = front & finite_uv
    if not np.any(front):
        return None

    # cast 경고를 피하기 위해 front 점들만 정수 픽셀로 변환한다.
    front_idx = np.nonzero(front)[0]
    u_front = u[front]
    v_front = v[front]
    z_front = z[front]

    ui_front = np.rint(u_front).astype(np.int32)
    vi_front = np.rint(v_front).astype(np.int32)
    inside_front = (ui_front >= 0) & (ui_front < w) & (vi_front >= 0) & (vi_front < h)
    inside = np.zeros_like(front, dtype=bool)
    inside[front_idx[inside_front]] = True
    if not np.any(inside):
        return None

    ui = ui_front[inside_front]
    vi = vi_front[inside_front]
    z = z_front[inside_front].astype(np.float32)

    flat = vi.astype(np.int64) * int(w) + ui.astype(np.int64)
    order = np.argsort(z)  # near -> far
    flat_sorted = flat[order]
    # 각 pixel의 첫 등장(가장 가까운 z)만 유지
    _, unique_first = np.unique(flat_sorted, return_index=True)
    keep = order[unique_first]

    return {
        "u": ui[keep].astype(np.int32),
        "v": vi[keep].astype(np.int32),
        "z": z[keep].astype(np.float32),
        "point_index": front_idx[inside_front][keep].astype(np.int32),
    }


def _dovsg_missing_evidence_for_object(
    *,
    obj_state: Dict[str, Any],
    camera_to_world: np.ndarray,
    intrinsics: Dict[str, Any],
    image_shape: Tuple[int, int],
    depth_m: np.ndarray,
    rgb_bgr: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    DovSG 스타일 point-wise depth/color 차이로 missing 근거를 계산한다.

    반환 decision:
    - `missing_evidence`: 실제 소실 근거가 강함 (missing 전이 가능)
    - `observed_or_occluded`: 관측됨/가려짐 가능성이 높음 (missing 금지)
    - `unknown`: 근거 부족 (보수적으로 missing 금지)
    """
    if o3d is None:
        return {"decision": "unknown", "reason": "open3d_unavailable"}

    source_path = _current_object_pcd_path(obj_state)
    if not source_path:
        return {"decision": "unknown", "reason": "missing_object_pcd_path"}

    pcd = o3d.io.read_point_cloud(str(source_path))
    if pcd is None or len(pcd.points) <= 0:
        return {"decision": "unknown", "reason": "empty_object_pcd"}

    points_world = np.asarray(pcd.points, dtype=np.float32)
    projected = _project_object_points_to_pixel_min_depth(
        points_world=points_world,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        image_shape=image_shape,
    )
    if projected is None:
        return {"decision": "unknown", "reason": "projection_failed_or_outside"}

    projected_count = int(projected["z"].shape[0])
    if projected_count < DOVSG_OBS_MIN_PROJECTED_PIXELS:
        return {
            "decision": "unknown",
            "reason": "too_few_projected_pixels",
            "projected_count": projected_count,
        }

    u = projected["u"]
    v = projected["v"]
    z_obj = projected["z"]
    obs_depth = depth_m[v, u].astype(np.float32)
    valid_depth = np.isfinite(obs_depth) & (obs_depth > 0.0)
    valid_count = int(np.count_nonzero(valid_depth))
    if valid_count < DOVSG_OBS_MIN_VALID_DEPTH_PIXELS:
        return {
            "decision": "unknown",
            "reason": "too_few_valid_observed_depth",
            "projected_count": projected_count,
            "valid_depth_count": valid_count,
        }

    u = u[valid_depth]
    v = v[valid_depth]
    z_obj = z_obj[valid_depth]
    obs_depth = obs_depth[valid_depth]
    depth_differ = obs_depth - z_obj

    missing_mask = depth_differ > float(DOVSG_OBS_DEPTH_DELETE_THRESH_M)
    secondary_mask = depth_differ > float(DOVSG_OBS_DEPTH_COLOR_THRESH_M)

    # DovSG의 보조 조건(depth + color)을 간단히 반영한다.
    if rgb_bgr is not None and pcd.has_colors():
        obj_colors = np.asarray(pcd.colors, dtype=np.float32)
        point_index = projected["point_index"][valid_depth]
        if obj_colors.shape[0] == points_world.shape[0]:
            obj_rgb = obj_colors[point_index]
            obs_rgb = rgb_bgr[v, u, ::-1].astype(np.float32) / 255.0
            color_differ = np.linalg.norm(obj_rgb - obs_rgb, axis=1)
            secondary_mask = secondary_mask & (color_differ > float(DOVSG_OBS_COLOR_DIFF_THRESH))
        else:
            color_differ = None
    else:
        color_differ = None

    delete_evidence = missing_mask | secondary_mask
    support_mask = np.abs(depth_differ) <= float(DOVSG_OBS_SUPPORT_DEPTH_TOL_M)
    occlusion_like_mask = depth_differ < -float(OCCLUSION_DEPTH_MARGIN_M)

    miss_ratio = float(np.mean(delete_evidence)) if delete_evidence.size > 0 else 0.0
    support_ratio = float(np.mean(support_mask)) if support_mask.size > 0 else 0.0
    occlusion_ratio = float(np.mean(occlusion_like_mask)) if occlusion_like_mask.size > 0 else 0.0

    # missing은 "삭제 근거가 충분하고, 동시에 support가 약할 때"만 발생시킨다.
    mark_missing = (
        miss_ratio >= float(DOVSG_OBS_MISSING_RATIO_MARK)
        and support_ratio <= float(DOVSG_OBS_SUPPORT_RATIO_MAX_FOR_MISS)
    )

    decision = "missing_evidence" if mark_missing else "observed_or_occluded"
    return {
        "decision": decision,
        "reason": "depth_color_projection_consistency",
        "projected_count": projected_count,
        "valid_depth_count": valid_count,
        "miss_ratio": miss_ratio,
        "support_ratio": support_ratio,
        "occlusion_ratio": occlusion_ratio,
        "mean_depth_differ": float(np.mean(depth_differ)) if depth_differ.size > 0 else None,
        "color_differ_mean": (
            None if color_differ is None or color_differ.size <= 0 else float(np.mean(color_differ))
        ),
    }


def _crop_points_near_center(
    *,
    points_world: np.ndarray,
    colors: Optional[np.ndarray],
    center_world: np.ndarray,
    extent_world: np.ndarray,
    half_extent_world: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points_world.size <= 0:
        return points_world, colors

    lower = center_world - half_extent_world
    upper = center_world + half_extent_world

    # 수직 방향(y)은 support surface가 섞이는 것을 줄이기 위해 조금 더 보수적으로 자른다.
    tight_half_y_lower = max(float(extent_world[1]) * 0.55, float(LOCAL_RGBD_RECONSTRUCTION_MIN_HALF_EXTENT_M[1]))
    tight_half_y_upper = max(float(extent_world[1]) * 0.70, float(LOCAL_RGBD_RECONSTRUCTION_MIN_HALF_EXTENT_M[1]))
    lower[1] = center_world[1] - tight_half_y_lower - LOCAL_RGBD_RECONSTRUCTION_MARGIN_M
    upper[1] = center_world[1] + tight_half_y_upper + LOCAL_RGBD_RECONSTRUCTION_MARGIN_M

    aabb_mask = np.all(points_world >= lower[None, :], axis=1) & np.all(
        points_world <= upper[None, :], axis=1
    )
    radius = max(float(np.linalg.norm(half_extent_world)), 0.08)
    radius_mask = np.linalg.norm(points_world - center_world[None, :], axis=1) <= radius * 1.2
    keep = aabb_mask & radius_mask

    cropped_points = points_world[keep]
    if colors is None:
        return cropped_points, None
    return cropped_points, colors[keep]


def _select_cluster_near_center(
    pcd: "o3d.geometry.PointCloud",
    *,
    center_world: np.ndarray,
) -> "o3d.geometry.PointCloud":
    if o3d is None or len(pcd.points) < LOCAL_RGBD_RECONSTRUCTION_DBSCAN_MIN_POINTS:
        return pcd

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=LOCAL_RGBD_RECONSTRUCTION_DBSCAN_EPS_M,
            min_points=LOCAL_RGBD_RECONSTRUCTION_DBSCAN_MIN_POINTS,
            print_progress=False,
        )
    )
    valid_labels = [int(label) for label in np.unique(labels) if int(label) >= 0]
    if not valid_labels:
        return pcd

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    best_label = valid_labels[0]
    best_score = float("inf")
    for label in valid_labels:
        cluster_points = points[labels == label]
        if cluster_points.shape[0] <= 0:
            continue
        centroid = cluster_points.mean(axis=0)
        dist = float(np.linalg.norm(centroid - center_world))
        score = dist - 0.0005 * float(cluster_points.shape[0])
        if score < best_score:
            best_score = score
            best_label = label

    keep = labels == best_label
    selected = o3d.geometry.PointCloud()
    selected.points = o3d.utility.Vector3dVector(points[keep])
    if colors is not None and colors.shape[0] == points.shape[0]:
        selected.colors = o3d.utility.Vector3dVector(colors[keep])
    return selected


def _reconstruct_object_pcd_from_event(
    *,
    event_dir: Path,
    event_meta: Optional[Dict[str, Any]],
    obj_state: Dict[str, Any],
    expected_center_world: np.ndarray,
    perception_payload: Optional[Dict[str, Any]] = None,
    camera_to_world_override: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    if o3d is None:
        return None

    payload = perception_payload or _load_event_rgbd_payload(event_dir, event_meta=event_meta)
    if payload is None:
        return None

    depth_m = payload["depth_m"]
    rgb_bgr = payload["rgb_bgr"]
    intrinsics = payload["intrinsics"]
    raw_camera_to_world = payload["raw_camera_to_world"]

    if camera_to_world_override is not None:
        camera_to_world = np.asarray(camera_to_world_override, dtype=np.float32).reshape(4, 4)
        pose_interpretation = "camera_to_world_shared_for_event"
    else:
        camera_to_world, pose_interpretation = _resolve_camera_to_world_matrix(
            raw_camera_to_world=raw_camera_to_world,
            intrinsics=intrinsics,
            image_shape=depth_m.shape[:2],
            reference_center_world=expected_center_world,
        )

    extent_world = _estimate_object_extent(obj_state)
    half_extent_world = _reconstruction_half_extent(extent_world)
    roi = _compute_reconstruction_roi(
        center_world=expected_center_world,
        half_extent_world=half_extent_world,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        image_shape=depth_m.shape[:2],
    )
    if roi is None:
        return None

    points_world, colors = _backproject_depth_roi_to_world(
        depth_m=depth_m,
        rgb_bgr=rgb_bgr,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        roi=roi,
    )
    points_world, colors = _crop_points_near_center(
        points_world=points_world,
        colors=colors,
        center_world=np.asarray(expected_center_world, dtype=np.float32),
        extent_world=extent_world,
        half_extent_world=half_extent_world,
    )
    if points_world.shape[0] < LOCAL_RGBD_RECONSTRUCTION_MIN_POINTS:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    if colors is not None and colors.shape[0] == points_world.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd = pcd.voxel_down_sample(voxel_size=LOCAL_RGBD_RECONSTRUCTION_VOXEL_SIZE_M)
    if len(pcd.points) >= LOCAL_RGBD_RECONSTRUCTION_DBSCAN_MIN_POINTS:
        pcd = _select_cluster_near_center(
            pcd,
            center_world=np.asarray(expected_center_world, dtype=np.float32),
        )

    if len(pcd.points) >= max(20, LOCAL_RGBD_RECONSTRUCTION_MIN_POINTS // 2):
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=min(20, max(5, len(pcd.points) - 1)), std_ratio=2.0)

    if len(pcd.points) < LOCAL_RGBD_RECONSTRUCTION_MIN_POINTS:
        return None

    bbox = pcd.get_axis_aligned_bounding_box()
    return {
        "pcd": pcd,
        "bbox_center": np.asarray(bbox.get_center(), dtype=np.float32),
        "bbox_extent": np.asarray(bbox.get_extent(), dtype=np.float32),
        "point_count": int(len(pcd.points)),
        "roi_xyxy": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
        "pose_interpretation": pose_interpretation,
        "geometry_update_method": "rgbd_local_reconstruction",
        "depth_clip_info": payload.get("depth_clip_info"),
    }


def _copy_pcd_to_event(source_path: Optional[str], dest_path: Path) -> Optional[str]:
    if not source_path:
        return None
    src = Path(source_path)
    if not src.exists():
        return None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    return str(dest_path)


def _current_object_pcd_path(obj_state: Dict[str, Any]) -> Optional[str]:
    source_path = obj_state.get("pcd_path_current") or obj_state.get("pcd_path_original")
    if not source_path:
        return None
    src = Path(source_path)
    if not src.exists():
        return None
    return str(src)


def _merge_point_cloud_files(source_paths: List[str], dest_path: Path) -> Optional[str]:
    valid_paths = [str(Path(p)) for p in source_paths if p and Path(p).exists()]
    if not valid_paths:
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if o3d is None:
        if len(valid_paths) == 1:
            shutil.copy2(valid_paths[0], dest_path)
            return str(dest_path)
        return None

    merged = o3d.geometry.PointCloud()
    loaded_any = False
    for source_path in valid_paths:
        pcd = o3d.io.read_point_cloud(str(source_path))
        if pcd is None:
            continue
        merged += pcd
        loaded_any = True

    if not loaded_any:
        return None

    o3d.io.write_point_cloud(str(dest_path), merged, write_ascii=False)
    return str(dest_path)


def _translate_pcd(
    source_path: Optional[str],
    dest_path: Path,
    translation_xyz: np.ndarray,
) -> Optional[str]:
    if not source_path:
        return None
    src = Path(source_path)
    if not src.exists():
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Open3D가 있으면 실제 world-coordinate PCD를 평행이동한다.
    # 없더라도 파일을 복사해서 "최신 geometry reference"는 유지한다.
    if o3d is None:
        shutil.copy2(src, dest_path)
        return str(dest_path)

    pcd = o3d.io.read_point_cloud(str(src))
    pcd.translate(np.asarray(translation_xyz, dtype=np.float64))
    o3d.io.write_point_cloud(str(dest_path), pcd, write_ascii=False)
    return str(dest_path)


def _touch_object_geometry(
    *,
    state_root: Path,
    event_dir: Path,
    obj_state: Dict[str, Any],
    new_center: Optional[np.ndarray],
    event_meta: Optional[Dict[str, Any]] = None,
    perception_payload: Optional[Dict[str, Any]] = None,
    camera_to_world_override: Optional[np.ndarray] = None,
) -> Dict[str, Optional[str]]:
    """
    object geometry를 업데이트한다.

    현재 v1의 목적:
    - full re-segmentation 없이도 "같은 물체가 이동했다"는 사실을 geometry에 반영
    - 따라서 기존 PCD를 rigid translation 하는 근사치를 사용
    - later stage에서 perception 기반 point update가 들어오면 이 부분만 교체하면 된다.
    """
    current_source_path = _current_object_pcd_path(obj_state)
    current_center = _as_point3(obj_state.get("bbox_center"))
    current_objects_dir = state_root / "current_objects"
    event_changed_dir = event_dir / "changed_objects"
    source_id = _safe_int(obj_state.get("source_id"))
    if source_id is None:
        return {"current_pcd_path": current_source_path, "event_pcd_path": None}

    current_dest = current_objects_dir / f"source_{source_id:04d}.ply"
    event_dest = event_changed_dir / f"source_{source_id:04d}.ply"

    if new_center is None or current_center is None:
        # pickup처럼 geometry 위치를 다시 계산하지 않는 이벤트에서는
        # 현재 geometry snapshot만 이벤트 폴더에 복사한다.
        event_pcd_path = _copy_pcd_to_event(current_source_path, event_dest)
        return {
            "current_pcd_path": current_source_path,
            "event_pcd_path": event_pcd_path,
            "geometry_update_method": "copied_snapshot",
        }

    reconstructed = _reconstruct_object_pcd_from_event(
        event_dir=event_dir,
        event_meta=event_meta,
        obj_state=obj_state,
        expected_center_world=new_center,
        perception_payload=perception_payload,
        camera_to_world_override=camera_to_world_override,
    )
    if reconstructed is not None:
        current_dest.parent.mkdir(parents=True, exist_ok=True)
        event_dest.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(current_dest), reconstructed["pcd"], write_ascii=False)
        shutil.copy2(current_dest, event_dest)
        return {
            "current_pcd_path": str(current_dest),
            "event_pcd_path": str(event_dest),
            "geometry_update_method": reconstructed.get("geometry_update_method"),
            "reconstructed_bbox_center": reconstructed.get("bbox_center"),
            "reconstructed_bbox_extent": reconstructed.get("bbox_extent"),
            "reconstructed_point_count": reconstructed.get("point_count"),
            "reconstruction_roi_xyxy": reconstructed.get("roi_xyxy"),
            "pose_interpretation": reconstructed.get("pose_interpretation"),
            "depth_clip_info": reconstructed.get("depth_clip_info"),
        }

    translation = new_center - current_center
    current_pcd_path = _translate_pcd(current_source_path, current_dest, translation)
    if current_pcd_path is None:
        return {
            "current_pcd_path": current_source_path,
            "event_pcd_path": None,
            "geometry_update_method": "translation_failed",
        }

    event_pcd_path = _copy_pcd_to_event(current_pcd_path, event_dest)
    return {
        "current_pcd_path": current_pcd_path,
        "event_pcd_path": event_pcd_path,
        "geometry_update_method": "translation_fallback",
    }


def _event_reference_center(
    *,
    state: Dict[str, Any],
    contexts: Dict[str, Any],
    extra: Dict[str, Any],
    preferred_source_ids: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """
    카메라 pose 해석(정방향/역방향 선택)을 안정화하기 위한 기준점을 찾는다.

    우선순위:
    1) 이번 이벤트의 primary source_id 객체 중심
    2) action context의 runtime center
    3) 아무 active 객체 중심
    """
    preferred_source_ids = preferred_source_ids or []
    for source_id in preferred_source_ids:
        obj_state = state.get("objects", {}).get(_source_key(source_id))
        center = _as_point3(None if obj_state is None else obj_state.get("bbox_center"))
        if center is not None:
            return center

    probe_contexts = [
        contexts.get("target_context"),
        contexts.get("held_context"),
        contexts.get("receptacle_target_context"),
    ]
    for ctx in probe_contexts:
        center = _extract_runtime_center(context=ctx, extra=extra)
        if center is not None:
            return center

    for obj_state in state.get("objects", {}).values():
        if str(obj_state.get("status", "active")) != "active":
            continue
        center = _as_point3(obj_state.get("bbox_center"))
        if center is not None:
            return center
    return None


def _persist_reconstructed_geometry(
    *,
    state_root: Path,
    event_dir: Path,
    source_id: int,
    reconstructed_pcd: "o3d.geometry.PointCloud",
) -> Dict[str, Optional[str]]:
    current_dest = state_root / "current_objects" / f"source_{source_id:04d}.ply"
    event_dest = event_dir / "changed_objects" / f"source_{source_id:04d}.ply"
    current_dest.parent.mkdir(parents=True, exist_ok=True)
    event_dest.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(current_dest), reconstructed_pcd, write_ascii=False)
    shutil.copy2(current_dest, event_dest)
    return {
        "current_pcd_path": str(current_dest),
        "event_pcd_path": str(event_dest),
    }


def _object_should_be_visible_in_event(
    *,
    obj_state: Dict[str, Any],
    camera_to_world: np.ndarray,
    intrinsics: Dict[str, Any],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    "이번 프레임에서 보였어야 했는가"를 기하적으로 판단한다.

    기준:
    - bbox_center가 카메라 앞(z>0)에 존재
    - projected 중심점이 이미지 안
    - 재구성 ROI 면적이 최소값 이상
    """
    center_world = _as_point3(obj_state.get("bbox_center"))
    if center_world is None:
        return {"should_be_visible": False, "reason": "missing_bbox_center"}

    projected = _project_world_point(center_world, camera_to_world, intrinsics)
    if projected is None:
        return {"should_be_visible": False, "reason": "projection_failed"}

    u, v, z = [float(x) for x in projected]
    height, width = image_shape
    if not (0.0 <= u < float(width) and 0.0 <= v < float(height)):
        return {
            "should_be_visible": False,
            "reason": "projected_outside_image",
            "projected_uvz": [u, v, z],
        }
    if z <= 1e-4:
        return {
            "should_be_visible": False,
            "reason": "non_positive_depth",
            "projected_uvz": [u, v, z],
        }

    extent_world = _estimate_object_extent(obj_state)
    half_extent_world = _reconstruction_half_extent(extent_world)
    roi = _compute_reconstruction_roi(
        center_world=center_world,
        half_extent_world=half_extent_world,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        image_shape=image_shape,
    )
    if roi is None:
        return {
            "should_be_visible": False,
            "reason": "roi_failed",
            "projected_uvz": [u, v, z],
        }

    x0, y0, x1, y1 = roi
    roi_area = int(max(0, x1 - x0) * max(0, y1 - y0))
    if roi_area < LOCAL_RGBD_RECONSTRUCTION_MIN_ROI_AREA_PX:
        return {
            "should_be_visible": False,
            "reason": "roi_too_small",
            "projected_uvz": [u, v, z],
            "roi_xyxy": [x0, y0, x1, y1],
            "roi_area_px": roi_area,
        }

    return {
        "should_be_visible": True,
        "reason": "frustum_and_roi_ok",
        "expected_center_world": center_world,
        "projected_uvz": [u, v, z],
        "roi_xyxy": [x0, y0, x1, y1],
        "roi_area_px": roi_area,
    }


def _mark_object_missing_or_deleted(
    *,
    state: Dict[str, Any],
    obj_state: Dict[str, Any],
    source_id: int,
    event_id: str,
    event_kind: str,
    reason: str,
) -> Dict[str, Any]:
    """
    보였어야 하는데 재관측되지 않은 객체의 상태를 전이한다.
    """
    old_status = str(obj_state.get("status", "active"))
    old_count = int(obj_state.get("missing_count", 0))
    new_count = old_count + 1
    new_status = "deleted" if new_count >= MISSING_TO_DELETED_COUNT else "missing"

    obj_state["missing_count"] = int(new_count)
    _update_object_common_fields(
        obj_state,
        event_id=event_id,
        event_kind=event_kind,
        change_type="not_observed_in_view",
        status=new_status,
        runtime_center=None,
    )

    removed_relation_ids = _deactivate_relations_for_object(
        state,
        source_id=source_id,
        reason=f"visibility_miss:{reason}",
        event_id=event_id,
    )

    return {
        "source_id": int(source_id),
        "status_before": old_status,
        "status_after": new_status,
        "change_type": "not_observed_in_view",
        "missing_count_before": old_count,
        "missing_count_after": new_count,
        "removed_relation_ids": removed_relation_ids,
        "reason": reason,
    }


def _assess_partial_observation_guard(
    *,
    obj_state: Dict[str, Any],
    reconstructed: Dict[str, Any],
    projected_depth_z: Optional[float],
    depth_clip_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    depth clip로 인해 "물체 일부만 남은 관측"인지 판단한다.

    판단 전략:
    1) 카메라 거리 z가 clip 경계(DEPTH_CLIP_MAX_M)에 충분히 가깝고,
    2) 관측된 bbox/point가 기존 대비 과도하게 작아졌다면 partial로 간주.

    partial로 간주되면:
    - 해당 프레임에서 center/bbox/pcd 덮어쓰기를 건너뛴다.
    - missing으로도 바로 전이하지 않는다.
    """
    clip_enabled = bool((depth_clip_info or {}).get("enabled", False))
    if not clip_enabled or projected_depth_z is None or projected_depth_z <= 0.0:
        return {"is_partial": False}

    near_clip = float(projected_depth_z) >= float(DEPTH_CLIP_MAX_M) * float(PARTIAL_OBSERVATION_NEAR_CLIP_RATIO)
    if not near_clip:
        return {"is_partial": False}

    prior_extent = _estimate_object_extent(obj_state)
    observed_extent = _as_point3(reconstructed.get("bbox_extent"))
    if observed_extent is None:
        return {"is_partial": False}

    prior_extent_safe = np.maximum(prior_extent.astype(np.float32), 1e-4)
    observed_extent_safe = np.maximum(observed_extent.astype(np.float32), 1e-4)

    prior_volume = float(np.prod(prior_extent_safe))
    observed_volume = float(np.prod(observed_extent_safe))
    volume_ratio = observed_volume / max(prior_volume, 1e-8)
    extent_ratio_min = float(np.min(observed_extent_safe / prior_extent_safe))
    point_count = int(reconstructed.get("point_count") or 0)

    low_volume = volume_ratio < float(PARTIAL_OBSERVATION_MIN_VOLUME_RATIO)
    low_extent = extent_ratio_min < float(PARTIAL_OBSERVATION_MIN_EXTENT_RATIO)
    low_points = point_count < int(PARTIAL_OBSERVATION_MIN_POINTS)

    weak_flags = int(low_volume) + int(low_extent) + int(low_points)
    is_partial = weak_flags >= 2
    if not is_partial:
        return {
            "is_partial": False,
            "projected_depth_z": float(projected_depth_z),
            "volume_ratio": volume_ratio,
            "extent_ratio_min": extent_ratio_min,
            "point_count": point_count,
        }

    reasons: List[str] = []
    if low_volume:
        reasons.append("low_volume_ratio")
    if low_extent:
        reasons.append("low_extent_ratio")
    if low_points:
        reasons.append("low_point_count")

    return {
        "is_partial": True,
        "reason": "+".join(reasons) if reasons else "partial_observation_guard",
        "projected_depth_z": float(projected_depth_z),
        "volume_ratio": volume_ratio,
        "extent_ratio_min": extent_ratio_min,
        "point_count": point_count,
        "depth_clip_info": depth_clip_info,
    }


def _run_frame_wide_perception_update(
    *,
    state: Dict[str, Any],
    event_id: str,
    event_kind: str,
    event_dir: Path,
    event_meta: Dict[str, Any],
    graph_index: Dict[str, Any],
    state_root: Path,
    contexts: Dict[str, Any],
    extra: Dict[str, Any],
    primary_source_ids: List[int],
) -> Dict[str, Any]:
    """
    Frame-wide dynamic adaptation.

    처리 순서:
    1) Remove obsolete indices (point-wise depth/color evidence)
       - 기존 객체별로 "실제로 사라졌는지"를 픽셀 단위 증거로 판단
    2) Update low-level memory
       - 현재는 missing 판정만 반영(거짓 업데이트 방지)
    """
    if o3d is None:
        return {
            "status": "skipped",
            "reason": "open3d_unavailable",
            "object_updates": [],
        }

    payload = _load_event_rgbd_payload(event_dir, event_meta=event_meta)
    if payload is None:
        return {
            "status": "skipped",
            "reason": "missing_event_rgbd_payload",
            "object_updates": [],
        }

    depth_m = payload["depth_m"]
    intrinsics = payload["intrinsics"]
    raw_camera_to_world = payload["raw_camera_to_world"]
    depth_clip_info = payload.get("depth_clip_info", {})
    image_shape = depth_m.shape[:2]

    reference_center = _event_reference_center(
        state=state,
        contexts=contexts,
        extra=extra,
        preferred_source_ids=primary_source_ids,
    )
    camera_to_world, pose_interpretation = _resolve_camera_to_world_matrix(
        raw_camera_to_world=raw_camera_to_world,
        intrinsics=intrinsics,
        image_shape=image_shape,
        reference_center_world=reference_center,
    )

    primary_set = set(int(x) for x in primary_source_ids)
    object_updates: List[Dict[str, Any]] = []
    checked_candidates = 0
    dovsg_missing_marked_candidates = 0
    dovsg_observed_or_occluded_candidates = 0

    ordered_objects = sorted(
        state.get("objects", {}).values(),
        key=lambda item: (_safe_int(item.get("source_id")) is None, _safe_int(item.get("source_id")) or 0),
    )

    for obj_state in ordered_objects:
        source_id = _safe_int(obj_state.get("source_id"))
        status = str(obj_state.get("status", "active"))
        if source_id is None:
            continue
        if source_id in primary_set:
            # primary target는 event-specific handler에서 이미 반영했으므로 중복 업데이트를 피한다.
            continue
        if status not in PERCEPTION_CANDIDATE_OBJECT_STATUSES:
            continue
        if status == "held":
            continue

        checked_candidates += 1
        missing_evidence = _dovsg_missing_evidence_for_object(
            obj_state=obj_state,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            image_shape=image_shape,
            depth_m=depth_m,
            rgb_bgr=payload.get("rgb_bgr"),
        )
        if missing_evidence.get("decision") == "missing_evidence":
            dovsg_missing_marked_candidates += 1
            missing_update = _mark_object_missing_or_deleted(
                state=state,
                obj_state=obj_state,
                source_id=source_id,
                event_id=event_id,
                event_kind=event_kind,
                reason="dovsg_pointwise_missing_evidence",
            )
            missing_update["dovsg_observation_evidence"] = missing_evidence
            object_updates.append(missing_update)
        else:
            dovsg_observed_or_occluded_candidates += 1

    return {
        "status": "updated",
        "pose_interpretation": pose_interpretation,
        "checked_candidates": checked_candidates,
        "dovsg_missing_marked_candidates": dovsg_missing_marked_candidates,
        "dovsg_observed_or_occluded_candidates": dovsg_observed_or_occluded_candidates,
        "depth_clip_info": depth_clip_info,
        "object_updates": object_updates,
    }


def _write_full_scene_snapshot(
    *,
    state: Dict[str, Any],
    event_id: str,
    event_dir: Path,
    state_root: Path,
) -> Dict[str, Any]:
    scene_objects_dir = event_dir / "scene_objects"
    scene_manifest_path = event_dir / "scene_objects_manifest.json"
    event_scene_full_path = event_dir / "scene_full.ply"
    current_scene_full_path = state_root / "current_scene_full.ply"
    current_scene_manifest_path = state_root / "current_scene_manifest.json"

    manifest_objects: List[Dict[str, Any]] = []
    current_scene_sources: List[str] = []

    ordered_objects = sorted(
        state.get("objects", {}).values(),
        key=lambda item: (_safe_int(item.get("source_id")) is None, _safe_int(item.get("source_id")) or 0),
    )

    for obj_state in ordered_objects:
        status = str(obj_state.get("status", "active"))
        if status not in WORLD_VISIBLE_OBJECT_STATUSES:
            continue

        source_id = _safe_int(obj_state.get("source_id"))
        if source_id is None:
            continue

        current_source_path = _current_object_pcd_path(obj_state)
        event_object_pcd_path = None
        if current_source_path is not None:
            event_object_pcd_path = _copy_pcd_to_event(
                current_source_path,
                scene_objects_dir / f"source_{source_id:04d}.ply",
            )
            current_scene_sources.append(current_source_path)

        manifest_objects.append(
            {
                "source_id": source_id,
                "status": status,
                "graph_object_name": obj_state.get("graph_object_name"),
                "conceptgraph_object_tag": obj_state.get("conceptgraph_object_tag"),
                "room_id": obj_state.get("room_id"),
                "support_source_id": obj_state.get("support_source_id"),
                "bbox_center": obj_state.get("bbox_center"),
                "current_pcd_path": current_source_path,
                "event_object_pcd_path": event_object_pcd_path,
            }
        )

    current_scene_pcd_path = _merge_point_cloud_files(current_scene_sources, current_scene_full_path)
    event_scene_pcd_path = _copy_pcd_to_event(current_scene_pcd_path, event_scene_full_path)

    snapshot_manifest = {
        "event_id": event_id,
        "object_count": len(manifest_objects),
        "visible_statuses": sorted(WORLD_VISIBLE_OBJECT_STATUSES),
        "scene_full_pcd_path": event_scene_pcd_path,
        "scene_objects_dir": str(scene_objects_dir),
        "objects": manifest_objects,
    }
    current_snapshot_manifest = {
        "event_id": event_id,
        "object_count": len(manifest_objects),
        "visible_statuses": sorted(WORLD_VISIBLE_OBJECT_STATUSES),
        "scene_full_pcd_path": current_scene_pcd_path,
        "scene_root": str(state_root),
        "objects": manifest_objects,
    }
    _write_json(scene_manifest_path, snapshot_manifest)
    _write_json(current_scene_manifest_path, current_snapshot_manifest)

    return {
        "scene_full_pcd_path": event_scene_pcd_path,
        "scene_objects_dir": str(scene_objects_dir),
        "scene_manifest_path": str(scene_manifest_path),
        "current_scene_full_pcd_path": current_scene_pcd_path,
        "current_scene_manifest_path": str(current_scene_manifest_path),
        "object_count": len(manifest_objects),
    }


def _deactivate_relations_for_object(
    state: Dict[str, Any],
    source_id: int,
    *,
    reason: str,
    event_id: str,
) -> List[str]:
    changed_relation_ids: List[str] = []
    for relation in state["relations"].values():
        if relation.get("status") != "active":
            continue
        if relation.get("source_id") == source_id or relation.get("target_id") == source_id:
            relation["status"] = "inactive"
            relation["deactivated_by_event"] = event_id
            relation["deactivated_reason"] = reason
            changed_relation_ids.append(str(relation.get("relation_id")))
    return changed_relation_ids


def _add_relation(
    state: Dict[str, Any],
    *,
    source_id: int,
    target_id: int,
    relationship: str,
    description: str,
    event_id: str,
) -> str:
    relation_id = f"dyn_edge_{int(state['next_relation_id']):06d}"
    state["next_relation_id"] = int(state["next_relation_id"]) + 1
    state["relations"][relation_id] = {
        "relation_id": relation_id,
        "source_id": int(source_id),
        "target_id": int(target_id),
        "relationship": relationship,
        "edge_description": description,
        "status": "active",
        "origin": "dynamic_update",
        "created_by_event": event_id,
    }
    return relation_id


def _extract_source_id(*contexts: Optional[Dict[str, Any]]) -> Optional[int]:
    for context in contexts:
        if not isinstance(context, dict):
            continue
        source_id = _safe_int(context.get("graph_source_id"))
        if source_id is not None:
            return source_id
    return None


def _source_ids_from_event_result(event_result: Dict[str, Any]) -> List[int]:
    ids: List[int] = []
    if not isinstance(event_result, dict):
        return ids
    updates = event_result.get("object_updates")
    if not isinstance(updates, list):
        return ids
    for row in updates:
        if not isinstance(row, dict):
            continue
        source_id = _safe_int(row.get("source_id"))
        if source_id is None:
            continue
        ids.append(source_id)
    return sorted(list(set(ids)))


def _is_effective_object_update(update_row: Dict[str, Any]) -> bool:
    """
    scene state를 실제로 바꾼 업데이트인지 판정한다.

    진단/스킵성 change_type은 object_updates 메인 목록에서 제외한다.
    """
    if not isinstance(update_row, dict):
        return False

    change_type = str(update_row.get("change_type", "") or "")
    status_before = update_row.get("status_before")
    status_after = update_row.get("status_after")
    geometry_method = str(update_row.get("geometry_update_method", "") or "")

    diagnostic_change_types = {
        "depth_clipped_skip",
        "occluded_skip",
        "partial_observation",
        "reconstruction_failed_no_missing_evidence",
        "new_object_candidate",
        "stale_new_object_candidate_removed",
    }
    if change_type in diagnostic_change_types:
        return False
    if geometry_method.startswith("skipped_"):
        return False
    if geometry_method.startswith("skipped_by_"):
        return False

    # 상태가 바뀌었다면 실제 업데이트로 본다.
    if status_before != status_after:
        return True

    # 상태는 같아도 geometry/room 등이 갱신된 경우가 있다.
    effective_change_types = {
        "picked_up",
        "placed",
        "reobserved",
        "minor_adjustment",
        "positional_shift",
        "appearance",
        "new_object",
        "not_observed_in_view",
    }
    if change_type in effective_change_types:
        return True

    # fallback: pcd path가 실제로 기록됐다면 업데이트로 취급
    if update_row.get("current_pcd_path") or update_row.get("event_pcd_path"):
        return True

    return False


def _split_effective_and_diagnostic_updates(
    updates: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    effective: List[Dict[str, Any]] = []
    diagnostic: List[Dict[str, Any]] = []
    for row in updates:
        if _is_effective_object_update(row):
            effective.append(row)
        else:
            diagnostic.append(row)
    return effective, diagnostic


def _extract_runtime_center(
    *,
    context: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    extra_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    if extra_key and isinstance(extra, dict):
        center = _as_point3(extra.get(extra_key))
        if center is not None:
            return center
    if isinstance(context, dict):
        center = _as_point3(context.get("runtime_position"))
        if center is not None:
            return center
        center = _as_point3(context.get("conceptgraph_bbox_center"))
        if center is not None:
            return center
        center = _as_point3(context.get("nav_xyz"))
        if center is not None:
            return center
    return None


def _build_initial_state(
    scene_id: str,
    *,
    hsg_root: str = DEFAULT_HSG_ROOT,
    hsg_dataset: str = DEFAULT_HSG_DATASET,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Dict[str, Any]:
    graph_index = load_hierarchical_graph_index(
        scene_id=scene_id,
        hsg_root=hsg_root,
        dataset_name=hsg_dataset,
    )
    conceptgraph_index = load_conceptgraph_index(
        scene_id=scene_id,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )
    conceptgraph_edges = _load_conceptgraph_edges(
        scene_id=scene_id,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )

    rooms_by_id = graph_index.get("rooms_by_id", {})
    graph_dir = Path(graph_index["graph_dir"])

    hsg_objects_by_source_id: Dict[int, Dict[str, Any]] = {}
    for graph_object in graph_index.get("objects_by_id", {}).values():
        source_id = _safe_int(graph_object.get("source_id"))
        if source_id is None:
            continue
        hsg_objects_by_source_id[source_id] = graph_object

    objects: Dict[str, Dict[str, Any]] = {}
    for source_id, conceptgraph_object in conceptgraph_index["objects_by_source_id"].items():
        graph_object = hsg_objects_by_source_id.get(int(source_id))
        room = None
        room_id = None
        room_name = None
        floor_id = None
        graph_object_id = None
        graph_object_name = None
        pcd_path_original = None
        semantic_embedding = None

        if graph_object is not None:
            graph_object_id = graph_object.get("object_id")
            graph_object_name = graph_object.get("name")
            room_id = graph_object.get("room_id")
            room = rooms_by_id.get(str(room_id)) if room_id is not None else None
            room_name = None if room is None else room.get("name")
            floor_id = graph_object.get("floor_id")
            if graph_object_id is not None:
                pcd_path_original = str(graph_dir / "objects" / f"{graph_object_id}.ply")
            semantic_embedding = graph_object.get("embedding")
            if semantic_embedding is None:
                graph_object_json = graph_object.get("json_path")
                if isinstance(graph_object_json, str) and Path(graph_object_json).exists():
                    try:
                        graph_object_meta = _read_json(Path(graph_object_json))
                        semantic_embedding = graph_object_meta.get("embedding")
                    except Exception:
                        semantic_embedding = None

        bbox_center = conceptgraph_object.get("bbox_center")
        bbox_extent = conceptgraph_object.get("bbox_extent")
        bbox_volume = conceptgraph_object.get("bbox_volume")
        semantic_embedding_norm = _as_feature_vector(semantic_embedding)
        semantic_embedding = None if semantic_embedding_norm is None else semantic_embedding_norm.tolist()

        objects[_source_key(source_id)] = {
            "source_id": int(source_id),
            "status": "active",
            "graph_object_id": graph_object_id,
            "graph_object_name": graph_object_name,
            "room_id": room_id,
            "room_name": room_name,
            "floor_id": floor_id,
            "support_source_id": None,
            "bbox_center": bbox_center,
            "bbox_extent": bbox_extent,
            "bbox_volume": bbox_volume,
            "conceptgraph_object_key": conceptgraph_object.get("object_key"),
            "conceptgraph_object_tag": conceptgraph_object.get("object_tag"),
            "conceptgraph_caption": conceptgraph_object.get("object_caption"),
            "semantic_embedding": semantic_embedding,
            "semantic_observation_count": 1 if semantic_embedding is not None else 0,
            "pcd_path_original": pcd_path_original,
            "pcd_path_current": pcd_path_original,
            "last_change_type": None,
            "last_event_kind": None,
            "last_event_id": None,
            "last_observed_event_id": None,
            "previous_room_id": None,
            "previous_support_source_id": None,
            "previous_bbox_center": None,
            "last_runtime_position": None,
            "missing_count": 0,
        }

    relations: Dict[str, Dict[str, Any]] = {}
    max_edge_id = 0
    for edge_key, edge in conceptgraph_edges.items():
        source_id = _safe_int(edge.get("object_1_id"))
        target_id = _safe_int(edge.get("object_2_id"))
        if source_id is None or target_id is None:
            continue
        if _source_key(source_id) not in objects or _source_key(target_id) not in objects:
            continue
        relation_id = str(edge_key)
        relations[relation_id] = {
            "relation_id": relation_id,
            "source_id": int(source_id),
            "target_id": int(target_id),
            "relationship": _relationship_label(edge.get("relationship")),
            "edge_description": edge.get("edge_description"),
            "status": "active",
            "origin": "conceptgraph",
        }
        if relations[relation_id]["relationship"] == "on_top_of":
            objects[_source_key(source_id)]["support_source_id"] = int(target_id)
        edge_id = _safe_int(edge.get("edge_id"))
        if edge_id is not None:
            max_edge_id = max(max_edge_id, edge_id)

    state = {
        "scene_id": scene_id,
        "scene_key": _scene_key_from_scene_id(scene_id),
        "hsg_root": hsg_root,
        "hsg_dataset": hsg_dataset,
        "conceptgraph_root": conceptgraph_root,
        "experiment_name": experiment_name,
        "objects": objects,
        "relations": relations,
        "next_source_id": (max([obj["source_id"] for obj in objects.values()]) + 1) if objects else 0,
        "next_relation_id": max_edge_id + 1,
        "pending_new_objects": [],
        "next_pending_object_id": 0,
        "last_event_id": None,
        "event_history": [],
        "notes": [
            "This state file is an overlay on top of ConceptGraph/HSG base memories.",
            "Action context is used first to avoid confusing moved objects with deleted objects.",
        ],
    }
    return state


def _state_paths(scene_id: str) -> Dict[str, Path]:
    scene_root = DYNAMIC_SCENE_GRAPH_ROOT / _scene_key_from_scene_id(scene_id)
    return {
        "scene_root": scene_root,
        "state_json": scene_root / "current_state.json",
        "current_objects_dir": scene_root / "current_objects",
    }


def _upgrade_state_schema_inplace(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    이전 버전 state 파일과의 하위 호환을 위해 필드를 보정한다.
    """
    objects = state.get("objects", {})
    if not isinstance(objects, dict):
        state["objects"] = {}
        objects = state["objects"]

    max_source_id = -1
    for obj_state in objects.values():
        source_id = _safe_int(obj_state.get("source_id"))
        if source_id is not None:
            max_source_id = max(max_source_id, source_id)
        if "missing_count" not in obj_state:
            obj_state["missing_count"] = 0
        if "semantic_embedding" not in obj_state:
            obj_state["semantic_embedding"] = None
        if "semantic_observation_count" not in obj_state:
            obj_state["semantic_observation_count"] = 0

    if "next_source_id" not in state or _safe_int(state.get("next_source_id")) is None:
        state["next_source_id"] = max_source_id + 1
    else:
        state["next_source_id"] = max(_safe_int(state.get("next_source_id")) or 0, max_source_id + 1)

    if "event_history" not in state or not isinstance(state.get("event_history"), list):
        state["event_history"] = []
    if "relations" not in state or not isinstance(state.get("relations"), dict):
        state["relations"] = {}
    if "pending_new_objects" not in state or not isinstance(state.get("pending_new_objects"), list):
        state["pending_new_objects"] = []
    else:
        state["pending_new_objects"] = [
            item for item in state["pending_new_objects"] if isinstance(item, dict)
        ]
    if "next_pending_object_id" not in state or _safe_int(state.get("next_pending_object_id")) is None:
        max_pending_id = -1
        for item in state.get("pending_new_objects", []):
            pending_id = _safe_int(item.get("pending_id"))
            if pending_id is not None:
                max_pending_id = max(max_pending_id, pending_id)
        state["next_pending_object_id"] = max_pending_id + 1
    else:
        state["next_pending_object_id"] = int(max(0, _safe_int(state.get("next_pending_object_id")) or 0))
    if "notes" not in state or not isinstance(state.get("notes"), list):
        state["notes"] = []
    return state


def _load_or_create_state(
    scene_id: str,
    *,
    hsg_root: str = DEFAULT_HSG_ROOT,
    hsg_dataset: str = DEFAULT_HSG_DATASET,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    paths = _state_paths(scene_id)
    if paths["state_json"].exists():
        state = _read_json(paths["state_json"])
        state = _upgrade_state_schema_inplace(state)
        return state, paths

    state = _build_initial_state(
        scene_id=scene_id,
        hsg_root=hsg_root,
        hsg_dataset=hsg_dataset,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )
    state = _upgrade_state_schema_inplace(state)
    paths["scene_root"].mkdir(parents=True, exist_ok=True)
    paths["current_objects_dir"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["state_json"], state)
    return state, paths


def _update_object_common_fields(
    obj_state: Dict[str, Any],
    *,
    event_id: str,
    event_kind: str,
    change_type: str,
    status: str,
    runtime_center: Optional[np.ndarray] = None,
):
    obj_state["status"] = status
    obj_state["last_change_type"] = change_type
    obj_state["last_event_kind"] = event_kind
    obj_state["last_event_id"] = event_id
    if runtime_center is not None:
        obj_state["last_runtime_position"] = runtime_center.tolist()


def _apply_pickup_event(
    state: Dict[str, Any],
    *,
    event_id: str,
    event_kind: str,
    event_dir: Path,
    contexts: Dict[str, Any],
    extra: Dict[str, Any],
    state_root: Path,
    event_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    target_context = contexts.get("target_context")
    held_context = contexts.get("held_context")
    source_id = _extract_source_id(held_context, target_context)
    if source_id is None:
        return {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": "pickup event did not include graph_source_id",
        }

    obj_key = _source_key(source_id)
    obj_state = state["objects"].get(obj_key)
    if obj_state is None:
        return {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": f"source_id={source_id} not found in current state",
        }

    obj_state["previous_room_id"] = obj_state.get("room_id")
    obj_state["previous_support_source_id"] = obj_state.get("support_source_id")
    obj_state["previous_bbox_center"] = obj_state.get("bbox_center")

    removed_relations = _deactivate_relations_for_object(
        state,
        source_id=source_id,
        reason="picked_up",
        event_id=event_id,
    )

    geometry_paths = _touch_object_geometry(
        state_root=state_root,
        event_dir=event_dir,
        obj_state=obj_state,
        new_center=None,
        event_meta=event_meta,
    )
    obj_state["pcd_path_current"] = (
        geometry_paths.get("current_pcd_path")
        or obj_state.get("pcd_path_current")
        or obj_state.get("pcd_path_original")
    )

    runtime_center = _extract_runtime_center(context=held_context, extra=extra)
    _update_object_common_fields(
        obj_state,
        event_id=event_id,
        event_kind=event_kind,
        change_type="picked_up",
        status="held",
        runtime_center=runtime_center,
    )
    obj_state["missing_count"] = 0
    obj_state["room_id"] = None
    obj_state["room_name"] = None
    obj_state["support_source_id"] = None

    return {
        "event_id": event_id,
        "event_kind": event_kind,
        "status": "updated",
        "object_updates": [
            {
                "source_id": source_id,
                "status_before": "active",
                "status_after": "held",
                "change_type": "picked_up",
                "removed_relation_ids": removed_relations,
                "event_pcd_path": geometry_paths.get("event_pcd_path"),
                "geometry_update_method": geometry_paths.get("geometry_update_method"),
            }
        ],
    }


def _apply_put_event(
    state: Dict[str, Any],
    *,
    event_id: str,
    event_kind: str,
    event_dir: Path,
    contexts: Dict[str, Any],
    extra: Dict[str, Any],
    graph_index: Dict[str, Any],
    state_root: Path,
    event_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    held_context = contexts.get("held_context")
    receptacle_context = contexts.get("receptacle_target_context")

    source_id = _extract_source_id(held_context)
    receptacle_source_id = _extract_source_id(receptacle_context)
    if source_id is None:
        return {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": "put event did not include held source_id",
        }

    obj_state = state["objects"].get(_source_key(source_id))
    if obj_state is None:
        return {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": f"source_id={source_id} not found in current state",
        }

    old_center = _as_point3(obj_state.get("bbox_center"))
    new_center = _extract_runtime_center(
        context=held_context,
        extra=extra,
        extra_key="placed_object_runtime_position",
    )
    if new_center is None:
        new_center = _extract_runtime_center(context=receptacle_context, extra=extra)

    removed_relations = _deactivate_relations_for_object(
        state,
        source_id=source_id,
        reason="replaced_after_put",
        event_id=event_id,
    )

    geometry_paths = _touch_object_geometry(
        state_root=state_root,
        event_dir=event_dir,
        obj_state=obj_state,
        new_center=new_center,
        event_meta=event_meta,
    )
    obj_state["pcd_path_current"] = (
        geometry_paths.get("current_pcd_path")
        or obj_state.get("pcd_path_current")
        or obj_state.get("pcd_path_original")
    )

    refined_center = _as_point3(geometry_paths.get("reconstructed_bbox_center"))
    if refined_center is None:
        refined_center = new_center
    refined_extent = _fuse_bbox_extent(
        obj_state.get("bbox_extent"),
        geometry_paths.get("reconstructed_bbox_extent"),
    )

    if refined_center is not None:
        obj_state["bbox_center"] = refined_center.tolist()
    if refined_extent is not None:
        obj_state["bbox_extent"] = refined_extent.tolist()
    room_id, room_name, floor_id = _choose_room_from_center(
        center_xyz=refined_center,
        graph_index=graph_index,
        preferred_room_id=(
            None
            if not isinstance(receptacle_context, dict)
            else receptacle_context.get("graph_room_id")
        ),
    )

    obj_state["room_id"] = room_id
    obj_state["room_name"] = room_name
    obj_state["floor_id"] = floor_id
    obj_state["support_source_id"] = receptacle_source_id

    relation_id = None
    if receptacle_source_id is not None and receptacle_source_id != source_id:
        relation_id = _add_relation(
            state,
            source_id=source_id,
            target_id=receptacle_source_id,
            relationship="on_top_of",
            description=f"{obj_state.get('conceptgraph_object_tag') or obj_state.get('graph_object_name') or source_id} on top of "
            f"{receptacle_context.get('conceptgraph_object_tag') if isinstance(receptacle_context, dict) else receptacle_source_id}",
            event_id=event_id,
        )

    change_type = _movement_change_type(old_center, refined_center, default="placed")
    _update_object_common_fields(
        obj_state,
        event_id=event_id,
        event_kind=event_kind,
        change_type=change_type,
        status="active",
        runtime_center=refined_center,
    )
    obj_state["missing_count"] = 0

    return {
        "event_id": event_id,
        "event_kind": event_kind,
        "status": "updated",
        "object_updates": [
            {
                "source_id": source_id,
                "status_before": "held",
                "status_after": "active",
                "change_type": change_type,
                "new_room_id": room_id,
                "new_support_source_id": receptacle_source_id,
                "removed_relation_ids": removed_relations,
                "new_relation_id": relation_id,
                "new_center": None if refined_center is None else refined_center.tolist(),
                "new_bbox_extent": None if refined_extent is None else refined_extent.tolist(),
                "current_pcd_path": geometry_paths.get("current_pcd_path"),
                "event_pcd_path": geometry_paths.get("event_pcd_path"),
                "geometry_update_method": geometry_paths.get("geometry_update_method"),
                "reconstructed_point_count": geometry_paths.get("reconstructed_point_count"),
                "reconstruction_roi_xyxy": geometry_paths.get("reconstruction_roi_xyxy"),
                "pose_interpretation": geometry_paths.get("pose_interpretation"),
                "depth_clip_info": geometry_paths.get("depth_clip_info"),
            }
        ],
    }


def _apply_goto_align_event(
    state: Dict[str, Any],
    *,
    event_id: str,
    event_kind: str,
    event_dir: Path,
    contexts: Dict[str, Any],
    extra: Dict[str, Any],
    graph_index: Dict[str, Any],
    state_root: Path,
    event_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    target_context = contexts.get("target_context")
    source_id = _extract_source_id(target_context)
    if source_id is None:
        return {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": "goto_align event did not include target source_id",
        }

    obj_state = state["objects"].get(_source_key(source_id))
    if obj_state is None:
        return {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": f"source_id={source_id} not found in current state",
        }

    old_status = obj_state.get("status", "active")
    old_center = _as_point3(obj_state.get("bbox_center"))
    observed_center = _extract_runtime_center(context=target_context, extra=extra)
    if observed_center is not None:
        geometry_paths = _touch_object_geometry(
            state_root=state_root,
            event_dir=event_dir,
            obj_state=obj_state,
            new_center=observed_center,
            event_meta=event_meta,
        )
        obj_state["pcd_path_current"] = (
            geometry_paths.get("current_pcd_path")
            or obj_state.get("pcd_path_current")
            or obj_state.get("pcd_path_original")
        )
        refined_center = _as_point3(geometry_paths.get("reconstructed_bbox_center"))
        if refined_center is None:
            refined_center = observed_center
        refined_extent = _fuse_bbox_extent(
            obj_state.get("bbox_extent"),
            geometry_paths.get("reconstructed_bbox_extent"),
        )
        obj_state["bbox_center"] = refined_center.tolist()
        if refined_extent is not None:
            obj_state["bbox_extent"] = refined_extent.tolist()
    else:
        geometry_paths = _touch_object_geometry(
            state_root=state_root,
            event_dir=event_dir,
            obj_state=obj_state,
            new_center=None,
            event_meta=event_meta,
        )
        obj_state["pcd_path_current"] = (
            geometry_paths.get("current_pcd_path")
            or obj_state.get("pcd_path_current")
            or obj_state.get("pcd_path_original")
        )
        refined_center = observed_center
        refined_extent = None

    room_id, room_name, floor_id = _choose_room_from_center(
        center_xyz=refined_center,
        graph_index=graph_index,
        preferred_room_id=obj_state.get("room_id"),
    )
    if room_id is not None:
        obj_state["room_id"] = room_id
        obj_state["room_name"] = room_name
        obj_state["floor_id"] = floor_id

    if old_status in {"missing", "deleted"}:
        change_type = "appearance"
    else:
        change_type = _movement_change_type(old_center, refined_center, default="reobserved")

    _update_object_common_fields(
        obj_state,
        event_id=event_id,
        event_kind=event_kind,
        change_type=change_type,
        status="active",
        runtime_center=refined_center,
    )
    obj_state["last_observed_event_id"] = event_id
    obj_state["missing_count"] = 0

    return {
        "event_id": event_id,
        "event_kind": event_kind,
        "status": "updated",
        "object_updates": [
            {
                "source_id": source_id,
                "status_before": old_status,
                "status_after": "active",
                "change_type": change_type,
                "observed_center": None if refined_center is None else refined_center.tolist(),
                "observed_bbox_extent": None if refined_extent is None else refined_extent.tolist(),
                "current_pcd_path": geometry_paths.get("current_pcd_path"),
                "event_pcd_path": geometry_paths.get("event_pcd_path"),
                "geometry_update_method": geometry_paths.get("geometry_update_method"),
                "reconstructed_point_count": geometry_paths.get("reconstructed_point_count"),
                "reconstruction_roi_xyxy": geometry_paths.get("reconstruction_roi_xyxy"),
                "pose_interpretation": geometry_paths.get("pose_interpretation"),
                "depth_clip_info": geometry_paths.get("depth_clip_info"),
            }
        ],
    }


def apply_dynamic_scene_graph_update(
    scene_id: str,
    event_meta_path: str,
    *,
    hsg_root: str = DEFAULT_HSG_ROOT,
    hsg_dataset: str = DEFAULT_HSG_DATASET,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Dict[str, Any]:
    """
    Action-context-first dynamic scene graph updater.

    설계 목적:
    1) DovSG의 dynamic adaptation 철학(삭제 후보 제거 -> local memory 갱신 -> high-level graph 갱신)을
       현재 ConceptGraph/HSG 기반 파이프라인에 맞게 단순화해서 가져온다.
    2) 아직 full open-vocabulary re-detection을 매 이벤트마다 돌리지 않더라도,
       pickup/put/goto_align 같은 "의미 있는 이벤트"만으로 scene graph를 안정적으로 갱신한다.
    3) 특히 action context를 사용해 "멀리 이동한 물체"를 "삭제 + 신규 객체"로 오인하지 않게 한다.
    """
    meta_path = Path(event_meta_path)
    event_dir = meta_path.parent
    meta = _read_json(meta_path)

    state, paths = _load_or_create_state(
        scene_id=scene_id,
        hsg_root=hsg_root,
        hsg_dataset=hsg_dataset,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )
    graph_index = load_hierarchical_graph_index(
        scene_id=scene_id,
        hsg_root=hsg_root,
        dataset_name=hsg_dataset,
    )

    event_id = event_dir.name
    event_kind = str(meta.get("event_kind", "unknown"))
    contexts = meta.get("contexts", {}) if isinstance(meta.get("contexts"), dict) else {}
    extra = meta.get("extra", {}) if isinstance(meta.get("extra"), dict) else {}

    if event_kind == "pickup":
        event_result = _apply_pickup_event(
            state,
            event_id=event_id,
            event_kind=event_kind,
            event_dir=event_dir,
            contexts=contexts,
            extra=extra,
            state_root=paths["scene_root"],
            event_meta=meta,
        )
    elif event_kind == "put":
        event_result = _apply_put_event(
            state,
            event_id=event_id,
            event_kind=event_kind,
            event_dir=event_dir,
            contexts=contexts,
            extra=extra,
            graph_index=graph_index,
            state_root=paths["scene_root"],
            event_meta=meta,
        )
    elif event_kind == "goto_align":
        event_result = _apply_goto_align_event(
            state,
            event_id=event_id,
            event_kind=event_kind,
            event_dir=event_dir,
            contexts=contexts,
            extra=extra,
            graph_index=graph_index,
            state_root=paths["scene_root"],
            event_meta=meta,
        )
    else:
        event_result = {
            "event_id": event_id,
            "event_kind": event_kind,
            "status": "skipped",
            "reason": f"unsupported event_kind={event_kind}",
        }

    # event handler가 만든 1차 업데이트 목록(주로 target object 중심)
    base_updates = list(event_result.get("object_updates", [])) if isinstance(event_result.get("object_updates"), list) else []

    frame_perception_result = {
        "status": "skipped",
        "reason": "event_kind_not_supported_for_frame_wide_update",
        "object_updates": [],
    }
    if event_kind in {"pickup", "put", "goto_align"}:
        primary_source_ids = _source_ids_from_event_result(event_result)
        frame_perception_result = _run_frame_wide_perception_update(
            state=state,
            event_id=event_id,
            event_kind=event_kind,
            event_dir=event_dir,
            event_meta=meta,
            graph_index=graph_index,
            state_root=paths["scene_root"],
            contexts=contexts,
            extra=extra,
            primary_source_ids=primary_source_ids,
        )

    frame_updates_raw = (
        list(frame_perception_result.get("object_updates", []))
        if isinstance(frame_perception_result.get("object_updates"), list)
        else []
    )
    all_updates_raw = base_updates + frame_updates_raw
    effective_updates, diagnostic_updates = _split_effective_and_diagnostic_updates(all_updates_raw)

    # 사용자 요구 반영:
    # - object_updates에는 "진짜 반영된 업데이트"만 둔다.
    # - skip/진단성 항목은 object_updates_diagnostic로 분리한다.
    event_result["object_updates"] = effective_updates
    event_result["object_updates_diagnostic"] = diagnostic_updates
    event_result["object_updates_count_effective"] = len(effective_updates)
    event_result["object_updates_count_diagnostic"] = len(diagnostic_updates)

    # frame_perception_update도 같은 방식으로 분리해서 해석을 쉽게 만든다.
    fp_effective, fp_diagnostic = _split_effective_and_diagnostic_updates(frame_updates_raw)
    frame_perception_result["object_updates_raw"] = frame_updates_raw
    frame_perception_result["object_updates"] = fp_effective
    frame_perception_result["object_updates_diagnostic"] = fp_diagnostic
    frame_perception_result["object_updates_count_raw"] = len(frame_updates_raw)
    frame_perception_result["object_updates_count_effective"] = len(fp_effective)
    frame_perception_result["object_updates_count_diagnostic"] = len(fp_diagnostic)

    event_result["frame_perception_update"] = frame_perception_result
    state["last_event_id"] = event_id
    scene_snapshot = _write_full_scene_snapshot(
        state=state,
        event_id=event_id,
        event_dir=event_dir,
        state_root=paths["scene_root"],
    )
    event_result["full_scene_snapshot"] = scene_snapshot
    state["latest_full_scene_snapshot"] = scene_snapshot
    state["event_history"].append(
        {
            "event_id": event_id,
            "event_kind": event_kind,
            "result_status": event_result.get("status"),
        }
    )

    _write_json(paths["state_json"], state)
    _write_json(event_dir / "scene_graph_update.json", event_result)
    return event_result
