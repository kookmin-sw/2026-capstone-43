#!/usr/bin/env python3
"""
REACT Inference Worker - runs in separate process with nightly PyTorch (sm_120 support).
Communicates with Isaac Sim main process via /tmp/react_* files.

Protocol:
  Main → Worker: writes /tmp/react_input.npz (rgb, depth, robot_pos, robot_quat, robot_yaw)
  Worker → Main: writes /tmp/react_output.json (changes list)
  Shutdown:      main writes /tmp/react_shutdown
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import json
import time
import math
import copy
import numpy as np

import torch
torch.set_num_threads(2) # [추가] CPU 스레드 최대 2개로 제한
import torch.nn as nn
import cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from PIL import Image

# ================= Config =================
YOLO_MODEL_NAME = "yolov8x-worldv2.pt"  # Open-vocabulary: detects any class by text
YOLO_CONF_THRESHOLD = 0.45  # Lower for YOLO-World open-vocab (post-filtering by embedding handles false positives)
EMBEDDING_DIM = 1408
MATCH_THRESHOLD = 0.45  # cosine similarity threshold for identity match
MATCH_THRESHOLD_TRAINED = 0.35  # lower threshold for objects with trained embeddings (more reliable)

# Object mobility categories (from REACT paper)
# static: walls, floors, built-in appliances - never move
# semi-static: furniture, large appliances - rarely move
# dynamic: small objects, books, plates - frequently move
MOBILITY = {
    "static": {
        "tags": ["wall", "floor", "ceiling", "vent", "window", "door", "light"],
        "miss_threshold": 999,  # never report as absent
        "move_threshold": 999.0,  # never report as moved
    },
    "semi_static": {
        "tags": ["refrigerator", "cabinet", "desk", "counter", "sink", "oven",
                 "stove", "couch", "sofa", "bed", "toilet", "bathtub",
                 "folded chair", "table", "shelf", "drawer", "dresser", "wardrobe"],
        "miss_threshold": 15,  # very strict: need many misses before ABSENT
        "move_threshold": 5.0,  # very strict: >5m to count as MOVED (furniture almost never moves)
    },
    "dynamic": {
        "tags": ["plate", "bowl", "cup", "mug", "glass", "bottle", "book",
                 "chair", "potted plant", "vase", "clock", "remote",
                 "scissors", "teddy bear", "backpack", "handbag", "umbrella",
                 "laptop", "keyboard", "mouse", "cell phone", "knife",
                 "fork", "spoon", "toothbrush"],
        "miss_threshold": 8,  # need consistent misses before ABSENT (YOLO can flicker)
        "move_threshold": 0.8,  # need larger displacement
    },
}

# Trained embedding model (EfficientNet-B2 + Triplet Loss from sam_crop data)
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_EMBEDDING_PATH = os.path.join(_WORKER_DIR, "data", "newscene", "best_embedding_model_newscene.pth")
# Reference crops directory (det-extracted per-object crops for new scene)
REF_CROPS_DIR = "/home/sungbin/extra/data/cg_replica/session_20260403_144334/training_crops_det"

CAMERA_FOCAL = 2.12
CAMERA_H_APERTURE = 6.02
CAMERA_V_APERTURE = 3.39
CAMERA_RESOLUTION = (1280, 720)

INPUT_PATH = "/tmp/react_input.npz"
OUTPUT_PATH = "/tmp/react_output.json"
SHUTDOWN_PATH = "/tmp/react_shutdown"
SG_SYNC_PATH = "/tmp/react_sg_current.json"
SG_UPDATE_PATH = "/tmp/react_sg_update.json"  # main → worker: position updates after pick/place

DEBUG_DIR = os.path.join(_WORKER_DIR, "react_debug")
# ==========================================


class EmbeddingNet(nn.Module):
    """Same architecture as embedding_train.py - EfficientNet-B2 + projection to 1408d."""
    def __init__(self, embed_dim=1408):
        super().__init__()
        from torchvision import models
        self.backbone = models.efficientnet_b2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 1408),
            nn.BatchNorm1d(1408),
            nn.ReLU(),
            nn.Linear(1408, embed_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeds = self.projection(features)
        embeds = nn.functional.normalize(embeds, p=2, dim=1)
        return embeds


def preprocess_image(bgr_image):
    """Same preprocessing as embedding_train.py for consistency."""
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    transform = Compose([
        Resize((288, 288)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)


def get_instance_crop(map_view_img, mask, padding=10):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    image = cv2.bitwise_and(map_view_img, map_view_img, mask=mask)
    img_h, img_w = map_view_img.shape[:2]
    crop = image[
        max(y - padding, 0): min(y + padding + h, img_h),
        max(x - padding, 0): min(x + padding + w, img_w),
    ]
    return crop


CAMERA_HEIGHT = 1.5  # must match sayplan_ui.py camera translation Z
CAMERA_TILT_DEG = 20.0  # must match sayplan_ui.py camera tilt (degrees downward)

def estimate_3d_extent(bbox_xyxy, depth_np, robot_pos=None, robot_quat=None):
    """Estimate 3D extent (width, depth_dim, height) via depth point cloud.

    Back-projects bbox pixels to 3D world coords, then computes
    axis-aligned bounding box in world frame.
    Works correctly even when viewing from diagonal angles (no foreshortening).
    Returns [w, d, h] in meters, or None if depth is unusable.
    """
    if depth_np is None or depth_np.shape[0] <= 1:
        return None
    if robot_pos is None or robot_quat is None:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    img_h, img_w = depth_np.shape[:2]

    # Clamp bbox to image
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Camera intrinsics
    fx = CAMERA_FOCAL * CAMERA_RESOLUTION[0] / CAMERA_H_APERTURE
    fy = CAMERA_FOCAL * CAMERA_RESOLUTION[1] / CAMERA_V_APERTURE
    ppx = CAMERA_RESOLUTION[0] / 2.0
    ppy = CAMERA_RESOLUTION[1] / 2.0

    # ── Step 1: Get reference depth from bbox center (the object itself) ──
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    ref_region = 10
    ref_crop = depth_np[max(0, center_y - ref_region):min(img_h, center_y + ref_region),
                        max(0, center_x - ref_region):min(img_w, center_x + ref_region)]
    ref_valid = ref_crop[(ref_crop > 0.1) & (ref_crop < 20.0)]
    if len(ref_valid) < 3:
        return None
    ref_depth = float(np.median(ref_valid))

    # ── Step 2: Sample full bbox, keep only pixels near object depth ──
    stride = max(1, int(math.sqrt(bbox_w * bbox_h / 300)))
    depth_crop = depth_np[y1:y2, x1:x2]

    ys = np.arange(0, bbox_h, stride)
    xs = np.arange(0, bbox_w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x.flatten() + x1
    py = grid_y.flatten() + y1
    depths = depth_crop[grid_y.flatten(), grid_x.flatten()]

    # Depth-based object segmentation: keep pixels near object depth
    # Adaptive tolerance: tighter for close objects, wider for far ones
    # Close objects have better depth resolution, so we can be strict
    if ref_depth < 1.0:
        depth_tolerance = ref_depth * 0.15
    elif ref_depth < 2.0:
        depth_tolerance = ref_depth * 0.20
    else:
        depth_tolerance = ref_depth * 0.25
    obj_mask = ((depths > 0.1) &
                (depths >= ref_depth - depth_tolerance) &
                (depths <= ref_depth + depth_tolerance))
    if obj_mask.sum() < 5:
        return None

    px = px[obj_mask].astype(float)
    py = py[obj_mask].astype(float)
    d = depths[obj_mask].astype(float)

    # Back-project to camera frame
    x_cam = (px - ppx) * d / fx
    y_cam = (py - ppy) * d / fy
    z_cam = d

    # Camera frame → robot frame (tilt compensation)
    tilt_rad = np.radians(CAMERA_TILT_DEG)
    cos_t, sin_t = np.cos(tilt_rad), np.sin(tilt_rad)
    fwd = z_cam * cos_t - y_cam * sin_t
    up = z_cam * sin_t + y_cam * cos_t
    # robot frame: [forward, -left, up]
    pts_robot = np.stack([fwd, -x_cam, -up + CAMERA_HEIGHT], axis=1)  # (N, 3)

    # Robot frame → world frame
    w_q, qx, qy, qz = robot_quat
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w_q*qz), 2*(qx*qz + w_q*qy)],
        [2*(qx*qy + w_q*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w_q*qx)],
        [2*(qx*qz - w_q*qy), 2*(qy*qz + w_q*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    pts_world = (R @ pts_robot.T).T + robot_pos  # (N, 3)

    # Use 10th~90th percentile for tighter extent estimation
    # (5-95 was too loose, included background/floor points)
    p_lo = np.percentile(pts_world, 10, axis=0)
    p_hi = np.percentile(pts_world, 90, axis=0)
    w_3d = float(p_hi[0] - p_lo[0])  # X span
    d_3d = float(p_hi[1] - p_lo[1])  # Y span
    h_3d = float(p_hi[2] - p_lo[2])  # Z span

    # Clamp to reasonable range
    w_3d = max(0.03, min(w_3d, 3.0))
    d_3d = max(0.03, min(d_3d, 3.0))
    h_3d = max(0.03, min(h_3d, 3.0))

    return [round(w_3d, 3), round(d_3d, 3), round(h_3d, 3)]


def estimate_3d_position(bbox_xyxy, depth_np, robot_pos, robot_quat):
    if depth_np is None or depth_np.shape[0] <= 1:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    img_h, img_w = depth_np.shape[:2]

    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)
    cx = int(np.clip((x1 + x2) // 2, 0, img_w - 1))
    # Projection point: bbox center (represents object center)
    cy_proj = int(np.clip((y1 + y2) // 2, 0, img_h - 1))
    # Depth sampling point: lower 60% of bbox (more likely to hit object surface, not background)
    cy_depth = int(np.clip(y1 + int(bbox_h * 0.6), 0, img_h - 1))

    # Adaptive region size based on bbox (min 5px, max 25% of bbox dimension)
    region_w = max(5, int(bbox_w * 0.25))
    region_h = max(5, int(bbox_h * 0.25))
    y_start = max(0, cy_depth - region_h)
    y_end = min(img_h, cy_depth + region_h)
    x_start = max(0, cx - region_w)
    x_end = min(img_w, cx + region_w)
    depth_region = depth_np[y_start:y_end, x_start:x_end]
    valid_depth = depth_region[(depth_region > 0.1) & (depth_region < 20.0)]

    if len(valid_depth) < 3:
        return None

    # IQR filtering to remove outliers (background/foreground objects)
    q1 = np.percentile(valid_depth, 25)
    q3 = np.percentile(valid_depth, 75)
    iqr = q3 - q1
    lower = q1 - 1.0 * iqr
    upper = q3 + 1.0 * iqr
    filtered = valid_depth[(valid_depth >= lower) & (valid_depth <= upper)]
    if len(filtered) == 0:
        filtered = valid_depth

    depth_val = float(np.median(filtered))

    fx = CAMERA_FOCAL * CAMERA_RESOLUTION[0] / CAMERA_H_APERTURE
    fy = CAMERA_FOCAL * CAMERA_RESOLUTION[1] / CAMERA_V_APERTURE
    ppx = CAMERA_RESOLUTION[0] / 2.0
    ppy = CAMERA_RESOLUTION[1] / 2.0

    # Use bbox CENTER for projection (object center), depth from lower region
    x_cam = (cx - ppx) * depth_val / fx
    y_cam = (cy_proj - ppy) * depth_val / fy
    z_cam = depth_val

    # Camera frame → robot frame, accounting for camera tilt
    # Camera is tilted CAMERA_TILT_DEG degrees downward from horizontal
    tilt_rad = np.radians(CAMERA_TILT_DEG)
    cos_t = np.cos(tilt_rad)
    sin_t = np.sin(tilt_rad)

    # Rotate camera z,y around X axis by tilt angle
    # Camera z (forward) → robot: (cos*z - sin*y) forward, (sin*z + cos*y) up
    fwd = z_cam * cos_t - y_cam * sin_t
    up = z_cam * sin_t + y_cam * cos_t
    point_robot = np.array([fwd, -x_cam, -up + CAMERA_HEIGHT])

    w_q, qx, qy, qz = robot_quat
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w_q*qz), 2*(qx*qz + w_q*qy)],
        [2*(qx*qy + w_q*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w_q*qx)],
        [2*(qx*qz - w_q*qy), 2*(qy*qz + w_q*qx), 1 - 2*(qx*qx + qy*qy)]
    ])

    point_world = R @ point_robot + robot_pos
    return point_world


class ReactSceneGraphManager:
    def __init__(self, reference_sg_path, embedding_model, yolo_model,
                 ref_embeddings=None):
        self.embedding_model = embedding_model
        self.yolo_model = yolo_model

        with open(reference_sg_path, 'r') as f:
            self.reference_sg = json.load(f)

        self.ref_clusters = {}
        self._build_reference_clusters()
        self.current_sg = copy.deepcopy(self.reference_sg)
        self.change_log = []
        self.frame_count = 0
        self._new_candidates = {}
        self.coco_names = yolo_model.names if hasattr(yolo_model, 'names') else {}

        # Pre-computed reference embeddings from SAM crops
        self._ref_embeddings = ref_embeddings or {}
        # Assign to ref_clusters
        for obj_id, emb in self._ref_embeddings.items():
            if obj_id in self.ref_clusters:
                self.ref_clusters[obj_id]["embedding"] = emb

    def _get_mobility(self, tag):
        """Classify object mobility based on REACT paper categories."""
        tag_lower = tag.lower().strip()
        for category, info in MOBILITY.items():
            if any(t in tag_lower or tag_lower in t for t in info["tags"]):
                return category
        return "dynamic"  # default: unknown objects are treated as dynamic

    def _build_reference_clusters(self):
        init_time = time.time()
        for key, obj in self.reference_sg.items():
            obj_id = obj.get("id")
            tag = obj.get("object_tag", "unknown").lower()
            position = np.array(obj.get("bbox_center", [0, 0, 0]))
            extent = np.array(obj.get("bbox_extent", [0, 0, 0]))
            mobility = self._get_mobility(tag)
            self.ref_clusters[obj_id] = {
                "key": key,
                "tag": tag,
                "position": position,
                "extent": extent.copy(),
                "_initial_extent": extent.copy(),      # SG 원본 (floor)
                "_max_observed_extent": extent.copy(),  # 관측된 최대값 추적
                "embedding": None,
                "matched": True,
                "last_seen_time": init_time,
                "miss_count": 0,
                "mobility": mobility,
                "_pos_history": [],  # recent matched positions (last 10)
                "_flickering": False,
            }
            print(f"[REACT Worker]   id:{obj_id:3d} tag:{tag:20s} mobility:{mobility}")

    def process_frame(self, rgb_np, depth_np, robot_pos, robot_quat, robot_yaw):
        self.frame_count += 1

        # Check for SG updates from main process (pick/place position changes)
        self._apply_sg_updates()

        if rgb_np is None or len(rgb_np.shape) < 3:
            return []

        bgr = cv2.cvtColor(rgb_np[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

        results = self.yolo_model(bgr, conf=YOLO_CONF_THRESHOLD, imgsz=1280, verbose=False)
        if not results or results[0].boxes is None:
            # Nothing detected (wall/floor/ceiling) - do NOT count as miss
            self._last_detections = []
            return []

        detections = []
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        # Batch embedding computation
        crop_list = []
        crop_indices = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = self.coco_names.get(cls_id, f"class_{cls_id}").lower()
            bbox_xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

            instance_mask = None
            if masks is not None and i < len(masks.data):
                mask_tensor = masks.data[i]
                instance_mask = mask_tensor.cpu().numpy().astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (bgr.shape[1], bgr.shape[0]))

            pos_3d = estimate_3d_position(bbox_xyxy, depth_np, robot_pos, robot_quat)
            extent_3d = estimate_3d_extent(bbox_xyxy, depth_np, robot_pos, robot_quat)

            # Prepare crop for batch embedding (mask crop or bbox crop fallback)
            crop = None
            if instance_mask is not None:
                crop = get_instance_crop(bgr, instance_mask, padding=10)
            if crop is None or crop.size == 0:
                # Bbox crop fallback (YOLO-World has no seg masks)
                x1c, y1c, x2c, y2c = bbox_xyxy
                pad = 10
                img_h, img_w = bgr.shape[:2]
                crop = bgr[max(y1c-pad,0):min(y2c+pad,img_h),
                           max(x1c-pad,0):min(x2c+pad,img_w)]

            if crop is not None and crop.size > 0:
                try:
                    img_tensor = preprocess_image(crop)
                    crop_list.append(img_tensor)
                    crop_indices.append(i)
                except Exception:
                    pass

            detections.append({
                "cls_name": cls_name,
                "cls_id": cls_id,
                "confidence": conf,
                "bbox": bbox_xyxy,
                "position_3d": pos_3d,
                "extent_3d": extent_3d,
                "embedding": None,
                "mask": instance_mask,
            })

        # Batch embedding inference on GPU
        if crop_list:
            batch = torch.cat(crop_list, dim=0).cuda()
            with torch.no_grad():
                embeddings = self.embedding_model(batch).cpu()
            for idx, det_i in enumerate(crop_indices):
                detections[det_i]["embedding"] = embeddings[idx:idx+1]

        self._last_detections = detections

        # Match and update
        changes, matched_obj_ids = self._match_and_update(detections, robot_pos)

        # Guided detection: for trained-embedding objects missed by YOLO,
        # project their SG position to image, crop, and compare embeddings
        guided_changes = self._guided_detect_trained(
            rgb_np, depth_np, robot_pos, robot_quat, matched_obj_ids)
        changes.extend(guided_changes)
        for gc in guided_changes:
            if gc.get("obj_id"):
                matched_obj_ids.add(gc["obj_id"])

        # Check absent objects (pass total detections count for context)
        absent_changes = self._check_absent_objects(
            robot_pos, robot_yaw, matched_obj_ids,
            total_dets=len(detections), detections=detections)
        changes.extend(absent_changes)

        if changes:
            self.change_log.extend(changes)
            # Cap change_log to prevent unbounded growth
            if len(self.change_log) > 500:
                self.change_log = self.change_log[-200:]

        # Sync current SG to disk for main process
        self._sync_sg_to_disk()

        return changes

    def _match_and_update(self, detections, robot_pos):
        """
        Hungarian algorithm matching (aligned with original REACT):
        1. Build cost matrix of spatial distances between detections and refs
        2. Use scipy linear_sum_assignment for globally optimal assignment
        3. Verify embedding similarity for each assigned pair
        This finds the globally optimal assignment minimizing total travel distance.
        """
        changes = []
        matched_obj_ids = set()

        # Filter detections that have embeddings
        valid_dets = [(i, det) for i, det in enumerate(detections) if det["embedding"] is not None]
        if not valid_dets:
            return changes, matched_obj_ids

        # Build list of reference objects that have embeddings
        ref_list = [(obj_id, ref) for obj_id, ref in self.ref_clusters.items()
                    if ref["embedding"] is not None]
        if not ref_list:
            return changes, matched_obj_ids

        # Pre-compute absent status for all refs
        absent_refs = set()
        for obj_id, ref in ref_list:
            for o in self.current_sg.values():
                if o.get("id") == obj_id and o.get("_status") == "absent":
                    absent_refs.add(obj_id)
                    break

        # --- Hungarian algorithm matching (from original REACT) ---
        # Group compatible (det, ref) pairs and build cost matrix
        # Cost = spatial distance; filtered by tag compatibility and max radius
        # After assignment, verify embedding similarity

        n_dets = len(valid_dets)
        n_refs = len(ref_list)

        # Build cost matrix: rows=detections, cols=refs
        # Use large cost (999) for incompatible pairs to effectively block them
        BLOCK_COST = 999.0
        cost_matrix = np.full((n_dets, n_refs), BLOCK_COST, dtype=np.float64)

        for di, (_, det) in enumerate(valid_dets):
            det_pos = det["position_3d"]
            det_cls = det["cls_name"]

            for ri, (obj_id, ref) in enumerate(ref_list):
                if not self._tags_compatible(det_cls, ref["tag"]):
                    continue

                is_absent = obj_id in absent_refs

                if det_pos is not None:
                    spatial_dist = float(np.linalg.norm(det_pos[:2] - ref["position"][:2]))
                    # Flickering refs get tight radius to prevent cross-matching
                    if ref.get("_flickering", False):
                        max_radius = 0.8
                    else:
                        max_radius = 4.0 if is_absent else 2.5
                    if spatial_dist > max_radius:
                        continue  # blocked
                else:
                    spatial_dist = 0.0

                # Check embedding similarity before allowing assignment
                ref_emb = ref["embedding"]
                det_emb = det["embedding"]
                cos_sim = float(torch.mm(det_emb, ref_emb.t()))
                thresh = MATCH_THRESHOLD_TRAINED if obj_id in self._ref_embeddings else MATCH_THRESHOLD
                if cos_sim < thresh:
                    continue  # blocked

                # Valid pair: set cost as spatial distance (Hungarian minimizes total)
                cost_matrix[di, ri] = spatial_dist

        # Run Hungarian algorithm for globally optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_det_indices = set()
        matched_ref_ids = set()

        for di, ri in zip(row_ind, col_ind):
            if cost_matrix[di, ri] >= BLOCK_COST:
                continue  # blocked pair, skip

            obj_id = ref_list[ri][0]
            is_absent = obj_id in absent_refs

            det = valid_dets[di][1]
            det_emb = det["embedding"]
            det_pos = det["position_3d"]
            ref = self.ref_clusters[obj_id]

            # Match found!
            matched_det_indices.add(di)
            matched_ref_ids.add(obj_id)
            matched_obj_ids.add(obj_id)

            # --- Flicker detection: track position history (last 10 frames) ---
            if det_pos is not None:
                pos_hist = ref.get("_pos_history", [])
                pos_hist.append(det_pos[:2].copy())
                if len(pos_hist) > 10:
                    pos_hist = pos_hist[-10:]
                ref["_pos_history"] = pos_hist

                # Check for oscillation: count large jumps between consecutive entries
                if len(pos_hist) >= 4:
                    jump_count = 0
                    for pi in range(1, len(pos_hist)):
                        if float(np.linalg.norm(pos_hist[pi] - pos_hist[pi - 1])) > 0.8:
                            jump_count += 1
                    # 10프레임 내 3번 이상 큰 점프 → flickering (두 물체를 하나로 보고 있음)
                    if jump_count >= 3:
                        if not ref.get("_flickering", False):
                            print(f"[REACT Worker] Flicker detected: {ref['tag']}(id:{obj_id}) - "
                                  f"{jump_count} jumps in {len(pos_hist)} frames, tightening match radius")
                        ref["_flickering"] = True
                    else:
                        ref["_flickering"] = False
                else:
                    ref["_flickering"] = False

            # Adaptive embedding EMA: faster adaptation for better viewpoint robustness
            ref["embedding"] = 0.85 * ref["embedding"] + 0.15 * det_emb
            ref["embedding"] = nn.functional.normalize(ref["embedding"], p=2, dim=1)

            ref["matched"] = True
            ref["last_seen_time"] = time.time()
            ref["miss_count"] = 0

            # Re-detect absent objects → RETURNED
            if is_absent:
                for key, obj in self.current_sg.items():
                    if obj.get("id") == obj_id and obj.get("_status") == "absent":
                        obj.pop("_status", None)
                        # Position/extent NOT updated — keep original SG values
                        ref["_returned_cooldown"] = 20
                        change = {
                            "type": "RETURNED",
                            "obj_id": obj_id,
                            "tag": ref["tag"],
                            "position": det_pos.tolist() if det_pos is not None else ref["position"].tolist(),
                            "timestamp": time.time(),
                        }
                        changes.append(change)
                        print(f"[REACT Worker] {ref['tag']}(id:{obj_id}) RETURNED!")
                        break

            # Position update: NO continuous EMA (aligned with original REACT)
            # Only update SG position on confirmed MOVED (large, consistent displacement).
            # Small noisy position differences from depth estimation are IGNORED
            # to prevent drift. Original REACT records positions per-scan, not per-frame.
            mob = ref.get("mobility", "dynamic")
            if det_pos is not None and mob == "dynamic":
                old_pos = ref["position"]
                if "origin_position" not in ref:
                    ref["origin_position"] = old_pos.copy()
                origin_pos = ref["origin_position"]

                shift_from_origin = float(np.linalg.norm(det_pos[:2] - origin_pos[:2]))

                # Track consecutive detections at a displaced location
                move_count = ref.get("_move_count", 0)
                if shift_from_origin > 0.5:
                    move_count += 1
                    ref["_move_count"] = move_count
                else:
                    ref["_move_count"] = 0

                # Confirmed MOVED: large displacement sustained over multiple frames
                if shift_from_origin > 0.5 and move_count >= 5:
                    ref["position"] = det_pos.copy()
                    ref["origin_position"] = det_pos.copy()
                    ref["_move_count"] = 0
                    self._update_sg_position(obj_id, det_pos)
                    change = {
                        "type": "MOVED",
                        "obj_id": obj_id,
                        "tag": ref["tag"],
                        "old_position": origin_pos.tolist(),
                        "new_position": det_pos.tolist(),
                        "distance": round(shift_from_origin, 2),
                        "timestamp": time.time(),
                    }
                    changes.append(change)
                # else: position NOT updated — no EMA drift

            # --- Extent (bbox size) update: DISABLED ---
            # Depth-based extent estimation from diagonal camera is unreliable.
            # Initial SG extent (from offline mapping) is more accurate.
            # Extent is kept fixed; only position is updated in real-time.

        # Unmatched detections → potential new objects
        for di, (_, det) in enumerate(valid_dets):
            if di not in matched_det_indices:
                self._handle_new_candidate(det, changes, robot_pos=robot_pos)

        return changes, matched_obj_ids

    def _project_world_to_image(self, world_pos, robot_pos, robot_quat):
        """Project a 3D world position to 2D image pixel coordinates."""
        # World → robot frame
        w_q, qx, qy, qz = robot_quat
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - w_q*qz), 2*(qx*qz + w_q*qy)],
            [2*(qx*qy + w_q*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - w_q*qx)],
            [2*(qx*qz - w_q*qy), 2*(qy*qz + w_q*qx), 1 - 2*(qx*qx + qy*qy)]
        ])
        point_robot = R.T @ (np.array(world_pos) - np.array(robot_pos))

        # Robot frame → camera frame (inverse of estimate_3d_position transform)
        # robot frame: [forward, -left, -up + CAMERA_HEIGHT]
        fwd, left_neg, up_neg_plus_h = point_robot
        up = -(up_neg_plus_h - CAMERA_HEIGHT)  # -up + H → up = -(val - H)

        tilt_rad = np.radians(CAMERA_TILT_DEG)
        cos_t, sin_t = np.cos(tilt_rad), np.sin(tilt_rad)
        # Inverse of: fwd = z*cos - y*sin, up = z*sin + y*cos
        z_cam = fwd * cos_t + up * sin_t
        y_cam = -fwd * sin_t + up * cos_t
        x_cam = -left_neg

        if z_cam <= 0.1:
            return None  # behind camera

        fx = CAMERA_FOCAL * CAMERA_RESOLUTION[0] / CAMERA_H_APERTURE
        fy = CAMERA_FOCAL * CAMERA_RESOLUTION[1] / CAMERA_V_APERTURE
        ppx = CAMERA_RESOLUTION[0] / 2.0
        ppy = CAMERA_RESOLUTION[1] / 2.0

        px = int(x_cam * fx / z_cam + ppx)
        py = int(y_cam * fy / z_cam + ppy)

        img_w, img_h = CAMERA_RESOLUTION
        if 0 <= px < img_w and 0 <= py < img_h:
            return (px, py, z_cam)
        return None

    def _guided_detect_trained(self, rgb_np, depth_np, robot_pos, robot_quat, matched_obj_ids):
        """For trained-embedding objects missed by YOLO, project SG position to image,
        crop around that location, and compare embeddings directly."""
        changes = []
        if not self._ref_embeddings:
            return changes

        bgr = cv2.cvtColor(rgb_np[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_h, img_w = bgr.shape[:2]

        for obj_id in self._ref_embeddings:
            if obj_id in matched_obj_ids:
                continue  # already matched by YOLO pipeline
            if obj_id not in self.ref_clusters:
                continue
            ref = self.ref_clusters[obj_id]

            # Project object's known position to image
            obj_pos_3d = ref["position"]
            proj = self._project_world_to_image(obj_pos_3d, robot_pos, robot_quat)
            if proj is None:
                continue
            px, py, z_cam = proj

            # Estimate crop size from object extent and depth
            ext = ref.get("extent", np.array([0.3, 0.3, 0.3]))
            obj_size = max(float(ext[0]), float(ext[1]), 0.15)
            fx = CAMERA_FOCAL * CAMERA_RESOLUTION[0] / CAMERA_H_APERTURE
            pixel_radius = int(obj_size * fx / z_cam * 0.7)
            pixel_radius = max(30, min(pixel_radius, 150))

            x1 = max(0, px - pixel_radius)
            y1 = max(0, py - pixel_radius)
            x2 = min(img_w, px + pixel_radius)
            y2 = min(img_h, py + pixel_radius)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = bgr[y1:y2, x1:x2]
            try:
                img_tensor = preprocess_image(crop)
                with torch.no_grad():
                    det_emb = self.embedding_model(img_tensor.cuda()).cpu()
                ref_emb = ref["embedding"]
                cos_sim = float(torch.mm(det_emb, ref_emb.t()))

                if cos_sim >= MATCH_THRESHOLD_TRAINED:
                    # Matched via guided detection — mark as seen but do NOT move position
                    # (depth-based position estimates are too noisy to trust)
                    if True:
                        was_absent = False
                        ref["matched"] = True
                        ref["miss_count"] = 0
                        ref["last_seen_time"] = time.time()
                        for _k, _obj in self.current_sg.items():
                            if _obj.get("id") == obj_id:
                                if _obj.get("_status") == "absent":
                                    was_absent = True
                                _obj.pop("_status", None)
                                break
                        matched_obj_ids.add(obj_id)
                        if was_absent:
                            changes.append({
                                "type": "RETURNED",
                                "obj_id": obj_id,
                                "tag": ref["tag"],
                                "position": new_pos.tolist(),
                                "timestamp": time.time(),
                            })
                        print(f"[REACT Guided] Matched {ref['tag']}(id:{obj_id}) sim={cos_sim:.3f} at px=({px},{py})")
            except Exception as e:
                print(f"[REACT Guided] Error for id:{obj_id}: {e}")
                continue

        return changes

    def _handle_new_candidate(self, det, changes, robot_pos=None):
        det_pos = det["position_3d"]
        det_name = det["cls_name"]
        det_emb = det.get("embedding", None)
        conf = det["confidence"]

        # 학습된 임베딩 태그와 매칭되는지 확인
        has_trained = any(
            self._tags_compatible(det_name, self.ref_clusters[oid]["tag"])
            for oid in self._ref_embeddings if oid in self.ref_clusters
        )

        # Distance-aware confidence threshold:
        # 학습된 임베딩 객체: 더 낮은 threshold (임베딩이 더 신뢰할 수 있으므로)
        # Close (< 4m): conf >= 0.4 (trained: 0.25)
        # Far (>= 4m): conf >= 0.6 (trained: 0.4)
        if det_pos is not None and robot_pos is not None:
            robot_dist = float(np.linalg.norm(det_pos[:2] - robot_pos[:2]))
            if has_trained:
                min_conf = 0.25 if robot_dist < 4.0 else 0.4
            else:
                min_conf = 0.4 if robot_dist < 4.0 else 0.6
        else:
            min_conf = 0.3 if has_trained else 0.5

        if conf < min_conf:
            return

        # Block NEW for static tags (walls, floors, etc.)
        det_mobility = self._get_mobility(det_name)
        if det_mobility == "static":
            return  # structural elements never truly "appear" as new

        # Semi-static (furniture): allow NEW only if no similar object exists nearby.
        # This prevents duplicate desk/shelf from YOLO tag confusion,
        # but allows genuinely new furniture placed in the scene.
        if det_mobility == "semi_static" and det_pos is not None:
            for obj_id, ref in self.ref_clusters.items():
                if self._tags_compatible(det_name, ref["tag"]):
                    dist = float(np.linalg.norm(det_pos[:2] - ref["position"][:2]))
                    if dist < 3.0:
                        return  # similar furniture already nearby — likely duplicate

        # First: check if an ABSENT object of same tag exists nearby → RETURNED
        if det_pos is not None:
            for obj_id, ref in self.ref_clusters.items():
                if not self._tags_compatible(det_name, ref["tag"]):
                    continue
                # Check if this object is ABSENT
                is_absent = False
                for _key, _obj in self.current_sg.items():
                    if _obj.get("id") == obj_id and _obj.get("_status") == "absent":
                        is_absent = True
                        break
                if not is_absent:
                    continue
                # ABSENT object of same tag found → treat as RETURNED
                # (skip distance check - it may have been moved to a new location)
                for _key, _obj in self.current_sg.items():
                    if _obj.get("id") == obj_id:
                        _obj.pop("_status", None)
                        # Position/extent NOT updated — keep original SG values
                        # Depth-based estimates are too noisy to trust
                        break
                # Keep original SG position for ref_clusters too
                # ref["position"] stays as-is (original SG position)
                ref["miss_count"] = 0
                ref["matched"] = True
                ref["last_seen_time"] = time.time()
                ref["_returned_cooldown"] = 20  # prevent re-ABSENT cycle
                # Update embedding if available (conservative)
                if det_emb is not None and ref["embedding"] is not None:
                    ref["embedding"] = 0.95 * ref["embedding"] + 0.05 * det_emb
                    ref["embedding"] = nn.functional.normalize(ref["embedding"], p=2, dim=1)
                elif det_emb is not None:
                    ref["embedding"] = det_emb

                change = {
                    "type": "RETURNED",
                    "obj_id": obj_id,
                    "tag": ref["tag"],
                    "position": det_pos.tolist(),
                    "timestamp": time.time(),
                }
                changes.append(change)
                print(f"[REACT Worker] {ref['tag']}(id:{obj_id}) RETURNED at new location!")
                # Clear any new_candidates for this tag to prevent duplicate NEW
                self._new_candidates = {k: v for k, v in self._new_candidates.items()
                                        if k[0] != det_name}
                return  # handled as RETURNED, not NEW

        # Check if a PRESENT object of same tag already exists nearby → skip (duplicate)
        if det_pos is not None:
            for obj_id, ref in self.ref_clusters.items():
                if not self._tags_compatible(det_name, ref["tag"]):
                    continue
                is_absent = any(
                    o.get("id") == obj_id and o.get("_status") == "absent"
                    for o in self.current_sg.values()
                )
                if is_absent:
                    continue  # skip absent ones (handled above)
                dist_to_ref = float(np.linalg.norm(det_pos[:2] - ref["position"][:2]))
                if dist_to_ref < 0.8:
                    # Already a present object of same type very nearby → not truly new
                    return

        # Grid key: use 3D position if available, otherwise bbox center as proxy
        if det_pos is not None:
            grid_key = (det_name,
                        round(det_pos[0] * 2) / 2,
                        round(det_pos[1] * 2) / 2)
        else:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            grid_key = (det_name, cx // 50, cy // 50)  # 50px grid

        if grid_key in self._new_candidates:
            count, first_time, last_pos = self._new_candidates[grid_key]
            self._new_candidates[grid_key] = (count + 1, first_time, det_pos)
            # Distance-adaptive strikes: close objects need fewer confirmations
            # Semi-static (furniture) requires more strikes to avoid false NEW
            if det_pos is not None and robot_pos is not None:
                rdist = float(np.linalg.norm(det_pos[:2] - robot_pos[:2]))
                if det_mobility == "semi_static":
                    required_strikes = 8 if rdist < 4.0 else 12
                else:
                    required_strikes = 3 if rdist < 4.0 else 6
            else:
                required_strikes = 8 if det_mobility == "semi_static" else 5
            if count + 1 >= required_strikes:
                # Use depth-based 3D extent if available, else rough pixel estimate
                est_extent = det.get("extent_3d")
                if est_extent is None:
                    bbox = det["bbox"]
                    bbox_w = float(bbox[2] - bbox[0])
                    bbox_h = float(bbox[3] - bbox[1])
                    scale = 0.003  # rough px-to-meter at ~1m depth
                    est_extent = [
                        max(0.05, min(bbox_w * scale, 0.5)),
                        max(0.05, min(bbox_h * scale, 0.5)),
                        max(0.05, min(min(bbox_w, bbox_h) * scale, 0.3)),
                    ]
                change = {
                    "type": "NEW",
                    "tag": det_name,
                    "position": det_pos.tolist() if det_pos is not None else None,
                    "confidence": det["confidence"],
                    "timestamp": time.time(),
                }
                changes.append(change)
                if det_pos is not None:
                    self._add_new_object(det_name, det_pos, det_emb, est_extent)
                del self._new_candidates[grid_key]
        else:
            self._new_candidates[grid_key] = (1, time.time(), det_pos)

        # Purge stale candidates older than 30s
        now = time.time()
        stale = [k for k, (_, t, _) in self._new_candidates.items() if now - t > 30.0]
        for k in stale:
            del self._new_candidates[k]

    def _tags_compatible(self, det_tag, ref_tag):
        det_tag = det_tag.lower().strip()
        ref_tag = ref_tag.lower().strip()
        if det_tag == ref_tag:
            return True

        tag_map = {
            "dining table": ["desk", "table", "counter", "dining table"],
            "cabinet": ["cabinet", "shelf", "dresser", "nightstand", "drawer",
                        "wardrobe", "bookshelf", "cupboard", "storage"],
            "couch": ["sofa", "couch"],
            "tv": ["tv", "monitor", "screen", "television"],
            "refrigerator": ["refrigerator", "fridge"],
            "microwave": ["microwave"],
            "oven": ["oven", "stove"],
            "sink": ["sink", "basin"],
            "chair": ["chair", "seat", "folded chair", "stool", "bench"],
            "bed": ["bed"],
            "toilet": ["toilet"],
            "bottle": ["bottle"],
            "cup": ["cup", "mug", "glass"],
            "bowl": ["bowl"],
            "book": ["book"],
            "clock": ["clock"],
            "vase": ["vase"],
            "potted plant": ["plant", "potted plant"],
        }

        for key, aliases in tag_map.items():
            det_matches = det_tag == key or det_tag in aliases
            ref_matches = ref_tag == key or ref_tag in aliases
            if det_matches and ref_matches:
                return True
        return False

    def _update_sg_position(self, obj_id, new_pos):
        for key, obj in self.current_sg.items():
            if obj.get("id") == obj_id:
                obj["bbox_center"] = new_pos.tolist()
                break

    def _update_sg_extent(self, obj_id, new_extent):
        """Update bbox_extent and bbox_volume in current_sg for a matched object."""
        ext = new_extent if isinstance(new_extent, list) else new_extent.tolist()
        ext = [round(v, 4) for v in ext]
        volume = round(ext[0] * ext[1] * ext[2], 6)
        for key, obj in self.current_sg.items():
            if obj.get("id") == obj_id:
                obj["bbox_extent"] = ext
                obj["bbox_volume"] = volume
                break

    def _add_new_object(self, tag, position, embedding=None, extent=None):
        if extent is None:
            extent = [0.15, 0.15, 0.15]  # small default for dynamic objects
        max_id = max((obj.get("id", 0) for obj in self.current_sg.values()), default=0)
        new_id = max_id + 1
        new_key = f"object_{len(self.current_sg) + 1}"

        volume = extent[0] * extent[1] * extent[2]
        self.current_sg[new_key] = {
            "id": new_id,
            "object_tag": tag,
            "object_caption": None,
            "bbox_extent": extent,
            "bbox_center": position.tolist(),
            "bbox_volume": round(volume, 4),
            "_is_new": True,  # flag: added by REACT, not original SG
        }

        self.ref_clusters[new_id] = {
            "key": new_key,
            "tag": tag,
            "position": np.array(position),
            "extent": np.array(extent),
            "embedding": embedding,  # store detection embedding so it can be matched
            "matched": True,
            "last_seen_time": time.time(),
            "miss_count": 0,
            "mobility": self._get_mobility(tag),
            "_pos_history": [],
            "_flickering": False,
        }

    def _check_absent_objects(self, robot_pos, robot_yaw, matched_obj_ids,
                              total_dets=0, fov_angle_deg=90.0,
                              detections=None):
        """
        ABSENT 판정 (mobility 반영):
        - static: ABSENT 판정 안 함 (벽/바닥/환풍구 등)
        - semi_static: 높은 miss 횟수 필요 (가구/대형 가전)
        - dynamic: 낮은 miss 횟수로 판정 (접시/책/컵 등)
        - FOV 내에서 다른 물체가 1개 이상 매칭되었을 때만 miss 카운트
        """
        changes = []
        half_fov = math.radians(fov_angle_deg / 2.0)

        # Count how many FOV objects were matched (not just YOLO dets)
        n_matched = len(matched_obj_ids)

        for obj_id, ref in self.ref_clusters.items():
            if self.current_sg.get(ref["key"], {}).get("_status") == "absent":
                continue

            # Cooldown after RETURNED: skip miss counting
            cooldown = ref.get("_returned_cooldown", 0)
            if cooldown > 0:
                ref["_returned_cooldown"] = cooldown - 1
                continue

            mob = ref.get("mobility", "dynamic")
            mob_info = MOBILITY.get(mob, MOBILITY["dynamic"])

            # Static objects never go absent
            if mob_info["miss_threshold"] >= 999:
                continue

            # Semi-static: also effectively never go absent
            # (YOLO-World often can't detect counter/sink/oven reliably)
            if mob == "semi_static":
                continue

            obj_pos = ref["position"]
            diff = obj_pos[:2] - robot_pos[:2]
            dist = np.linalg.norm(diff)

            if dist > 3.5 or dist < 0.3:
                ref["miss_count"] = 0
                continue

            angle_to_obj = math.atan2(diff[1], diff[0])
            angle_diff = abs(math.atan2(math.sin(angle_to_obj - robot_yaw),
                                        math.cos(angle_to_obj - robot_yaw)))

            if angle_diff > half_fov:
                ref["miss_count"] = 0
                continue

            # In FOV - check detection status
            if obj_id in matched_obj_ids:
                ref["miss_count"] = 0
                continue

            # NOT matched - count miss only when at least 1 other object was matched
            # (proves camera is seeing real objects, not just wall)
            if n_matched == 0 and total_dets == 0:
                continue

            # If YOLO detected a same-tag object in this frame (even if embedding
            # didn't match), the object is likely still there — don't count as miss.
            # This prevents ABSENT when embedding drifts but YOLO still sees it.
            det_tags_in_frame = [d["cls_name"].lower().strip() for d in detections] if detections else []
            ref_tag_lower = ref["tag"].lower().strip()
            tag_seen = any(self._tags_compatible(dt, ref_tag_lower) for dt in det_tags_in_frame)
            if tag_seen:
                ref["miss_count"] = max(0, ref.get("miss_count", 0) - 1)
                continue

            miss_threshold = mob_info["miss_threshold"]
            ref["miss_count"] = ref.get("miss_count", 0) + 1
            print(f"[REACT Worker] {ref['tag']}(id:{obj_id}) [{mob}] in FOV @{dist:.1f}m "
                  f"not matched (dets={total_dets}, matched={n_matched}) "
                  f"miss {ref['miss_count']}/{miss_threshold}")

            if ref["miss_count"] >= miss_threshold:
                # Check if this was a REACT-added object (false positive)
                is_react_added = False
                for key, obj in list(self.current_sg.items()):
                    if obj.get("id") == obj_id:
                        if obj.get("_is_new"):
                            # False positive: remove entirely instead of marking absent
                            is_react_added = True
                            del self.current_sg[key]
                            print(f"[REACT Worker] FALSE POSITIVE removed: {ref['tag']}(id:{obj_id})")
                        else:
                            obj["_status"] = "absent"
                        break

                change = {
                    "type": "ABSENT",
                    "obj_id": obj_id,
                    "tag": ref["tag"],
                    "expected_position": ref["position"].tolist(),
                    "mobility": mob,
                    "timestamp": time.time(),
                    "_was_false_positive": is_react_added,
                }
                changes.append(change)

                # Clean up ref_clusters for false positives
                if is_react_added:
                    ref["_removed"] = True

        return changes

    def _apply_sg_updates(self):
        """Read position updates from main process (after pick/place) and apply to internal state."""
        try:
            if not os.path.exists(SG_UPDATE_PATH):
                return
            with open(SG_UPDATE_PATH, 'r') as f:
                updates = json.load(f)
            os.remove(SG_UPDATE_PATH)

            for upd in updates:
                obj_id = upd.get("obj_id")
                action = upd.get("action")  # "moved", "removed", "returned"
                new_pos = upd.get("position")

                if obj_id is None:
                    continue

                if action == "moved" and new_pos is not None:
                    new_pos_arr = np.array(new_pos, dtype=np.float64)
                    # Update ref_clusters
                    if obj_id in self.ref_clusters:
                        ref = self.ref_clusters[obj_id]
                        ref["position"] = new_pos_arr
                        ref["origin_position"] = new_pos_arr.copy()
                        ref["_move_count"] = 0
                        ref["_pos_history"] = []
                        print(f"[REACT Worker] SG update: {ref['tag']}(id:{obj_id}) moved to {new_pos}")
                    # Update current_sg
                    for _k, _obj in self.current_sg.items():
                        if _obj.get("id") == obj_id:
                            _obj["bbox_center"] = new_pos
                            _obj.pop("_status", None)
                            break

                elif action == "removed":
                    # Object picked up — mark absent
                    if obj_id in self.ref_clusters:
                        self.ref_clusters[obj_id]["matched"] = False
                    for _k, _obj in self.current_sg.items():
                        if _obj.get("id") == obj_id:
                            _obj["_status"] = "absent"
                            break
                    print(f"[REACT Worker] SG update: id:{obj_id} removed (picked)")

        except (json.JSONDecodeError, IOError):
            pass
        except Exception as e:
            print(f"[REACT Worker] SG update error: {e}")

    def _sync_sg_to_disk(self):
        """Write current scene graph to disk so main process can read it."""
        tmp_path = SG_SYNC_PATH + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(self.current_sg, f)
            os.replace(tmp_path, SG_SYNC_PATH)
        except Exception:
            pass

    def get_change_summary(self):
        new_objs = [c for c in self.change_log if c["type"] == "NEW"]
        absent = [c for c in self.change_log if c["type"] == "ABSENT"]
        returned = [c for c in self.change_log if c["type"] == "RETURNED"]
        moved = [c for c in self.change_log if c["type"] == "MOVED"]

        summary = ""
        if new_objs:
            summary += f"NEW({len(new_objs)}): "
            summary += ", ".join([f"{c['tag']}" for c in new_objs[-5:]])
            summary += "\n"
        if moved:
            summary += f"MOVED({len(moved)}): "
            summary += ", ".join([f"{c['tag']}(id:{c['obj_id']}) {c['distance']}m" for c in moved[-5:]])
            summary += "\n"
        if returned:
            summary += f"RETURNED({len(returned)}): "
            summary += ", ".join([f"{c['tag']}(id:{c['obj_id']})" for c in returned[-5:]])
            summary += "\n"
        if absent:
            summary += f"ABSENT({len(absent)}): "
            summary += ", ".join([f"{c['tag']}(id:{c['obj_id']})" for c in absent[-5:]])
            summary += "\n"
        return summary if summary else "No scene changes detected."


def _save_debug_frame(rgb_np, manager, debug_dir, frame_num):
    """Save annotated debug image with YOLO boxes + embedding match results."""
    try:
        os.makedirs(debug_dir, exist_ok=True)
        vis = cv2.cvtColor(rgb_np[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)

        dets = manager._last_detections if hasattr(manager, '_last_detections') else []

        # Build match info: which detection matched which reference object
        match_info = {}
        if dets:
            valid_dets = [(i, d) for i, d in enumerate(dets) if d["embedding"] is not None]
            ref_list = [(oid, r) for oid, r in manager.ref_clusters.items()
                        if r["embedding"] is not None]
            for di, (det_idx, det) in enumerate(valid_dets):
                best_sim, best_tag, best_oid = -1.0, None, None
                best_compat = False
                for oid, ref in ref_list:
                    if not manager._tags_compatible(det["cls_name"], ref["tag"]):
                        continue  # only show tag-compatible matches
                    cos_sim = float(torch.mm(det["embedding"], ref["embedding"].t()))
                    if cos_sim > best_sim:
                        best_sim = cos_sim
                        best_tag = ref["tag"]
                        best_oid = oid
                        best_compat = True
                if best_compat:
                    match_info[det_idx] = (best_tag, best_oid, best_sim)
                else:
                    # No compatible ref → show as NEW candidate
                    match_info[det_idx] = (None, None, -1.0)

        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            yolo_cls = det["cls_name"]
            conf = det["confidence"]

            if i in match_info:
                ref_tag, ref_oid, cos_sim = match_info[i]
                if ref_tag is None:
                    # No tag-compatible match → NEW candidate
                    color = (0, 200, 255)  # yellow-orange
                    label = f"YOLO:{yolo_cls}({conf:.2f})"
                    match_label = "-> NEW (no compatible ref)"
                else:
                    matched = cos_sim >= MATCH_THRESHOLD
                    color = (0, 200, 0) if matched else (0, 128, 255)
                    label = f"YOLO:{yolo_cls}({conf:.2f})"
                    match_label = f"-> {ref_tag}(id:{ref_oid}) sim:{cos_sim:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, label, (x1, y1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis, match_label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (128, 128, 128)
                label = f"YOLO:{yolo_cls}({conf:.2f}) [no emb]"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                cv2.putText(vis, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status text
        info = f"F:{frame_num} dets:{len(dets)} matched:{sum(1 for v in match_info.values() if v[2]>=MATCH_THRESHOLD)}"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save latest + numbered
        cv2.imwrite(os.path.join(debug_dir, "latest.jpg"), vis)
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_num:04d}.jpg"), vis)
        print(f"[REACT Worker] Debug saved: {debug_dir}/frame_{frame_num:04d}.jpg "
              f"(img shape={rgb_np.shape}, min={rgb_np.min()}, max={rgb_np.max()})")
    except Exception as e:
        print(f"[REACT Worker] Debug save error: {e}")


def _save_change_log(manager, new_changes, debug_dir):
    """Save cumulative change log + current scene graph diff to JSON."""
    try:
        log_path = os.path.join(debug_dir, "scene_changes.json")

        # Build current state summary for each object
        sg_state = {}
        for key, obj in manager.current_sg.items():
            obj_id = obj.get("id")
            ref = manager.ref_clusters.get(obj_id, {})
            sg_state[key] = {
                "id": obj_id,
                "tag": obj.get("object_tag", "unknown"),
                "position": obj.get("bbox_center"),
                "extent": obj.get("bbox_extent"),
                "volume": obj.get("bbox_volume"),
                "status": obj.get("_status", "present"),
                "mobility": ref.get("mobility", "unknown"),
                "miss_count": ref.get("miss_count", 0),
            }

        log_data = {
            "frame_count": manager.frame_count,
            "timestamp": time.time(),
            "total_changes": len(manager.change_log),
            "change_summary": manager.get_change_summary(),
            "all_changes": manager.change_log,
            "current_scene_graph": sg_state,
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"[REACT Worker] Change log saved: {log_path}")
    except Exception as e:
        print(f"[REACT Worker] Change log save error: {e}")


def _compute_reference_embeddings(embedding_model):
    """Pre-compute average embedding for each object from SAM training crops."""
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    transform = Compose([
        Resize((288, 288)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ref_embeddings = {}

    if not os.path.isdir(REF_CROPS_DIR):
        print(f"[REACT Worker] WARNING: Crops dir not found: {REF_CROPS_DIR}")
        return ref_embeddings

    for folder in sorted(os.listdir(REF_CROPS_DIR)):
        if not folder.startswith("obj_"):
            continue
        parts = folder.split('_', 2)
        if len(parts) < 3:
            continue
        obj_id = int(parts[1])
        folder_path = os.path.join(REF_CROPS_DIR, folder)

        imgs = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        if not imgs:
            continue

        # Sample up to 20 crops for efficiency
        sample_imgs = imgs[:20] if len(imgs) > 20 else imgs
        tensors = []
        for img_name in sample_imgs:
            try:
                pil_img = Image.open(os.path.join(folder_path, img_name)).convert("RGB")
                tensors.append(transform(pil_img))
            except Exception:
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).cuda()
        with torch.no_grad():
            embeds = embedding_model(batch)
        avg_embed = embeds.mean(dim=0, keepdim=True)  # (1, 512)
        avg_embed = nn.functional.normalize(avg_embed, p=2, dim=1)
        ref_embeddings[obj_id] = avg_embed.cpu()
        print(f"[REACT Worker]   obj_{obj_id:03d}: {len(tensors)} crops -> embedding OK")

    return ref_embeddings


def main():
    if len(sys.argv) < 2:
        print("Usage: react_worker.py <scene_graph_json_path>")
        sys.exit(1)

    sg_path = sys.argv[1]
    print(f"[REACT Worker] Loading models on GPU...")

    # Extract unique tags from scene graph for YOLO-World classes
    with open(sg_path, 'r') as f:
        sg_data = json.load(f)
    sg_tags = list(set(obj.get("object_tag", "unknown").lower()
                       for obj in sg_data.values()))

    # COCO 80 classes + kitchen/household objects for broad detection
    # YOLO-World is open-vocab but needs class list - make it comprehensive
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    KITCHEN_EXTRA = [
        "plate", "pan", "pot", "mug", "glass", "jar", "container",
        "cutting board", "tray", "napkin", "towel", "sponge",
        "food", "fruit", "vegetable", "bread", "egg", "meat", "cheese",
        "can", "box", "bag", "basket", "bucket",
        "paper", "newspaper", "magazine",
        "toy", "doll", "ball",
        "shoe", "hat", "glove",
        "pillow", "blanket", "cloth",
        "lamp", "candle", "flower",
        "cabinet", "shelf", "drawer", "counter", "desk", "table",
        "stove", "vent", "folded chair",
    ]
    all_classes = list(set(sg_tags + COCO_CLASSES + KITCHEN_EXTRA))
    print(f"[REACT Worker] YOLO-World classes ({len(all_classes)}): {all_classes}")

    # Load YOLO-World (open-vocabulary detection)
    yolo_model = YOLO(YOLO_MODEL_NAME)
    yolo_model.set_classes(all_classes)
    yolo_model(np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
               conf=YOLO_CONF_THRESHOLD, imgsz=1280, verbose=False)

    # Load trained embedding model (same arch as embedding_train.py)
    embedding_model = EmbeddingNet(embed_dim=EMBEDDING_DIM)
    state_dict = torch.load(TRAINED_EMBEDDING_PATH, map_location="cuda", weights_only=True)
    embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.cuda().eval()

    # Warm up
    with torch.no_grad():
        embedding_model(torch.randn(1, 3, 288, 288).cuda())

    print(f"[REACT Worker] Models loaded. GPU: {torch.cuda.get_device_name(0)}")
    print(f"[REACT Worker] Trained embedding: {TRAINED_EMBEDDING_PATH}")

    # Pre-compute reference embeddings from SAM crops
    ref_embeddings = _compute_reference_embeddings(embedding_model)
    print(f"[REACT Worker] Reference embeddings computed: {len(ref_embeddings)} objects")

    manager = ReactSceneGraphManager(
        reference_sg_path=sg_path,
        embedding_model=embedding_model,
        yolo_model=yolo_model,
        ref_embeddings=ref_embeddings,
    )
    print(f"[REACT Worker] Reference SG: {len(manager.reference_sg)} objects")

    # Clean up old files
    for f in [INPUT_PATH, OUTPUT_PATH, SHUTDOWN_PATH, SG_SYNC_PATH]:
        if os.path.exists(f):
            os.remove(f)

    # Clean debug folder from previous run
    import shutil
    if os.path.isdir(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"[REACT Worker] Debug folder cleaned: {DEBUG_DIR}")

    # Write initial SG
    manager._sync_sg_to_disk()

    print(f"[REACT Worker] Ready. Waiting for frames...")
    last_input_mtime = 0

    while True:
        # Check shutdown signal
        if os.path.exists(SHUTDOWN_PATH):
            print("[REACT Worker] Shutdown signal received.")
            break

        # Check for new input
        if not os.path.exists(INPUT_PATH):
            time.sleep(0.1)
            continue

        try:
            current_mtime = os.path.getmtime(INPUT_PATH)
        except OSError:
            time.sleep(0.1)
            continue

        if current_mtime <= last_input_mtime:
            time.sleep(0.1)
            continue

        last_input_mtime = current_mtime

        # try:
        #     data = np.load(INPUT_PATH, allow_pickle=True)
        #     rgb = data["rgb"]
        #     depth = data["depth"]
        #     robot_pos = data["robot_pos"]
        #     robot_quat = data["robot_quat"]
        #     robot_yaw = float(data["robot_yaw"])
        try:
            # [수정] np.load가 파일을 계속 물고 있어 꼬이는 현상 방지 (with문 + copy)
            with np.load(INPUT_PATH, allow_pickle=True) as data:
                rgb = data["rgb"].copy()
                depth = data["depth"].copy()
                robot_pos = data["robot_pos"].copy()
                robot_quat = data["robot_quat"].copy()
                robot_yaw = float(data["robot_yaw"])
        except Exception as e:
            print(f"[REACT Worker] Failed to read input: {e}")
            time.sleep(0.1)
            continue

        # Process frame
        t0 = time.time()
        changes = manager.process_frame(rgb, depth, robot_pos, robot_quat, robot_yaw)
        dt = time.time() - t0

        # Debug: save annotated frame every 10 frames
        if manager.frame_count % 3 == 0 or manager.frame_count <= 3:
            _save_debug_frame(rgb, manager, DEBUG_DIR, manager.frame_count)

        # Count YOLO detections from latest result
        n_det = len(manager._last_detections) if hasattr(manager, '_last_detections') else 0
        has_depth = depth is not None and depth.shape[0] > 1
        print(f"[REACT Worker] Frame {manager.frame_count}: {int(dt*1000)}ms, "
              f"pos=({robot_pos[0]:.1f},{robot_pos[1]:.1f}), "
              f"depth={'OK '+str(depth.shape) if has_depth else 'NONE'}, "
              f"dets={n_det}, changes={len(changes)}")

        # Write output
        output = {
            "changes": changes,
            "frame_count": manager.frame_count,
            "object_count": len(manager.current_sg),
            "total_changes": len(manager.change_log),
            "change_summary": manager.get_change_summary(),
            "inference_time_ms": int(dt * 1000),
            "timestamp": time.time(),
        }

        tmp_path = OUTPUT_PATH + ".tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(output, f)
            os.replace(tmp_path, OUTPUT_PATH)
        except Exception:
            pass

        if changes:
            for c in changes:
                extra = ""
                if c["type"] == "RETURNED":
                    pos = c.get("position")
                    extra = f" at ({pos[0]:.1f},{pos[1]:.1f})" if pos else ""
                print(f"[REACT Worker] [{c['type']}] {c.get('tag', '?')}{extra}")

            # Save scene graph change log
            _save_change_log(manager, changes, DEBUG_DIR)

    print("[REACT Worker] Exiting.")


if __name__ == "__main__":
    main()
