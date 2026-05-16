"""
SceneUpdate 설정 파일 - 모든 경로/파라미터를 여기서 관리

Planner modes:
  sayplan  — SayPlan brain (static SG only, no REACT)
  pred     — PRED brain (static SG only, no REACT)
  predreact — PRED brain + REACT (real-time SG updates)
  custom   — Custom brain + REACT + depth freespace
"""
import os
import numpy as np

# ================= [Paths] =================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SG_JSON_FILE = os.path.join(_BASE_DIR, "data", "newscene", "obj_json_newscene.json")
HIERARCHY_JSON_FILE = os.path.join(_BASE_DIR, "data", "newscene", "hierarchical_scene_graph.json")
EDGE_JSON_FILE = os.path.join(_BASE_DIR, "data", "newscene", "edge_json_newscene.json")
SURFACE_JSON_FILE = os.path.join(_BASE_DIR, "data", "newscene", "surface_freespace_from_pcd.json")
PRECOMPUTED_FREESPACE_FILE = os.path.join(_BASE_DIR, "data", "newscene", "surface_json_precomputed.json")
# Per-room split scene graph
ROOMS_DIR = os.path.join(_BASE_DIR, "data", "newscene", "rooms")
ROOMS_INDEX_FILE = os.path.join(ROOMS_DIR, "_index.json")

# REACT Worker
REACT_WORKER_PYTHON = os.path.join(_BASE_DIR, "..", "react_venv", "bin", "python3")
REACT_WORKER_SCRIPT = os.path.join(_BASE_DIR, "react_worker.py")

# REACT IPC paths
REACT_INPUT_PATH = "/tmp/react_input.npz"
REACT_OUTPUT_PATH = "/tmp/react_output.json"
REACT_SHUTDOWN_PATH = "/tmp/react_shutdown"
REACT_SG_SYNC_PATH = "/tmp/react_sg_current.json"

# Model paths (react_worker에서 사용)
YOLO_MODEL_PATH = os.path.join(_BASE_DIR, "models", "yolov8x-worldv2.pt")
EMBEDDING_MODEL_PATH = os.path.join(_BASE_DIR, "data", "newscene", "best_embedding_model_newscene.pth")
REF_CROPS_DIR = "/home/sungbin/extra/data/cg_replica/session_20260403_144334/training_crops_det"

# ================= [Gemini via Vertex AI] =================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "gen-lang-client-0633975915")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Camera intrinsics
CAMERA_FOCAL = 2.12
CAMERA_H_APERTURE = 6.02
CAMERA_V_APERTURE = 3.39
CAMERA_RESOLUTION = (1280, 720)

# REACT Config
REACT_UPDATE_INTERVAL = 1.2

# ================= [Robot] =================
USD_PATH = os.path.join(_BASE_DIR, "scene", "behavior1", "World0.usd")
ROBOT_NAME_KEYWORD = "fetch"
ARRIVAL_TOLERANCE = 0.5
APPROACH_DISTANCE = 0.5
MOVE_LINEAR_SPEED = 0.5
MOVE_ANGULAR_SPEED = 2.0
SIM_DT = 1.0 / 60.0

# Fetch wheel geometry (from og_dataset/models/fetch/fetch.yaml)
WHEEL_RADIUS = 0.0613   # metres
WHEEL_BASE   = 0.372    # wheel axle length (metres)
# Fetch is a true differential-drive robot (no skid-steer compensation needed)
SKID_STEER_MULTIPLIER = 1.0
