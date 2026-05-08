import math
import subprocess
import queue
# from queue import Queue
import re
import shutil
import time
import threading
import cv2
import numpy as np
import habitat
import habitat_sim
import habitat_sim.agent
from habitat.utils.visualizations import maps
from habitat.config import read_write
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0, RearrangeEpisode
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
    HeadRGBSensorConfig,
    HeadDepthSensorConfig,
)
import random
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Optional, Tuple
from collections import deque
import random
import os
from glob import glob
from mapping import first_map
from mapping import first_map_for_next_time
from mapping import second_map
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from memory_save import compare_objects_location
from memory_save import read_json_file
from longterm_save import get_divided_positions
from longterm_save import get_static_objects_in_regions
from longterm_save import extract_regions_from_json
from semantic_utils import object_type_from_handle
from actions_utils import (
    wrap_to_pi,
    get_fetch_base_pose,
    find_target_object_position,
    teleport_fetch_base,
    debug_teleport_to_object,
    distance_xz,
    is_finite_point,
    try_shortest_path,
    project_goal_to_navmesh,
    sync_follower_agent_state,
    dest_obj_to_xyz,
    drop_pending_nav_actions_for_agent,
    find_target_object_position_live,
    resolve_hierarchical_target_with_conceptgraph,
    normalize_navigation_target,
    require_fully_specified_navigation_target,
    resolve_runtime_object_from_target_context,
    object_labels_semantically_match,
    score_object_query_against_text,
)
from dynamic_scene_graph_updater import apply_dynamic_scene_graph_update
import json
import importlib
import sys
import argparse
from types import SimpleNamespace
import openai
import base64
import requests
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_magnum
from habitat_sim.agent.agent import SixDOFPose


_INIT_T0 = time.perf_counter()


def _init_log(message: str):
    elapsed = time.perf_counter() - _INIT_T0
    print(f"[init +{elapsed:7.2f}s] {message}", flush=True)


# 목적지까지 남은 거리 디버그용
_DEBUG_OVERLAY_LINES = []
_DEBUG_OVERLAY_LOCK = threading.Lock()
_NAVMESH_TOPDOWN_CACHE = {}

def set_debug_overlay(*lines):
    global _DEBUG_OVERLAY_LINES
    with _DEBUG_OVERLAY_LOCK:
        _DEBUG_OVERLAY_LINES = [str(x) for x in lines if x is not None]

def get_debug_overlay():
    with _DEBUG_OVERLAY_LOCK:
        return list(_DEBUG_OVERLAY_LINES)



HABITAT_LAB_SRC = "/home/yuchaehee/long_term_memory_project/habitat-lab/habitat-lab"
if HABITAT_LAB_SRC not in sys.path:
    sys.path.insert(0, HABITAT_LAB_SRC)
from habitat.articulated_agents.robots.fetch_robot import FetchRobot

task_queue = queue.Queue()
ENABLE_INTERACTIVE_VIEW = os.environ.get("KARMA_INTERACTIVE_VIEW", "1") == "1"
START_TASK_EXECUTOR = os.environ.get("KARMA_START_TASK_EXECUTOR", "1") == "1"
ENABLE_HOV_MULTICAM = os.environ.get("KARMA_ENABLE_HOV_MULTICAM", "0") == "1"
HOV_OBSERVER_RGB_UUID = "observer_up_rgb"
HOV_OBSERVER_DEPTH_UUID = "observer_up_depth"
HOV_OBSERVER_HEIGHT_OFFSET_M = 0.35
HOV_OBSERVER_PITCH_DEG = 40.0
HOV_DYNAMIC_SENSOR_CONFIGS = {}
T_HABITAT_TO_OPENCV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

api_key = os.environ.get("OPENAI_API_KEY")
directory_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/short_term'
task_description_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/logs/task_description.json'
results_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/analysis_results.json'
_DYNAMIC_SCENE_UPDATE_ROOT = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/dynamic_scene_updates"
_DYNAMIC_SCENE_UPDATE_COUNTER = None
_FETCH_VISION_TUCK_ARM_JOINTS = np.array(
    # Fetch official tuck pose with torso_lift removed.
    [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0],
    dtype=np.float32,
)
_PUT_OBJECT_SETTLE_STEPS_BEFORE_TUCK = 15
_PUT_OBJECT_POST_RESTORE_SETTLE_STEPS = 5

def load_task_description():
    try:
        with open(task_description_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get("task_description", "")
    except FileNotFoundError:
        return ""
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# Function to analyze an image with a given task
def analyze_image(image_path, task):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required.")
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 아, 이미지 상태 분석할 때, 리스트를 줬구낭 
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "As an image analysis expert, your task is to infer the state of objects in the image through step-by-step reasoning."},
            {"role": "user", "content": f"1. Provide a detailed description of this image.\n2. From the given task [Task], extract the relevant content from the first step's image description that pertains to the mentioned objects.\n3. Based on the object descriptions extracted in the second step, match each object to one of the following states: heated, cooked, sliced, cleaned, dirty, filled, used up, off, on, opened, closed, none.\n4. Summarize the results from step three in the following format: object: state. Please only output the content of the last step summary."},
            {"role": "user", "content": f"[Task]: {task}"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def analyze_specific_image(directory, filename, task):
    if os.path.exists(results_file_path):
        with open(results_file_path, 'r', encoding='utf-8') as file:
            results = json.load(file)
    else:
        results = {}

    image_path = os.path.join(directory, filename)
    if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
        result = analyze_image(image_path, task)
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        # 결과를 따로 저장한다
        objects_states = {}
        for line in content.split('\n'):
            if ': ' in line:
                obj, state = line.split(': ', 1) # 한 번만 분할하도록 수정. 즉, 객체 이름에 ": "가 포함될 수 있으므로, 첫 번째 ": "만 분할하도록 변경
                objects_states[obj] = state
        results[filename] = objects_states

    # 합쳐진 결과를 JSON 파일에 저장
    with open(results_file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    return results

def save_agent_view(image, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, filename), image)


def rgb_observation_to_bgr(rgb_obs):
    """
    Habitat color observation(RGB/RGBA)을 OpenCV BGR 이미지로 변환.
    """
    img = np.asarray(rgb_obs)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def save_rgb_observation_png(rgb_obs, save_path, filename):
    """
    Habitat color observation(RGB/RGBA)을 OpenCV 저장용 BGR로 변환 후 PNG 저장.
    """
    img = rgb_observation_to_bgr(rgb_obs)
    save_agent_view(img, save_path, filename)


def _safe_step_action_name(action_name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(action_name or "").strip())
    return safe or "unknown"


def save_initial_rgb_camera_views(observations, save_path, prefix="init"):
    """
    초기 관측에서 RGB 카메라별 이미지를 저장한다.
    파일명 예시: init_head_rgb.png
    """
    saved = []
    for key in sorted(observations.keys()):
        val = np.asarray(observations[key])
        # RGB/RGBA 형태의 관측만 저장
        if val.ndim == 3 and val.shape[2] in (3, 4):
            filename = f"{prefix}_{key}.png"
            save_rgb_observation_png(val, save_path, filename)
            saved.append(filename)
    return saved


def _obs_by_suffix_optional(obs, suffix: str):
    if suffix in obs:
        return obs[suffix]
    for k, v in obs.items():
        if k.endswith(suffix):
            return v
    return None


def _jsonify_for_log(value):
    if isinstance(value, dict):
        return {str(k): _jsonify_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_for_log(v) for v in value]
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


def _write_json_file(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resize_to_height(img, target_h: int):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = float(target_h) / float(max(h, 1))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def _build_navmesh_topdown_panel(sim_obj, map_resolution: int = 512):
    if sim_obj is None or not hasattr(sim_obj, "pathfinder"):
        return None
    if not sim_obj.pathfinder.is_loaded:
        return None

    try:
        if hasattr(sim_obj, "articulated_agent"):
            agent_pos = np.array(sim_obj.articulated_agent.base_pos, dtype=np.float32)
        else:
            agent_pos = np.array(sim_obj.get_agent(0).get_state().position, dtype=np.float32)
    except Exception:
        return None

    floor_y = float(agent_pos[1])
    cache_key = (round(floor_y, 2), int(map_resolution))
    if cache_key not in _NAVMESH_TOPDOWN_CACHE:
        try:
            top_down = maps.get_topdown_map(
                sim_obj.pathfinder,
                height=floor_y,
                map_resolution=map_resolution,
                draw_border=True,
            )
            top_down_rgb = maps.colorize_topdown_map(top_down)
            _NAVMESH_TOPDOWN_CACHE[cache_key] = top_down_rgb
        except Exception:
            return None

    try:
        navmesh_bgr = cv2.cvtColor(
            _NAVMESH_TOPDOWN_CACHE[cache_key].copy(), cv2.COLOR_RGB2BGR
        )
    except Exception:
        return None

    try:
        gx, gy = maps.to_grid(
            float(agent_pos[2]),
            float(agent_pos[0]),
            navmesh_bgr.shape[:2],
            pathfinder=sim_obj.pathfinder,
        )
        gx = int(np.clip(gx, 0, navmesh_bgr.shape[0] - 1))
        gy = int(np.clip(gy, 0, navmesh_bgr.shape[1] - 1))
        cv2.circle(navmesh_bgr, (gy, gx), 6, (0, 0, 255), -1)
        cv2.circle(navmesh_bgr, (gy, gx), 9, (255, 255, 255), 1)
    except Exception:
        pass

    cv2.putText(
        navmesh_bgr,
        f"navmesh top-down | y={floor_y:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return navmesh_bgr


def _render_spf_realtime_frame(
    obs,
    step_idx: int,
    action_name: str,
    nav_dist: float,
    best_dist: float,
    goal_radius: float,
    success: bool,
):
    head_rgb = _obs_by_suffix_optional(obs, "head_rgb")
    scene_rgb = _obs_by_suffix_optional(obs, "scene_camera_rgb")
    down_rgb = _obs_by_suffix_optional(obs, "head_down_rgb")

    head_bgr = rgb_observation_to_bgr(head_rgb) if head_rgb is not None else None
    scene_bgr = rgb_observation_to_bgr(scene_rgb) if scene_rgb is not None else None
    down_bgr = rgb_observation_to_bgr(down_rgb) if down_rgb is not None else None

    panel_imgs = [x for x in [head_bgr, scene_bgr, down_bgr] if x is not None]
    if not panel_imgs:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    target_h = min(img.shape[0] for img in panel_imgs)
    panel_imgs = [_resize_to_height(img, target_h) for img in panel_imgs]
    panel = cv2.hconcat(panel_imgs)

    status = "SUCCESS" if success else "RUNNING"
    lines = [
        f"SPF step={step_idx} action={action_name}",
        f"nav_dist={nav_dist:.3f}m best={best_dist:.3f}m goal_radius={goal_radius:.3f}m",
        f"status={status} | Q or ESC: stop",
    ]
    y = 30
    for line in lines:
        cv2.putText(
            panel,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return panel


def run_interactive_navmesh_debug_view(sim, fetch_robots):
    """
    viewer2.py의 N키 navmesh 토글과 유사한 디버그 뷰.
    - N: navmesh 시각화 on/off
    - Q 또는 ESC: 디버그 뷰 종료 후 스크립트 계속 진행
    """
    if not ENABLE_INTERACTIVE_VIEW:
        print("[init] interactive view disabled (KARMA_INTERACTIVE_VIEW=0)")
        return

    window_name = "KARMA Debug View (N: navmesh toggle, Q/ESC: continue)"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
    except cv2.error:
        print("[init] OpenCV window unavailable. Skip interactive navmesh debug view.")
        return

    print("[init] interactive debug view started: N=toggle navmesh, Q/ESC=continue")
    while True:
        if fetch_robots:
            fetch_robots[0].update()

        obs = get_current_observations()
        head_rgb = _obs_by_suffix_optional(obs, "head_rgb")
        head_bgr = rgb_observation_to_bgr(head_rgb)
        top_rgb = _obs_by_suffix_optional(obs, HOV_OBSERVER_RGB_UUID)
        if top_rgb is None:
            top_rgb = _obs_by_suffix_optional(obs, "top_rgb")
        if top_rgb is None:
            raise KeyError(f"Observer RGB not found. available={list(obs.keys())}")
        top_bgr = rgb_observation_to_bgr(top_rgb)

        cv2.putText(
            head_bgr,
            "head_rgb",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            top_bgr,
            f"top_rgb | navmesh={'ON' if sim.navmesh_visualization else 'OFF'}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        panel = cv2.hconcat([head_bgr, top_bgr])
        cv2.imshow(window_name, panel)

        key = cv2.waitKey(10) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord("N")):
            if sim.pathfinder.is_loaded:
                sim.navmesh_visualization = not sim.navmesh_visualization
                print(f"[init] toggle navmesh -> {sim.navmesh_visualization}")
            else:
                print("[init] navmesh not loaded")

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

def save_regions_to_json(regions, filename='/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/longterm_memory.json'):
    data = {}
    for center, objects in regions.items():
        center_key = f'({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})'
        packed = []
        for obj in objects:
            row = {"objectType": obj["objectType"], "position": obj["position"]}
            if obj.get("objectName"):
                row["objectName"] = obj["objectName"]
            packed.append(row)
        data[center_key] = packed

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def _center_clearance(pathfinder, center, max_search_radius: float = 2.0):
    p = np.array(pathfinder.snap_point(np.array(center, dtype=np.float32)), dtype=np.float32)
    if not np.all(np.isfinite(p)):
        return center, p, float("-inf")
    d = float(pathfinder.distance_to_closest_obstacle(p, max_search_radius))
    if not np.isfinite(d):
        d = max_search_radius
    return center, p, d


def _pick_spawn_from_centers_with_clearance(
    pathfinder,
    centers,
    used_centers=None,
    min_clearance: float = 0.35,
    max_search_radius: float = 2.0,
):
    """
    center 후보 중 obstacle 여유거리가 충분한 지점을 우선 선택.
    충분한 후보가 없으면 clearance가 가장 큰 center를 fallback으로 반환.
    """
    if used_centers is None:
        used_centers = set()

    candidates = [c for c in centers if c not in used_centers]
    random.shuffle(candidates)
    if not candidates:
        return None

    evaluated = [
        _center_clearance(pathfinder, c, max_search_radius=max_search_radius)
        for c in candidates
    ]

    safe = [e for e in evaluated if e[2] >= min_clearance]
    if safe:
        return random.choice(safe)

    return max(evaluated, key=lambda x: x[2])

# 이 함수 쓰는 곳 없는 것 같아서 일단 주석 처리
# def generate_random_position_from_ilst(available_positions):
#     rand_position = random.choice(available_positions)
#     available_positions.remove(rand_position)
#     return rand_position, available_positions

# ---------------------------------- 아래 세 개 함수는 일단 잘 모르겠음. 나중에 다시 봐야할듯 ----------------------------------
def closest_node(node, nodes, no_robot, clost_node_location):
    """
    목표점(node) 기준으로 robots마다 가까운 후보 좌표를 하나씩 뽑아 반환

    Args:
        node: 목표 좌표 [x, y, z] (예: 가고 싶은 물체 주변 점)
        nodes: 후보 좌표 리스트 [(x,y,z), ...] (예: reachable_positions)
        no_robot: 로봇 수
        clost_node_location: 로봇별 오프셋 리스트
            - stuck 발생 시 이 값을 증가시켜 "다음으로 가까운 후보"를 고르게 함

    Returns:
        crps: 로봇별 선택된 후보 좌표 리스트
    """
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i*5) + clost_node_location[i]]
        crps.append(nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def generate_video(input_path, prefix, char_id=0, image_synthesis=['normal'], frame_rate=5, output_path=None):
    """ Generate a video of an episode """
    if output_path is None:
        output_path = input_path

    vid_folder = '{}/{}/{}/'.format(input_path, prefix, char_id)
    if not os.path.isdir(vid_folder):
        print("The input path: {} you specified does not exist.".format(input_path))
    else:
        for vid_mod in image_synthesis:
            command_set = ['ffmpeg', '-i',
                             '{}/Action_%04d_0_{}.png'.format(vid_folder, vid_mod), 
                             '-framerate', str(frame_rate),
                             '-pix_fmt', 'yuv420p',
                             '{}/video_{}.mp4'.format(output_path, vid_mod)]
            subprocess.call(command_set)
            print("Video generated at ", '{}/video_{}.mp4'.format(output_path, vid_mod))

# action/obs 키 해석 헬퍼 추가
def _resolve_action_name(env, short_name: str) -> str:
    keys = env.action_space.spaces.keys()
    if short_name in keys:
        return short_name
    for k in keys:
        if k.endswith(short_name):
            return k
    raise KeyError(f"Action '{short_name}' not found. available={list(keys)}")

def _obs_by_suffix(obs, suffix: str):
    if suffix in obs:
        return obs[suffix]
    for k, v in obs.items():
        if k.endswith(suffix):
            return v
    raise KeyError(f"Obs '{suffix}' not found. available={list(obs.keys())}")


def _sensor_state_by_suffix(agent, suffix: str):
    state = agent.get_state()
    if suffix in state.sensor_states:
        return state.sensor_states[suffix]
    for key, sensor_state in state.sensor_states.items():
        if str(key).endswith(suffix):
            return sensor_state
    raise KeyError(f"Sensor '{suffix}' not found. available={list(state.sensor_states.keys())}")


def _get_sensor_view_pose(sim_obj, agent_id: int = 0, sensor_suffix: str = "head_rgb"):
    """
    특정 sensor가 "실제로 어디에서, 어느 방향을 보고 있는지"를 읽는다.

    왜 필요한가:
    - 지금까지 AlignToTarget은 base_rot만 보고 회전 오차를 계산했다.
    - 하지만 사용자가 보는 건 head_rgb 이미지이므로, 정렬 기준도
      "베이스 정면"이 아니라 "헤드 카메라 optical axis"여야 자연스럽다.

    반환:
    - pos: sensor의 world position
    - yaw_rad: sensor optical axis의 world yaw

    구현 메모:
    - Habitat sensor local frame에서 optical axis는 -Z 방향이다.
    - 따라서 sensor rotation matrix에 [0, 0, -1]을 곱해 world forward vector를 만든 뒤,
      그 xz 방향으로 yaw를 계산한다.
    """
    agent = sim_obj.get_agent(agent_id)
    sensor_state = _sensor_state_by_suffix(agent, sensor_suffix)

    pos = np.array(sensor_state.position, dtype=np.float32)
    rot = np.array(quat_to_magnum(sensor_state.rotation).to_matrix(), dtype=np.float32)

    # Habitat camera optical axis: local -Z
    forward_world = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    forward_xz = np.array([forward_world[0], forward_world[2]], dtype=np.float32)
    if not np.all(np.isfinite(forward_xz)) or float(np.linalg.norm(forward_xz)) < 1e-8:
        raise ValueError(f"Invalid sensor forward vector for suffix={sensor_suffix}")

    # 중요:
    # - sensor forward vector에서 직접 atan2를 하면 "표준 world yaw(CCW)" 기준 각도가 나온다.
    # - 그런데 현재 코드베이스의 base_rot / teleport / AlignToTarget은
    #   그 반대 부호 convention을 사실상 사용하고 있다.
    # - 실제 디버그 로그에서도 RotateLeft 시
    #     base_yaw: +1.91deg 증가
    #     view_yaw: -1.91deg 감소
    #   로 읽혀서, sensor yaw만 부호가 반대로 해석되고 있었다.
    #
    # 따라서 여기서 sensor-derived yaw를 한 번 뒤집어서
    # get_fetch_base_pose()가 반환하는 yaw convention과 맞춘다.
    yaw_rad = float(np.arctan2(-forward_world[2], forward_world[0]))
    yaw_rad = wrap_to_pi(yaw_rad)
    return pos, yaw_rad


def sync_hov_observer_state() -> bool:
    if not ENABLE_HOV_MULTICAM:
        return False
    if HOV_OBSERVER_RGB_UUID not in sim._sensors:
        return False

    try:
        main_agent = sim.get_agent(0)
        agent_state = main_agent.get_state()
        head_state = _sensor_state_by_suffix(main_agent, "head_rgb")
    except Exception:
        return False

    observer_pos = np.array(head_state.position, dtype=np.float32)
    observer_pos[1] += HOV_OBSERVER_HEIGHT_OFFSET_M
    observer_rot = head_state.rotation * quat_from_angle_axis(
        np.deg2rad(HOV_OBSERVER_PITCH_DEG),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    agent_state.sensor_states = {}

    for sensor_uuid in (
        HOV_OBSERVER_RGB_UUID,
        HOV_OBSERVER_DEPTH_UUID,
    ):
        if sensor_uuid not in sim._sensors:
            continue
        agent_state.sensor_states[sensor_uuid] = SixDOFPose(
            position=np.array(observer_pos, dtype=np.float32),
            rotation=observer_rot,
        )

    main_agent.set_state(
        agent_state,
        reset_sensors=False,
        infer_sensor_states=False,
    )
    return True


def get_current_observations(base_obs=None):
    if not ENABLE_HOV_MULTICAM:
        return base_obs if base_obs is not None else sim.get_sensor_observations()

    sync_hov_observer_state()
    sim_obs = sim.get_sensor_observations()
    if base_obs is None:
        return sim_obs

    merged = dict(base_obs)
    for sensor_uuid in (
        HOV_OBSERVER_RGB_UUID,
        HOV_OBSERVER_DEPTH_UUID,
    ):
        if sensor_uuid in sim_obs:
            merged[sensor_uuid] = sim_obs[sensor_uuid]
    return merged


def _register_hov_dynamic_sensor_config(
    config_key: str,
    *,
    width: int,
    height: int,
    hfov: float,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    normalize_depth: bool = True,
):
    HOV_DYNAMIC_SENSOR_CONFIGS[config_key] = SimpleNamespace(
        width=int(width),
        height=int(height),
        hfov=float(hfov),
        min_depth=float(min_depth),
        max_depth=float(max_depth),
        normalize_depth=bool(normalize_depth),
    )


def _add_hov_observer_sensor(
    uuid: str,
    sensor_type,
    *,
    width: int = 1000,
    height: int = 1000,
    hfov: float = 90.0,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    normalize_depth: bool = True,
):
    if uuid in sim._sensors:
        return

    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = [height, width]
    spec.position = [0.0, 0.0, 0.0]
    spec.orientation = [0.0, 0.0, 0.0]
    spec.hfov = hfov
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    if sensor_type == habitat_sim.SensorType.DEPTH:
        spec.min_depth = min_depth
        spec.max_depth = max_depth
        spec.normalize_depth = normalize_depth
    sim.add_sensor(spec, agent_id=0)


def setup_hov_observer_sensors() -> None:
    if not ENABLE_HOV_MULTICAM:
        return

    _add_hov_observer_sensor(HOV_OBSERVER_RGB_UUID, habitat_sim.SensorType.COLOR)
    _register_hov_dynamic_sensor_config(
        "observer_up_rgb_sensor",
        width=1000,
        height=1000,
        hfov=90.0,
    )

    _add_hov_observer_sensor(
        HOV_OBSERVER_DEPTH_UUID,
        habitat_sim.SensorType.DEPTH,
        normalize_depth=False,
    )
    _register_hov_dynamic_sensor_config(
        "observer_up_depth_sensor",
        width=1000,
        height=1000,
        hfov=90.0,
        min_depth=0.0,
        max_depth=10.0,
        # sim.add_sensor() depth observations come back in raw meters here,
        # unlike Habitat-Lab wrapped env.step() depth which is normalized.
        normalize_depth=False,
    )

    sync_hov_observer_state()


def _env_step(action_dict):
    global obs
    raw_obs = env.step(action_dict)
    obs = get_current_observations(base_obs=raw_obs)
    return obs


def _sensor_cfg_by_suffix(cfg_suffix: str):
    suffixes = [str(cfg_suffix)]
    if not str(cfg_suffix).endswith("_sensor"):
        suffixes.append(f"{cfg_suffix}_sensor")

    agents = cfg.habitat.simulator.agents
    for agent_name in agents.keys():
        sensors = getattr(getattr(agents, agent_name), "sim_sensors", {})
        for key in sensors.keys():
            key_str = str(key)
            if any(key_str.endswith(suffix) for suffix in suffixes):
                return getattr(sensors, key)

    dynamic_cfgs = getattr(sys.modules[__name__], "HOV_DYNAMIC_SENSOR_CONFIGS", {})
    for key, value in dynamic_cfgs.items():
        key_str = str(key)
        if any(key_str.endswith(suffix) for suffix in suffixes):
            return value

    raise KeyError(f"Sensor config with suffix '{cfg_suffix}' not found.")


def _quat_to_rotmat(quat) -> np.ndarray:
    try:
        return np.array(quat_to_magnum(quat).to_matrix(), dtype=np.float32)
    except Exception:
        pass

    if hasattr(quat, "to_matrix"):
        return np.array(quat.to_matrix(), dtype=np.float32)

    if hasattr(quat, "vector") and hasattr(quat, "scalar"):
        vec = np.array(quat.vector, dtype=np.float32).reshape(3)
        coeffs_xyzw = np.array(
            [float(vec[0]), float(vec[1]), float(vec[2]), float(quat.scalar)],
            dtype=np.float32,
        )
        return R.from_quat(coeffs_xyzw).as_matrix().astype(np.float32)

    if all(hasattr(quat, x) for x in ("x", "y", "z", "w")):
        coeffs_xyzw = np.array(
            [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
            dtype=np.float32,
        )
        return R.from_quat(coeffs_xyzw).as_matrix().astype(np.float32)

    coeffs = np.asarray(quat, dtype=np.float32).reshape(-1)
    if coeffs.size != 4:
        raise TypeError(f"Unsupported quaternion value: {quat!r}")
    return R.from_quat(coeffs).as_matrix().astype(np.float32)


def _camera_transform_from_sensor_state(sensor_state) -> np.ndarray:
    pos = np.array(sensor_state.position, dtype=np.float32)
    rot = _quat_to_rotmat(sensor_state.rotation)

    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, :3] = rot
    T_world_cam[:3, 3] = pos
    return T_world_cam @ T_HABITAT_TO_OPENCV


def _camera_intrinsics_from_cfg(rgb_cfg_suffix: str) -> Dict[str, float]:
    rgb_cfg = _sensor_cfg_by_suffix(rgb_cfg_suffix)
    width = int(rgb_cfg.width)
    height = int(rgb_cfg.height)
    hfov_deg = float(getattr(rgb_cfg, "hfov", 90.0))
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0 - 0.5
    cy = height / 2.0 - 0.5
    return {
        "width": width,
        "height": height,
        "hfov_deg": hfov_deg,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


def _depth_obs_to_meters(depth_obs, depth_cfg_suffix: str) -> Tuple[np.ndarray, Dict[str, float]]:
    depth_cfg = _sensor_cfg_by_suffix(depth_cfg_suffix)
    depth_min = float(getattr(depth_cfg, "min_depth", 0.0))
    depth_max = float(getattr(depth_cfg, "max_depth", 10.0))
    normalize_depth = bool(getattr(depth_cfg, "normalize_depth", True))

    depth = np.squeeze(np.asarray(depth_obs, dtype=np.float32))
    depth = np.nan_to_num(
        depth,
        nan=depth_max,
        posinf=depth_max,
        neginf=depth_min,
    )
    if normalize_depth:
        depth = depth_min + np.clip(depth, 0.0, 1.0) * (depth_max - depth_min)
    depth = np.clip(depth, depth_min, depth_max).astype(np.float32)

    return depth, {
        "depth_min": depth_min,
        "depth_max": depth_max,
        "normalize_depth": normalize_depth,
    }


def _write_manual_control_compatible_camera_files(
    *,
    event_dir: str,
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_u16: np.ndarray,
    T_cam: np.ndarray,
    intrinsics: Dict[str, float],
    depth_scale: float,
):
    results_dir = os.path.join(event_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    rgb_frame_path = os.path.join(results_dir, "frame000000.jpg")
    depth_frame_path = os.path.join(results_dir, "depth000000.png")
    traj_path = os.path.join(event_dir, "traj.txt")
    params_path = os.path.join(event_dir, "cam_params.json")
    render_camera_path = os.path.join(event_dir, "render_camera.json")
    depth_m_path = os.path.join(results_dir, "depth000000.npy")

    cv2.imwrite(rgb_frame_path, rgb_bgr)
    cv2.imwrite(depth_frame_path, depth_u16)
    np.save(depth_m_path, depth_m.astype(np.float32))

    params_payload = {
        "camera": {
            "w": int(intrinsics["width"]),
            "h": int(intrinsics["height"]),
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "scale": float(depth_scale),
        }
    }
    _write_json_file(params_path, params_payload)

    render_camera_payload = {
        "class_name": "PinholeCameraParameters",
        "extrinsic": T_cam.flatten().tolist(),
        "intrinsic": {
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "intrinsic_matrix": [
                float(intrinsics["fx"]),
                0.0,
                float(intrinsics["cx"]),
                0.0,
                float(intrinsics["fy"]),
                float(intrinsics["cy"]),
                0.0,
                0.0,
                1.0,
            ],
        },
    }
    _write_json_file(render_camera_path, render_camera_payload)

    with open(traj_path, "w", encoding="utf-8") as f:
        f.write(" ".join(format(x, ".18e") for x in T_cam.flatten()) + "\n")

    return {
        "results_dir": results_dir,
        "rgb_frame_path": rgb_frame_path,
        "depth_frame_path": depth_frame_path,
        "depth_m_path": depth_m_path,
        "traj_path": traj_path,
        "params_path": params_path,
        "render_camera_path": render_camera_path,
    }


def _set_arm_pose(agent_idx: int, joint_pose, *, close_gripper: bool = True):
    art = sim.get_agent_data(agent_idx).articulated_agent
    pose = np.asarray(joint_pose, dtype=np.float32)
    art.arm_joint_pos = pose
    art.arm_motor_pos = pose
    if close_gripper:
        art.gripper_joint_pos = np.array(art.params.gripper_closed_state, dtype=np.float32)
    art.update()


def _tuck_arm_for_vision(agent_idx: int = 0):
    global obs
    _set_arm_pose(agent_idx, _FETCH_VISION_TUCK_ARM_JOINTS, close_gripper=True)
    obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})
    return obs


def _settle_physics(agent_idx: int = 0, steps: int = 1):
    global obs
    steps = max(0, int(steps))
    for _ in range(steps):
        obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})
    return obs


def _dynamic_scene_event_label(event_kind: str) -> str:
    raw = str(event_kind).strip().lower()
    alias = {
        "goto_align": "align",
        "align_done": "align",
    }.get(raw, raw)
    label = re.sub(r"[^a-z0-9]+", "_", alias).strip("_")
    return label or "event"


def _discover_dynamic_scene_update_counter() -> int:
    os.makedirs(_DYNAMIC_SCENE_UPDATE_ROOT, exist_ok=True)
    max_idx = -1
    for name in os.listdir(_DYNAMIC_SCENE_UPDATE_ROOT):
        path = os.path.join(_DYNAMIC_SCENE_UPDATE_ROOT, name)
        if not os.path.isdir(path):
            continue

        match = re.match(r"^frame_(\d+)_", name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
            continue

        # Legacy naming support: goto_align_0000 / pickup_0001 / put_0002 ...
        match = re.search(r"_(\d+)$", name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    return max_idx + 1


def _next_dynamic_scene_update_index() -> int:
    global _DYNAMIC_SCENE_UPDATE_COUNTER
    if _DYNAMIC_SCENE_UPDATE_COUNTER is None:
        _DYNAMIC_SCENE_UPDATE_COUNTER = _discover_dynamic_scene_update_counter()
    event_idx = int(_DYNAMIC_SCENE_UPDATE_COUNTER)
    _DYNAMIC_SCENE_UPDATE_COUNTER += 1
    return event_idx


def _capture_posed_rgbd_snapshot(
    *,
    event_kind: str,
    agent_id: int,
    obs_snapshot,
    sensor_state_suffix: str = "head_rgb",
    obs_rgb_suffix: str = "head_rgb",
    obs_depth_suffix: str = "head_depth",
    rgb_cfg_suffix: str = "head_rgb_sensor",
    depth_cfg_suffix: str = "head_depth_sensor",
    arm_pose_mode: Optional[str] = None,
    contexts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    global _DYNAMIC_SCENE_UPDATE_COUNTER

    # obs 없으면 바로 실패. 이 함수는 무조건 한 프레임 필요
    if obs_snapshot is None:
        raise ValueError("obs_snapshot is required for posed RGB-D capture.")

    # obs에서 RGB/Depth를 뽑아내고, agent sensor state(pose) 획득
    rgb_obs = _obs_by_suffix(obs_snapshot, obs_rgb_suffix)
    depth_obs = _obs_by_suffix(obs_snapshot, obs_depth_suffix)
    sensor_state = _sensor_state_by_suffix(sim.get_agent(agent_id), sensor_state_suffix)

    # RGB를 OpenCV용 BGR로 변환
    rgb_bgr = rgb_observation_to_bgr(rgb_obs)
    # depth를 meter 스케일로 정규화/클리핑한 배열과 meta(depth_min/max) 획득
    depth_m, depth_meta = _depth_obs_to_meters(depth_obs, depth_cfg_suffix=depth_cfg_suffix)
    intrinsics = _camera_intrinsics_from_cfg(rgb_cfg_suffix=rgb_cfg_suffix)
    # 카메라 외부 파라미터(월드 -> 카메라 관련 4x4) 계산
    T_cam = _camera_transform_from_sensor_state(sensor_state)

    # 이벤트 번호 증가, label 생성, 이벤트 ID(ex: frame_00012_pickup) 생성
    event_idx = _next_dynamic_scene_update_index()
    event_label = _dynamic_scene_event_label(event_kind)
    event_id = f"frame_{event_idx:05d}_{event_label}"

    # 이벤트 폴더 생성
    event_dir = os.path.join(
        _DYNAMIC_SCENE_UPDATE_ROOT,
        event_id,
    )
    os.makedirs(event_dir, exist_ok=True)

    # 저장 파일 경로 확정
    rgb_path = os.path.join(event_dir, "rgb.png")
    depth_npy_path = os.path.join(event_dir, "depth.npy")
    depth_vis_path = os.path.join(event_dir, "depth_u16.png") # 시각화용 u16 png
    meta_path = os.path.join(event_dir, "meta.json")

    cv2.imwrite(rgb_path, rgb_bgr)
    np.save(depth_npy_path, depth_m.astype(np.float32))

    depth_scale = 65535.0 / max(depth_meta["depth_max"], 1e-6)
    depth_u16 = np.clip(depth_m * depth_scale, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(depth_vis_path, depth_u16)
    manual_control_files = _write_manual_control_compatible_camera_files(
        event_dir=event_dir,
        rgb_bgr=rgb_bgr,
        depth_m=depth_m,
        depth_u16=depth_u16,
        T_cam=T_cam, # camera transform 저장
        intrinsics=intrinsics,
        depth_scale=depth_scale,
    )

    metadata = {
        "event_kind": str(event_kind),
        "event_id": event_id,
        "event_index": event_idx,
        "event_label": event_label,
        "agent_id": int(agent_id),
        "sensor_state_suffix": str(sensor_state_suffix),
        "obs_rgb_suffix": str(obs_rgb_suffix),
        "obs_depth_suffix": str(obs_depth_suffix),
        "arm_pose_mode": None if arm_pose_mode is None else str(arm_pose_mode),
        "rgb_path": rgb_path,
        "depth_npy_path": depth_npy_path,
        "depth_u16_path": depth_vis_path,
        "camera_transform_world_to_opencv_camera": T_cam.flatten().tolist(),
        "intrinsics": {
            "width": intrinsics["width"],
            "height": intrinsics["height"],
            "hfov_deg": intrinsics["hfov_deg"],
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
            "depth_scale_u16": float(depth_scale),
            **depth_meta,
        },
        "manual_control_compatible_files": manual_control_files,
        "contexts": _jsonify_for_log(contexts or {}),
        "extra": _jsonify_for_log(extra or {}),
    }

    _write_json_file(meta_path, metadata)

    return {
        "event_dir": event_dir,
        "event_id": event_id,
        "meta_path": meta_path,
        "rgb_path": rgb_path,
        "depth_npy_path": depth_npy_path,
        "depth_u16_path": depth_vis_path,
        "event_index": event_idx,
        "event_kind": str(event_kind),
        "manual_control_files": manual_control_files,
    }


def _dispatch_dynamic_scene_update(
    *,
    event_kind: str,
    agent_id: int,
    obs_snapshot=None,
    refresh_observation: bool = False,
    arm_pose_mode: Optional[str] = None,
    target_context: Optional[Dict[str, Any]] = None,
    held_context: Optional[Dict[str, Any]] = None,
    receptacle_target_context: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    # 입력으로 받은 obs_snapshot을 기본 관측으로 사용
    latest_obs = obs_snapshot
    if arm_pose_mode == "vision_tuck": # vision_tuck이면 팔을 접고 새관측을 강제로 얻음. 로봇 팔이 카메라를 가리는 현상을 줄이려는 의도
        latest_obs = _tuck_arm_for_vision(agent_idx=agent_id)
    elif refresh_observation or latest_obs is None: # refresh_observation 또는 스냅샷이 없는 경우에 EMPTY_ACTION 한 스텝으로 최신 obs 생성
        latest_obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})

    # contexts 딕셔너리 구성
    # dict()로 복사해서 외부 객체 참조 부작용 방지
    contexts = {
        "target_context": None if target_context is None else dict(target_context),
        "held_context": None if held_context is None else dict(held_context),
        "receptacle_target_context": (
            None if receptacle_target_context is None else dict(receptacle_target_context)
        ),
    }

    try:
        # RGB/Depth/카메라 파라미터/컨텍스트를 disk에 저장
        capture_info = _capture_posed_rgbd_snapshot(
            event_kind=event_kind,
            agent_id=agent_id,
            obs_snapshot=latest_obs,
            arm_pose_mode=arm_pose_mode,
            contexts=contexts,
            extra=extra,
        )
        # 캡처 성공 로그 출력
        print(
            f"[dynamic-update] captured event={event_kind} "
            f"idx={capture_info['event_index']} dir={capture_info['event_dir']}"
        )
        try:
            # apply_dynamic_scene_graph_update 호출을 통해 실제 scene graph overlay 갱신
            update_result = apply_dynamic_scene_graph_update(
                scene_id=SCENE_ID,
                event_meta_path=capture_info["meta_path"],
            )
            # updater 결과를 capture_info에 붙이기
            capture_info["scene_graph_update"] = update_result
            # 업데이트 성공 로그
            print(
                f"[dynamic-update] scene graph updated | "
                f"event={event_kind} status={update_result.get('status')}"
            )
        # 업데이트 실패 처리
        except Exception as update_exc:
            failure_path = os.path.join(capture_info["event_dir"], "scene_graph_update.json")
            failure_payload = {
                "event_id": os.path.basename(capture_info["event_dir"]),
                "event_kind": event_kind,
                "status": "failed",
                "error": str(update_exc),
            }
            try:
                with open(failure_path, "w", encoding="utf-8") as f:
                    json.dump(failure_payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            print(
                f"[dynamic-update] scene graph update failed | "
                f"event={event_kind}, err={update_exc}"
            )
        return capture_info
    except Exception as exc:
        print(f"[dynamic-update] capture failed | event={event_kind}, err={exc}")
        return None


def _maybe_dispatch_post_align_update(act: dict, agent_id: int):
    post_align_update = act.get("post_align_update")
    if not isinstance(post_align_update, dict):
        return None

    return _dispatch_dynamic_scene_update(
        event_kind=str(post_align_update.get("event_kind", "align_done")),
        agent_id=agent_id,
        refresh_observation=bool(post_align_update.get("refresh_observation", True)),
        arm_pose_mode=post_align_update.get("arm_pose_mode"),
        target_context=post_align_update.get("target_context"),
        held_context=post_align_update.get("held_context"),
        receptacle_target_context=post_align_update.get("receptacle_target_context"),
        extra=post_align_update.get("extra"),
    )
# ------------------------------------------------------------------------------------------------------------------

# ----------------------------- Main Execution Loop -----------------------------
# 1) robot/task meta 정보
robots = [{
    "name": "fetch1",
    "skills": [
        "GoToObject", "OpenObject", "CloseObject", "BreakObject", "SliceObject",
        "SwitchOn", "SwitchOff", "PickupObject", "PutObject",
        "DropHandObject", "ThrowObject", "PushObject", "PullObject"
    ]
}]

no_robot = len(robots)
objects_locations1 = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json'
# 로봇이 잡고 놓을 수 있는 물체 리스트, 이걸 rigid_objs에 등록해야 최종적으로 로봇이 상호작용할 수 있는 물체가 됨
RIGID_OBJS_POSE_PATH = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/rigid_objs.json"
_RIGID_OBJ_SETTINGS_BY_HANDLE = {}


# ============================== [MOD: agent execution context cache] ==============================
# 왜 필요한가:
# - GoToObject / Explore는 "semantic target"을 해석해서 특정 물체/리셉터클 근처로 이동시킨다.
# - PickupObject / PutObject는 실제 Habitat runtime object id가 필요하다.
# - 따라서 "방금 어디로 갔는지", "지금 무엇을 들고 있는지"를 agent_id 기준으로 안정적으로 저장해야 한다.
#
# 이 캐시는 robot dict 유무와 상관없이 항상 사용할 수 있는 실행 컨텍스트 저장소다.
# 필요하면 bot["_karma"]에도 같이 미러링해서 디버깅할 수 있게 유지한다.
_AGENT_EXEC_CONTEXTS = {}


def _get_agent_exec_context(agent_id: int) -> dict:
    """
    agent_id별 실행 컨텍스트를 가져온다.

    보관 항목:
    - last_nav_target:
        가장 최근 GoToObject / Explore 등이 resolve한 target
    - held_object:
        현재 grasp 중인 물체의 memory/runtime 정보
    """
    agent_id = int(agent_id)
    if agent_id not in _AGENT_EXEC_CONTEXTS:
        _AGENT_EXEC_CONTEXTS[agent_id] = {
            "last_nav_target": None,
            "held_object": None,
        }
    return _AGENT_EXEC_CONTEXTS[agent_id]


def _mirror_exec_context_to_bot(bot, agent_id: int):
    """
    디버깅 편의를 위해 module-level 컨텍스트를 bot["_karma"]에도 복사한다.

    주의:
    - 실제 authoritative source는 _AGENT_EXEC_CONTEXTS다.
    - bot이 dict가 아닐 수도 있으므로, dict일 때만 미러링한다.
    """
    if not isinstance(bot, dict):
        return

    ctx = _get_agent_exec_context(agent_id)
    bot.setdefault("_karma", {})
    bot["_karma"]["last_nav_target"] = ctx.get("last_nav_target")
    # 기존 키 이름을 참조하는 코드가 남아 있을 수 있어 backward compatibility로 유지
    bot["_karma"]["last_resolved_target"] = ctx.get("last_nav_target")
    bot["_karma"]["held_object"] = ctx.get("held_object")


def _set_last_nav_target_context(agent_id: int, target_context: dict, bot=None):
    """
    가장 최근 navigation target context를 저장한다.
    """
    ctx = _get_agent_exec_context(agent_id)
    ctx["last_nav_target"] = target_context
    _mirror_exec_context_to_bot(bot, agent_id)


def _get_last_nav_target_context(agent_id: int):
    """
    가장 최근 navigation target context를 반환한다.
    """
    return _get_agent_exec_context(agent_id).get("last_nav_target")


def _set_held_object_context(agent_id: int, held_context: dict, bot=None):
    """
    현재 grasp 중인 물체의 context를 저장한다.
    """
    ctx = _get_agent_exec_context(agent_id)
    ctx["held_object"] = held_context
    _mirror_exec_context_to_bot(bot, agent_id)


def _get_held_object_context(agent_id: int):
    """
    현재 grasp 중인 물체의 context를 반환한다.
    """
    return _get_agent_exec_context(agent_id).get("held_object")


def _clear_held_object_context(agent_id: int, bot=None):
    """
    grasp 해제 후 held_object context를 비운다.
    """
    ctx = _get_agent_exec_context(agent_id)
    ctx["held_object"] = None
    _mirror_exec_context_to_bot(bot, agent_id)


def _agent_is_holding_object(agent_id: int) -> bool:
    try:
        gm = sim.get_agent_data(int(agent_id)).grasp_mgr
        if bool(getattr(gm, "is_grasped", False)):
            return True
    except Exception:
        pass
    return _get_held_object_context(int(agent_id)) is not None


def _query_matches_target_context(obj_query, target_context: dict) -> bool:
    """
    사용자/플래너가 넘긴 object query가 현재 target context와 일치하는지 느슨하게 확인한다.

    왜 필요한가:
    - GoToObject(robot, apple...) 다음에 PickupObject(robot, "Apple")은 허용
    - 하지만 GoToObject(robot, table...) 직후 PickupObject(robot, "Apple")은 막고 싶다.

    규칙:
    - 1차: object canonicalization(alias-aware) 기준으로 같은 물체인지 확인
    - 2차: 과거 planner 예시 호환을 위해 substring 계열 fallback도 허용
      예: "Table" vs "dining table"
    """
    if target_context is None:
        return False

    try:
        spec = normalize_navigation_target(obj_query)
        raw_query = str(spec.get("object", "")).strip()
    except Exception:
        raw_query = str(obj_query or "").strip()

    if not raw_query:
        return False

    name_candidates = [
        target_context.get("graph_object_name"),
        target_context.get("conceptgraph_object_tag"),
        target_context.get("target_object_type"),
    ]

    query = target_context.get("query", {})
    if isinstance(query, dict):
        name_candidates.append(query.get("object"))

    for candidate in name_candidates:
        if not candidate:
            continue

        # 1) alias-aware semantic equality
        # 예: Mug <-> cup
        if object_labels_semantically_match(raw_query, str(candidate)):
            return True

        # 2) substring / free-text fallback
        # 예: Table <-> dining table
        if score_object_query_against_text(raw_query, str(candidate)) >= 75:
            return True

    return False


def _runtime_rigid_handle(template_name: str, instance_idx: int) -> str:
    return f"{template_name.split('.')[0]}_:{int(instance_idx):04d}"


def _motion_type_from_name(name: str):
    if not name:
        return None
    key = str(name).strip().upper()
    if key == "STATIC":
        return habitat_sim.physics.MotionType.STATIC
    if key == "KINEMATIC":
        return habitat_sim.physics.MotionType.KINEMATIC
    if key == "DYNAMIC":
        return habitat_sim.physics.MotionType.DYNAMIC
    print(f"[rigid_objs] unknown motion_type='{name}', ignore.")
    return None


def _zero_rigid_body_motion(obj):
    if obj is None:
        return
    try:
        obj.angular_velocity = np.zeros(3, dtype=np.float32)
    except Exception:
        pass
    try:
        obj.linear_velocity = np.zeros(3, dtype=np.float32)
    except Exception:
        pass


def _apply_rigid_obj_motion_settings(sim):
    rom = sim.get_rigid_object_manager()
    applied = 0
    missing = []
    for handle, cfg in _RIGID_OBJ_SETTINGS_BY_HANDLE.items():
        obj = rom.get_object_by_handle(handle)
        if obj is None:
            missing.append(handle)
            continue
        motion_type = _motion_type_from_name(cfg.get("motion_type", ""))
        if motion_type is None:
            continue
        try:
            obj.motion_type = motion_type
            _zero_rigid_body_motion(obj)
            applied += 1
        except Exception as exc:
            print(f"[rigid_objs] failed to set motion_type for {handle}: {exc}")
    print(f"[rigid_objs] applied runtime motion settings={applied}")
    if missing:
        print(f"[rigid_objs] missing runtime handles: {missing}")


def _set_rigid_object_dynamic_for_pick(sim, object_id: int):
    rom = sim.get_rigid_object_manager()
    obj = rom.get_object_by_id(int(object_id))
    if obj is None:
        return
    try:
        if obj.motion_type != habitat_sim.physics.MotionType.DYNAMIC:
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            _zero_rigid_body_motion(obj)
            print(f"[rigid_objs] {obj.handle} -> DYNAMIC for pickup")
    except Exception as exc:
        print(f"[rigid_objs] failed to switch {obj.handle} to DYNAMIC: {exc}")


def _restore_rigid_object_motion_type(sim, obj_handle: str):
    if not obj_handle:
        return
    cfg = _RIGID_OBJ_SETTINGS_BY_HANDLE.get(str(obj_handle))
    if cfg is None:
        return
    motion_type = _motion_type_from_name(cfg.get("motion_type", ""))
    if motion_type is None:
        return
    rom = sim.get_rigid_object_manager()
    obj = rom.get_object_by_handle(str(obj_handle))
    if obj is None:
        return
    try:
        obj.motion_type = motion_type
        _zero_rigid_body_motion(obj)
        print(f"[rigid_objs] restored {obj.handle} -> {cfg.get('motion_type', '')}")
    except Exception as exc:
        print(f"[rigid_objs] failed to restore motion_type for {obj_handle}: {exc}")


def _pose_entry_to_rigid_obj(entry: dict):
    """
    scene_instance object entry(translation + rotation[w,x,y,z])를
    RearrangeEpisode.rigid_objs 포맷 (handle, 4x4 transform)으로 변환
    """
    handle = str(entry["template_name"])
    tx, ty, tz = [float(v) for v in entry["translation"]]
    qw, qx, qy, qz = [float(v) for v in entry["rotation"]]

    # scipy Rotation은 [x, y, z, w] 순서를 사용한다.
    rot_m = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot_m
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return (handle, T.tolist())

def _load_rigid_objs_from_pose_json(path: str):
    """
    rigid_objs_*.json에서 objects_for_rigid_objs를 읽어
    RearrangeEpisode.rigid_objs 리스트를 생성
    """
    global _RIGID_OBJ_SETTINGS_BY_HANDLE
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("objects_for_rigid_objs", [])
    handle_counts = {}
    settings = {}
    rigid_objs = [_pose_entry_to_rigid_obj(e) for e in entries]
    for entry in entries:
        template_name = str(entry["template_name"])
        instance_idx = int(handle_counts.get(template_name, 0))
        runtime_handle = _runtime_rigid_handle(template_name, instance_idx)
        settings[runtime_handle] = {
            "template_name": template_name,
            "motion_type": str(entry.get("motion_type", "")).strip().upper(),
        }
        handle_counts[template_name] = instance_idx + 1
    _RIGID_OBJ_SETTINGS_BY_HANDLE = settings
    print(f"[rigid_objs] loaded={len(rigid_objs)} from {path}")
    return rigid_objs


# 2) Habitat-Sim init
HAB_CFG_ROOT = "/home/yuchaehee/long_term_memory_project/habitat-lab/habitat-lab/habitat/config"
# SCENE_DATASET_CFG = "/home/yuchaehee/long_term_memory_project/scene_data/scene_datasets/ai2thorhab/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
SCENE_DATASET_CFG = "/home/yuchaehee/long_term_memory_project/scene_data/scene_datasets/hssd-hab/hssd-hab-articulated.scene_dataset_config.json"
# SCENE_ID = "FloorPlan1_physics.scene_instance"
# SCENE_ID = "FloorPlan1_physics_no_apple_bread_potato.scene_instance"
# SCENE_ID = "ArchitecTHOR-Test-00.scene_instance"
# SCENE_ID = "102343992.scene_instance"
SCENE_ID = "108736872_177263607.scene_instance"
# SCENE_ID = "102344193"

cfg = habitat.get_config(
    config_path="benchmark/rearrange/play/play.yaml",
    configs_dir=HAB_CFG_ROOT, 
)

with read_write(cfg):
    agent = cfg.habitat.simulator.agents.main_agent

    down_pitch = math.radians(-20.0) # Python float
    agent.sim_sensors["head_down_rgb_sensor"] = HeadRGBSensorConfig(
        uuid="head_down_rgb",
        width=1000,
        height=1000,
        orientation=[down_pitch, 0.0, 0.0], # 20도 아래 쳐다보도록
    )
    agent.sim_sensors["head_down_depth_sensor"] = HeadDepthSensorConfig(
        uuid="head_down_depth",
        width=1000,
        height=1000,
        orientation=[down_pitch, 0.0, 0.0], # 20도 아래 쳐다보도록
    )
    agent.sim_sensors["third_rgb_sensor"] = ThirdRGBSensorConfig(
        uuid="third_rgb",
        width=512,
        height=512,
    )
    agent.sim_sensors["scene_camera_rgb_sensor"] = ThirdRGBSensorConfig(
        uuid="scene_camera_rgb",
        width=512,
        height=512,
        position=[0.0, 0.35, 0.0],
        orientation=[-np.pi / 2, 0.0, 0.0], # 탑뷰 느낌
    )
    hcfg = cfg.habitat
    hcfg.environment.max_episode_steps = 0
    hcfg.simulator.scene_dataset = SCENE_DATASET_CFG
    hcfg.simulator.scene = SCENE_ID
    hcfg.simulator.habitat_sim_v0.enable_physics = True
    hcfg.simulator.navmesh_include_static_objects = True

    sensors = hcfg.simulator.agents.main_agent.sim_sensors
    sensors.head_rgb_sensor.width = 1000
    sensors.head_rgb_sensor.height = 1000
    sensors.head_depth_sensor.width = 1000
    sensors.head_depth_sensor.height = 1000
    sensors.arm_rgb_sensor.width = 1000
    sensors.arm_rgb_sensor.height = 1000
    sensors.arm_depth_sensor.width = 1000
    sensors.arm_depth_sensor.height = 1000

dataset = RearrangeDatasetV0()
dataset.config = cfg.habitat.dataset
_init_log("loading rigid object episode config")
episode_rigid_objs = _load_rigid_objs_from_pose_json(RIGID_OBJS_POSE_PATH)
_init_log(f"loaded rigid object episode config | count={len(episode_rigid_objs)}")
dataset.episodes = [
    RearrangeEpisode(
        episode_id='0',
        scene_id=SCENE_ID,
        scene_dataset_config=SCENE_DATASET_CFG,
        additional_obj_config_paths=list(hcfg.simulator.additional_object_paths),
        start_position=[0.0, 0.0, 0.0],
        start_rotation=[0.0, 0.0, 0.0, 1.0],
        info={"object_labels": {}},
        ao_states={},
        rigid_objs=episode_rigid_objs,
        targets={},
    )
]
_init_log("creating Habitat env")
env = habitat.Env(config=cfg, dataset=dataset)
_init_log("Habitat env created")
_init_log("resetting env")
obs = env.reset()
_init_log("env reset complete")
sim = env.sim
_init_log("applying runtime rigid-object settings")
_apply_rigid_obj_motion_settings(sim)
_init_log("runtime rigid-object settings applied")
_init_log("setting up observer sensors")
setup_hov_observer_sensors()
_init_log("observer sensors ready")
_init_log("fetching initial observations")
obs = get_current_observations(base_obs=obs)
_init_log("initial observations fetched")

_init_log("collecting initial sensor list")
observations = get_current_observations()
want = ["third_rgb", "head_rgb", "scene_camera_rgb", "head_down_rgb"]
if ENABLE_HOV_MULTICAM:
    want.extend([HOV_OBSERVER_RGB_UUID, HOV_OBSERVER_DEPTH_UUID])
print([k for k in want if k in observations])
_init_log("initial sensor list ready")


BASE_ACTION = _resolve_action_name(env, "base_velocity")
ARM_ACTION = _resolve_action_name(env, "arm_action")
EMPTY_ACTION = _resolve_action_name(env, "empty")

# 초기 관측 
_init_log("running initial EMPTY_ACTION step")
obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})
_init_log("initial EMPTY_ACTION step complete")
head_rgb = _obs_by_suffix(obs, "head_rgb")
head_depth = _obs_by_suffix(obs, "head_depth")

init_view_dir = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/logs/init_views"
saved_files = save_initial_rgb_camera_views(observations, init_view_dir, prefix="init")
print(f"[init] saved camera views ({len(saved_files)}): {saved_files}")
_init_log(f"saved initial camera views | count={len(saved_files)}")

# long-term memory에서 쓸 3x3 center를 먼저 계산 (Fetch 스폰에도 재사용)
try:
    # ltm_centers = get_divided_positions(sim, divisions=3)
    _init_log("computing long-term-memory centers")
    ltm_centers = get_divided_positions(sim, divisions=5)
except Exception:
    ref_p = np.array(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
    ref_island = int(sim.pathfinder.get_island(ref_p))
    ltm_centers = get_divided_positions(sim, divisions=3, island_index=ref_island)
print(f"[init] ltm_centers={len(ltm_centers)}")
_init_log(f"long-term-memory centers ready | count={len(ltm_centers)}")

# -------------------- Random spawn (same scene) --------------------
RANDOMIZE_SPAWN = os.environ.get("KARMA_RANDOMIZE_SPAWN", "1") == "1"

if RANDOMIZE_SPAWN:
    if not ltm_centers:
        raise RuntimeError("ltm_centers is empty.")
    
    # ltm_centers 중 하나를 진짜 랜덤으로 선택
    center_xyz = random.choice(ltm_centers)
    spawn_xyz = np.array(center_xyz, dtype=np.float32)

    # yaw도 랜덤
    spawn_yaw = random.uniform(-math.pi, math.pi)

    _init_log("randomizing initial spawn pose")
    obs = teleport_fetch_base(
        sim,
        env,
        EMPTY_ACTION,
        spawn_xyz,
        yaw_rad=spawn_yaw,
        agent_idx=0,
    )
    print(
        f"[spawn] ltm-center-random | center={np.round(spawn_xyz, 3).tolist()}, "
        f"yaw_deg={np.degrees(spawn_yaw):.1f}"
    )

# map for memory (event -> sim)
first_map(sim) # 맨 처음 시뮬레이터 상태에서 맵을 만들어서 저장 (objects_locations1.json)
first_map_for_next_time(sim) 

centers = ltm_centers
regions = get_static_objects_in_regions(sim, centers)

# 1) regions -> json 저장
save_regions_to_json(
    regions,
    filename='/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/longterm_memory.json'
)

# 2) json -> 문장 리스트 변환
filename = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/longterm_memory.json"
sentences = extract_regions_from_json(filename)

# 3) 콘솔 출력
for sentence in sentences:
    print(sentence)

# 4) 프롬프트 파일로 저장(llm_as_planner.py가 읽는 파일)
os.makedirs(os.path.dirname("/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/prompts/long_term_memory.txt"), exist_ok=True)
with open ("/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/prompts/long_term_memory.txt", "w", encoding="utf=8") as f:
    if sentences:
        f.write("\n".join(sentences) + "\n")
    else:
        f.write("")
    f.write("\nIf you are using the `Explore` function, you should consider which of the aforementioned points the object is likely to be located at when passing parameters, and sort all eight points in the order of the likelihood of finding the object.")    

print(f"[init] long-term memory prompt saved: /home/yuchaehee/long_term_memory_project/my_long_term_mem_project/prompts/long_term_memory.txt (lines={len(sentences)})")


action_queue = []
task_over = False


"""
여기는 action 수행 관련 함수 & 변수
"""
MOVE_STEP_M = 0.08         # 한 번 MoveAhead를 몇 m로 볼지 (작게 = 더 안정)
TURN_CHUNK_DEG = 5.0       # 회전을 잘게 쪼개기
BASE_LIN_SCALE = 0.40      # 전진 속도 스케일 
BASE_ANG_SCALE = 0.40      # 회전 속도 스케일 
ACTION_SLEEP_SEC = 0.01    # 액션 사이 딜레이
ROT_DIR_SIGN = 1.0          # 회전 부호 기본값(필요 시 디버그 루프에서 자동 반전)


# 1) 추가: 팔 접기 헬퍼 ( _step_base 위/아래 아무데나 )
def _tuck_arm_for_nav(agent_idx: int = 0):
    """
    내비 중 팔을 기본 접힘 자세로 유지한다.
    물체를 들고 있을 때는 carry/vision tuck 자세를 써서
    head camera에 물체가 보이는 양을 줄인다.
    """
    global obs
    art = sim.get_agent_data(agent_idx).articulated_agent
    target_pose = (
        _FETCH_VISION_TUCK_ARM_JOINTS
        if _agent_is_holding_object(agent_idx)
        else np.array(art.params.arm_init_params, dtype=np.float32)
    )
    _set_arm_pose(
        agent_idx,
        target_pose,
        close_gripper=True,
    )
    obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})


def _step_base(lin: float, ang: float, repeat: int=1):
    global obs
    for _ in range(repeat):
        obs = _env_step(
            {
                "action": BASE_ACTION,
                "action_args": {
                    "base_vel": np.array(
                        [lin * BASE_LIN_SCALE, ang * BASE_ANG_SCALE],
                        dtype=np.float32,
                    )
                },
            }
        )
        time.sleep(ACTION_SLEEP_SEC)


# def _step_arm(grip_action=None, arm_action=None, repeat=1):
#     """
#     1. arm_action이 없으면 0벡터 넣음
#         - 팔 관절은 안 움직이고 유지

#     2. grip_action이 있으면 같이 넣음
#         - MagicGraspAction 기준:
#             -- >=0: grasp 시도
#             -- <0: release 시도
    
#     3. repeat만큼 같은 액션 반복 실행
#     4. 매 step마다 전역 obs를 최신 관측으로 갱신
#     """
#     global obs
#     arm_dim = env.action_space.spaces[ARM_ACTION].spaces['arm_action'].shape[0]
#     if arm_action is None:
#         arm_action = np.zeros(arm_dim, dtype=np.float32)
    
#     args = {"arm_action": np.asarray(arm_action, dtype=np.float32)}
#     if grip_action is not None:
#         args["grip_action"] = float(grip_action)
    
#     for _ in range(repeat):
#         obs = env.step({"action": ARM_ACTION, "action_args": args})

def _show_realtime_panel(obs, banner="IDLE"):
    # real-time 으로 맵 환경 시각화 하는 코드
    global task_over
    if (not ENABLE_INTERACTIVE_VIEW) or (obs is None):
        return

    head_rgb = _obs_by_suffix_optional(obs, "head_rgb")
    # top_rgb = _obs_by_suffix_optional(obs, "scene_camera_rgb")
    # 로봇 팔 카메라 우선 사용 (예: articulated_agent_arm_rgb)
    # arm 카메라가 없을 때만 기존 scene_camera_rgb로 fallback
    top_rgb = _obs_by_suffix_optional(obs, "arm_rgb")
    if top_rgb is None:
        top_rgb = _obs_by_suffix_optional(obs, "scene_camera_rgb")
    third_rgb = _obs_by_suffix_optional(obs, "third_rgb")

    head_bgr = rgb_observation_to_bgr(head_rgb) if head_rgb is not None else None
    top_bgr = rgb_observation_to_bgr(top_rgb) if top_rgb is not None else None
    third_bgr = rgb_observation_to_bgr(third_rgb) if third_rgb is not None else None

    sim_obj = globals().get("sim", None)
    navmesh_bgr = _build_navmesh_topdown_panel(sim_obj)

    panel_imgs = [img for img in [head_bgr, top_bgr, third_bgr, navmesh_bgr] if img is not None]
    if not panel_imgs:
        return

    try:
        target_h = min(img.shape[0] for img in panel_imgs)
        panel_imgs = [_resize_to_height(img, target_h) for img in panel_imgs]
        panel = cv2.hconcat(panel_imgs)
        cv2.putText(
            panel,
            f"{banner} | N:toggle navmesh",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        debug_lines = get_debug_overlay()
        y = 62
        for line in debug_lines[:8]:
            cv2.putText(
                panel, 
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),   # 노란색
                2,
                cv2.LINE_AA,
            )
            y += 24
        cv2.imshow("KARMA Realtime", panel)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            task_over = True
        elif key in (ord("n"), ord("N")):
            if sim_obj is not None and getattr(sim_obj, "pathfinder", None) is not None and sim_obj.pathfinder.is_loaded:
                sim_obj.navmesh_visualization = not bool(sim_obj.navmesh_visualization)
                print(f"[realtime] toggle navmesh -> {sim_obj.navmesh_visualization}")
            else:
                print("[realtime] navmesh not loaded")
    except cv2.error:
        # headless / 창 생성 불가 환경
        return

def _save_step_views(obs, img_counter):
    base_dir = os.path.dirname(__file__)

    agent_dir = os.path.join(base_dir, "agent_1")
    top_dir = os.path.join(base_dir, "top_view")
    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(top_dir, exist_ok=True)

    # 1인칭(head)
    head_rgb = _obs_by_suffix(obs, "head_rgb")
    head_bgr = rgb_observation_to_bgr(head_rgb)
    cv2.imwrite(os.path.join(agent_dir, f"img_{img_counter:05d}.png"), head_bgr)
    return True


_EXEC_ACTION_IO_READY = False
_EXEC_IMG_COUNTER = 0
_EXEC_SHORT_MEMORY_COUNTER = 0


def _prepare_action_output_dirs():
    global _EXEC_ACTION_IO_READY
    if _EXEC_ACTION_IO_READY:
        return

    base_dir = os.path.dirname(__file__)
    cleanup_dirs = []
    for i in range(no_robot):
        cleanup_dirs.append(os.path.join(base_dir, f"agent_{i+1}"))
        cleanup_dirs.append(os.path.join(base_dir, f"third_{i+1}"))
    cleanup_dirs.append(os.path.join(base_dir, "top_view"))
    for d in cleanup_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)

    for i in range(no_robot):
        os.makedirs(os.path.join(base_dir, f"agent_{i+1}"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f"third_{i+1}"), exist_ok=True)

    _EXEC_ACTION_IO_READY = True


def exec_actions_tick():
    """
    action_queue의 액션 1개를 처리한다.
    Tkinter 메인 스레드에서 주기적으로 호출하기 위해 만든 tick 함수.
    """
    global task_over, obs, _EXEC_IMG_COUNTER, _EXEC_SHORT_MEMORY_COUNTER

    _prepare_action_output_dirs()

    if task_over:
        # task 끝나면 시각화 화면 끄기
        _show_realtime_panel(obs, banner="STOP")
        return False
    
    if not action_queue:
        # action_queue 빈 상태면 시각화 화면 멈춘 상태로 놓기
        _show_realtime_panel(obs, banner="IDLE")
        return False

    act = action_queue.pop(0)
    name = act["action"]
    print(f"[tick] action={name}, qlen={len(action_queue)}")

    if name == "ObjectNavExpertAction":
        _object_nav_expert_tick(act) # 1회 next-action 실행만
    elif name == "AlignToTarget":
        aid = int(act.get("agent_id", 0))
        # 로봇이 바라봐야 하는 target 위치 (x, y, z)
        target_xyz = np.asarray(act.get("target_xyz", []), dtype=np.float32)
        align_thresh_deg = float(act.get("align_thresh_deg", 2.0))
        camera_yaw_offset_deg = float(act.get("camera_yaw_offset_deg", 0.0))
        max_step_deg = max(1.0, float(act.get("max_step_deg", TURN_CHUNK_DEG)))

        if target_xyz.shape != (3,) or not np.all(np.isfinite(target_xyz)):
            print(f"[AlignToTarget] invalid target_xyz={act.get('target_xyz')}")
        else:
            # 현재 로봇 베이스 pose는 계속 읽어두되,
            # 실제 정렬 기준은 "사용자가 보는 head camera"의 pose를 우선 사용한다.
            cur_base_pos, cur_base_yaw_rad = get_fetch_base_pose(sim, aid)

            try:
                cur_view_pos, cur_view_yaw_rad = _get_sensor_view_pose(
                    sim,
                    agent_id=aid,
                    sensor_suffix="head_rgb",
                )
                view_source = "head_rgb"
            except Exception as exc:
                # sensor pose를 못 읽으면 기존 base 기준으로 fallback
                cur_view_pos, cur_view_yaw_rad = cur_base_pos, cur_base_yaw_rad
                view_source = f"base_fallback:{exc}"

            # 로봇 위치와 target 위치의 x, z 차이 계산 
            dx = float(target_xyz[0] - cur_view_pos[0])
            dz = float(target_xyz[2] - cur_view_pos[2])

            # 오브젝트 점을 보려면 몇 라디안 방향을 봐야 하는지 계산하고
            # 현재 yaw와의 차이 계산
            # 왼쪽/오른쪽으로 조금씩 돌면서 점점 정렬해가는 방식으로, 한 번에 다 돌지 않고 작은 step으로 나눠서 회전한다.
            if abs(dx) + abs(dz) < 1e-8:
                print("[AlignToTarget] skip | target too close to current pose")
                _maybe_dispatch_post_align_update(act, aid)
            else:
                # Fetch base_rot / _get_sensor_view_pose()와 같은 yaw convention 사용:
                # - yaw = 0   -> world +X 방향
                # - yaw = +90 -> world -Z 방향
                # - yaw = -90 -> world +Z 방향
                # 즉 Habitat world에서 "forward=-Z"라는 사실과
                # Fetch base의 local +X forward를 함께 고려하면
                # target yaw는 atan2(-dz, dx)로 계산해야 부호가 맞는다.
                target_yaw_rad = float(np.arctan2(-dz, dx))
                target_yaw_rad = wrap_to_pi(
                    target_yaw_rad + np.radians(camera_yaw_offset_deg)
                )
                yaw_err_rad = wrap_to_pi(target_yaw_rad - float(cur_view_yaw_rad))
                yaw_err_signed_deg = float(np.degrees(yaw_err_rad))
                yaw_err_deg = abs(yaw_err_signed_deg)
                cur_base_yaw_deg = float(np.degrees(cur_base_yaw_rad))
                cur_view_yaw_deg = float(np.degrees(cur_view_yaw_rad))
                target_yaw_deg = float(np.degrees(target_yaw_rad))

                # [MOD: align debug]
                # 정렬이 반대로 도는지 확인하려면 아래 로그만 보면 된다.
                # - 현재 로봇 위치/목표 위치
                # - 현재 yaw / 목표 yaw
                # - signed yaw error
                #
                # 해석 포인트:
                # - target이 분명 왼쪽에 있는데 yaw_err_signed_deg가 음수라면
                #   yaw convention 또는 atan2 축 정의가 뒤집혀 있을 가능성이 있다.
                print(
                    "[align-debug] "
                    f"view_src={view_source} "
                    f"base_pos={np.round(cur_base_pos, 3).tolist()} "
                    f"view_pos={np.round(cur_view_pos, 3).tolist()} "
                    f"target={np.round(target_xyz, 3).tolist()} "
                    f"dx={dx:.3f} dz={dz:.3f} "
                    f"base_yaw_deg={cur_base_yaw_deg:.2f} "
                    f"view_yaw_deg={cur_view_yaw_deg:.2f} "
                    f"target_yaw_deg={target_yaw_deg:.2f} "
                    f"yaw_err_signed_deg={yaw_err_signed_deg:.2f}"
                )

                if yaw_err_deg <= align_thresh_deg:
                    print(
                        f"[AlignToTarget] done | err_deg={yaw_err_deg:.2f}, "
                        f"target={np.round(target_xyz, 3).tolist()}"
                    )
                    _maybe_dispatch_post_align_update(act, aid)
                else:
                    # 한 번에 다 돌지 않고, 작은 회전만 한 뒤 다음 tick에서 다시 오차 측정
                    step_deg = min(yaw_err_deg, max_step_deg)

                    # 현재 codebase에서 AlignToTarget이 쓰는 yaw convention 기준:
                    # positive error -> RotateLeft, negative error -> RotateRight
                    rotate_name = "RotateLeft" if yaw_err_rad > 0.0 else "RotateRight"

                    print(
                        "[align-debug] "
                        f"choose_action={rotate_name} "
                        f"step_deg={step_deg:.2f} "
                        f"align_thresh_deg={align_thresh_deg:.2f}"
                    )

                    # 다음 tick에 다시 오차를 측정하도록 align action을 재삽입
                    action_queue.insert(0, dict(act))
                    action_queue.insert(
                        0,
                        {
                            "action": rotate_name,
                            "degrees": step_deg,
                            "agent_id": aid,
                            "from_align_debug": True,
                        },
                    )

                    print(
                        f"[AlignToTarget] step | err_deg={yaw_err_deg:.2f}, "
                        f"step_deg={step_deg:.2f}, action={rotate_name}, "
                        f"target={np.round(target_xyz, 3).tolist()}"
                    )
    elif name == "MoveAhead":
        _tuck_arm_for_nav()
        _step_base(1.0, 0.0, repeat=int(act.get("repeat", 4)))
    elif name == "MoveBack":
        _tuck_arm_for_nav()
        _step_base(-1.0, 0.0, repeat=int(act.get("repeat", 4)))
    elif name == "RotateLeft":
        debug_before_yaw = None
        debug_before_view_yaw = None
        if bool(act.get("from_align_debug", False)):
            _, debug_before_yaw = get_fetch_base_pose(sim, int(act.get("agent_id", 0)))
            try:
                _, debug_before_view_yaw = _get_sensor_view_pose(
                    sim,
                    agent_id=int(act.get("agent_id", 0)),
                    sensor_suffix="head_rgb",
                )
            except Exception:
                debug_before_view_yaw = None
        _tuck_arm_for_nav()
        rep = max(1, int(np.ceil(float(act.get("degrees", TURN_CHUNK_DEG)) / TURN_CHUNK_DEG)))
        _step_base(0.0, +ROT_DIR_SIGN, repeat=rep)
        if debug_before_yaw is not None:
            _, debug_after_yaw = get_fetch_base_pose(sim, int(act.get("agent_id", 0)))
            debug_view_delta_line = ""
            if debug_before_view_yaw is not None:
                try:
                    _, debug_after_view_yaw = _get_sensor_view_pose(
                        sim,
                        agent_id=int(act.get("agent_id", 0)),
                        sensor_suffix="head_rgb",
                    )
                    debug_view_delta_line = (
                        f" view_delta_deg="
                        f"{np.degrees(wrap_to_pi(debug_after_view_yaw - debug_before_view_yaw)):.2f}"
                    )
                except Exception:
                    pass
            print(
                "[align-debug] "
                f"after RotateLeft "
                f"before_yaw_deg={np.degrees(debug_before_yaw):.2f} "
                f"after_yaw_deg={np.degrees(debug_after_yaw):.2f} "
                f"delta_deg={np.degrees(wrap_to_pi(debug_after_yaw - debug_before_yaw)):.2f}"
                f"{debug_view_delta_line}"
            )
    elif name == "RotateRight":
        debug_before_yaw = None
        debug_before_view_yaw = None
        if bool(act.get("from_align_debug", False)):
            _, debug_before_yaw = get_fetch_base_pose(sim, int(act.get("agent_id", 0)))
            try:
                _, debug_before_view_yaw = _get_sensor_view_pose(
                    sim,
                    agent_id=int(act.get("agent_id", 0)),
                    sensor_suffix="head_rgb",
                )
            except Exception:
                debug_before_view_yaw = None
        _tuck_arm_for_nav()
        rep = max(1, int(np.ceil(float(act.get("degrees", TURN_CHUNK_DEG)) / TURN_CHUNK_DEG)))
        _step_base(0.0, -ROT_DIR_SIGN, repeat=rep)
        if debug_before_yaw is not None:
            _, debug_after_yaw = get_fetch_base_pose(sim, int(act.get("agent_id", 0)))
            debug_view_delta_line = ""
            if debug_before_view_yaw is not None:
                try:
                    _, debug_after_view_yaw = _get_sensor_view_pose(
                        sim,
                        agent_id=int(act.get("agent_id", 0)),
                        sensor_suffix="head_rgb",
                    )
                    debug_view_delta_line = (
                        f" view_delta_deg="
                        f"{np.degrees(wrap_to_pi(debug_after_view_yaw - debug_before_view_yaw)):.2f}"
                    )
                except Exception:
                    pass
            print(
                "[align-debug] "
                f"after RotateRight "
                f"before_yaw_deg={np.degrees(debug_before_yaw):.2f} "
                f"after_yaw_deg={np.degrees(debug_after_yaw):.2f} "
                f"delta_deg={np.degrees(wrap_to_pi(debug_after_yaw - debug_before_yaw)):.2f}"
                f"{debug_view_delta_line}"
            )
    elif name == "PickupObject":
        aid = int(act.get("agent_id", 0))
        pick_id = act.get("pick_obj_id", None)

        if pick_id is None:
            print("[PickupObject] failed | no pick_obj_id")
        else:
            gm = sim.get_agent_data(aid).grasp_mgr
            try:
                pid = int(pick_id)
                _set_rigid_object_dynamic_for_pick(sim, pid)
                # 이미 다른 물체 잡고 있으면 먼저 해제
                if gm.is_grasped and gm.snap_idx != pid:
                    gm.desnap(True)
                
                gm.snap_to_obj(pid, force=True)
                obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})
                print(f"[PickupObject] success | target_id={pid}, snapped={gm.snap_idx}")

                # ============================== [MOD: update held-object context on successful grasp] ==============================
                # 큐 등록 시점에는 아직 grasp가 진짜 성공할지 모르므로,
                # held_object context는 실제 snap 성공 후에만 갱신한다.
                memory_target_context = act.get("memory_target_context")
                held_context = None
                if isinstance(memory_target_context, dict):
                    held_context = dict(memory_target_context)
                    held_context["runtime_object_id"] = str(pid)
                    held_context["runtime_handle"] = act.get("target_handle", held_context.get("runtime_handle"))
                    held_context["runtime_object_type"] = act.get("target_object_type", held_context.get("runtime_object_type"))
                    held_context["held_via_action"] = "PickupObject"
                    _set_held_object_context(aid, held_context)

                _dispatch_dynamic_scene_update(
                    event_kind="pickup",
                    agent_id=aid,
                    obs_snapshot=obs,
                    refresh_observation=False,
                    arm_pose_mode="vision_tuck",
                    target_context=memory_target_context if isinstance(memory_target_context, dict) else None,
                    held_context=held_context,
                    extra={
                        "pick_obj_id": str(pid),
                        "target_handle": act.get("target_handle"),
                        "target_object_type": act.get("target_object_type"),
                    },
                )

            except Exception as e:
                print(f"[PickupObject] failed | target_id={pick_id}, err={e}")

    elif name == "PutObject":
        aid = int(act.get("agent_id", 0))
        recp_id = act.get("recp_object_id", None)
        gm = sim.get_agent_data(aid).grasp_mgr

        put_success = False
        released_obj = None

        if not gm.is_grasped:
            print("[PutObject] failed | nothing is grasped")
        else:
            try:
                held_handle = ""
                if gm.snap_rigid_obj is not None:
                    held_handle = str(getattr(gm.snap_rigid_obj, "handle", ""))
                if recp_id not in (None, ""):
                    target_xyz = _place_held_object_on_receptacle(aid, int(recp_id), y_offset=0.01)
                    print(f"[PutObject] target recp_id={recp_id}, place_xyz={target_xyz.tolist()}")
                # release (원본 PutObject 실행에 해당)
                released_obj = gm.snap_rigid_obj
                gm.desnap(True)
                _zero_rigid_body_motion(released_obj)
                obs = _settle_physics(
                    agent_idx=aid,
                    steps=_PUT_OBJECT_SETTLE_STEPS_BEFORE_TUCK,
                )

                _restore_rigid_object_motion_type(sim, held_handle)
                _zero_rigid_body_motion(released_obj)
                obs = _settle_physics(
                    agent_idx=aid,
                    steps=_PUT_OBJECT_POST_RESTORE_SETTLE_STEPS,
                )
                put_success = not gm.is_grasped
                print(f"[PutObject] success | recp_id={recp_id}, released={not gm.is_grasped}")
            except Exception as e:
                print(f"[PutObject] failed | recp_id={recp_id}, err={e}")

        if not put_success:
            # 실패 시 후처리/메모리 업데이트 생략 (원하면 여기서 return)
            return

        held_context_for_update = act.get("held_object_context")
        receptacle_context_for_update = act.get("receptacle_target_context")

        # ============================== [MOD: clear held-object context after successful put] ==============================
        _clear_held_object_context(aid)

        _dispatch_dynamic_scene_update(
            event_kind="put",
            agent_id=aid,
            obs_snapshot=obs,
            refresh_observation=False,
            arm_pose_mode="vision_tuck",
            held_context=held_context_for_update if isinstance(held_context_for_update, dict) else None,
            receptacle_target_context=(
                receptacle_context_for_update if isinstance(receptacle_context_for_update, dict) else None
            ),
            extra={
                "recp_object_id": recp_id,
                "recp_handle": act.get("recp_handle"),
                "recp_object_type": act.get("recp_object_type"),
                "placed_object_runtime_position": (
                    None
                    if released_obj is None
                    else np.asarray(released_obj.translation, dtype=np.float32).tolist()
                ),
                "placed_object_runtime_handle": (
                    None
                    if released_obj is None
                    else str(getattr(released_obj, "handle", ""))
                ),
            },
        )

        second_map(sim, output_path='/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations2.json')
        compare_objects_location('/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json', '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations2.json', '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/memory3.json')
        first_map(sim, '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json')

        down_rgb = _obs_by_suffix(obs, "head_down_rgb")
        filename = f"short_memory_{_EXEC_SHORT_MEMORY_COUNTER:04d}.png"
        save_rgb_observation_png(down_rgb, "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/short_term", filename)

        _EXEC_SHORT_MEMORY_COUNTER += 1

        task_description = load_task_description()
        print(f"Executing task: {task_description}")
        analyze_specific_image(directory_path, filename, task_description)
        print("Analysis complete. Results saved to 'analysis_results.json'.")

    elif name == "Done":
        obs = _env_step({"action": EMPTY_ACTION, "action_args": {}})

    if not _save_step_views(obs, _EXEC_IMG_COUNTER):
        task_over = True
        _show_realtime_panel(obs, banner="STOP")
        return False
    _EXEC_IMG_COUNTER += 1
    _show_realtime_panel(obs, banner=f"RUN {name} | q={len(action_queue)}")
    return True


def exec_actions(stop_when_empty: bool = False, timeout_sec: float = None):
    """
    연속 처리 루프. 기본적으로 exec_actions_tick()을 반복 호출한다.
    """
    global task_over
    start_t = time.time()
    while not task_over:
        if timeout_sec is not None and (time.time() - start_t) > float(timeout_sec):
            print(f"[queue] exec_actions timeout ({float(timeout_sec):.1f}s)")
            break
        processed = exec_actions_tick()
        if not processed:
            if stop_when_empty and not action_queue:
                break
            time.sleep(0.01)

# """
# 여기는 action 보조 함수 (action이 필요로하는 값 얻을 때 사용하는 함수들)
# """
def _robot_to_agent_id(bot) -> int:
    name = str(bot.get("name", "fetch1"))
    m = re.search(r"(\d+)$", name)
    if m:
        return max(0, int(m.group(1)) - 1)
    return 0

def _wrap_deg_pm180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def _resolve_best_align_target_xyz(
    agent_id: int,
    fallback_target_xyz: np.ndarray,
) -> np.ndarray:
    """
    최종 시선 정렬에 사용할 look-at target을 결정한다.

    왜 필요한가:
    - GoToObject는 HSG / ConceptGraph 기준 좌표로 목적지까지 이동한다.
    - 하지만 최종 정렬은 가능한 한 "실제 sim runtime object" 기준으로 보는 편이 정확하다.
    - 특히 table, cabinet, sofa 같은 큰 물체는 ConceptGraph bbox_center보다
      runtime object의 AABB 중심이 더 안정적인 시선 목표가 된다.

    우선순위:
    1) last_nav_target 안의 runtime_object_id가 있으면 그 object의 world AABB 중심
    2) runtime_position
    3) conceptgraph_bbox_center
    4) fallback_target_xyz

    주의:
    - yaw 정렬에서는 x,z만 실제로 사용하므로 y는 큰 의미가 없다.
    - 그래도 디버깅 편의를 위해 항상 3D point로 반환한다.
    """
    fallback = np.asarray(fallback_target_xyz, dtype=np.float32)
    if fallback.shape != (3,) or not np.all(np.isfinite(fallback)):
        raise ValueError(f"Invalid fallback_target_xyz: {fallback_target_xyz}")

    target_context = _get_last_nav_target_context(agent_id)
    if not isinstance(target_context, dict):
        return fallback

    runtime_obj_id = target_context.get("runtime_object_id")
    if runtime_obj_id not in (None, ""):
        try:
            obj_id = int(runtime_obj_id)
            rom = sim.get_rigid_object_manager()
            aom = sim.get_articulated_object_manager()
            runtime_obj = rom.get_object_by_id(obj_id)
            if runtime_obj is None:
                runtime_obj = aom.get_object_by_id(obj_id)

            if runtime_obj is not None:
                bb_min, bb_max = _world_aabb_minmax(runtime_obj)
                aabb_center = np.asarray(
                    [
                        float((bb_min[0] + bb_max[0]) * 0.5),
                        float((bb_min[1] + bb_max[1]) * 0.5),
                        float((bb_min[2] + bb_max[2]) * 0.5),
                    ],
                    dtype=np.float32,
                )
                if np.all(np.isfinite(aabb_center)):
                    return aabb_center
        except Exception as exc:
            print(f"[align] runtime AABB center lookup failed: {exc}")

    for key in ("runtime_position", "conceptgraph_bbox_center", "nav_xyz"):
        value = target_context.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (3,) and np.all(np.isfinite(arr)):
            return arr

    return fallback

def _enqueue_align_to_target(
    agent_id: int,
    target_xyz: np.ndarray,
    align_thresh_deg: float = 2.0,
    camera_yaw_offset_deg: float = 0.0,
    post_align_update: Optional[Dict[str, Any]] = None,
):
    """
    최종 정렬을 위해 closed-loop AlignToTarget 액션을 큐에 넣는다.

    예전 방식의 문제:
    - 현재 pose에서 각도를 한 번 계산한 뒤
      RotateLeft/RotateRight를 "한 방"에 큰 각도로 넣는 open-loop 방식이었다.
    - 그런데 실제 회전량은 sim dynamics / BASE_ANG_SCALE / tick timing에 좌우되어
      계산한 degrees만큼 정확히 돌지 않을 수 있다.
    - 그래서 목적지에는 잘 도착했는데 최종 시선만 이상하게 틀어지는 현상이 생겼다.

    새 방식:
    - AlignToTarget 액션이 매 tick마다 현재 yaw와 target yaw를 다시 계산한다.
    - 오차가 남아 있으면 TURN_CHUNK_DEG 단위로 조금씩 회전한다.
    - 즉 "작게 돌고 다시 측정"하는 closed-loop 정렬 방식이다.
    """
    arr = np.asarray(target_xyz, dtype=np.float32)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        print(f"[align] skip invalid target_xyz={target_xyz}")
        return

    action_queue.insert(
        0,
        {
            "action": "AlignToTarget",
            "target_xyz": arr.tolist(),
            "align_thresh_deg": float(align_thresh_deg),
            "camera_yaw_offset_deg": float(camera_yaw_offset_deg),
            "max_step_deg": float(TURN_CHUNK_DEG),
            "agent_id": int(agent_id),
            "post_align_update": None if post_align_update is None else dict(post_align_update),
        },
    )

def _object_nav_expert_tick(act: dict):
    """
    ObjectNavExpertAction 1개를 받아서
    ShortestPathFollower의 next action 1스텝만 즉시 실행하고 종료한다.

    핵심 원칙:
    - low-level 액션(MoveAhead/Rotate...)을 queue에 넣지 않음
    - 재삽입도 하지 않음
    - 반복 enqueue는 상위 GoToObject/ExploreObject while 루프가 담당
    -> 기존 KARMA 시스템 구조랑 똑같이 만들기 위함
    """
    agent_id = int(act.get("agent_id", 0))
    goal_radius = float(act.get("goal_radius", 0.30)) # 도착 판정 반경 (목표점까지 xz 거리)
    # GoToObject에서 오브젝트 찾을 때 쓰는 JSON 경로 옵션
    objects_json_path = str(
        act.get(
            "objects_json_path",
            "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json",
        )
    )
    # raw_target/nav_goal 없으면 이번 액션에서 계산
    if ("raw_target" not in act) or ("nav_goal" not in act):
        if "position" in act: # Explore 원본 스타일 호환
            p = act["position"]
            raw_target = np.array([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float32)
            label = str(act.get("dest_obj_regex", f"ExplorePoint({raw_target[0]:.2f},{raw_target[2]:.2f})"))
        else: # GoToObject 스타일 (dest_obj_regex ex: "Sink"로 객체 좌표 찾는 경우)
            raw_target = find_target_object_position(
                sim,
                str(act.get("dest_obj_regex", "")),
                objects_json_path=objects_json_path,
                agent_idx=agent_id,
            )
            label = str(act.get("dest_obj_regex", "target"))

        # raw_target을 실제 navmesh 상 이동가능 목표점으로 투영
        nav_goal, goal_src, geo_dist, obj_gap = project_goal_to_navmesh(sim, raw_target, agent_idx=agent_id)
        # 다음 tick에서도 재사용하도록 act에 저장
        act["raw_target"] = np.asarray(raw_target, dtype=np.float32).tolist()
        act["nav_goal"] = np.asarray(nav_goal, dtype=np.float32).tolist()
        act["_label"] = label
        # 디버그 로그
        print(
            f"[ObjectNavExpertAction] target={label}, source={goal_src}, "
            f"geo_dist={'inf' if not np.isfinite(geo_dist) else f'{geo_dist:.3f}'}, obj_gap={obj_gap:.3f}m"
        )

    # act에서 최종 목표/라벨 꺼냄
    raw_target = np.asarray(act["raw_target"], dtype=np.float32)
    nav_goal = np.asarray(act["nav_goal"], dtype=np.float32)
    label = str(act.get("_label", act.get("dest_obj_regex", "target")))

    # 수행 전 도착 체크
    cur_pos, _ = get_fetch_base_pose(sim, agent_id) # 현재 로봇 base 위치
    nav_dist = distance_xz(cur_pos, nav_goal) # 목표 nav 점까지 xz 거리
    if nav_dist <= goal_radius:
        if bool(act.get("align_on_reach", True)):
            _enqueue_align_to_target(agent_id, raw_target)
        print(f"Reached: {label}")
        return
        
    # follower next action 1개 결정
    sync_follower_agent_state(sim, agent_id) # follower가 보는 pose와 현재 articulated base 동기화
    follower = ShortestPathFollower(sim, goal_radius=goal_radius, return_one_hot=False, stop_on_error=True)
    next_action = follower.get_next_action(nav_goal) # move_forward / turn_left / turn_right / stop

    # next action 즉시 실행 
    _tuck_arm_for_nav() # 팔 원 상태로 접기
    if next_action == HabitatSimActions.move_forward:
        _step_base(1.0, 0.0, repeat=int(act.get("forward_repeat", 2)))
    elif next_action == HabitatSimActions.turn_left:
        rep = max(1, int(np.ceil(float(act.get("turn_chunk_deg", TURN_CHUNK_DEG)) / TURN_CHUNK_DEG)))
        _step_base(0.0, +ROT_DIR_SIGN, repeat=rep)
    elif next_action == HabitatSimActions.turn_right:
        rep = max(1, int(np.ceil(float(act.get("turn_chunk_deg", TURN_CHUNK_DEG)) / TURN_CHUNK_DEG)))
        _step_base(0.0, -ROT_DIR_SIGN, repeat=rep)
    else:
        print(f"[ObjectNavExpertAction] stop/unknown for {label}")
        return

def _get_reachable_positions_for_explore(
    agent_idx: int = 0,
    island_index: int = None, # 어느 연결 컴포넌트의 점들만 가져올지 (None: 현재 로봇이 서있는 island)
    dedup_decimals: int = 3,
    snap_to_navmesh: bool = True,
):
    """
    AI2-THOR GetReachablePositions 대체:
    현재 agent가 있는 island의 navmesh vertices 전체를 반환한다.

    - 랜덤 샘플링(get_random_navigable_point) 사용 안 함
    - 현재 island의 모든 정점(build_navmesh_vertices) 사용
    - 필요 시 각 후보를 snap_point로 같은 island navmesh 위로 보정
    - 중복점은 좌표 반올림 기준으로 제거
    """
    pf = sim.pathfinder
    if not pf.is_loaded:
        raise RuntimeError("NavMesh is not loaded in the simulator.")

    # island를 지정하지 않으면 현재 agent가 서 있는 island 사용
    if island_index is None:
        cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_idx)
        island_index = int(pf.get_island(np.asarray(cur_pos, dtype=np.float32)))

    verts = np.asarray(pf.build_navmesh_vertices(island_index), dtype=np.float32)
    if verts.size == 0:
        raise RuntimeError(f"No navmesh vertices on island {island_index}.")

    # 비정상 좌표 제거
    verts = verts[np.all(np.isfinite(verts), axis=1)]
    if verts.size == 0:
        raise RuntimeError(f"All navmesh vertices are non-finite on island {island_index}.")

    # 안전성 강화: 후보점을 동일 island navmesh 위로 재스냅
    # (일반적으로 navmesh vertices는 이미 유효하지만, 버전/수치 이슈 대비)
    if snap_to_navmesh:
        snapped = []
        for v in verts:
            s = np.array(pf.snap_point(v, island_index), dtype=np.float32)
            if np.all(np.isfinite(s)):
                snapped.append(s)
        if not snapped:
            raise RuntimeError(f"Failed to snap reachable points on island {island_index}.")
        verts = np.asarray(snapped, dtype=np.float32)

    # 너무 촘촘한 중복 정점을 정리해 closest_node의 후보 중복을 줄인다.
    rounded = np.round(verts, decimals=int(dedup_decimals))
    _, keep_idx = np.unique(rounded, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    verts = verts[keep_idx]

    pts = [(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    print(
        f"[reachable] island={island_index}, vertices={len(pts)}, "
        f"snap={'on' if snap_to_navmesh else 'off'}"
    )
    return pts

def _world_aabb_minmax(obj):
    # 입력 obj(리셉터클 후보 객체)의 AABB를 월드 좌표계 기준 min/max로 반환하는 유틸 함수
    try:
        # obj.aabb는 보통 객체 로컬 좌표계 기준 박스라서,
        # 실제 씬에서의 위치/회전을 반영한 월드 AABB로 변환
        bb_world = habitat_sim.geo.get_transformed_bb(obj.aabb, obj.transformation)
    except Exception:
        # 변환 실패 시(객체 타입/버전 이슈 등) 로컬 aabb라도 fallback으로 사용
        bb_world = obj.aabb

    # Habitat 버전에 따라 bb.min / bb.max가
    # 프로퍼티일 수도 있고, 호출 가능한 메서드일 수도 있어서 둘 다 처리
    bb_min = bb_world.min() if callable(getattr(bb_world, "min", None)) else bb_world.min
    bb_max = bb_world.max() if callable(getattr(bb_world, "max", None)) else bb_world.max

    # 이후 계산 편의를 위해 numpy float32로 통일해서 반환
    return np.array(bb_min, dtype=np.float32), np.array(bb_max, dtype=np.float32)

def _place_held_object_on_receptacle(agent_id: int, recp_obj_id: int, y_offset: float=0.01):
    """
    원본 AI2-THOR은 PutObject(objectId=recp_obj_id) 자체가 고수준 액션이라,
    시뮬레이터가 내부에서 리셉터클 위 적절한 배치 위치를 계산함
        - 원본 실행부: execute_LLM_plan.py (line 304)
    Habitat-sim에는 그와 1:1 대응되는 PutObject(objectId) primitive가 없음. 
    그래서 recp_id를 실제로 반영하려면 직접 목표 좌표를 계산해서(예: AABB top-center) 배치해야 함
    즉, 좌표 계산 로직은 AI2-THOR 내장 기능을 Habitat에서 수동으로 대체한 것
    """
    # 현재 agent가 잡고 있는 오브젝트를
    # receptacle(recp_obj_id) AABB 상단 중앙으로 위치를 맞추는 함수

    # agent의 grasp manager 핸들 획득
    gm = sim.get_agent_data(agent_id).grasp_mgr

    # 손에 아무것도 없으면 둘 게 없으므로 에러
    if not gm.is_grasped:
        raise RuntimeError("No grasped object.")
    
    # 씬 오브젝트 매니저
    # - ROM: rigid object
    # - AOM: articulated object
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()

    # 먼저 rigid에서 receptacle id 탐색
    recp_obj = rom.get_object_by_id(recp_obj_id)

    # rigid에 없으면 articulated에서 탐색
    if recp_obj is None:
        recp_obj = aom.get_object_by_id(recp_obj_id)

    # 둘 다 없으면 잘못된 id이므로 중단
    if recp_obj is None:
        raise RuntimeError(f"Receptacle object id not found: {recp_obj_id}")
    
    # receptacle의 월드 AABB min/max 계산
    bb_min, bb_max = _world_aabb_minmax(recp_obj)

    # 현재 손에 붙어 있는(잡힌) rigid object 핸들
    held = gm.snap_rigid_obj

    # 잡힌 물체의 현재 월드 AABB 높이를 이용해
    # 중심점이 아니라 "밑면"이 리셉터클 표면 위에 오도록 배치한다.
    held_bb_min, held_bb_max = _world_aabb_minmax(held)
    held_half_height = max(float(held_bb_max[1] - held_bb_min[1]) * 0.5, 0.01)

    # 놓을 목표 좌표 계산:
    # x,z는 AABB 중앙
    # y는 receptacle 상단 + object 절반 높이 + 작은 clearance
    target_xyz = np.array(
        [
            (bb_min[0] + bb_max[0]) * 0.5,
            bb_max[1] + held_half_height + y_offset,
            (bb_min[2] + bb_max[2]) * 0.5,
        ],
        dtype=np.float32,
    )

    # 잡힌 물체를 계산한 위치로 즉시 이동
    # (아직 release는 하지 않은 상태)
    held.translation = target_xyz
    _zero_rigid_body_motion(held)

    # 디버깅/로그용으로 최종 배치 좌표 반환
    return target_xyz


# """
# 여기는 action 정의
# """
def GoToObject(robot, dest_obj):
    print("Going to ", dest_obj)

    # 단일 robot 입력도 리스트로 통일
    bots = robot if isinstance(robot, list) else [robot]
    no_agents = len(bots)
    if no_agents == 0:
        return

    # karma 원본과 동일한 상태 배열
    dist_goals = [10.0] * no_agents
    prev_dist_goals = [10.0] * no_agents
    count_since_update = [0] * no_agents
    clost_node_location = [0] * no_agents

    # 각 로봇의 agent_id 추출
    agent_ids = [_robot_to_agent_id(bot) for bot in bots]

    # ============================== [MOD: GoToObject strict target policy] ==============================
    # 이제 GoToObject는 "정확히 어디로 갈지 알고 있는 경우"에만 쓰는 액션이다.
    # 즉 object + floor + room/room_instance가 모두 지정되어 있어야 하며,
    # 정보가 덜 주어진 경우는 이후 Explore 계열이 처리하도록 역할을 분리한다.
    try:
        strict_spec = require_fully_specified_navigation_target(dest_obj)

        # 1) floor / room / object가 모두 지정된 target만 resolve
        # 2) HSG로 계층적으로 후보를 좁히고
        # 3) 선택된 object의 source_id로 ConceptGraph object를 복원한 뒤
        # 4) bbox_center를 raw reference target으로 사용한다.
        resolved = resolve_hierarchical_target_with_conceptgraph(
            sim=sim,
            scene_id=SCENE_ID,
            dest_obj=strict_spec,
            agent_idx=agent_ids[0],
        )
        dest_obj_pos = resolved["nav_xyz"].astype(np.float32).tolist()

        # PickupObject / PutObject가 이어받을 수 있도록
        # graph/ConceptGraph 기준 target context를 agent execution cache에 저장한다.
        target_context = {
            "target_kind": "object",
            "query": resolved["query"],
            "graph_floor_id": None if resolved["graph_floor"] is None else resolved["graph_floor"]["floor_id"],
            "graph_room_id": None if not resolved["graph_room_candidates"] else resolved["graph_room_candidates"][0]["room_id"],
            "graph_room_name": None if not resolved["graph_room_candidates"] else resolved["graph_room_candidates"][0]["name"],
            "graph_room_instance": None if not resolved["graph_room_candidates"] else resolved["graph_room_candidates"][0]["instance_name"],
            "graph_object_id": resolved["graph_object"]["object_id"],
            "graph_object_name": resolved["graph_object"]["name"],
            "graph_source_id": resolved["graph_object"]["source_id"],
            "conceptgraph_object_key": resolved["conceptgraph_object"].get("object_key"),
            "conceptgraph_object_tag": resolved["conceptgraph_object"].get("object_tag"),
            "conceptgraph_caption": resolved["conceptgraph_object"].get("object_caption"),
            "conceptgraph_bbox_center": resolved["conceptgraph_object"].get("bbox_center"),
            "nav_xyz": dest_obj_pos,
            "nav_source": resolved["nav_source"],
            # runtime object는 navigation 후점에 resolve해서 채울 예정
            "runtime_object_id": None,
            "runtime_handle": None,
            "runtime_object_type": None,
        }
        _set_last_nav_target_context(
            agent_ids[0],
            target_context,
            bot=bots[0],
        )
    except Exception as e:
        print(f"[GoToObject] target lookup failed: {e}")
        return
    
    # reachable 후보점: 현재 island navmesh vertices 전체
    reachable_positions = _get_reachable_positions_for_explore(agent_idx=agent_ids[0])
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

    # 목표 근접 임계값
    goal_thresh = 0.30

    # 원본 패턴: 모든 로봇이 아직 목표보다 멀면 루프 지속
    while all(d > goal_thresh for d in dist_goals):
        for ia, agent_id in enumerate(agent_ids):
            # 현재 로봇 위치
            cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_id)
            location = [float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])]

            # 거리 갱신
            prev_dist_goals[ia] = dist_goals[ia]

            # 종료 기준은 enqueue한 탐색점(crp) 기준으로 맞추기
            dist_to_crp = distance_pts(location, crp[ia])
            dist_to_obj = distance_pts(location, dest_obj_pos)
            dist_goals[ia] = dist_to_crp

            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])

            ## 남은 거리 디버그용
            set_debug_overlay(
            f"mode=GoToObject target={dest_obj} q={len(action_queue)}",
            f"agent={agent_id} dist_goals={dist_goals[ia]:.3f} thresh={goal_thresh:.3f}",
            f"dist_to_obj={dist_to_obj:.3f} dist_to_crp={dist_to_crp:.3f}",
            f"dist_del={dist_del:.3f} stuck={count_since_update[ia]}",
        )

            # 원본 정체 판단
            # if dist_del < -100: # 이거는 사실 정체 판단 취급 안 하는 거..
            # if dist_del < 0.2: # 원본 karma
            if dist_del < -100:
                count_since_update[ia] += 1
            else:
                count_since_update[ia] = 0

            # 정체가 아니면 ObjectNavExpertAction enqueue
            if count_since_update[ia] < 15:
                action_queue.append(
                    {
                        "action": "ObjectNavExpertAction",
                        "position": {
                            "x": float(crp[ia][0]),
                            "y": float(crp[ia][1]),
                            "z": float(crp[ia][2]),
                        },
                        "align_on_reach": False, # _object_nav_expert_tick의 자동 align을 GoToObject에서는 끄기
                        "dest_obj_regex": str(dest_obj),  # 로그 라벨 용도
                        "agent_id": agent_id,
                        "goal_radius": goal_thresh,
                        "forward_repeat": 2,
                        "turn_chunk_deg": TURN_CHUNK_DEG,
                        "objects_json_path": "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json",
                    }
                )
            else:
                # 정체면 후보점 갱신
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

            # 원본 템포 유지
            time.sleep(0.2)
        
    # while 끝난 뒤, 남아있을 수 있는 nav 액션 제거 (반복 Reached 방지)
    for aid in agent_ids:
        drop_pending_nav_actions_for_agent(action_queue, aid)

    # ============================== [MOD: eager runtime-object resolution after arrival] ==============================
    # navigation이 끝난 시점은 로봇이 target object 근처에 가장 가까운 순간이므로,
    # graph target을 실제 sim runtime object id로 연결하기 가장 좋은 타이밍이다.
    #
    # 실패해도 navigation 자체는 성공으로 간주하고,
    # PickupObject / PutObject에서 한 번 더 resolve할 수 있게 한다.
    try:
        current_target_context = _get_last_nav_target_context(agent_ids[0])
        if current_target_context is not None:
            runtime_match = resolve_runtime_object_from_target_context(
                sim=sim,
                target_context=current_target_context,
                agent_idx=agent_ids[0],
                max_xz_dist=1.75,
                allow_nearest_fallback=False,
            )
            current_target_context["runtime_object_id"] = runtime_match["runtime_object_id"]
            current_target_context["runtime_handle"] = runtime_match["runtime_handle"]
            current_target_context["runtime_object_type"] = runtime_match["runtime_object_type"]
            current_target_context["runtime_position"] = runtime_match["runtime_position"]
            current_target_context["runtime_match_source"] = runtime_match["match_source"]
            _set_last_nav_target_context(agent_ids[0], current_target_context, bot=bots[0])
            print(
                "[GoToObject] runtime match =",
                {
                    "object_id": runtime_match["runtime_object_id"],
                    "object_type": runtime_match["runtime_object_type"],
                    "dist_xz": runtime_match["reference_dist_xz"],
                },
            )
    except Exception as e:
        print(f"[GoToObject] runtime match skipped/failed: {e}")

    # ============================== [MOD: runtime-aware final align] ==============================
    # 예전에는 graph/ConceptGraph에서 나온 dest_obj_pos를 바로 바라보게 했는데,
    # 큰 물체(table, cabinet 등)에서는 그 reference point가 실제 runtime object 중심과
    # 어긋날 수 있어서 최종 시선이 이상해지는 경우가 있었다.
    #
    # 이제는:
    # 1) 가능하면 runtime object의 world AABB 중심을 바라보도록 하고
    # 2) 회전은 open-loop 한 방이 아니라 closed-loop AlignToTarget으로 수행한다.
    align_target_xyz = _resolve_best_align_target_xyz(
        agent_ids[0],
        np.asarray(dest_obj_pos, dtype=np.float32),
    )
    current_target_context = _get_last_nav_target_context(agent_ids[0])
    _enqueue_align_to_target(
        agent_ids[0],
        align_target_xyz,
        post_align_update={
            "event_kind": "goto_align",
            "refresh_observation": True,
            "arm_pose_mode": "vision_tuck",
            "target_context": None if current_target_context is None else dict(current_target_context),
            "extra": {
                "dest_obj": dest_obj,
                "align_target_xyz": np.asarray(align_target_xyz, dtype=np.float32).tolist(),
            },
        },
    )

    print("Reached: ", dest_obj)


def ExploreObject(robots, dest_obj, dest_obj2):
    """
    탐색 포인트로 이동 + 이동 중에 목표 객체 발견하면 조기 종료
    - dest_obj_pos2: dest_obj2(찾고 싶은 객체) 중심 좌표 찾기 
    - dest_obj: (탐색할 포인트 좌표) 정리
    - closest_node(...)로 각 로봇의 실제 이동 목표점(crp) 선택
    - 루프 진행
        -- dist_goals[ia] = distance(robot, crp[ia])
        -- dist_goals2[ia] = distance(robot, dest_obj_pos2) -> 객체 발견 여부
        -- dist_goals2 < 1.7면 exit_flag=True (객체 발견)
        -- 아니라면 dist_del = |현재 dist_goals - 이전 dist_goals|로 정체 판정
        -- 정체가 길어지면 다른 closest_node 후보로 갈아탐
    - 루프 종료 후 정렬(회전)하고 exit_goto 반환
        -- True: 이동 중 객체 찾음
        -- False: 못 찾고 탐색 포인트만 도달/종료
    """
    print("Explore", dest_obj2)

    if not isinstance(robots, list):
        robots = [robots]
    no_agents = len(robots)
    if no_agents == 0:
        return False

    # 원본과 동일한 상태 배열
    dist_goals = [10.0] * no_agents # 로봇이 현재 탐색 포인트 crp로 잘 가고 있는지 보는 이동 제어용 거리
    dist_goals2 = [10.0] * no_agents # 로봇이 찾고 싶은 객체(dest_obj2)에 충분히 가까워졌는지 보는 발견 판정용 거리
    prev_dist_goals = [10.0] * no_agents
    count_since_update = [0] * no_agents
    clost_node_location = [0] * no_agents

    agent_ids = [_robot_to_agent_id(bot) for bot in robots]

    # 원본처럼 탐색 시작 시점의 목표 객체 위치를 기준으로 사용
    try:
        target_xyz, target_meta = find_target_object_position_live(
            sim, str(dest_obj2), agent_idx=agent_ids[0]
        )
        dest_obj_pos2 = [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])]
    except Exception as e:
        print(f"[ExploreObject] target lookup failed: {e}")
        return False

    # 탐색할 포인트 좌표 (dest_obj) 정규화
    p0, _ = get_fetch_base_pose(sim, agent_idx=agent_ids[0])
    dxyz = dest_obj_to_xyz(sim, dest_obj, agent_id=agent_ids[0])
    dest_obj_pos = [float(dxyz[0]), float(p0[1]), float(dxyz[2])]

    # 원본 closest_node 로직 유지 (후보는 island navmesh vertices 전체)
    reachable_positions = _get_reachable_positions_for_explore(agent_idx=agent_ids[0])
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

    goal_thresh = 0.3
    obj_detect_thresh = 1.7
    # at least one robot is far away from the goal
    exit_flag = False
    exit_goto = False
    while all(d > goal_thresh for d in dist_goals):
        for ia, agent_id in enumerate(agent_ids):
            cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_id)
            location = [float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2])]

            prev_dist_goals[ia] = dist_goals[ia]
            dist_goals[ia] = distance_pts(location, crp[ia]) # 탐색 포인트 거리
            dist_goals2[ia] = distance_pts(location, dest_obj_pos2) # 오브젝트 거리

            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            # 디버깅용 시각화
            set_debug_overlay(
                f"mode=ExploreObject target={dest_obj2} q={len(action_queue)}",
                f"agent={agent_id} d_point={dist_goals[ia]:.3f} thresh_point={goal_thresh:.3f}",
                f"agent={agent_id} d_obj={dist_goals2[ia]:.3f} thresh_obj={obj_detect_thresh:.3f}",
                f"dist_del={dist_del:.3f} stuck={count_since_update[ia]}",
            )

            # 원본: 목표 객체 근접 발견 시 종료
            if dist_goals2[ia] < obj_detect_thresh:
                print(f"옹 목표 객체 근접 확인")
                # time.sleep(1)
                exit_flag = True
                break

            # 원본 stuck 판단
            # 일단 이 로직 무시할 수 있도록 -100으로 해놓기
            # if dist_del < 0.2:
            if dist_del < -100:
                count_since_update[ia] += 1
            else:
                count_since_update[ia] = 0

            # 원본: stuck 아니면 ObjectNavExpertAction enqueue
            if count_since_update[ia] < 15:
                raw_target = np.array(crp[ia], dtype=np.float32)
                nav_goal, _, _, _ = project_goal_to_navmesh(sim, raw_target, agent_idx=agent_id)

                action_queue.append(
                    {
                        "action": "ObjectNavExpertAction",
                        # 로그 혼선 방지: 이 액션은 "탐색 포인트"로 가는 액션
                        "dest_obj_regex": f"ExplorePoint({raw_target[0]:.2f},{raw_target[2]:.2f})",
                        "agent_id": agent_id,
                        "raw_target": raw_target.tolist(),
                        "nav_goal": np.asarray(nav_goal, dtype=np.float32).tolist(),
                        "goal_radius": 0.30,
                        "forward_repeat": 2,
                        "turn_chunk_deg": TURN_CHUNK_DEG,
                    }
                )
            else:
                # 원본: 목표 후보점 업데이트
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

            time.sleep(0.2)
        if exit_flag:
            exit_goto = True
            print("find", dest_obj2)
            break

    # while 끝난 뒤, 남아있을 수 있는 nav 액션 제거 (반복 Reached 방지)
    for aid in agent_ids:
        drop_pending_nav_actions_for_agent(action_queue, aid)

    # 원본 align 의도 유지 (탐색 포인트 방향 정렬)
    _enqueue_align_to_target(agent_ids[0], np.array(dest_obj_pos, dtype=np.float32))
    set_debug_overlay("mode=ExploreObject done", f"found={exit_goto}", f"target={dest_obj2}") # 디버깅용 시각화
    print("Reached explore point:", tuple(round(v, 2) for v in dest_obj_pos))
    return exit_goto


def Explore(robot, sw_obj, available_positions):
    """
    탐색 points 순회 + 발견 시 즉시 확정 이동
    GoToObject랑 다른 점은, available positions을 돌면서 원하는 오브젝트가 진짜로 있는지 확인한 후 가도록 한다는 점.
    중간 확인 단계가 추가됐다고 생각하면 됨.
        - 있다면 GoToObject O (성공 확정)
        - 없다면 GoToObject X (걍 호출 자체를 안 해)

    입력:
        - sw_obj: 최종적으로 찾고 싶은 물체 이름/패턴 (ex: "Apple")
        - available_positions: 탐색 후보 좌표 리스트 (long-term memory points)
    """
    exit_goto = False # 현재 탐색 point에서 물체를 찾았는지
    exit_goto_finish = False # 전체 탐색 끝낼지 여부

    explore_point_count = 0 # 실제로 몇 개 포인트 방문/시도했는지 기록

    for positions in available_positions:
        if exit_goto_finish:
            break

        exit_goto = ExploreObject(robot, positions, sw_obj) # 현재 포인트 positions로 가는 과정에서 sw_obj를 발견하면 True 반환
        explore_point_count += 1

        if exit_goto: # 이미 찾았으면 나머지 포인트는 안 감
            GoToObject(robot, sw_obj)
            exit_goto_finish = True
    print(explore_point_count)


def PickupObject(robot, pick_obj):
    """
    [MODIFIED]
    PickupObject는 더 이상 sim 전체를 regex scan해서 "첫 번째 맞는 물체"를 집지 않는다.

    새 동작:
    1) 가장 최근 navigation target(last_nav_target)을 확인
    2) 사용자가 요청한 pick_obj와 그 target이 semantic하게 맞는지 검증
    3) 필요하면 graph target -> runtime object id를 다시 resolve
    4) 실제 runtime object id를 큐에 넣고, grasp 성공 시 exec tick에서 held_object context를 저장

    즉:
    - GoToObject / Explore가 "어디로 갔는지"를 정하고
    - PickupObject는 "방금 찾아간 target을 실제로 집는 실행"만 담당한다.
    """
    bots = robot if isinstance(robot, list) else [robot]
    if not bots:
        return False

    agent_id = _robot_to_agent_id(bots[0])
    pick_regex = str(pick_obj or "")

    # ------------------------------------------------------------------
    # 1) 마지막 navigation target context 확인
    # ------------------------------------------------------------------
    last_target_context = _get_last_nav_target_context(agent_id)
    if last_target_context is None:
        print(
            "[PickupObject] failed | no last_nav_target context. "
            "Call GoToObject/Explore first."
        )
        return False

    # semantic query와 현재 target context가 어긋나면 잘못된 물체를 집을 위험이 크므로 중단
    if not _query_matches_target_context(pick_regex, last_target_context):
        print(
            "[PickupObject] failed | requested object does not match last_nav_target. "
            f"pick_obj={pick_regex}, last_target={last_target_context.get('query')}"
        )
        return False

    # ------------------------------------------------------------------
    # 2) runtime object id 확보
    # ------------------------------------------------------------------
    # GoToObject가 navigation 직후 eager resolve를 시도했을 수 있지만,
    # 여기서는 pickup 직전에 한 번 더 보장해주는 쪽이 안전하다.
    runtime_match = None
    if last_target_context.get("runtime_object_id"):
        runtime_match = {
            "runtime_object_id": str(last_target_context.get("runtime_object_id")),
            "runtime_handle": str(last_target_context.get("runtime_handle", "")),
            "runtime_object_type": str(last_target_context.get("runtime_object_type", "")),
            "runtime_position": last_target_context.get("runtime_position"),
            "reference_dist_xz": float("nan"),
            "match_source": str(last_target_context.get("runtime_match_source", "cached")),
        }
    else:
        try:
            runtime_match = resolve_runtime_object_from_target_context(
                sim=sim,
                target_context=last_target_context,
                agent_idx=agent_id,
                max_xz_dist=1.75,
                allow_nearest_fallback=False,
            )
            last_target_context["runtime_object_id"] = runtime_match["runtime_object_id"]
            last_target_context["runtime_handle"] = runtime_match["runtime_handle"]
            last_target_context["runtime_object_type"] = runtime_match["runtime_object_type"]
            last_target_context["runtime_position"] = runtime_match["runtime_position"]
            last_target_context["runtime_match_source"] = runtime_match["match_source"]
            _set_last_nav_target_context(agent_id, last_target_context, bot=bots[0])
        except Exception as e:
            print(f"[PickupObject] runtime resolve failed: {e}")
            return False

    pick_obj_id = str(runtime_match["runtime_object_id"])

    # ------------------------------------------------------------------
    # 3) 현재 로봇과 타겟의 거리 로그용 계산
    # ------------------------------------------------------------------
    dist_xz = None
    runtime_position = runtime_match.get("runtime_position")
    if runtime_position is not None:
        try:
            cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_id)
            tgt = np.asarray(runtime_position, dtype=np.float32)
            dist_xz = distance_xz(cur_pos, tgt)
        except Exception:
            dist_xz = None

    # ------------------------------------------------------------------
    # 4) 실제 집기 액션 payload 생성
    # ------------------------------------------------------------------
    # 실제 grasp는 exec_actions_tick의 PickupObject 분기에서 수행하고,
    # 여기서는 그 분기가 사용할 runtime object id와 memory context를 큐에 등록한다.
    act = {
        "action": "PickupObject",
        "agent_id": agent_id,
        "pick_obj_regex": pick_regex,
        "pick_obj_id": str(pick_obj_id),
        "pick_match_key": str(runtime_match.get("match_source", "memory_context")),
        "target_handle": str(runtime_match.get("runtime_handle", "")),
        "target_object_type": str(runtime_match.get("runtime_object_type", "")),
        "memory_target_context": dict(last_target_context),
    }
    if dist_xz is not None:
        act["target_dist_xz"] = float(dist_xz)

    # 큐 등록
    action_queue.append(act)

    # 디버그 로그
    if dist_xz is None:
        print(
            f"[PickupObject] queued pick via memory context: "
            f"{pick_regex} -> objectId={pick_obj_id}"
        )
    else:
        print(
            f"[PickupObject] queued pick via memory context: {pick_regex} -> objectId={pick_obj_id}, "
            f"dist_xz={dist_xz:.3f}m"
        )
    return True

def PutObject(robot, put_obj, recp):
    """
    [MODIFIED]
    PutObject는 이제 sim 전체에서 receptacle을 직접 regex scan하지 않는다.

    새 동작:
    1) 현재 grasp 중인 held_object context 확인
    2) 마지막 navigation target(last_nav_target)을 receptacle target으로 사용
    3) 필요하면 그 target context를 runtime receptacle object로 resolve
    4) queue에는 runtime receptacle id를 넣고, 실제 release는 exec tick에서 수행
    """
    bots = robot if isinstance(robot, list) else [robot]
    if not bots:
        return False

    agent_id = _robot_to_agent_id(bots[0])
    recp_regex = str(recp)

    gm = sim.get_agent_data(agent_id).grasp_mgr
    if not gm.is_grasped:
        print(f"[PutObject] failed | nothing is grasped")
        return False

    # ------------------------------------------------------------------
    # 1) 현재 손에 든 물체와 put_obj 요청이 semantic하게 맞는지 확인
    # ------------------------------------------------------------------
    held_context = _get_held_object_context(agent_id)
    if held_context is not None and put_obj and not _query_matches_target_context(put_obj, held_context):
        print(
            "[PutObject] failed | put_obj does not match held_object context. "
            f"put_obj={put_obj}, held={held_context.get('query')}"
        )
        return False

    # ------------------------------------------------------------------
    # 2) receptacle target은 마지막 navigation target에서 이어받는다
    # ------------------------------------------------------------------
    recp_target_context = _get_last_nav_target_context(agent_id)
    if recp_target_context is None:
        print(
            "[PutObject] failed | no last_nav_target context for receptacle. "
            "Call GoToObject/Explore for the receptacle first."
        )
        return False

    if not _query_matches_target_context(recp_regex, recp_target_context):
        print(
            "[PutObject] failed | receptacle query does not match last_nav_target. "
            f"recp={recp_regex}, last_target={recp_target_context.get('query')}"
        )
        return False

    # ------------------------------------------------------------------
    # 3) receptacle runtime object id 확보
    # ------------------------------------------------------------------
    runtime_match = None
    if recp_target_context.get("runtime_object_id"):
        runtime_match = {
            "runtime_object_id": str(recp_target_context.get("runtime_object_id")),
            "runtime_handle": str(recp_target_context.get("runtime_handle", "")),
            "runtime_object_type": str(recp_target_context.get("runtime_object_type", "")),
        }
    else:
        try:
            runtime_match = resolve_runtime_object_from_target_context(
                sim=sim,
                target_context=recp_target_context,
                agent_idx=agent_id,
                max_xz_dist=2.0,
                allow_nearest_fallback=False,
            )
            recp_target_context["runtime_object_id"] = runtime_match["runtime_object_id"]
            recp_target_context["runtime_handle"] = runtime_match["runtime_handle"]
            recp_target_context["runtime_object_type"] = runtime_match["runtime_object_type"]
            recp_target_context["runtime_position"] = runtime_match.get("runtime_position")
            recp_target_context["runtime_match_source"] = runtime_match.get("match_source")
            _set_last_nav_target_context(agent_id, recp_target_context, bot=bots[0])
        except Exception as e:
            print(f"[PutObject] receptacle runtime resolve failed ({recp_regex}): {e}")
            return False

    action_queue.append(
        {
            "action": "PutObject",
            "agent_id": agent_id,
            "put_obj_regex": str(put_obj),
            "recp_regex": recp_regex,
            "recp_object_id": str(runtime_match.get("runtime_object_id", "")),
            "recp_object_type": str(runtime_match.get("runtime_object_type", "")),
            "recp_handle": str(runtime_match.get("runtime_handle", "")),
            "held_object_context": None if held_context is None else dict(held_context),
            "receptacle_target_context": dict(recp_target_context),
        }
    )
    print(
        f"[PutObject] queued via memory context | "
        f"recp={recp_regex}, recp_id={runtime_match.get('runtime_object_id')}"
    )
    return True


# """
# 여기는 task 관련 함수
# """
def execute_tasks(robot):
    while True:
        task = task_queue.get()
        if task is None:
            break
        task_function = task['function']
        task_function(robot)
        task_queue.task_done()

def add_task_to_queue(task_function, robot):
    ensure_task_executor_started()
    task_queue.put({'function': task_function, 'robot': robot})

def parse_and_execute_task(robot):
    try:
        # Load the generated function name from the .json file
        with open('/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/logs/generated_function_name.json', 'r') as file:
            data = json.load(file)
            function_name = data["function_name"]
        
        # Dynamically import the module and reload it
        module = importlib.import_module('task_functions')
        importlib.reload(module)
        task_function = getattr(module, function_name)

        # Add the task to the queue
        add_task_to_queue(task_function, robot)
    except ImportError as e:
        print(f"No task function machine the description, error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def parse_task(robot, task_description):
    print(f"Executing task: {task_description}")
    return

def run_scripts():
    py = sys.executable if sys.executable else "python"
    query_cmd = [py, '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/scripts/query_with_short_term_memory.py']
    planner_cmd = [py, '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/scripts/llm_as_planner.py']
    print(f"[run_scripts] python={py}")

    query_ok = True
    try:
        subprocess.run(query_cmd, check=True)
    except subprocess.CalledProcessError as e:
        query_ok = False
        print(f"[run_scripts] warning: query_with_short_term_memory failed: {e}")

    planner_ok = True
    try:
        subprocess.run(planner_cmd, check=True)
    except subprocess.CalledProcessError as e:
        planner_ok = False
        print(f"[run_scripts] error: llm_as_planner failed: {e}")

    if planner_ok and query_ok:
        print("[run_scripts] query + planner completed")
    elif planner_ok:
        print("[run_scripts] planner completed (query step was skipped/failed)")
    else:
        print("[run_scripts] planner failed")

    return planner_ok

actions_thread = None
task_executor_thread = None


def ensure_task_executor_started():
    global task_executor_thread
    if task_executor_thread is not None and task_executor_thread.is_alive():
        return task_executor_thread

    task_executor_thread = threading.Thread(target=execute_tasks, args=(robots[0], ))
    task_executor_thread.start()
    return task_executor_thread


def shutdown_task_executor(timeout: float = None):
    global task_executor_thread
    if task_executor_thread is None:
        return

    if task_executor_thread.is_alive():
        task_queue.put(None)
        task_executor_thread.join(timeout=timeout)

    task_executor_thread = None


if START_TASK_EXECUTOR:
    ensure_task_executor_started()



# 아래는 debuging용 코드
# def _run_follower_step(
#     next_action: int,
#     follower: ShortestPathFollower,
#     nav_goal: np.ndarray,
#     forward_repeat: int,
#     turn_repeat: int,
#     rot_dir_sign: float,
#     max_turn_sweep_deg: float = 360.0,
# ):
#     if next_action == HabitatSimActions.move_forward:
#         _tuck_arm_for_nav()
#         _step_base(1.0, 0.0, repeat=forward_repeat)
#         return "move_forward"

#     if next_action in (HabitatSimActions.turn_left, HabitatSimActions.turn_right):
#         _tuck_arm_for_nav()
#         turn_cmd = +rot_dir_sign if next_action == HabitatSimActions.turn_left else -rot_dir_sign
#         action_name = "turn_left" if next_action == HabitatSimActions.turn_left else "turn_right"
#         max_turn_rad = float(np.deg2rad(max_turn_sweep_deg))

#         _, prev_yaw = get_fetch_base_pose(sim, 0)
#         swept_rad = 0.0
#         # follower 판단이 바뀔 때까지 같은 방향으로 회전시켜 좌/우 진동을 줄임
#         while swept_rad < max_turn_rad:
#             _step_base(0.0, turn_cmd, repeat=turn_repeat)
#             _, cur_yaw = get_fetch_base_pose(sim, 0)
#             swept_rad += abs(wrap_to_pi(cur_yaw - prev_yaw))
#             prev_yaw = cur_yaw

#             sync_follower_agent_state(sim, 0)
#             updated = follower.get_next_action(nav_goal)
#             if updated != next_action:
#                 break

#         return action_name

#     if next_action == HabitatSimActions.stop:
#         return "stop"
#     return f"unknown({next_action})"


# def debug_shortest_path_to_object(
#     dest_obj_regex: str,
#     objects_json_path: str = "/home/yuchaehee/long_term_memory_project/habitat-karma/memory/objects_locations1.json",
#     goal_radius: float = 0.60,
#     max_steps: int = 300,
#     forward_repeat: int = 2,
#     turn_repeat: int = 1,
#     realtime: bool = True,
#     max_turn_sweep_deg: float = 360.0,
# ):
#     raw_target = find_target_object_position(
#         sim,
#         dest_obj_regex,
#         objects_json_path=objects_json_path,
#     )
#     nav_goal, goal_source, geo_dist, obj_gap = project_goal_to_navmesh(sim, raw_target)
#     follower = ShortestPathFollower(sim, goal_radius=goal_radius, return_one_hot=False, stop_on_error=True)

#     out_dir = "/home/yuchaehee/long_term_memory_project/habitat-karma/logs/shortest_path_debug"
#     os.makedirs(out_dir, exist_ok=True)
#     safe = re.sub(r"[^0-9A-Za-z_-]+", "_", dest_obj_regex)

#     print(
#         f"[spf] target_regex={dest_obj_regex}, raw_target={raw_target.tolist()}, "
#         f"nav_goal={nav_goal.tolist()}, source={goal_source}, "
#         f"path_geodesic={'inf' if not np.isfinite(geo_dist) else f'{geo_dist:.3f}'}, "
#         f"obj_gap={obj_gap:.3f}m, goal_radius={goal_radius:.2f}"
#     )
#     global obs
#     obs = env.step({"action": EMPTY_ACTION, "action_args": {}})
#     sync_follower_agent_state(sim, 0)

#     success = False
#     rot_dir_sign = float(ROT_DIR_SIGN)
#     best_dist = float("inf")
#     window_name = "KARMA SPF Realtime (Q/ESC: stop)"
#     show_realtime = bool(realtime and ENABLE_INTERACTIVE_VIEW)
#     if show_realtime:
#         try:
#             cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#             cv2.resizeWindow(window_name, 1800, 900)
#         except cv2.error:
#             print("[spf] OpenCV window unavailable. Disable realtime visualization.")
#             show_realtime = False

#     for step_idx in range(max_steps):
#         cur_pos, _ = get_fetch_base_pose(sim, 0)
#         dist = distance_xz(nav_goal, cur_pos)
#         best_dist = min(best_dist, dist)

#         if dist <= goal_radius:
#             success = True
#             obj_dist = distance_xz(raw_target, cur_pos)
#             print(f"[spf] goal reached at step={step_idx}, nav_dist={dist:.3f}m, obj_dist={obj_dist:.3f}m")
#             break

#         sync_follower_agent_state(sim, 0)
#         next_action = follower.get_next_action(nav_goal)
#         action_name = _run_follower_step(
#             next_action=next_action,
#             follower=follower,
#             nav_goal=nav_goal,
#             forward_repeat=forward_repeat,
#             turn_repeat=turn_repeat,
#             rot_dir_sign=rot_dir_sign,
#             max_turn_sweep_deg=max_turn_sweep_deg,
#         )
#         cur_pos_after, _ = get_fetch_base_pose(sim, 0)
#         dist_after = distance_xz(nav_goal, cur_pos_after)
#         print(
#             f"[spf] step={step_idx:03d}, action={action_name}, nav_dist={dist_after:.3f}m, "
#             f"best={best_dist:.3f}m, rot_sign={rot_dir_sign:+.1f}"
#         )
#         best_dist = min(best_dist, dist_after)

#         if action_name in ("stop",) or action_name.startswith("unknown"):
#             print("[spf] follower returned stop/unknown action. end debug run.")
#             break

#         if show_realtime:
#             panel = _render_spf_realtime_frame(
#                 obs=obs,
#                 step_idx=step_idx,
#                 action_name=action_name,
#                 nav_dist=dist_after,
#                 best_dist=best_dist,
#                 goal_radius=goal_radius,
#                 success=False,
#             )
#             cv2.imshow(window_name, panel)
#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord("q"), 27):
#                 print("[spf] realtime window requested stop.")
#                 break

#         if step_idx % 10 == 0:
#             head_rgb = _obs_by_suffix(obs, "head_rgb")
#             save_rgb_observation_png(head_rgb, out_dir, f"{safe}_step_{step_idx:04d}.png")

#     final_pos, _ = get_fetch_base_pose(sim, 0)
#     final_nav_dist = distance_xz(nav_goal, final_pos)
#     final_obj_dist = distance_xz(raw_target, final_pos)

#     head_rgb = _obs_by_suffix(obs, "head_rgb")
#     save_rgb_observation_png(head_rgb, out_dir, f"{safe}_final.png")
#     if show_realtime:
#         try:
#             panel = _render_spf_realtime_frame(
#                 obs=obs,
#                 step_idx=step_idx if "step_idx" in locals() else -1,
#                 action_name="done",
#                 nav_dist=final_nav_dist,
#                 best_dist=best_dist,
#                 goal_radius=goal_radius,
#                 success=success,
#             )
#             cv2.imshow(window_name, panel)
#             cv2.waitKey(300)
#             cv2.destroyWindow(window_name)
#         except cv2.error:
#             pass
#     print(
#         f"[spf] done success={success}, final_nav_dist={final_nav_dist:.3f}m, "
#         f"final_obj_dist={final_obj_dist:.3f}m, "
#         f"max_steps={max_steps}, log_dir={out_dir}"
#     )


# def run_queue_navigation_to_object(dest_obj_regex: str, timeout_sec: float = 180.0):
#     """
#     KARMA-style queue 실행:
#     - exec_actions 스레드 실행
#     - GoToObject가 queue에 ObjectNavExpertAction enqueue
#     - queue가 빌 때까지 대기
#     """
#     global task_over
#     task_over = False
#     action_queue.clear()

#     actions_thread = threading.Thread(target=exec_actions, daemon=True)
#     actions_thread.start()

#     GoToObject(robots[0], dest_obj_regex)
#     print(f"[queue] started dest_obj={dest_obj_regex}, timeout={timeout_sec:.1f}s")

#     start_t = time.time()
#     timed_out = False
#     while True:
#         if not actions_thread.is_alive():
#             print("[queue] exec_actions thread stopped unexpectedly.")
#             break

#         if not action_queue:
#             # thread가 마지막 액션 처리 중일 수 있어 짧게 한번 더 대기
#             time.sleep(0.2)
#             if not action_queue:
#                 print("[queue] action_queue empty -> done")
#                 break

#         if (time.time() - start_t) > float(timeout_sec):
#             timed_out = True
#             print("[queue] timeout reached.")
#             break

#         time.sleep(0.05)

#     task_over = True
#     actions_thread.join(timeout=2.0)
#     if timed_out:
#         print("[queue] finished with timeout.")
#     else:
#         print("[queue] finished successfully.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Debug helpers for object navigation.")
#     parser.add_argument(
#         "dest_obj_pos",
#         nargs="?",
#         default=None,
#         help="Destination object name or regex (positional, e.g. Apple)",
#     )
#     parser.add_argument(
#         "--dest_obj",
#         dest="dest_obj_opt",
#         default=None,
#         help="Destination object name or regex (optional flag)",
#     )
#     parser.add_argument(
#         "--mode",
#         choices=["spf", "teleport", "queue"],
#         default="spf",
#         help="Debug mode: shortest-path-follower(spf), teleport, or queue-style execution",
#     )
#     parser.add_argument(
#         "--objects_json_path",
#         default="/home/yuchaehee/long_term_memory_project/habitat-karma/memory/objects_locations1.json",
#         help="Path to objects_locations JSON file",
#     )
#     parser.add_argument("--goal_radius", type=float, default=0.60, help="Goal radius for shortest path follower")
#     parser.add_argument("--max_steps", type=int, default=300, help="Maximum follower steps")
#     parser.add_argument("--forward_repeat", type=int, default=2, help="Base forward repeats per move_forward action")
#     parser.add_argument("--turn_repeat", type=int, default=1, help="Base turn repeats per turn action")
#     parser.add_argument("--max_turn_sweep_deg", type=float, default=360.0, help="Max turn sweep per follower turn action")
#     parser.add_argument("--no_realtime", action="store_true", help="Disable realtime SPF visualization window")
#     parser.add_argument("--stand_off", type=float, default=0.7, help="Teleport standoff distance")
#     parser.add_argument("--exact", action="store_true", help="Teleport exactly to object center")
#     parser.add_argument("--queue_timeout", type=float, default=180.0, help="Timeout seconds for queue mode")
#     args = parser.parse_args()
#     dest_obj = args.dest_obj_opt if args.dest_obj_opt is not None else (args.dest_obj_pos or "Apple")

#     if args.mode == "teleport":
#         obs = debug_teleport_to_object(
#             sim=sim,
#             env=env,
#             empty_action=EMPTY_ACTION,
#             obs=obs,
#             dest_obj_regex=dest_obj,
#             stand_off=args.stand_off,
#             exact=args.exact,
#             objects_json_path=args.objects_json_path,
#             obs_by_suffix_fn=_obs_by_suffix,
#             save_rgb_fn=save_rgb_observation_png,
#         )
#     elif args.mode == "spf":
#         debug_shortest_path_to_object(
#             dest_obj,
#             objects_json_path=args.objects_json_path,
#             goal_radius=args.goal_radius,
#             max_steps=args.max_steps,
#             forward_repeat=args.forward_repeat,
#             turn_repeat=args.turn_repeat,
#             realtime=(not args.no_realtime),
#             max_turn_sweep_deg=args.max_turn_sweep_deg,
#         )
#     else:
#         run_queue_navigation_to_object(
#             dest_obj_regex=dest_obj,
#             timeout_sec=args.queue_timeout,
#         )
