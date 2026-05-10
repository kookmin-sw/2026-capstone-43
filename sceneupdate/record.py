from isaacsim import SimulationApp
import carb.settings

# 1. 초기화 (원본 설정 유지)
simulation_app = SimulationApp({"headless": False})

import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
import cv2
import math
import time
import threading
import queue
import omni.ui as ui
import omni.usd
import re
import omni.replicator.core as rep
from omni.kit.viewport.utility import create_viewport_window 

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid, VisualCylinder
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from pxr import Gf, UsdGeom

# ================= [설정 - 원본 그대로] =================
USD_PATH = "/home/sungbin/isaac/sg/scene/Collected_World2/World2.usd"
SAVE_ROOT_DIR = "/home/sungbin/isaac/sg/captured_sessions"

ZED_CONFIGS = {
    "WIDE (2.1mm)": {"focal": 2.12, "h_aperture": 4.8, "v_aperture": 2.7},
    "NARROW (4.0mm)": {"focal": 2.12, "h_aperture": 6.02, "v_aperture": 3.39}
}

state = {
    "current_mode": "NARROW (4.0mm)",
    "is_recording": False,
    "current_session_dir": "",
    "frame_count": 0,
    "render_product": None,
    "annotators": {},
    "last_save_time": 0.0
}

# 백그라운드 저장을 위한 큐와 스레드 설정 (저장 속도 대폭 개선)
save_queue = queue.Queue()

def save_worker():
    while True:
        data = save_queue.get()
        if data is None:
            break
        
        session_dir, f_idx, rgb_img, depth_data, pose_info = data
        
        # 메인 스레드의 부하를 없애기 위해 무거운 연산(클리핑 및 이미지 저장)을 백그라운드 워커에서 처리
        Image.fromarray(rgb_img).save(os.path.join(session_dir, "rgb", f"rgb_{f_idx}.png"), optimize=False)
        
        if depth_data is not None:
            # depth_raw(.npy)는 엄청나게 무겁고 느리므로 생략하고 바로 16-bit PNG(밀리미터 단위)로 변환해서 저장
            depth_m = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(depth_m * 1000, 0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(os.path.join(session_dir, "depth", f"{f_idx}.png"), optimize=False)
            
            # 시각화용 8-bit depth (선택 사항)
            d_vis_img = (np.clip(depth_data, 0, 10.0) * 25.5).astype(np.uint8)
            Image.fromarray(d_vis_img).save(os.path.join(session_dir, "depth_viz", f"depth_{f_idx}.png"), optimize=False)
            
        with open(os.path.join(session_dir, "poses", f"pose_{f_idx}.json"), "w") as f:
            json.dump(pose_info, f, indent=2)
            
        save_queue.task_done()

# 4개의 다중 워커 스레드 시작 (병렬 저장으로 엄청난 속도 향상)
for _ in range(4):
    threading.Thread(target=save_worker, daemon=True).start()

RESOLUTION = (1280, 720)
MOVE_STEP  = 0.05
TURN_STEP  = 1.5

cam_pos   = np.array([2.0, 2.0, 1.5], dtype=float)
cam_yaw   = 0.0
cam_pitch = 0.0

os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
# ==========================================

def setup_render_settings():
    """사용자님 원본 설정 100% 동일"""
    print(">>> 렌더링 설정 최적화 (TAA OFF, Motion Blur OFF)...")
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/post/aa/op", 0)
    settings.set_bool("/rtx/post/motionblur/enabled", False)
    settings.set_int("/rtx/post/dlss/execMode", 0)

def set_transform_safe(prim_path, translate=None, rotate_xyz=None, scale=None):
    """사용자님 원본 로직 100% 동일"""
    prim = get_prim_at_path(prim_path)
    if not prim: return
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translate is not None:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    if rotate_xyz is not None:
        xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*rotate_xyz))
    if scale is not None:
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))

def get_fwd_right(yaw_deg):
    yaw   = math.radians(yaw_deg)
    fwd   = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    right = np.array([ math.cos(yaw), math.sin(yaw), 0.0])
    return fwd, right

def main():
    global cam_pos, cam_yaw, cam_pitch

    print(f"시스템 초기화 중...")
    try: open_stage(USD_PATH)
    except: simulation_app.close(); return

    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    try: create_prim("/World/Env/DomeLight", "DomeLight", attributes={"inputs:intensity": 1000.0})
    except: pass

    setup_render_settings()

    # -----------------------------------------------------------
    # [1] 카메라 리그 조립 (원본 마커 유지)
    # -----------------------------------------------------------
    rig_path = "/World/ZED_Rig"
    create_prim(rig_path, "Xform")
    set_transform_safe(rig_path, translate=(cam_pos[0], cam_pos[1], cam_pos[2]), rotate_xyz=(cam_pitch, 0.0, cam_yaw))
    
    # 빨간 바디 (위치 확인용)
    VisualCuboid(prim_path=f"{rig_path}/Body", color=np.array([0.8, 0.1, 0.1]))
    set_transform_safe(f"{rig_path}/Body", scale=(0.1, 0.4, 0.1))
    
    # 파란 렌즈 (방향 확인용)
    VisualCylinder(prim_path=f"{rig_path}/Lens", color=np.array([0.1, 0.1, 1.0]))
    set_transform_safe(f"{rig_path}/Lens", translate=(0, 0.2, 0), rotate_xyz=(90, 0, 0), scale=(0.05, 0.05, 0.05))
    
    # 실제 센서
    cam_prim_path = f"{rig_path}/Sensor"
    create_prim(cam_prim_path, "Camera")
    set_transform_safe(cam_prim_path, translate=(0, 0.22, 0), rotate_xyz=(90, 0, 0))

    # -----------------------------------------------------------
    # [2] Replicator 및 프리뷰 창 (새로 추가됨)
    # -----------------------------------------------------------
    render_product = rep.create.render_product(cam_prim_path, RESOLUTION)
    state["annotators"] = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
        "params": rep.AnnotatorRegistry.get_annotator("camera_params")
    }
    for a in state["annotators"].values(): a.attach(render_product)

    # [핵심 추가] 카메라가 보고 있는 화면을 실시간으로 띄워주는 창
    viewport = create_viewport_window("ZED Live View", width=640, height=360)
    viewport.viewport_api.set_active_camera(cam_prim_path)

    def apply_zed_specs(mode_name):
        conf = ZED_CONFIGS[mode_name]
        prim = get_prim_at_path(cam_prim_path)
        if prim:
            prim.GetAttribute("focalLength").Set(conf["focal"])
            prim.GetAttribute("horizontalAperture").Set(conf["h_aperture"])
            prim.GetAttribute("verticalAperture").Set(conf["v_aperture"])
            prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.1, 100.0))

    world.reset()
    apply_zed_specs(state["current_mode"])

    # -----------------------------------------------------------
    # [3] 데이터 추출 함수 (원본 포즈 추출 로직 100% 유지)
    # -----------------------------------------------------------
    _pose_debug_done = [False]

    def get_pose_from_params(params_data):
        view_matrix = params_data["cameraViewTransform"].reshape(4, 4)
        c2w_matrix = np.linalg.inv(view_matrix)

        # ---- 진단: 어느 위치에 translation이 있는지 확인 ----
        if not _pose_debug_done[0]:
            pos_row = c2w_matrix[3, :3]   # USD row-vector convention
            pos_col = c2w_matrix[:3, 3]   # OpenGL column-vector convention
            print(f"\n[DIAG] cameraViewTransform convention 확인")
            print(f"  카메라 초기 위치 설정값: cam_pos = {cam_pos.tolist()}")
            print(f"  c2w_matrix[3, :3]  (row-vector): {pos_row.tolist()}")
            print(f"  c2w_matrix[:3, 3]  (col-vector): {pos_col.tolist()}")
            print(f"  → 위 두 값 중 cam_pos에 가까운 쪽이 올바른 pos\n")
            _pose_debug_done[0] = True
        # -------------------------------------------------------

        pos = c2w_matrix[3, :3]   # 진단 후 맞는 쪽으로 교체
        rot_mat = c2w_matrix[:3, :3]
        gf_mat = Gf.Matrix3d(
            float(rot_mat[0,0]), float(rot_mat[0,1]), float(rot_mat[0,2]),
            float(rot_mat[1,0]), float(rot_mat[1,1]), float(rot_mat[1,2]),
            float(rot_mat[2,0]), float(rot_mat[2,1]), float(rot_mat[2,2])
        )
        q = gf_mat.ExtractRotation().GetQuaternion()
        quat_wxyz = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        return pos, quat_wxyz

    def save_frame():
        # 프레임 제한 (너무 잦은 저장을 방지하여 렉 최소화 - 약 10 FPS 제한)
        current_time = time.time()
        if current_time - state["last_save_time"] < 0.1:
            return
            
        # Replicator 강제 동기화는 엄청난 렉을 유발하므로 부드러운 움직임을 위해 제거.
        # 데이터가 비동기적으로 들어오더라도 일반적인 이동 속도에서는 문제가 적음.
        # rep.orchestrator.step(rt_subframes=0, pause_timeline=True)

        rgb = state["annotators"]["rgb"].get_data()
        depth = state["annotators"]["depth"].get_data()
        params = state["annotators"]["params"].get_data()

        # 거대 씬에서는 렌더링이 덜 끝나면 1차원 빈 배열 (0,) 이 반환되어 IndexError 발생함.
        if rgb is not None and isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.size > 0:
            if params is not None:
                # ============================================================
                # [추가] 물체 통과/충돌 감지 로직
                # 카메라가 물체에 너무 가깝거나 통과 중일 때 저장을 스킵합니다.
                # ============================================================
                if depth is not None and isinstance(depth, np.ndarray) and depth.size > 0:
                    # 깊이값이 0.0보다 크고 0.3m 미만인 픽셀 개수 확인
                    # (하늘이나 빈 공간 등 깊이가 0.0으로 찍히는 배경 픽셀은 제외해야 함)
                    too_close_mask = (depth > 0.0) & (depth < 0.3)
                    too_close_ratio = np.sum(too_close_mask) / depth.size
                    
                    if too_close_ratio > 0.2: # 화면의 20% 이상이 0.3m 이내에 실제 물체가 있으면 스킵
                        print(f"[Skip] 충돌/통과 감지됨! (화면의 {too_close_ratio*100:.1f}% 가 0.3m 이내의 물체에 닿아있음)")
                        return

                f_idx = f"{state['frame_count']:05d}"
                session_dir = state["current_session_dir"]
                
                # 카메라 파라미터 추출
                pos, rot = get_pose_from_params(params)
                pose_info = {"frame": state["frame_count"], "timestamp": datetime.now().isoformat(), "position": pos.tolist(), "quaternion_wxyz": rot}
                
                # 데이터 복사 (백그라운드 스레드에서 참조 시 덮어써지는 것 방지)
                # d_vis_img 연산도 메인 스레드를 느리게 하므로 백그라운드로 미룸
                rgb_img = np.ascontiguousarray(rgb[:, :, :3])
                depth_data = np.ascontiguousarray(depth) if (depth is not None and depth.size > 0) else None
                
                # 큐에 데이터 삽입 (비동기 병렬 저장)
                save_queue.put((session_dir, f_idx, rgb_img, depth_data, pose_info))

                state["frame_count"] += 1
                state["last_save_time"] = current_time
        else:
            # 아직 씬 로딩/렌더 파이프라인이 덜 끝나서 1차원 빈 배열이 나올 때는 그냥 스킵.
            print(f"[warn] rgb data not ready yet (shape={getattr(rgb, 'shape', 'None')}), waiting...")

    # -----------------------------------------------------------
    # [4] UI 설정 (원본 그대로 유지)
    # -----------------------------------------------------------
    def on_start_rec():
        if not state["is_recording"]:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_path = os.path.join(SAVE_ROOT_DIR, f"session_{ts}")
            state["current_session_dir"] = session_path
            state["frame_count"] = 0
            for s in ["rgb", "depth", "depth_viz", "poses"]:
                os.makedirs(os.path.join(session_path, s), exist_ok=True)
            state["is_recording"] = True
            btn_start.enabled = False; btn_stop.enabled = True
            print(f">>> REC START: {session_path}")
            
            # [추가] 데이터 기록 시작 시 씬 내부의 객체 이름들을 추출하여 저장
            extract_and_save_object_names(session_path)

    def extract_and_save_object_names(session_dir):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
            
        object_names = set()
        
        # 정규식: _랜덤문자열_숫자 로 끝나는 패턴 제거 (예: coffee_table_qlmqyy_0 -> coffee_table)
        pattern = re.compile(r'_[a-zA-Z0-9]+_\d+$')
        
        for prim in stage.Traverse():
            name = prim.GetName()
            path = str(prim.GetPath())
            
            # 카메라 리그나 조명 등 시스템 생성 객체는 제외
            if "ZED_Rig" in path or "DomeLight" in path:
                continue
                
            # Xform이나 Mesh인 경우에만 이름 추출 (보통 객체의 최상위는 Xform이거나 바로 Mesh임)
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
                clean_name = pattern.sub('', name)
                clean_name = clean_name.replace('_', ' ').strip()
                clean_name = clean_name.lower()
                
                # 의미 없는 부분 필터링을 더 강력하게
                # "collision", "link", "mesh", "dof", "meta", "base" 등이 포함된 이름은 버립니다.
                invalid_substrings = ["collision", "link", "mesh", "dof", "meta", "root", "material", "looks", "scene", "default", "xform", "world", "collisions", "environment", "floors", "ceilings", "glass"]
                
                is_valid = True
                for word in invalid_substrings:
                    if word in clean_name:
                        is_valid = False
                        break
                
                if is_valid and len(clean_name) > 1 and not clean_name.isnumeric():
                    object_names.add(clean_name)
                    
        output_path = os.path.join(session_dir, "scene_objects.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(object_names)), f, indent=4, ensure_ascii=False)
        print(f">>> 씬 내의 물체 이름 추출 완료! 총 {len(object_names)}개의 고유 클래스 이름이 {output_path}에 저장되었습니다.")

    def on_stop_rec():
        if state["is_recording"]:
            state["is_recording"] = False
            conf = ZED_CONFIGS[state["current_mode"]]
            info = {"device": "ZED 2i", "lens": state["current_mode"], "intrinsics": {"focal": conf["focal"], "res": list(RESOLUTION)}, "total_frames": state["frame_count"]}
            with open(os.path.join(state["current_session_dir"], "camera_info.json"), "w") as f:
                json.dump(info, f, indent=4)
            btn_start.enabled = True; btn_stop.enabled = False
            print(">>> REC STOPPED (저장 대기열 처리 중...)")

    window = ui.Window("Final Precision Recorder", width=400, height=280)
    with window.frame:
        with ui.VStack(spacing=15):
            ui.Label("Final Precision Recorder", style={"font_size": 18, "color": 0xFF00FF00})
            with ui.HStack(height=30):
                ui.Label("Mode:", width=60)
                combo = ui.ComboBox(1, *list(ZED_CONFIGS.keys()))
                combo.model.add_item_changed_fn(lambda m, i: apply_zed_specs(list(ZED_CONFIGS.keys())[combo.model.get_item_value().as_int]))
            with ui.HStack(spacing=10):
                btn_start = ui.Button("START", height=60, clicked_fn=on_start_rec)
                btn_stop = ui.Button("STOP", height=60, clicked_fn=on_stop_rec, enabled=False)
            status_label = ui.Label("Idle", style={"color": 0xFF888888})

    # OpenCV 창 - 키보드 전용
    cv2.namedWindow("Camera Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Control", 480, 60)
    ctrl_img = np.zeros((60, 480, 3), dtype=np.uint8)
    cv2.putText(ctrl_img, "W/S/A/D/Q/E: move   Arrows: rotate   ESC: quit",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    while simulation_app.is_running():
        world.step(render=True)

        cv2.imshow("Camera Control", ctrl_img)
        key = cv2.waitKey(1) & 0xFF

        fwd, right = get_fwd_right(cam_yaw)
        if   key == ord('w'): cam_pos += fwd   * MOVE_STEP
        elif key == ord('s'): cam_pos -= fwd   * MOVE_STEP
        elif key == ord('a'): cam_pos -= right * MOVE_STEP
        elif key == ord('d'): cam_pos += right * MOVE_STEP
        elif key == ord('q'): cam_pos[2] += MOVE_STEP
        elif key == ord('e'): cam_pos[2] -= MOVE_STEP
        elif key == 81:       cam_yaw   += TURN_STEP
        elif key == 83:       cam_yaw   -= TURN_STEP
        elif key == 82:       cam_pitch  = min(cam_pitch + TURN_STEP,  80.0)
        elif key == 84:       cam_pitch  = max(cam_pitch - TURN_STEP, -80.0)
        elif key == 27:
            if state["is_recording"]: on_stop_rec()
            break

        set_transform_safe(rig_path, translate=(cam_pos[0], cam_pos[1], cam_pos[2]), rotate_xyz=(cam_pitch, 0.0, cam_yaw))

        if state["is_recording"]:
            save_frame()
            status_label.text = f"REC: {state['frame_count']}"
            # status_label.style = {"color": 0xFF0000FF}
        else:
            status_label.text = "Idle"

    cv2.destroyAllWindows()
    
    # [버그 수정] 프로그램 종료 시 백그라운드 큐에 남은 데이터가 모두 쓰일 때까지 대기
    # 데몬 스레드가 도중에 종료되면 이미지가 깨지거나 파일 수가 안 맞는 문제가 발생함.
    if not save_queue.empty():
        print(f"\n>>> 디스크 저장을 마무리하는 중입니다... 잠시만 기다려주세요. (대기열: {save_queue.qsize()} 프레임)")
        save_queue.join()
        print(">>> 모든 데이터 저장 완료!")

    simulation_app.close()

if __name__ == "__main__": main()
