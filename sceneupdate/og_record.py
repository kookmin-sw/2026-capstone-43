from isaacsim import SimulationApp
import carb.settings

simulation_app = SimulationApp({"headless": False})

import os
import json
import math
import numpy as np
from PIL import Image
from datetime import datetime
import omni.ui as ui
import omni.replicator.core as rep
import carb
import carb.input
import omni.appwindow
from omni.kit.viewport.utility import create_viewport_window

from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid, VisualCylinder
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from pxr import Gf, UsdGeom

# ================= [설정] =================
USD_PATH      = "/home/sungbin/extra/omnigibscene0/beechwood_flat.usd"
SAVE_ROOT_DIR = "/home/sungbin/isaac/sg/captured_sessions"

ZED_CONFIGS = {
    "WIDE (2.1mm)":   {"focal": 2.12, "h_aperture": 4.8,  "v_aperture": 2.7},
    "NARROW (4.0mm)": {"focal": 2.12, "h_aperture": 6.02, "v_aperture": 3.39}
}

RESOLUTION       = (1280, 720)
MOVE_STEP        = 0.05   # m/frame
TURN_STEP        = 1.5    # deg/frame
COLLISION_RADIUS = 0.12   # m

state = {
    "current_mode":     "NARROW (4.0mm)",
    "is_recording":     False,
    "current_session_dir": "",
    "frame_count":      0,
    "render_product":   None,
    "annotators":       {}
}

# 카메라 위치/회전 상태
cam_pos   = np.array([0.0, 0.0, 1.5], dtype=float)
cam_yaw   = 0.0
cam_pitch = 0.0

key_events = {"r": False, "space": False}

os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
# ==========================================

def setup_render_settings():
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/post/aa/op", 0)
    settings.set_bool("/rtx/post/motionblur/enabled", False)
    settings.set_int("/rtx/post/dlss/execMode", 0)

def set_transform_safe(prim_path, translate=None, rotate_xyz=None, scale=None):
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

def apply_cam_transform(rig_path):
    set_transform_safe(rig_path,
        translate=(cam_pos[0], cam_pos[1], cam_pos[2]),
        rotate_xyz=(cam_pitch, 0.0, cam_yaw))

def get_fwd_right(yaw_deg):
    yaw   = math.radians(yaw_deg)
    fwd   = np.array([ math.cos(yaw),  math.sin(yaw), 0.0])
    right = np.array([ math.sin(yaw), -math.cos(yaw), 0.0])
    return fwd, right

def is_inside_geometry():
    # PhysX overlap_sphere segfaults on static (non-physics) scenes.
    # Use a simple floor/ceiling sanity check instead.
    return cam_pos[2] < 0.2 or cam_pos[2] > 4.0

def main():
    global cam_pos, cam_yaw, cam_pitch

    print(f"시스템 초기화 중...")
    try: open_stage(USD_PATH)
    except: simulation_app.close(); return

    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    try: create_prim("/World/Env/DomeLight", "DomeLight", attributes={"inputs:intensity": 1000.0})
    except: pass

    setup_render_settings()

    # 카메라 리그
    rig_path = "/World/ZED_Rig"
    create_prim(rig_path, "Xform")

    VisualCuboid(prim_path=f"{rig_path}/Body", color=np.array([0.8, 0.1, 0.1]))
    set_transform_safe(f"{rig_path}/Body", scale=(0.1, 0.4, 0.1))

    VisualCylinder(prim_path=f"{rig_path}/Lens", color=np.array([0.1, 0.1, 1.0]))
    set_transform_safe(f"{rig_path}/Lens", translate=(0, 0.2, 0), rotate_xyz=(90, 0, 0), scale=(0.05, 0.05, 0.05))

    cam_prim_path = f"{rig_path}/Sensor"
    create_prim(cam_prim_path, "Camera")
    set_transform_safe(cam_prim_path, translate=(0, 0.22, 0), rotate_xyz=(90, 0, 0))

    try:
        import omni.usd as _ousd
        _stage = _ousd.get_context().get_stage()
        for _path in ["/Render/PostProcess/SDGPipeline", "/Render"]:
            if _stage.GetPrimAtPath(_path).IsValid():
                _stage.RemovePrim(_path)
                break
    except Exception as _e:
        print(f"[warn] SDGPipeline 제거 실패 (무시): {_e}")

    render_product = rep.create.render_product(cam_prim_path, RESOLUTION)
    state["annotators"] = {
        "rgb":    rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth":  rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
        "params": rep.AnnotatorRegistry.get_annotator("camera_params")
    }
    for a in state["annotators"].values(): a.attach(render_product)

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
    apply_cam_transform(rig_path)

    # 키보드
    input_iface = carb.input.acquire_input_interface()
    appwindow   = omni.appwindow.get_default_app_window()
    keyboard    = appwindow.get_keyboard()
    K           = carb.input.KeyboardInput

    def on_key_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == K.R:     key_events["r"]     = True
            if event.input == K.SPACE: key_events["space"] = True
        return True
    input_iface.subscribe_to_keyboard_events(keyboard, on_key_event)

    def held(k):
        return input_iface.get_keyboard_value(keyboard, k) > 0.5

    def get_pose_from_params(params_data):
        view_matrix = params_data["cameraViewTransform"].reshape(4, 4)
        c2w_matrix  = np.linalg.inv(view_matrix)
        pos         = c2w_matrix[3, :3]
        rot_mat     = c2w_matrix[:3, :3]
        gf_mat = Gf.Matrix3d(
            float(rot_mat[0,0]), float(rot_mat[0,1]), float(rot_mat[0,2]),
            float(rot_mat[1,0]), float(rot_mat[1,1]), float(rot_mat[1,2]),
            float(rot_mat[2,0]), float(rot_mat[2,1]), float(rot_mat[2,2]))
        q = gf_mat.ExtractRotation().GetQuaternion()
        quat_wxyz = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        return pos, quat_wxyz

    def save_frame():
        if is_inside_geometry():
            print(f"  [SKIP] 물체 내부 — frame {state['frame_count']}")
            return

        rgb    = state["annotators"]["rgb"].get_data()
        depth  = state["annotators"]["depth"].get_data()
        params = state["annotators"]["params"].get_data()

        if rgb is not None and params is not None:
            f_idx       = f"{state['frame_count']:05d}"
            session_dir = state["current_session_dir"]

            Image.fromarray(rgb[:, :, :3]).save(
                os.path.join(session_dir, "rgb", f"rgb_{f_idx}.png"))
            np.save(os.path.join(session_dir, "depth_raw", f"depth_{f_idx}.npy"), depth)
            d_vis = np.clip(depth, 0, 10.0) / 10.0
            Image.fromarray((d_vis * 255).astype(np.uint8)).save(
                os.path.join(session_dir, "depth_viz", f"depth_{f_idx}.png"))

            pos, rot = get_pose_from_params(params)
            pose_info = {"frame": state["frame_count"], "timestamp": datetime.now().isoformat(),
                         "position": pos.tolist(), "quaternion_wxyz": rot}
            with open(os.path.join(session_dir, "poses", f"pose_{f_idx}.json"), "w") as f:
                json.dump(pose_info, f, indent=2)

            state["frame_count"] += 1

    # UI
    def on_start_rec():
        if not state["is_recording"]:
            ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_path = os.path.join(SAVE_ROOT_DIR, f"session_{ts}")
            state["current_session_dir"] = session_path
            state["frame_count"]         = 0
            for s in ["rgb", "depth_raw", "depth_viz", "poses"]:
                os.makedirs(os.path.join(session_path, s), exist_ok=True)
            state["is_recording"] = True
            btn_start.enabled = False; btn_stop.enabled = True
            print(f">>> REC START: {session_path}")

    def on_stop_rec():
        if state["is_recording"]:
            state["is_recording"] = False
            conf = ZED_CONFIGS[state["current_mode"]]
            info = {"device": "ZED 2i", "lens": state["current_mode"],
                    "intrinsics": {"focal": conf["focal"], "res": list(RESOLUTION)},
                    "total_frames": state["frame_count"]}
            with open(os.path.join(state["current_session_dir"], "camera_info.json"), "w") as f:
                json.dump(info, f, indent=4)
            btn_start.enabled = True; btn_stop.enabled = False
            print(">>> REC STOPPED")

    window = ui.Window("Recorder", width=400, height=300)
    with window.frame:
        with ui.VStack(spacing=10):
            ui.Label("Scene Recorder", style={"font_size": 18, "color": 0xFF00FF00})
            ui.Label("W/S/A/D/Q/E: 이동  |  ←→↑↓: 회전  |  R: 녹화  |  SPACE: 단일캡처",
                     style={"font_size": 11})
            with ui.HStack(height=30):
                ui.Label("Mode:", width=60)
                combo = ui.ComboBox(1, *list(ZED_CONFIGS.keys()))
                combo.model.add_item_changed_fn(
                    lambda m, i: apply_zed_specs(
                        list(ZED_CONFIGS.keys())[combo.model.get_item_value().as_int]))
            with ui.HStack(spacing=10):
                btn_start = ui.Button("START", height=60, clicked_fn=on_start_rec)
                btn_stop  = ui.Button("STOP",  height=60, clicked_fn=on_stop_rec, enabled=False)
            status_label = ui.Label("Idle", style={"color": 0xFF888888})

    print("\n=== Recorder 준비 완료 ===")
    print("W/S/A/D: 이동  Q/E: 높이  ←→: 좌우회전  ↑↓: 시점")
    print("R: 녹화 토글  SPACE: 단일 캡처\n")

    while simulation_app.is_running():
        world.step(render=True)

        # 카메라 이동
        fwd, right = get_fwd_right(cam_yaw)
        if held(K.W):     cam_pos += fwd   * MOVE_STEP
        if held(K.S):     cam_pos -= fwd   * MOVE_STEP
        if held(K.A):     cam_pos -= right * MOVE_STEP
        if held(K.D):     cam_pos += right * MOVE_STEP
        if held(K.Q):     cam_pos[2] += MOVE_STEP
        if held(K.E):     cam_pos[2] -= MOVE_STEP
        if held(K.LEFT):  cam_yaw   += TURN_STEP
        if held(K.RIGHT): cam_yaw   -= TURN_STEP
        if held(K.UP):    cam_pitch  = min(cam_pitch + TURN_STEP,  80.0)
        if held(K.DOWN):  cam_pitch  = max(cam_pitch - TURN_STEP, -80.0)
        apply_cam_transform(rig_path)

        # R 토글
        if key_events["r"]:
            key_events["r"] = False
            if state["is_recording"]: on_stop_rec()
            else: on_start_rec()

        # SPACE 단일 캡처
        if key_events["space"]:
            key_events["space"] = False
            was = state["is_recording"]
            if not was: on_start_rec()
            save_frame()
            if not was: on_stop_rec()

        # 녹화
        if state["is_recording"]:
            save_frame()
            in_col = is_inside_geometry()
            status_label.text  = f"⚠ 충돌(스킵) | {state['frame_count']}" if in_col else f"REC: {state['frame_count']}"
            status_label.style = {"color": 0xFF0088FF if in_col else 0xFF0000FF}
        else:
            status_label.text  = "Idle"
            status_label.style = {"color": 0xFF888888}

    simulation_app.close()

if __name__ == "__main__": main()
