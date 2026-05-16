"""
SceneUpdate 엔트리포인트

Usage:
  python main.py --planner sayplan     # SayPlan (static SG, no REACT)
  python main.py --planner pred        # PRED (static SG, no REACT)
  python main.py --planner predreact   # PRED + REACT (real-time SG)
  python main.py --planner custom      # Custom + REACT + depth freespace

Options:
  --nav dwa|ros2    Navigator (default: dwa)
  --llm gemini|qwen|llama   LLM backend (default: gemini)
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from pxr import UsdGeom
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core import World

from config import USD_PATH, SG_JSON_FILE
from robot_utils import (find_robot_path, find_articulation_root,
                         get_prim_transform, quat_to_euler_z,
                         attach_fresh_lidar, setup_wheel_drives)
from arm_controller import FrankaArmController
from navigator import LocalNavigator
from llm_wrapper import create_llm
from sayplan_ui import SayPlanUI

import argparse


PLANNER_CHOICES = ["sayplan", "pred", "predreact", "custom"]

# Which planners use REACT
REACT_PLANNERS = {"predreact", "custom"}

# Which planners use depth freespace
FREESPACE_PLANNERS = {"custom"}


def main():
    parser = argparse.ArgumentParser(description="SceneUpdate Main")
    parser.add_argument("--planner", type=str, choices=PLANNER_CHOICES,
                        default="custom", help="Planner mode")
    parser.add_argument("--nav", type=str, choices=["dwa", "ros2"],
                        default="dwa", help="Navigator: dwa or ros2")
    parser.add_argument("--llm", type=str, choices=["gemini", "qwen", "llama"],
                        default="gemini", help="LLM backend")
    args = parser.parse_args()

    use_react = args.planner in REACT_PLANNERS
    use_freespace = args.planner in FREESPACE_PLANNERS

    print(f">>> Planner: {args.planner} | Nav: {args.nav} | LLM: {args.llm}")
    print(f">>> REACT: {'ON' if use_react else 'OFF'} | Freespace: {'ON' if use_freespace else 'OFF'}")

    # ── Extensions ──
    enable_extension("isaacsim.core.nodes")
    try: enable_extension("omni.isaac.sensor")
    except: pass
    try: enable_extension("omni.isaac.range_sensor")
    except: pass
    try: enable_extension("isaacsim.sensors.physx")
    except: pass
    if args.nav == "ros2":
        try: enable_extension("isaacsim.ros2.bridge")
        except: pass
        try: enable_extension("omni.isaac.ros2_bridge")
        except: pass
    for _ in range(60): simulation_app.update()

    # ── Stage ──
    from omni.isaac.core.utils.stage import open_stage
    try: open_stage(USD_PATH)
    except: pass

    stage = omni.usd.get_context().get_stage()
    robot_path = find_robot_path(stage)
    if not robot_path:
        print("Robot not found!")
        simulation_app.close()
        return

    base_link_path = f"{robot_path}/base_link"
    if not stage.GetPrimAtPath(base_link_path).IsValid():
        base_link_path = robot_path

    robot_pos, robot_quat = get_prim_transform(base_link_path)
    print(f">>> Robot: {robot_path} pos={robot_pos} yaw={np.degrees(quat_to_euler_z(robot_quat)):.1f}")

    # ── Arm ──
    arm_controller = FrankaArmController(None, robot_path)
    arm_controller.prepare_articulation()
    for _ in range(20): simulation_app.update()

    # ── World ──
    world = World(stage_units_in_meters=1.0)
    world.reset()
    for _ in range(60): world.step(render=True)

    # Disable existing lidar vis
    for prim in stage.Traverse():
        if prim.GetTypeName() in ("PhysicsLidar", "Lidar", "RangeSensor"):
            for attr_name in ["drawLines", "drawPoints"]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid(): attr.Set(False)
            imageable = UsdGeom.Imageable(prim)
            if imageable: imageable.MakeInvisible()

    lidar_path, lidar_frame = attach_fresh_lidar(robot_path, base_link_path, simulation_app)
    if not lidar_path:
        print("Failed to attach lidar!")
        simulation_app.close()
        return

    dc = _dynamic_control.acquire_dynamic_control_interface()

    # ── REACT (only for predreact, custom) ──
    react_bridge = None
    if use_react:
        from react_bridge import ReactWorkerBridge
        react_bridge = ReactWorkerBridge(SG_JSON_FILE)
        if not react_bridge.start_worker():
            print("WARNING: REACT Worker failed to start.")
            react_bridge = None

    # ── LLM ──
    llm = create_llm(args.llm)

    # ── Brain ──
    if args.planner == "custom":
        from custom_brain import CustomBrain
        brain = CustomBrain(SG_JSON_FILE, react_bridge=react_bridge, llm=llm)
    elif args.planner in ("pred", "predreact"):
        from pred_brain import PREDBrain
        brain = PREDBrain(SG_JSON_FILE, react_bridge=react_bridge, llm=llm)
    else:  # sayplan
        from sayplan_brain import SayPlanBrain
        brain = SayPlanBrain(SG_JSON_FILE, react_bridge=react_bridge, llm=llm)
    print(f">>> Brain: {type(brain).__name__}")

    # ── Play ──
    world.play()
    for _ in range(100): world.step(render=True)

    art, art_path = find_articulation_root(robot_path, dc)
    left_idxs, right_idxs = setup_wheel_drives(robot_path, dc, art)

    if not arm_controller.initialize():
        print("WARNING: Arm init failed.")

    # ── Navigator ──
    if args.nav == "ros2":
        from navigator_ros2 import ROS2Navigator
        navigator = ROS2Navigator(
            robot_path, base_link_path, dc, art,
            left_idxs, right_idxs, lidar_path,
            robot_start_pos=robot_pos,
            robot_start_yaw=quat_to_euler_z(robot_quat),
            lidar_frame=lidar_frame)
    else:
        navigator = LocalNavigator(
            robot_path, base_link_path, dc, art,
            left_idxs, right_idxs, lidar_path)

    # ── UI ──
    ui_app = SayPlanUI(brain, robot_path, base_link_path, navigator,
                       arm_controller, react_bridge=react_bridge,
                       enable_depth_freespace=use_freespace)

    try:
        while simulation_app.is_running():
            world.step(render=True)
            if not world.is_playing(): continue
            try:
                dt = world.get_physics_dt()
                ui_app.update(dt)
            except Exception as e:
                import traceback
                print(f"\n{'='*60}\n[Error] {e}\n{traceback.format_exc()}{'='*60}\n")
                ui_app.log_chat("Error occurred! Check terminal.")
                ui_app.state = "IDLE"
                navigator.cancel()
    finally:
        if react_bridge: react_bridge.shutdown()
        ui_app.cleanup()
        simulation_app.close()


if __name__ == "__main__":
    main()
