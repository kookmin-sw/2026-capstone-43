"""
SayPlanUI - Omni UI for SayPlan + REACT mobile manipulator
"""
import gc
import os
import time
import json
import threading
import numpy as np
import omni.ui as ui
import omni.replicator.core as rep
from pxr import Gf
from datetime import datetime
from omni.kit.viewport.utility import create_viewport_window
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.prims import create_prim

from robot_utils import get_prim_transform, quat_to_euler_z, find_scene_prim_near
from config import (CAMERA_FOCAL, CAMERA_H_APERTURE, CAMERA_V_APERTURE,
                     CAMERA_RESOLUTION, REACT_UPDATE_INTERVAL, SURFACE_JSON_FILE)
from pred_brain import PREDBrain
from custom_brain import CustomBrain


class SayPlanUI:
    def __init__(self, brain, robot_path, base_link_path, navigator,
                 arm_controller, react_bridge=None,
                 enable_depth_freespace=False):
        self.brain = brain
        self.robot_path = robot_path
        self.base_link_path = base_link_path
        self.navigator = navigator
        self.arm = arm_controller
        self.react_bridge = react_bridge

        # Depth-based free space (precomputed + live)
        self._depth_fs_enabled = enable_depth_freespace
        self._depth_fs_analyzer = None
        self._last_depth_fs_time = 0
        self._depth_fs_interval = 1.5
        self._depth_fs_frame_count = 0
        self._depth_fs_cached_rviz = []
        if enable_depth_freespace:
            from depth_freespace import DepthFreeSpaceAnalyzer
            self._depth_fs_analyzer = DepthFreeSpaceAnalyzer(
                debug=False, stride=6)
            self._depth_fs_analyzer.load_precomputed()
            self._seed_precomputed_depth_grids()

        self._planner_name = "Custom" if isinstance(brain, CustomBrain) else ("PRED" if isinstance(brain, PREDBrain) else "SayPlan")
        self._nav_name = "Nav2" if type(navigator).__name__ == "ROS2Navigator" else "DWA"
        self._title = f"{self._planner_name} + {self._nav_name} + REACT"
        self.window = ui.Window(self._title, width=1100, height=700)
        self.instruction_model = ui.SimpleStringModel("")
        self.chat_log_model = ui.SimpleStringModel("")
        self.interaction_log_model = ui.SimpleStringModel("")
        self.react_log_model = ui.SimpleStringModel("")

        self.plan_queue = []
        self.current_action = None
        self.state = "IDLE"
        self.current_instruction = ""
        self.feedback = ""
        self.nav_target_name = None
        self._current_arm_target_id = None
        self._current_arm_action = None
        self._nav_tracking_id = None
        self._picked_this_plan = set()  # IDs already picked in current plan (avoid re-picking)
        self._last_nav_refresh = 0
        self._pending_replan_reason = None
        self._last_relevance_check = 0
        self._replan_cooldown = 0
        self._last_sg_viz_export = 0
        self._last_gc_time = 0

        # Pre-place observation: robot stops for 1.5s before placing to get fresh freespace
        self._observe_start_time = 0
        self._observe_duration = 1.5  # seconds
        self._observe_place_target_id = None
        self._observe_place_target_name = None

        # Pending new surface observation queue
        # Each entry: {"id": obj_id, "tag": str, "center": [x,y,z], "extent": [w,d,h]}
        self._pending_surface_queue = []
        self._known_surface_ids = set()  # already-observed or precomputed surface IDs
        self._obs_surface_current = None  # currently observing surface info
        self._obs_surface_viewpoints = []  # list of (x, y, yaw) viewpoints
        self._obs_surface_vp_idx = 0  # which viewpoint we're on
        self._obs_surface_observe_start = 0  # when observation at current viewpoint started
        self._obs_surface_observe_duration = 2.0  # seconds per viewpoint
        self._obs_surface_nav_start = 0  # when nav to viewpoint started
        self._obs_surface_nav_timeout = 15.0  # max seconds to reach a viewpoint
        self._obs_surface_nav_failures = 0  # consecutive nav failures for current surface
        self._obs_surface_max_failures = 2  # after this many failures, skip remaining viewpoints

        # Disable automatic GC — collect manually during safe moments
        # Python GC pauses can freeze Isaac Sim's physics loop for 50-200ms
        gc.disable()

        # [FIX] 마커 크기 축소 + 흰색으로 변경하여 눈에 안띄게
        self.marker = VisualSphere("/World/DebugTarget", radius=0.03,
                                   color=np.array([1, 1, 1]), visible=False)

        # Camera on base_link (1.5m up, tilted 20 degrees down)
        self.camera_path = f"{base_link_path}/HeadCamera"
        create_prim(
            self.camera_path, "Camera", translation=(0.0, 0.0, 1.5),
            orientation=np.array([0.5792, 0.4056, -0.4056, -0.5792]),
            attributes={
                "focalLength": CAMERA_FOCAL,
                "horizontalAperture": CAMERA_H_APERTURE,
                "verticalAperture": CAMERA_V_APERTURE,
                "clippingRange": Gf.Vec2f(0.01, 100000.0),
            })

        self.viewport = create_viewport_window("Robot Camera View", width=640, height=360)
        self.viewport.viewport_api.set_active_camera(self.camera_path)

        self.render_product = rep.create.render_product(self.camera_path, CAMERA_RESOLUTION)
        self.rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annot.attach(self.render_product)
        self.depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth_annot.attach(self.render_product)

        self.last_react_time = time.time() + 10.0
        self.react_enabled = True

        # Session log (saved as JSON)
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_start_time = time.time()
        self._log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(self._log_dir, exist_ok=True)
        self._session_log = []
        self._last_log_save = time.time()
        self._LOG_SAVE_INTERVAL = 30.0  # auto-save every 30s

        # PRED vs SayPlan mode detection
        self._is_pred = isinstance(brain, (PREDBrain, CustomBrain))
        print(f"[UI] Planner mode: {self._planner_name}")

        self._build_ui()

    def _build_ui(self):
        with self.window.frame:
            with ui.HStack(spacing=10):
                with ui.VStack(width=ui.Fraction(1), spacing=8, padding=10):
                    ui.Label(self._title, height=30,
                             style={"font_size": 20, "color": 0xFFFFAA00})
                    ui.Label("Instruction:", height=20)
                    with ui.HStack(height=40):
                        ui.StringField(model=self.instruction_model)
                        ui.Button("EXECUTE", width=80, clicked_fn=self.on_submit)
                    self.status_lbl = ui.Label("Idle", height=30)
                    self.nav_status_lbl = ui.Label("Nav: idle", height=20,
                                                    style={"color": 0xFF88CCFF})
                    self.arm_status_lbl = ui.Label("Arm: idle", height=20,
                                                    style={"color": 0xFFFFCC88})
                    ui.Label("Chat & Vision Log:", height=20)
                    ui.StringField(model=self.chat_log_model, multiline=True, read_only=True)
                    with ui.HStack(height=40, spacing=5):
                        ui.Button("STOP", width=100,
                                  style={"background_color": 0xFF880000},
                                  clicked_fn=self.on_stop)
                        self.react_btn = ui.Button("REACT: ON", width=120,
                                                    style={"background_color": 0xFF006600},
                                                    clicked_fn=self.toggle_react)

                with ui.VStack(width=ui.Fraction(1), spacing=8, padding=10):
                    ui.Label("REACT Scene Monitor", height=20,
                             style={"color": 0xFF00FF88})
                    self.react_status_lbl = ui.Label("Initializing...", height=25,
                                                      style={"color": 0xFF88FF88})
                    self.sg_count_lbl = ui.Label("Objects: --", height=20)

    def log_chat(self, text):
        s = self.chat_log_model.as_string
        s += f"{text}\n"
        if len(s) > 8000:
            s = s[-4000:]
        self.chat_log_model.as_string = s
        print(f"[UI Log] {text}")
        self._append_session_log("chat", text)

    def log_inter(self, text):
        # UI panel removed for performance — log to session + terminal only
        entry = {"raw": text}
        if text.startswith("LLM Response:"):
            try:
                json_str = text[len("LLM Response:"):].strip()
                entry = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
        self._append_session_log("llm", entry)

    def log_react(self, text):
        # UI panel removed for performance — log to session + terminal only
        self._append_session_log("react", text)

    def _append_session_log(self, category, data):
        self._session_log.append({
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.last_react_time + 10.0, 2),
            "category": category,
            "state": self.state,
            "data": data,
        })
        # Cap to prevent memory growth
        if len(self._session_log) > 1000:
            self._save_session_log()
            self._session_log = self._session_log[-100:]

    def _save_session_log(self):
        path = os.path.join(self._log_dir, f"session_{self._session_start}.json")
        # Copy data for background write
        log_copy = list(self._session_log)
        start = self._session_start
        def _write():
            try:
                with open(path, 'w') as f:
                    json.dump({
                        "session_start": start,
                        "total_entries": len(log_copy),
                        "entries": log_copy,
                    }, f, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                print(f"[Log] Save error: {e}")
        threading.Thread(target=_write, daemon=True).start()

    def toggle_react(self):
        self.react_enabled = not self.react_enabled
        if self.react_enabled:
            self.react_btn.text = "REACT: ON"
            self.react_btn.set_style({"background_color": 0xFF006600})
        else:
            self.react_btn.text = "REACT: OFF"
            self.react_btn.set_style({"background_color": 0xFF444444})

    def on_submit(self):
        instr = self.instruction_model.as_string
        if not instr: return
        # Reset previous task state
        if self.state == "NAVIGATING":
            self.navigator.cancel()
        self.plan_queue = []
        self.current_action = None
        self._nav_tracking_id = None
        self._llm_future = None
        self._vision_future = None
        self._step_cooldown = None
        self.state = "THINKING"
        self.log_chat(f"User: {instr}")
        self.current_instruction = instr
        self.feedback = ""

    def on_stop(self):
        self.state = "IDLE"
        self.plan_queue = []
        self.navigator.cancel()
        self.arm.go_home()
        self.status_lbl.text = "Stopped"

    def update(self, dt=0.0):
        now = time.time()

        # Manual GC: only when IDLE, max once per 10s — avoids physics stutter
        if self.state == "IDLE" and now - self._last_gc_time > 10.0:
            self._last_gc_time = now
            gc.collect()

        # Auto-observe new surfaces when IDLE and pending queue has entries
        if (self.state == "IDLE" and self._depth_fs_enabled
                and self._pending_surface_queue):
            self._start_surface_observation()

        # Idle update: generate edges for new objects + update surface JSON
        # DISABLED — LLM edge generation blocks main thread and freezes Isaac Sim.
        # To re-enable, uncomment below.
        # if (self.state == "IDLE" and hasattr(self.brain, 'idle_update')
        #         and now - getattr(self, '_session_start_time', now) > 30.0):
        #     if now - getattr(self, '_last_idle_update', 0) > 10.0:
        #         self._last_idle_update = now
        #         try:
        #             n = self.brain.idle_update()
        #             if n > 0:
        #                 self.log_chat(f"[idle] Generated {n} edges for new objects, surface JSON updated")
        #         except Exception as e:
        #             print(f"[UI] idle_update error: {e}")

        # Auto-save session log
        if self._session_log and now - self._last_log_save > self._LOG_SAVE_INTERVAL:
            self._save_session_log()
            self._last_log_save = now

        # REACT
        if self.react_enabled and self.react_bridge:
            t = time.time()
            if t - self.last_react_time > REACT_UPDATE_INTERVAL:
                self._send_react_frame()
                self.last_react_time = t
            # Throttle poll to every 0.5s
            if t - getattr(self, '_last_poll_time', 0) > 0.5:
                self._last_poll_time = t
                self._poll_react_results()

        # Periodic scene graph sync from REACT (every 3s, regardless of state)
        # coords_only=True: only update positions/extents, skip expensive graph string rebuild
        if self.react_enabled and self.react_bridge:
            if now - getattr(self, '_last_periodic_refresh', 0) > 3.0:
                self._last_periodic_refresh = now
                try:
                    self.brain.refresh_from_react(coords_only=True)
                except Exception:
                    pass

        # Depth-based free space test
        if self._depth_fs_enabled and self._depth_fs_analyzer:
            if now - self._last_depth_fs_time > self._depth_fs_interval:
                self._last_depth_fs_time = now
                self._run_depth_freespace()

        # Export scene graph to file for RViz visualization (1Hz)
        if hasattr(self.navigator, '_isaac_to_odom') and now - self._last_sg_viz_export > 1.0:
            self._last_sg_viz_export = now
            self._export_sg_for_rviz()

        try:
            nav_status = self.navigator.update()
        except Exception as e:
            nav_status = f"nav error: {str(e)[:30]}"
            print(f"[Nav] update() error (likely LiDAR): {e}")
            # Don't let LiDAR errors kill the sim — just stop and retry next frame
        self.nav_status_lbl.text = f"Nav: {nav_status}"

        arm_idle = self.arm.update(dt)
        holding = " [HOLDING]" if self.arm.holding_object else ""
        self.arm_status_lbl.text = f"Arm: {'idle' if arm_idle else 'busy'}{holding}"

        # === State Machine ===
        if self.state == "THINKING":
            if not hasattr(self, '_llm_future') or self._llm_future is None:
                self.status_lbl.text = "Thinking... (LLM)"
                self._llm_future = {"done": False, "result": None}
                ref = self._llm_future
                instr, fb = self.current_instruction, self.feedback
                def _call():
                    try: ref["result"] = self.brain.process_turn(instr, fb)
                    except Exception as e:
                        ref["result"] = {"mode": "error", "reasoning": str(e), "command": {}}
                    ref["done"] = True
                threading.Thread(target=_call, daemon=True).start()
                return
            if not self._llm_future["done"]: return
            response = self._llm_future["result"]
            self._llm_future = None
            self.log_inter(f"LLM Response:\n{json.dumps(response, indent=2)}")
            self._append_session_log("plan", {
                "instruction": self.current_instruction,
                "llm_response": response,
            })
            mode = response.get("mode")
            command = response.get("command", {})
            if mode == "exploring":
                self.brain.execute_graph_api(command.get("command_name"),
                                            command.get("node_name"))
                self.feedback = "Graph Expanded."
            elif mode == "planning":
                self.plan_queue = self._parse_plan(command.get("plan", []))
                self._step_cooldown = None
                self._picked_this_plan = set()  # reset for new plan
                self.state = "EXECUTING"
            elif mode == "searching":
                # PRED search mode: navigate to LLM-guessed locations to find missing object
                missing_name = response.get("missing_object", "unknown")
                search_targets = response.get("search_targets", [])
                original_instr = response.get("original_instruction", self.current_instruction)
                self.log_chat(f"Object '{missing_name}' not found. Searching {len(search_targets)} likely locations...")
                if search_targets:
                    self._search_missing_name = missing_name
                    self._search_original_instr = original_instr
                    # Build goto plan for each search target, then re-plan after search
                    search_plan = []
                    for sid in search_targets:
                        search_plan.append({"type": "goto", "target": str(sid),
                                            "_search_for": missing_name})
                    self.plan_queue = search_plan
                    self.state = "EXECUTING"
                else:
                    self.log_chat(f"No search locations available for '{missing_name}'.")
                    self.state = "IDLE"
            else:
                self.state = "IDLE"

        elif self.state == "EXECUTING":
            # Check if new relevant objects were found → replan (PRED only)
            if self._is_pred and self._pending_replan_reason:
                reason = self._pending_replan_reason
                self._pending_replan_reason = None
                self._replan_cooldown = time.time() + 30  # prevent rapid re-triggers
                self.log_chat(f"[PRED] New relevant object discovered! Re-planning...")
                self.brain.refresh_from_react()
                self.feedback = reason
                self.plan_queue = []
                self.current_action = None
                self.state = "THINKING"
                return

            if not self.current_action:
                if self.plan_queue:
                    # Cooldown between steps: wait 1s to let Isaac Sim settle
                    if not hasattr(self, '_step_cooldown') or self._step_cooldown is None:
                        self._step_cooldown = time.time()
                        return
                    if time.time() - self._step_cooldown < 1.0:
                        return
                    self._step_cooldown = None

                    self.current_action = self.plan_queue.pop(0)
                    self.log_chat(f">> Executing: {self.current_action['type']}"
                                  f"({self.current_action.get('target', '')})")
                else:
                    # If we just finished a PRED search sweep, check if the missing object appeared (PRED only)
                    if self._is_pred and hasattr(self, '_search_missing_name'):
                        missing = self._search_missing_name
                        original_instr = getattr(self, '_search_original_instr', self.current_instruction)
                        # Refresh scene graph and check
                        self.brain.refresh_from_react()
                        found_objs = self.brain._find_object_by_name(missing) if hasattr(self.brain, '_find_object_by_name') else []
                        if not found_objs:
                            # Also check raw_data directly for SayPlanBrain
                            for info in self.brain.raw_data.values():
                                if missing.lower() in info.get("object_tag", "").lower() and info.get("_status", "present") == "present":
                                    found_objs.append(info)
                        del self._search_missing_name
                        if hasattr(self, '_search_original_instr'):
                            del self._search_original_instr
                        if found_objs:
                            self.log_chat(f"Found '{missing}' after search! Re-planning...")
                            self.current_instruction = original_instr
                            self.feedback = f"FOUND_AFTER_SEARCH: '{missing}' is now visible in the scene graph."
                            self.state = "THINKING"
                        else:
                            self.log_chat(f"Could not find '{missing}' after searching all locations.")
                            self.state = "IDLE"
                            self.status_lbl.text = f"Search failed: {missing}"
                    else:
                        self.state = "IDLE"
                        self.status_lbl.text = "Finished"
                    return

            action_type = self.current_action['type']
            target_id = self.current_action.get('target')

            if action_type == 'goto':
                # PRED: prefer closest duplicate when goto(X) → pick(X)
                # with the SAME target (duplicates of the same object tag).
                # Do NOT substitute when:
                #   - goto(surface) → pick(different_obj): LLM chose that surface
                #   - goto(surface) → [open] → place: LLM chose placement target
                if self._is_pred and target_id and self.plan_queue:
                    next_pick = None
                    for a in self.plan_queue:
                        if a.get('type') == 'pick':
                            next_pick = a
                            break
                        if a.get('type') in ('place', 'done'):
                            break

                    # Only substitute if goto and next pick share the same target ID
                    if (next_pick and str(next_pick.get('target')) == str(target_id)):
                        target_id = self._pred_pick_closest_duplicate(target_id)
                        self.current_action['target'] = target_id
                        next_pick['target'] = target_id

                approach_info = self.brain.get_nav_approach_info(target_id)
                target_name = self.brain.get_obj_name_by_id(target_id)

                # PRED-style dynamic searching using LLM-guessed locations (PRED only)
                if approach_info is None and self._is_pred and hasattr(self.brain, 'generate_search_targets'):
                    # Initialize search mode if not already searching
                    if not hasattr(self, '_search_target_queue'):
                        if not hasattr(self, '_search_future') or self._search_future is None:
                            self.log_chat(f"Object {target_name} not found! Asking LLM for likely locations...")
                            self.status_lbl.text = f"Generating search plan for {target_name}..."
                            self._search_future = {"done": False, "result": None}
                            ref = self._search_future
                            def _gen_search():
                                try:
                                    ref["result"] = self.brain.generate_search_targets(target_id, num_targets=3)
                                except Exception as e:
                                    print(f"[UI] Search gen error: {e}")
                                    ref["result"] = []
                                ref["done"] = True
                            threading.Thread(target=_gen_search, daemon=True).start()
                            return # Wait for thread to finish
                        
                        if not self._search_future["done"]:
                            return # Still waiting for LLM
                            
                        guessed_ids = self._search_future["result"] or []
                        self._search_future = None
                        self._search_target_queue = guessed_ids
                        self._current_search_idx = 0
                        if guessed_ids:
                            self.log_chat(f"Will search near: {', '.join([self.brain.get_obj_name_by_id(gid) for gid in guessed_ids])}")
                        else:
                            self.log_chat(f"LLM returned no search locations for {target_name}.")
                    
                    if self._current_search_idx >= len(self._search_target_queue):
                        self.log_chat(f"Searched all likely locations but could not find {target_name}. Aborting.")
                        delattr(self, '_search_target_queue')
                        delattr(self, '_current_search_idx')
                        self.state = "IDLE"
                        self.status_lbl.text = f"Failed to find {target_name}"
                        self.current_action = None
                        return
                        
                    # Get the next guessed location
                    guess_id = str(self._search_target_queue[self._current_search_idx])
                    guess_approach = self.brain.get_nav_approach_info(guess_id)
                    guess_name = self.brain.get_obj_name_by_id(guess_id)
                    
                    print(f"[UI] Trying to search near ID={guess_id} ({guess_name}), approach={guess_approach}")

                    if guess_approach is not None:
                        # Navigate to the guessed location
                        nav_pos = guess_approach["nav_pos"]
                        arrival_dist = guess_approach["arrival_dist"]
                        
                        robot_pos, _ = get_prim_transform(self.robot_path)
                        direction = nav_pos[:2] - robot_pos[:2]
                        distance = np.linalg.norm(direction)
                        if distance > arrival_dist:
                            goal_pos = nav_pos[:2] - (direction / distance) * arrival_dist
                        else:
                            goal_pos = robot_pos[:2]
                        
                        goal_yaw = np.arctan2(direction[1], direction[0])
                        self.navigator.set_goal(goal_pos, yaw=goal_yaw, object_pos=nav_pos[:2], arrival_dist=arrival_dist)
                        self.nav_target_name = f"SEARCH({target_name})_AT_{guess_name}"
                        self._nav_tracking_id = guess_id
                        self.log_chat(f"Searching for {target_name} near {guess_name}...")
                        self._arrive_time = None
                        self.state = "NAVIGATING"
                        
                        # Increment index for next time (in case it's not found here)
                        self._current_search_idx += 1
                        # Do NOT clear current_action, so it re-evaluates goto(target) next
                    else:
                        # Skip invalid guess
                        self.log_chat(f"Invalid search location ID {guess_id}. Skipping...")
                        self._current_search_idx += 1

                elif approach_info is None and not self._is_pred:
                    # SayPlan: object not in scene graph → cannot navigate
                    target_name = self.brain.get_obj_name_by_id(target_id)
                    self.log_chat(f"[SayPlan] Object '{target_name}' (id:{target_id}) not found in scene graph. Stopping.")
                    self.current_action = None
                    self.state = "IDLE"
                    self.status_lbl.text = f"Object not found: {target_name}"
                    return

                elif approach_info is not None:
                    if hasattr(self, '_search_target_queue'):
                        self.log_chat(f"Found {target_name}! Heading to it.")
                        delattr(self, '_search_target_queue')
                        delattr(self, '_current_search_idx')

                    nav_pos = approach_info["nav_pos"]
                    face_pos = np.array(approach_info["face_pos"], dtype=np.float64)
                    arrival_dist = approach_info["arrival_dist"]
                    parent_name = approach_info["parent_name"]

                    # Marker shows current goto target (the object we're heading to)

                    robot_pos, _ = get_prim_transform(self.robot_path)
                    direction = nav_pos[:2] - robot_pos[:2]
                    distance = np.linalg.norm(direction)
                    if distance > arrival_dist:
                        goal_pos = nav_pos[:2] - (direction / distance) * arrival_dist
                    else:
                        goal_pos = robot_pos[:2]

                    # Face toward the actual target (not nav target)
                    face_dir = face_pos[:2] - goal_pos
                    if np.linalg.norm(face_dir) > 0.01:
                        goal_yaw = np.arctan2(face_dir[1], face_dir[0])
                    else:
                        goal_yaw = np.arctan2(direction[1], direction[0])

                    self.marker.set_world_pose(position=face_pos)
                    self.marker.set_visibility(True)
                    self.navigator.set_goal(goal_pos, yaw=goal_yaw,
                                            object_pos=nav_pos[:2],
                                            arrival_dist=arrival_dist)
                    self._append_session_log("action", {
                        "type": "goto",
                        "target_id": target_id,
                        "target_name": target_name,
                        "parent_name": parent_name,
                        "nav_pos": nav_pos.tolist() if hasattr(nav_pos, 'tolist') else list(nav_pos),
                        "face_pos": face_pos.tolist() if hasattr(face_pos, 'tolist') else list(face_pos),
                        "goal_pos": goal_pos.tolist() if hasattr(goal_pos, 'tolist') else list(goal_pos),
                        "arrival_dist": arrival_dist,
                    })
                    # Check if this is a search-mode goto
                    search_for = self.current_action.get('_search_for')
                    if search_for:
                        self.nav_target_name = f"SEARCH({search_for})_AT_{target_name}"
                        self.log_chat(f"Searching for {search_for} near {target_name}...")
                    elif parent_name:
                        self.log_chat(f"Going to {target_name} "
                                      f"(approaching {parent_name})")
                        self.nav_target_name = target_name
                    else:
                        self.log_chat(f"Going to {target_name}")
                        self.nav_target_name = target_name
                    self.status_lbl.text = f"Navigating to {target_name}..."
                    self._nav_tracking_id = target_id
                    self._arrive_time = None
                    self._nav_start_time = time.time()
                    self.state = "NAVIGATING"
                    self.current_action = None

            elif action_type == 'pick':
                pos = self.brain.get_coords_by_id(target_id)
                target_name = self.brain.get_obj_name_by_id(target_id)
                if pos is not None:
                    # Marker at pick target
                    marker_pos = np.array([pos[0], pos[1], pos[2] + 0.05], dtype=np.float64)
                    self.marker.set_world_pose(position=marker_pos)
                    self.marker.set_visibility(True)

                    prim_path = find_scene_prim_near(pos, radius=0.8, object_name=target_name)
                    if prim_path:
                        self.log_chat(f"Picking {target_name} ({prim_path})")
                    else:
                        self.log_chat(f"Picking {target_name} (no prim found, simulating)")
                    self.arm.pick(prim_path, target_world_pos=pos)
                    self.status_lbl.text = f"Picking {target_name}..."
                    self.state = "ARM_BUSY"
                    self._arm_start_time = time.time()
                    self._current_arm_action = "pick"
                    self._current_arm_target_id = target_id
                    self.current_action = None
                else:
                    self.log_chat(f"Cannot find {target_name} position")
                    self.current_action = None

            elif action_type == 'place':
                target_name = self.brain.get_obj_name_by_id(target_id)
                # Enter observation mode: stop for 1.5s to accumulate fresh freespace
                if self._depth_fs_enabled and self._depth_fs_analyzer:
                    self.log_chat(f"Observing {target_name} surface for placement...")
                    self.status_lbl.text = f"Observing {target_name}..."
                    self._observe_start_time = time.time()
                    self._observe_place_target_id = target_id
                    self._observe_place_target_name = target_name
                    self.state = "OBSERVING_FOR_PLACE"
                    self.current_action = None
                else:
                    # No depth freespace — execute place immediately
                    self._execute_place(target_id, target_name)

            elif action_type in ('open', 'close'):
                pos = self.brain.get_coords_by_id(target_id)
                target_name = self.brain.get_obj_name_by_id(target_id)
                if pos is not None:
                    # Marker at open/close target
                    marker_pos = np.array([pos[0], pos[1], pos[2] + 0.05], dtype=np.float64)
                    self.marker.set_world_pose(position=marker_pos)
                    self.marker.set_visibility(True)

                    prim_path = find_scene_prim_near(
                        pos, radius=1.5, object_name=target_name)
                    verb = "Opening" if action_type == 'open' else "Closing"
                    self.log_chat(f"{verb} {target_name}")
                    if action_type == 'open':
                        self.arm.open_object(prim_path, target_world_pos=pos)
                    else:
                        self.arm.close_object(prim_path, target_world_pos=pos)
                    self.status_lbl.text = f"{verb} {target_name}..."
                    self.state = "ARM_BUSY"
                    self._arm_start_time = time.time()
                    self._current_arm_action = action_type
                    self._current_arm_target_id = target_id
                    self.current_action = None
                else:
                    self.log_chat(f"Cannot find {target_name} position")
                    self.current_action = None

            elif action_type == 'done':
                self.current_action = None

        elif self.state == "NAVIGATING":
            # Navigation timeout: if stuck for >60s, force arrival
            _nav_elapsed = time.time() - getattr(self, '_nav_start_time', time.time())
            if _nav_elapsed > 60.0:
                self.log_chat(f"Navigation timeout ({_nav_elapsed:.0f}s). Forcing arrival.")
                self.navigator.cancel()
                self.navigator.is_arrived = True
                self._arrive_time = time.time() - 2.0  # skip settle delay
                # fall through to arrival check below

            # Check if new relevant objects → stop nav and replan (PRED only)
            if self._is_pred and self._pending_replan_reason:
                reason = self._pending_replan_reason
                self._pending_replan_reason = None
                self._replan_cooldown = time.time() + 30
                self.log_chat(f"[PRED] New relevant object discovered during navigation! Re-planning...")
                self.navigator.cancel()
                self.brain.refresh_from_react()
                self.feedback = reason
                self.plan_queue = []
                self.current_action = None
                self._nav_tracking_id = None
                self.state = "THINKING"
                return

            # Throttled live-update: refresh scene graph + marker every 0.5s (not every frame)
            _now = time.time()
            if _now - getattr(self, '_last_nav_refresh', 0) > 0.5:
                self._last_nav_refresh = _now
                try:
                    self.brain.refresh_from_react(coords_only=True)
                except Exception as e:
                    print(f"[Nav] refresh_from_react error: {e}")

                # Update marker + nav goal using latest REACT data
                if self._nav_tracking_id:
                    try:
                        approach_info = self.brain.get_nav_approach_info(self._nav_tracking_id)
                        if approach_info is not None:
                            face_pos = np.array(approach_info["face_pos"], dtype=np.float64)
                            nav_pos = np.array(approach_info["nav_pos"], dtype=np.float64)
                            arrival_dist = float(approach_info["arrival_dist"])

                            # Marker tracks current goto target (real-time REACT update)

                            # Skip if coordinates are invalid
                            if np.any(np.isnan(face_pos)) or np.any(np.isinf(face_pos)):
                                print(f"[Nav] Invalid face_pos: {face_pos}, skipping marker update")
                            else:
                                self.marker.set_world_pose(position=face_pos)

                                robot_pos, _ = get_prim_transform(self.robot_path)
                                direction = nav_pos[:2] - robot_pos[:2]
                                distance = np.linalg.norm(direction)
                                if distance > arrival_dist:
                                    goal_pos = nav_pos[:2] - (direction / distance) * arrival_dist
                                else:
                                    goal_pos = robot_pos[:2]
                                face_dir = face_pos[:2] - goal_pos
                                if np.linalg.norm(face_dir) > 0.01:
                                    goal_yaw = np.arctan2(face_dir[1], face_dir[0])
                                else:
                                    goal_yaw = np.arctan2(direction[1], direction[0])
                                self.navigator.update_goal(goal_pos, yaw=goal_yaw,
                                                           object_pos=nav_pos[:2],
                                                           arrival_dist=arrival_dist)
                    except Exception as e:
                        print(f"[Nav] Marker/goal update error: {e}")

                # Search mode: check if missing object appeared (PRED only)
                if self._is_pred and self.nav_target_name and self.nav_target_name.startswith("SEARCH("):
                    try:
                        missing_name = self.nav_target_name.split("SEARCH(")[1].split(")")[0]
                        found = False
                        for info in self.brain.raw_data.values():
                            if (missing_name.lower() in info.get("object_tag", "").lower()
                                    and info.get("_status", "present") == "present"):
                                found = True
                                break
                        if found:
                            self.log_chat(f"Found '{missing_name}' during search! Stopping search and re-planning...")
                            self.navigator.cancel()
                            self.plan_queue = []
                            self.current_action = None
                            self._nav_tracking_id = None
                            if hasattr(self, '_search_target_queue'):
                                delattr(self, '_search_target_queue')
                            if hasattr(self, '_current_search_idx'):
                                delattr(self, '_current_search_idx')
                            original_instr = getattr(self, '_search_original_instr', self.current_instruction)
                            if hasattr(self, '_search_missing_name'):
                                del self._search_missing_name
                            if hasattr(self, '_search_original_instr'):
                                del self._search_original_instr
                            self.current_instruction = original_instr
                            self.feedback = f"FOUND_AFTER_SEARCH: '{missing_name}' is now visible in the scene graph."
                            self.state = "THINKING"
                            return
                    except Exception as e:
                        print(f"[Nav] Search check error: {e}")

            if self.navigator.arrived:
                # Settle delay: wait 1s after arrival before acting
                if not hasattr(self, '_arrive_time') or self._arrive_time is None:
                    self._arrive_time = time.time()
                    # Refresh REACT coords while we wait
                    try:
                        self.brain.refresh_from_react(coords_only=True)
                    except Exception:
                        pass
                    return

                if time.time() - self._arrive_time < 1.0:
                    return  # still settling

                self._arrive_time = None
                self._append_session_log("action", {
                    "type": "arrived", "target_name": self.nav_target_name,
                })
                self.log_chat(f"Arrived at {self.nav_target_name}!")

                # If it was a search waypoint, just go back to EXECUTING to pick next search point
                if self.nav_target_name and self.nav_target_name.startswith("SEARCH("):
                    self.log_chat("Looked around but haven't found it yet...")
                    self.state = "EXECUTING"
                else:
                    self.state = "VERIFYING"
                    self._vision_future = {"done": False, "result": None}
                    ref = self._vision_future
                    img = self.rgb_annot.get_data()
                    target = self.nav_target_name
                    def _verify():
                        try: ref["result"] = self.brain.verify_visual(img, target)
                        except Exception as e: ref["result"] = f"Vision Error: {e}"
                        ref["done"] = True
                    threading.Thread(target=_verify, daemon=True).start()
            elif not self.navigator.navigating:
                self.log_chat(f"Navigation ended ({self.navigator.status_text})")
                self.state = "EXECUTING"

        elif self.state == "VERIFYING":
            if hasattr(self, '_vision_future') and self._vision_future and self._vision_future["done"]:
                result_text = str(self._vision_future['result']).strip()
                self._append_session_log("action", {
                    "type": "vision_verify",
                    "target_name": self.nav_target_name,
                    "result": result_text,
                })
                self.log_chat(f"Vision: {result_text}")
                self._vision_future = None
                self.marker.set_visibility(False)

                # If vision says NO, trust scene graph — object is there but hard to see
                # Just proceed with the plan instead of replanning
                if result_text.upper().startswith("NO"):
                    self.log_chat(f"Object may be hard to see, but scene graph confirms it's here. Proceeding.")
                self.state = "EXECUTING"

        elif self.state == "NAV_TO_OBSERVE_SURFACE":
            # Navigating to a viewpoint to observe a new surface
            info = self._obs_surface_current
            vp_idx = self._obs_surface_vp_idx
            if self.navigator.is_arrived:
                # Arrived at viewpoint — start observing
                self._obs_surface_nav_failures = 0  # reset on success
                self.log_chat(f"[Surface] Arrived at viewpoint {vp_idx+1}/3 for {info['tag']}(id:{info['id']})")
                self._obs_surface_observe_start = time.time()
                self.state = "OBSERVING_NEW_SURFACE"
            elif time.time() - self._obs_surface_nav_start > self._obs_surface_nav_timeout:
                # Timed out reaching viewpoint (likely unreachable — wall in the way)
                self._obs_surface_nav_failures += 1
                self.navigator.cancel()
                self.log_chat(f"[Surface] Viewpoint {vp_idx+1}/3 unreachable (timeout), failures={self._obs_surface_nav_failures}")

                if self._obs_surface_nav_failures >= self._obs_surface_max_failures:
                    # Too many failures — finalize with whatever data we have
                    self.log_chat(f"[Surface] Too many nav failures for {info['tag']}(id:{info['id']}), finishing observation")
                    self._finalize_surface_observation(info)
                else:
                    # Skip to next viewpoint
                    self._obs_surface_vp_idx += 1
                    if self._obs_surface_vp_idx < len(self._obs_surface_viewpoints):
                        vp = self._obs_surface_viewpoints[self._obs_surface_vp_idx]
                        vx, vy, vyaw = vp
                        n = self._obs_surface_vp_idx + 1
                        cx, cy = info["center"][0], info["center"][1]
                        self.log_chat(f"[Surface] Skipping to viewpoint {n}/3")
                        self.navigator.set_goal([vx, vy], yaw=vyaw, object_pos=[cx, cy])
                        self._obs_surface_nav_start = time.time()
                    else:
                        self._finalize_surface_observation(info)

        elif self.state == "OBSERVING_NEW_SURFACE":
            # Observing new surface: collect depth frames for Z detection + freespace
            elapsed = time.time() - self._obs_surface_observe_start
            info = self._obs_surface_current
            sid = info["id"]

            # Process depth every frame during observation
            if self._depth_fs_analyzer:
                depth_data = self.depth_annot.get_data()
                if depth_data is not None:
                    robot_pos, robot_quat = get_prim_transform(self.base_link_path)
                    
                    # Collect Z samples for surface height detection (in background!)
                    if not getattr(self, '_z_sample_busy', False):
                        self._z_sample_busy = True
                        depth_copy = depth_data.copy()
                        rpos = np.array(robot_pos, dtype=np.float64)
                        rquat = np.array(robot_quat, dtype=np.float64)
                        def _bg_z():
                            try:
                                self._depth_fs_analyzer.collect_z_sample(sid, depth_copy, rpos, rquat)
                            except Exception as e:
                                print(f"[Z-Sample] BG error: {e}")
                            finally:
                                self._z_sample_busy = False
                        threading.Thread(target=_bg_z, daemon=True).start()

                    # Also run normal freespace processing
                    self._run_depth_freespace()

            if elapsed >= self._obs_surface_observe_duration:
                # Viewpoint observation done — move to next viewpoint or finish
                self._obs_surface_vp_idx += 1
                if self._obs_surface_vp_idx < len(self._obs_surface_viewpoints):
                    # Navigate to next viewpoint
                    vp = self._obs_surface_viewpoints[self._obs_surface_vp_idx]
                    vx, vy, vyaw = vp
                    n = self._obs_surface_vp_idx + 1
                    cx, cy = info["center"][0], info["center"][1]
                    self.log_chat(f"[Surface] Moving to viewpoint {n}/3")
                    self.navigator.set_goal([vx, vy], yaw=vyaw, object_pos=[cx, cy])
                    self._obs_surface_nav_start = time.time()
                    self.state = "NAV_TO_OBSERVE_SURFACE"
                else:
                    self._finalize_surface_observation(info)

        elif self.state == "OBSERVING_FOR_PLACE":
            # Robot stops and observes for 1.5s to accumulate fresh freespace
            elapsed = time.time() - self._observe_start_time
            # Force depth freespace processing every frame during observation
            if self._depth_fs_enabled and self._depth_fs_analyzer:
                self._run_depth_freespace()
            if elapsed >= self._observe_duration:
                # Observation done — now execute place with fresh freespace data
                self._execute_place(
                    self._observe_place_target_id,
                    self._observe_place_target_name)
                self._observe_place_target_id = None
                self._observe_place_target_name = None

        elif self.state == "ARM_BUSY":
            # Arm timeout: if stuck for >30s, force completion
            _arm_elapsed = time.time() - getattr(self, '_arm_start_time', time.time())
            if _arm_elapsed > 30.0:
                self.log_chat(f"Arm timeout ({_arm_elapsed:.0f}s). Forcing completion.")
                arm_idle = True

            if arm_idle:
                self.log_chat("Arm action completed.")

                if self._current_arm_action == "pick" and self._current_arm_target_id:
                    obj_name = self.brain.get_obj_name_by_id(self._current_arm_target_id)
                    self.brain.holding_object = self._current_arm_target_id
                    self._picked_this_plan.add(str(self._current_arm_target_id))
                    # Mark object as absent in scene graph (picked up)
                    self._sg_mark_status(self._current_arm_target_id, "absent")
                    # Notify react_worker: object removed
                    self._notify_react_sg_update(int(self._current_arm_target_id), "removed")
                    # Update cached freespace grid: mark picked object's area as free
                    if self._depth_fs_analyzer:
                        pick_pos = self.brain.get_coords_by_id(self._current_arm_target_id)
                        if pick_pos is not None:
                            for sid in self._depth_fs_analyzer.surfaces:
                                self._depth_fs_analyzer.mark_cells_free(
                                    sid, float(pick_pos[0]), float(pick_pos[1]), radius_m=0.12)
                    self.log_chat(f"Holding {obj_name} — removed from scene graph")

                elif self._current_arm_action == "place":
                    placed_id = self.brain.holding_object
                    placed_name = self.brain.get_obj_name_by_id(placed_id) if placed_id else "object"
                    self.brain.holding_object = None
                    # Update scene graph: object placed at arm's target position
                    if placed_id and hasattr(self.arm, 'last_place_pos') and self.arm.last_place_pos is not None:
                        place_pos = self.arm.last_place_pos
                        self._sg_update_position(placed_id, place_pos)
                        self._sg_mark_status(placed_id, "present")
                        # Notify react_worker: object moved to new position
                        self._notify_react_sg_update(
                            int(placed_id), "moved",
                            [float(place_pos[0]), float(place_pos[1]), float(place_pos[2])])

                        # Update cached freespace grid: mark placement area as occupied
                        if self._depth_fs_analyzer:
                            for sid in self._depth_fs_analyzer.surfaces:
                                self._depth_fs_analyzer.mark_cells_occupied(
                                    sid, float(place_pos[0]), float(place_pos[1]), radius_m=0.12)

                        # Re-run surface analysis so free spaces immediately reflect the placed object
                        if hasattr(self.brain, 'surface_analyzer'):
                            react_sg = getattr(self.brain.react_bridge, 'current_sg', None) if hasattr(self.brain, 'react_bridge') else None
                            self.brain.surface_analyzer.update(self.brain.raw_data, getattr(self.brain, 'edges', []), react_sg)

                        self.log_chat(f"Placed {placed_name} — scene graph updated at ({place_pos[0]:.2f}, {place_pos[1]:.2f}, {place_pos[2]:.2f})")
                    elif placed_id:
                        self._sg_mark_status(placed_id, "present")
                        self.log_chat(f"Placed {placed_name} — marked present")
                    # Trigger REACT re-check after placing
                    if self.react_enabled and self.react_bridge:
                        try:
                            self._send_react_frame()
                        except Exception:
                            pass

                self._current_arm_action = None
                self._current_arm_target_id = None

                self.status_lbl.text = "Arm done"
                self.state = "EXECUTING"

    def _seed_precomputed_depth_grids(self):
        """Seed persistent cache + surface_analyzer._depth_grids from precomputed freespace.

        This lets placement and RViz work immediately using precomputed data,
        before any live depth frames are processed.
        """
        if not self._depth_fs_analyzer:
            return

        # Seed persistent cache in depth_freespace analyzer
        n_cached = 0
        for sid in self._depth_fs_analyzer._precomputed_rects:
            self._depth_fs_analyzer.seed_cache_from_precomputed(sid)
            if sid in self._depth_fs_analyzer._cached_grids:
                n_cached += 1

        # Also seed surface_analyzer._depth_grids for placement integration
        sa = getattr(self.brain, 'surface_analyzer', None) if hasattr(self.brain, 'surface_analyzer') else None
        if sa:
            for sid, cached in self._depth_fs_analyzer._cached_grids.items():
                sa._depth_grids[sid] = {
                    "free_grid": cached["free"].copy(),
                    "bounds": {
                        "x_min": cached["x_min"], "y_min": cached["y_min"],
                        "nx": cached["nx"], "ny": cached["ny"],
                        "cell_size": self._depth_fs_analyzer.cell_size,
                    },
                }

        # Mark precomputed + registered surfaces as known (no observation needed)
        self._known_surface_ids = set(self._depth_fs_analyzer._precomputed_rects.keys())
        self._known_surface_ids.update(self._depth_fs_analyzer.surfaces.keys())

        if n_cached > 0:
            print(f"[UI] Seeded {n_cached} persistent grid caches from precomputed freespace")

    def _run_depth_freespace(self):
        """Run depth-based free space analysis.

        Main thread: capture data + sync surfaces (fast).
        Background thread: process depth, build RViz cells, save overlay (heavy).
        """
        # Skip if previous background work is still running
        if getattr(self, '_depth_fs_busy', False):
            return
        try:
            assert self._depth_fs_analyzer is not None
            depth_data = self.depth_annot.get_data()
            rgb_data = self.rgb_annot.get_data()
            if depth_data is None:
                return
            robot_pos, robot_quat = get_prim_transform(self.base_link_path)

            # Sync surfaces from scene graph (lightweight dict merge — keep on main thread)
            if (not self._depth_fs_analyzer.surfaces or
                    self._depth_fs_frame_count % 10 == 0):
                merged_sg = {}
                if hasattr(self.brain, 'raw_data') and self.brain.raw_data:
                    merged_sg.update(self.brain.raw_data)
                react_sg = None
                if hasattr(self.brain, 'react_bridge') and self.brain.react_bridge:
                    react_sg = getattr(self.brain.react_bridge, 'current_sg', None)
                if react_sg:
                    for k, v in react_sg.items():
                        rid = v.get("id")
                        matched = False
                        if rid is not None:
                            for mk, mv in merged_sg.items():
                                if mv.get("id") == rid:
                                    if "bbox_center" in v:
                                        mv["bbox_center"] = v["bbox_center"]
                                    if "bbox_extent" in v:
                                        mv["bbox_extent"] = v["bbox_extent"]
                                    if "_status" in v:
                                        mv["_status"] = v["_status"]
                                    matched = True
                                    break
                        if not matched:
                            merged_sg[k] = v
                if merged_sg:
                    self._depth_fs_analyzer.register_from_scene_graph(merged_sg)

            if not self._depth_fs_analyzer.surfaces:
                return

            # Snapshot data for background thread
            depth_copy = depth_data.copy()
            rgb_copy = rgb_data.copy() if rgb_data is not None else None
            rpos = np.array(robot_pos, dtype=np.float64)
            rquat = np.array(robot_quat, dtype=np.float64)
            has_odom = hasattr(self.navigator, '_isaac_to_odom')
            odom_fn = self.navigator._isaac_to_odom if has_odom else None

            # Collect SG objects for overlay (small dicts, fast)
            sg_objects_map = {}
            for sid in self._depth_fs_analyzer.surfaces:
                sg_objects_map[sid] = self._get_sg_objects_for_surface(sid)

            self._depth_fs_frame_count += 1
            self._depth_fs_busy = True

            def _bg_work():
                try:
                    # Heavy: process depth + accumulate
                    self._depth_fs_analyzer.process_depth(depth_copy, rpos, rquat)

                    # Build RViz cells + update depth grids
                    rviz_cells = []
                    cell_step = 3
                    cs = self._depth_fs_analyzer.cell_size

                    for sid in list(self._depth_fs_analyzer.surfaces.keys()):
                        surf = self._depth_fs_analyzer.surfaces.get(sid, {})
                        grid_data = self._depth_fs_analyzer.get_best_grid(sid)
                        if grid_data is None:
                            continue

                        sz = grid_data.get("surface_z", surf.get("z", 0))
                        nx, ny = grid_data["nx"], grid_data["ny"]
                        x_min, y_min = grid_data["x_min"], grid_data["y_min"]
                        free = grid_data["free"]
                        occupied = grid_data["occupied"]

                        # Update placement grids
                        if hasattr(self.brain, 'surface_analyzer') and self.brain.surface_analyzer:
                            self.brain.surface_analyzer._depth_grids[sid] = {
                                "free_grid": free.copy(),
                                "bounds": {"x_min": x_min, "y_min": y_min,
                                           "nx": nx, "ny": ny, "cell_size": cs},
                            }
                            # Inject hybrid free rects for JSON saving
                            if hasattr(self.brain.surface_analyzer, 'surfaces'):
                                brain_surf = self.brain.surface_analyzer.surfaces.get(sid)
                                if brain_surf and hasattr(brain_surf, 'set_custom_free_spaces'):
                                    # Get hybrid rects (depth + SG) from analyzer
                                    # SG objects provided via sg_objects_map (from main thread snapshot)
                                    sg_objs = sg_objects_map.get(sid)
                                    hybrid_rects = self._depth_fs_analyzer.get_hybrid_free_rects(sid, sg_objs)
                                    brain_surf.set_custom_free_spaces(hybrid_rects)

                        # RViz export
                        if odom_fn is not None:
                            surf_tag = surf.get("tag", "?")
                            cell_size_m = cs * cell_step
                            for gy in range(0, ny, cell_step):
                                for gx in range(0, nx, cell_step):
                                    gy_end = min(gy + cell_step, ny)
                                    gx_end = min(gx + cell_step, nx)
                                    n_free = int(free[gy:gy_end, gx:gx_end].sum())
                                    n_occ = int(occupied[gy:gy_end, gx:gx_end].sum())
                                    if n_free == 0 and n_occ == 0:
                                        continue
                                    wx = x_min + (gx + cell_step * 0.5) * cs
                                    wy = y_min + (gy + cell_step * 0.5) * cs
                                    odom_xy = odom_fn([wx, wy])
                                    rviz_cells.append({
                                        "x": round(float(odom_xy[0]), 3),
                                        "y": round(float(odom_xy[1]), 3),
                                        "z": round(float(sz), 3),
                                        "sx": round(cell_size_m, 3),
                                        "sy": round(cell_size_m, 3),
                                        "state": "free" if n_free > n_occ else "occupied",
                                        "surface_tag": surf_tag,
                                        "surface_id": sid,
                                    })

                    self._depth_fs_cached_rviz = rviz_cells

                    # Save camera overlay
                    if rgb_copy is not None:
                        sa = getattr(self.brain, 'surface_analyzer', None)
                        if sa and hasattr(sa, '_last_placement') and sa._last_placement:
                            self._depth_fs_analyzer.last_placement = sa._last_placement
                        self._depth_fs_analyzer.save_camera_overlay(
                            rgb_copy, rpos, rquat,
                            sg_objects_on_map=sg_objects_map)

                    # Save updated surface analysis to JSON (persistent log of free space changes)
                    # Only save if we actually updated something
                    if hasattr(self.brain, 'surface_analyzer') and self.brain.surface_analyzer:
                        try:
                            self.brain.surface_analyzer.save_to_json(SURFACE_JSON_FILE)
                        except Exception as e:
                            print(f"[DepthFS] JSON save error: {e}")

                except Exception as e:
                    print(f"[DepthFS] BG error: {e}")
                finally:
                    self._depth_fs_busy = False

            threading.Thread(target=_bg_work, daemon=True).start()

        except Exception as e:
            self._depth_fs_busy = False
            print(f"[DepthFS] Error: {e}")

    def _get_sg_objects_for_surface(self, surface_id):
        """Get scene graph objects on a surface (helper for depth FS)."""
        if hasattr(self.brain, 'surface_analyzer') and self.brain.surface_analyzer:
            sg_surf = self.brain.surface_analyzer.surfaces.get(surface_id)
            if sg_surf:
                return sg_surf.objects_on
        return []

    def _execute_place(self, target_id, target_name):
        """Execute the actual place action (called after observation or immediately)."""
        place_pos = None
        if hasattr(self.brain, 'get_placement_pos'):
            place_pos = self.brain.get_placement_pos(target_id)
        if place_pos is not None:
            place_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.05]
            self.marker.set_world_pose(position=np.array(place_pos, dtype=np.float64))
            self.marker.set_visibility(True)
            self.log_chat(f"Placing on {target_name} (free-space: {place_pos[0]:.2f}, {place_pos[1]:.2f})")
            self.arm.place(place_pos)
            self.status_lbl.text = f"Placing on {target_name}..."
            self.state = "ARM_BUSY"
            self._arm_start_time = time.time()
            self._current_arm_action = "place"
            self._current_arm_target_id = target_id
        else:
            # Fallback: surface center
            pos = self.brain.get_coords_by_id(target_id)
            if pos is not None:
                surface_z = self.brain.get_obj_surface_z(target_id)
                place_pos = [pos[0], pos[1], surface_z + 0.05]
                self.marker.set_world_pose(position=np.array(place_pos, dtype=np.float64))
                self.marker.set_visibility(True)
                self.log_chat(f"Placing on {target_name} (center fallback)")
                self.arm.place(place_pos)
                self.status_lbl.text = f"Placing on {target_name}..."
                self.state = "ARM_BUSY"
                self._arm_start_time = time.time()
                self._current_arm_action = "place"
                self._current_arm_target_id = target_id
            else:
                self.log_chat(f"Cannot find {target_name} position")
                self.state = "EXECUTING"

    # ================================================================
    # New surface observation (REACT-discovered surfaces)
    # ================================================================

    def _check_new_surface_objects(self):
        """Scan REACT SG for new surface-type objects not yet in our known set."""
        from depth_freespace import DEPTH_SUITABLE_TAGS
        react_sg = self.react_bridge.current_sg
        if not react_sg:
            return
        for key, obj in react_sg.items():
            obj_id = obj.get("id")
            if obj_id is None:
                continue
            if int(obj_id) in self._known_surface_ids:
                continue
            if not obj.get("_is_new", False):
                # Not a REACT-discovered object — was in original SG
                self._known_surface_ids.add(int(obj_id))
                continue
            tag = obj.get("object_tag", "").lower().strip()
            suitable = any(st in tag or tag in st for st in DEPTH_SUITABLE_TAGS)
            if not suitable:
                self._known_surface_ids.add(int(obj_id))
                continue
            center = obj.get("bbox_center")
            extent = obj.get("bbox_extent")
            if center is None or extent is None:
                continue
            # Check not already in queue or currently being observed
            already_queued = any(p["id"] == int(obj_id) for p in self._pending_surface_queue)
            currently_observing = (self._obs_surface_current is not None
                                   and self._obs_surface_current["id"] == int(obj_id))
            if already_queued or currently_observing:
                continue
            self._pending_surface_queue.append({
                "id": int(obj_id),
                "tag": tag,
                "center": [float(c) for c in center],
                "extent": [float(e) for e in extent],
            })
            self.log_chat(f"[Surface] New {tag}(id:{obj_id}) queued for observation")

    def _compute_observation_viewpoints(self, center, extent):
        """Compute 2 viewpoints near a surface for observation.

        Returns list of (x, y, yaw) — front and side, close to the surface.
        """
        import math
        cx, cy = float(center[0]), float(center[1])
        half_w = float(extent[0]) / 2.0 if extent else 0.5
        half_d = float(extent[1]) / 2.0 if extent else 0.5
        obs_radius = max(half_w, half_d) + 1.0  # 1.0m from edge

        viewpoints = []
        for angle_deg in [0.0, 90.0]:
            angle_rad = math.radians(angle_deg)
            vx = cx + obs_radius * math.cos(angle_rad)
            vy = cy + obs_radius * math.sin(angle_rad)
            face_yaw = math.atan2(cy - vy, cx - vx)
            viewpoints.append((vx, vy, face_yaw))
        return viewpoints

    def _start_surface_observation(self):
        """Pop next surface from queue and start navigating to first viewpoint."""
        if not self._pending_surface_queue:
            return False
        info = self._pending_surface_queue.pop(0)
        self._obs_surface_current = info
        self._obs_surface_viewpoints = self._compute_observation_viewpoints(
            info["center"], info["extent"])
        self._obs_surface_vp_idx = 0

        # Register surface temporarily with bbox-based Z (will be corrected after observation)
        if self._depth_fs_analyzer:
            sid = info["id"]
            cx, cy, cz = info["center"]
            ew, ed, eh = info["extent"]
            bbox_z = cz + eh / 2.0 - 0.02
            self._depth_fs_analyzer.register_surface(
                sid, bbox_z,
                cx - ew * 0.45, cx + ew * 0.45,
                cy - ed * 0.45, cy + ed * 0.45,
                info["tag"])

        # Navigate to first viewpoint
        vx, vy, vyaw = self._obs_surface_viewpoints[0]
        self.log_chat(f"[Surface] Observing {info['tag']}(id:{info['id']}) — viewpoint 1/3")
        cx, cy = info["center"][0], info["center"][1]
        self.navigator.set_goal([vx, vy], yaw=vyaw, object_pos=[cx, cy])
        self._obs_surface_nav_start = time.time()
        self._obs_surface_nav_failures = 0
        self.state = "NAV_TO_OBSERVE_SURFACE"
        return True

    def _finalize_surface_observation(self, info):
        """Finalize observation of a new surface: correct Z, mark known, move on."""
        sid = info["id"]
        if self._depth_fs_analyzer:
            refined_z = self._depth_fs_analyzer.finalize_observed_z(sid)
            if refined_z is not None:
                delta = refined_z - (info["center"][2] + info["extent"][2] / 2.0 - 0.02)
                self.log_chat(f"[Surface] {info['tag']}(id:{sid}) Z corrected: {refined_z:.3f}m (delta={delta:+.3f}m)")
            else:
                self.log_chat(f"[Surface] {info['tag']}(id:{sid}) Z detection failed — using bbox estimate")

        self._known_surface_ids.add(sid)
        self._obs_surface_current = None
        self._obs_surface_viewpoints = []

        # Start next pending surface or return to IDLE
        if self._pending_surface_queue:
            self._start_surface_observation()
        else:
            self.log_chat("[Surface] All new surfaces observed")
            self.state = "IDLE"

    def _send_react_frame(self):
        try:
            # get_data() must run on main thread (Isaac Sim API),
            # but copy to numpy is fast; send_frame writes to disk in background thread
            rgb_data = self.rgb_annot.get_data()
            depth_data = self.depth_annot.get_data()
            if rgb_data is None:
                return
            robot_pos, robot_quat = get_prim_transform(self.base_link_path)
            robot_yaw = quat_to_euler_z(robot_quat)
            self.react_bridge.send_frame(rgb_data, depth_data, robot_pos, robot_quat, robot_yaw)
        except Exception as e:
            self.react_status_lbl.text = f"Frame error: {str(e)[:50]}"

    def _poll_react_results(self):
        result = self.react_bridge.poll_results()
        if result is None: return
        new_objects = []
        for c in result.get("changes", []):
            if c["type"] == "MOVED":
                self.log_react(f"[MOVED] {c['tag']}(id:{c['obj_id']}) {c['distance']:.2f}m")
            elif c["type"] == "NEW":
                self.log_react(f"[NEW] {c['tag']} conf:{c['confidence']:.2f}")
                new_objects.append(c["tag"])
            elif c["type"] == "ABSENT":
                if c.get("_was_false_positive"):
                    self.log_react(f"[REVERTED] {c['tag']}(id:{c['obj_id']}) — false positive removed")
                else:
                    self.log_react(f"[ABSENT] {c['tag']}(id:{c['obj_id']})")

        # Check if new objects are surface-type → add to pending observation queue
        if self._depth_fs_enabled and self._depth_fs_analyzer and self.react_bridge:
            self._check_new_surface_objects()

        # Pred/Custom brain: check false positive reverts — only replan if
        # the reverted object is mentioned in current plan actions
        if (self._is_pred
                and self.current_instruction
                and self.state in ("EXECUTING", "NAVIGATING")
                and self._replan_cooldown < time.time()):
            reverted = self.brain.check_false_positive_reverts()
            if reverted:
                # Only replan if reverted object is relevant to current plan
                plan_text = " ".join(
                    str(a) for a in (self.plan_queue + ([self.current_action] if self.current_action else []))
                ).lower()
                relevant_reverts = [r for r in reverted if r.lower() in plan_text]
                if relevant_reverts:
                    reverted_str = ", ".join(relevant_reverts)
                    self.log_chat(f"[REACT] Plan-relevant object reverted: {reverted_str}. Re-planning...")
                    self._replan_cooldown = time.time() + 30
                    self.feedback = f"REVERTED_FALSE_DETECTION: {reverted_str} were false positives. Re-plan using only confirmed objects."
                    self.plan_queue = []
                    self.current_action = None
                    self.state = "THINKING"
                    return
                else:
                    self.log_react(f"[REACT] Reverted {', '.join(reverted)} (not relevant to current plan, skipping replan)")

        # Check if newly discovered objects are relevant to current instruction → replan (PRED only)
        # Only trigger for dynamic objects (not furniture — furniture gets observation queue instead)
        if (self._is_pred and new_objects and self.current_instruction
                and self.state in ("EXECUTING", "NAVIGATING")
                and self._replan_cooldown < time.time()):
            # Filter out semi_static/static tags — furniture discovery doesn't warrant replan
            from react_worker import MOBILITY
            semi_static_tags = set(MOBILITY.get("semi_static", {}).get("tags", []))
            static_tags = set(MOBILITY.get("static", {}).get("tags", []))
            skip_tags = semi_static_tags | static_tags
            dynamic_new = [t for t in new_objects if t.lower() not in skip_tags]
            if dynamic_new:
                self._check_new_object_relevance(dynamic_new)

        oc = result.get("object_count", 0)
        tc = result.get("total_changes", 0)
        fc = result.get("frame_count", 0)
        it = result.get("inference_time_ms", 0)
        self.sg_count_lbl.text = f"Objects: {oc} | Changes: {tc}"
        self.react_status_lbl.text = f"REACT Active | Frame #{fc} | {it}ms"

    def _check_new_object_relevance(self, new_tags):
        """Check if newly discovered objects are relevant to the current instruction.
        If so, trigger replanning to include them."""
        if not hasattr(self.brain, 'model'):
            return
        # Avoid repeated checks: only check once per 10 seconds
        if self._last_relevance_check > time.time() - 10:
            return
        self._last_relevance_check = time.time()

        instr = self.current_instruction
        new_str = ", ".join(new_tags)
        prompt = (
            f"Instruction: '{instr}'\n"
            f"Newly discovered objects: [{new_str}]\n"
            f"Are any of these new objects directly relevant to the instruction? "
            f"For example, if the instruction is 'wash all dishes' and a new 'bowl' appears, "
            f"that bowl is relevant. But if a 'chair' appears, it's not.\n"
            f"Answer ONLY 'RELEVANT: <object1>, <object2>' or 'NOT_RELEVANT'."
        )

        def _check():
            try:
                resp = self.brain.model.generate(prompt)
                text = resp.text.strip().upper()
                if text.startswith("RELEVANT"):
                    print(f"[UI] New relevant objects found: {new_str} for '{instr}'")
                    self._pending_replan_reason = f"FOUND_NEW_RELEVANT_OBJECT: Discovered {new_str} during execution."
            except Exception as e:
                print(f"[UI] Relevance check error: {e}")

        threading.Thread(target=_check, daemon=True).start()

    # ── Scene Graph manipulation for pick/place ──

    def _sg_mark_status(self, obj_id, status):
        """Mark an object as 'absent' or 'present' in brain + react_bridge SG."""
        obj_id_str = str(obj_id)
        try:
            for obj in self.brain.raw_data.values():
                if str(obj.get("id")) == obj_id_str:
                    if status == "absent":
                        obj["_status"] = "absent"
                    else:
                        obj.pop("_status", None)
                    break
        except RuntimeError:
            pass
        try:
            if self.react_bridge and self.react_bridge.current_sg:
                for obj in self.react_bridge.current_sg.values():
                    if str(obj.get("id")) == obj_id_str:
                        if status == "absent":
                            obj["_status"] = "absent"
                        else:
                            obj.pop("_status", None)
                        break
        except RuntimeError:
            pass

    def _notify_react_sg_update(self, obj_id, action, position=None):
        """Write SG update to /tmp so react_worker picks it up next frame."""
        import json as _json
        update_path = "/tmp/react_sg_update.json"
        try:
            obj_id_int = int(obj_id)
        except (ValueError, TypeError):
            return
        # Append to existing updates (multiple actions in one frame possible)
        updates = []
        try:
            if os.path.exists(update_path):
                with open(update_path, 'r') as f:
                    updates = _json.load(f)
        except Exception:
            updates = []
        entry = {"obj_id": obj_id_int, "action": action}
        if position is not None:
            entry["position"] = position
        updates.append(entry)
        try:
            with open(update_path, 'w') as f:
                _json.dump(updates, f)
        except Exception as e:
            print(f"[UI] Failed to write react SG update: {e}")

    def _sg_update_position(self, obj_id, new_pos):
        """Update an object's position in brain + react_bridge SG.
        Sets _manual_pos flag so refresh_from_react won't overwrite with stale REACT data."""
        obj_id_str = str(obj_id)
        pos_list = [float(new_pos[0]), float(new_pos[1]), float(new_pos[2])]
        try:
            for obj in self.brain.raw_data.values():
                if str(obj.get("id")) == obj_id_str:
                    obj["bbox_center"] = pos_list
                    obj["_manual_pos"] = True
                    break
        except RuntimeError:
            pass
        try:
            if self.react_bridge and self.react_bridge.current_sg:
                for obj in self.react_bridge.current_sg.values():
                    if str(obj.get("id")) == obj_id_str:
                        obj["bbox_center"] = pos_list
                        obj["_manual_pos"] = True
                        break
        except RuntimeError:
            pass

    def _pred_pick_closest_duplicate(self, target_id):
        """PRED: If multiple objects share the same tag, pick the closest one to the robot."""
        target_name = self.brain.get_obj_name_by_id(target_id)
        if not target_name or target_name == "Unknown":
            return target_id

        # Find all objects with same tag that are present and not already picked this plan
        candidates = []
        try:
            for info in self.brain.raw_data.values():
                tag = info.get("object_tag", "").lower()
                status = info.get("_status", "present")
                oid = str(info.get("id", ""))
                if (tag == target_name.lower() and status == "present"
                        and oid not in self._picked_this_plan):
                    pos = info.get("bbox_center", info.get("center"))
                    if pos:
                        candidates.append((oid, np.array(pos)))
        except RuntimeError:
            return target_id

        if len(candidates) <= 1:
            return target_id  # no alternatives

        # Get robot position and find closest
        robot_pos, _ = get_prim_transform(self.robot_path)
        robot_xy = robot_pos[:2]

        best_id = target_id
        best_dist = float('inf')
        for cid, cpos in candidates:
            dist = float(np.linalg.norm(cpos[:2] - robot_xy))
            if dist < best_dist:
                best_dist = dist
                best_id = cid

        if best_id != str(target_id):
            self.log_chat(f"[PRED] Multiple '{target_name}' found — picking closest (id:{best_id}, {best_dist:.1f}m)")
        return best_id

    def _parse_plan(self, plan):
        actions = []
        for s in plan:
            if "pick" in s and "(" in s:
                actions.append({"type": "pick", "target": s.split("(")[1].split(")")[0]})
            elif "place" in s and "(" in s:
                actions.append({"type": "place", "target": s.split("(")[1].split(")")[0]})
            elif "open" in s and "(" in s:
                actions.append({"type": "open", "target": s.split("(")[1].split(")")[0]})
            elif "close" in s and "(" in s:
                actions.append({"type": "close", "target": s.split("(")[1].split(")")[0]})
            elif "goto" in s:
                actions.append({"type": "goto", "target": s.split("(")[1].split(")")[0]})
            elif "done" in s:
                actions.append({"type": "done", "target": None})
        return actions

    SG_VIZ_FILE = "/tmp/nav2_sg_viz.json"

    def _export_sg_for_rviz(self):
        """Export scene graph objects to file for RViz MarkerArray visualization (non-blocking)."""
        if getattr(self, '_sg_viz_writing', False):
            return
        try:
            # Build data on main thread (fast dict iteration)
            objects = []
            for key, obj in self.brain.raw_data.items():
                if obj.get("_status", "present") != "present":
                    continue
                center = obj.get("bbox_center", obj.get("center"))
                extent = obj.get("bbox_extent", [0.3, 0.3, 0.3])
                tag = obj.get("object_tag", key)
                obj_id = obj.get("id", 0)
                if center is None:
                    continue
                odom_xy = self.navigator._isaac_to_odom(center)
                objects.append({
                    "id": obj_id, "tag": tag,
                    "x": round(float(odom_xy[0]), 3),
                    "y": round(float(odom_xy[1]), 3),
                    "z": round(float(center[2]), 3),
                    "ex": round(float(extent[0]), 3),
                    "ey": round(float(extent[1]), 3),
                    "ez": round(float(extent[2]), 3),
                    "is_new": bool(obj.get("_is_new", False)),
                })

            # Free space: only use depth-based (precomputed + live).
            # Old SG-based free space is no longer exported.
            depth_free_spaces = self._depth_fs_cached_rviz if self._depth_fs_enabled else []

            # File write in background thread
            self._sg_viz_writing = True
            data = {
                "objects": objects,
                "free_spaces": depth_free_spaces,
                "depth_free_spaces": depth_free_spaces,
                "t": time.time(),
            }
            def _write():
                try:
                    tmp = self.SG_VIZ_FILE + ".tmp"
                    with open(tmp, 'w') as f:
                        json.dump(data, f)
                    os.replace(tmp, self.SG_VIZ_FILE)
                except Exception:
                    pass
                self._sg_viz_writing = False
            threading.Thread(target=_write, daemon=True).start()
        except Exception:
            self._sg_viz_writing = False

    def cleanup(self):
        self._save_session_log()
        try:
            os.remove(self.SG_VIZ_FILE)
        except Exception:
            pass
        print(f"[Log] Session saved: {len(self._session_log)} entries "
              f"-> logs/session_{self._session_start}.json")
        try: self.rgb_annot.detach()
        except: pass
        try: self.depth_annot.detach()
        except: pass
        try: self.viewport.destroy()
        except: pass
