"""
SayPlanBrain - LLM-based task planner with pick/place support
"""
import os
import re
import json
import numpy as np

from config import HIERARCHY_JSON_FILE, EDGE_JSON_FILE


def _repair_json(text: str) -> dict:
    """Try to parse JSON with common LLM output fixes."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r'//[^\n]*', '', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    fixed2 = text.replace("'", '"')
    fixed2 = re.sub(r',\s*([}\]])', r'\1', fixed2)
    return json.loads(fixed2)

class SayPlanBrain:
    def __init__(self, json_path, react_bridge=None, llm=None):
        self.raw_data = {}
        self.full_graph_str = ""
        self.collapsed_graph_str = ""
        self.current_graph_state = ""
        self.memory = []
        self.action_history = []
        self.holding_object = None  # id or name of the object currently held
        self.react_bridge = react_bridge

        # Load hierarchy (room structure) and edge relations
        self.hierarchy = {}
        self.edges = []
        self.room_map = {}  # obj_index → room_label
        self._load_hierarchy_and_edges()

        self._load_and_transform_scene_graph(json_path)
        self.model = llm
        self.system_prompt = self._build_system_prompt()

    def get_nav_approach_info(self, target_id):
        """Determine approach position for navigation. Direct coordinate-based, no edge lookup."""
        target_pos = self.get_coords_by_id(target_id)
        target_size = self.get_obj_size_by_id(target_id)

        if target_pos is None:
            return None

        arrival_dist = target_size / 2.0 + 0.5
        arrival_dist = max(arrival_dist, 0.6)
        return {
            "nav_pos": target_pos,
            "face_pos": target_pos,
            "arrival_dist": arrival_dist,
            "parent_name": None,
            "parent_id": None,
        }

    def _load_hierarchy_and_edges(self):
        """Load hierarchy (room assignments) and edge relations from Gemini v2 data."""
        # Hierarchy → room_map: index → room_label
        if os.path.exists(HIERARCHY_JSON_FILE):
            with open(HIERARCHY_JSON_FILE, 'r') as f:
                self.hierarchy = json.load(f)
            # Parse building→floor→room→object
            for floor in self.hierarchy.get("children", []):
                for room in floor.get("children", []):
                    room_label = room.get("room_label", "unknown_room")
                    for obj in room.get("children", []):
                        if obj.get("type") == "object":
                            self.room_map[obj["index"]] = room_label
            print(f"[SayPlan] Loaded hierarchy: {len(self.room_map)} objects in rooms: "
                  f"{set(self.room_map.values())}")

        # Edges → object relations
        if os.path.exists(EDGE_JSON_FILE):
            with open(EDGE_JSON_FILE, 'r') as f:
                edge_data = json.load(f)
            self.edges = edge_data.get("edges", [])
            print(f"[SayPlan] Loaded {len(self.edges)} edge relations")

    def _load_and_transform_scene_graph(self, path):
        if not os.path.exists(path): return
        with open(path, 'r') as f: self.raw_data = json.load(f)
        self._rebuild_graph_strings(self.raw_data)

    def _rebuild_graph_strings(self, data):
        # Collect unique rooms from hierarchy
        room_labels = sorted(set(self.room_map.values())) if self.room_map else ["main_area"]
        nodes = {"room": [{"id": r, "type": "room"} for r in room_labels], "asset": []}
        links_list = []

        for key, info in data.items():
            obj_id = info.get("id")
            tag = info.get("object_tag", "unknown")
            bbox_center = info.get("bbox_center", info.get("center", [0, 0, 0]))
            desc = info.get("caption", info.get("object_caption", ""))
            desc = str(desc)[:50] if desc else ""
            status = info.get("_status", "present")

            # Assign room from hierarchy (fall back to first room)
            room = self.room_map.get(obj_id, room_labels[0])

            nodes["asset"].append({
                "id": str(obj_id), "name": tag, "type": "asset",
                "room": room, "state": status,
                "affordances": ["goto", "pick", "place", "inspect"],
                "attributes": desc, "position": bbox_center
            })
            links_list.append(f"{room} <-> {obj_id}")

        # Add edge relations (spatial relationships between objects)
        for edge in self.edges:
            src_idx = edge.get("src_index")
            dst_idx = edge.get("dst_index")
            relation = edge.get("object_relation", "near")
            src_tag = edge.get("src_tag", "?")
            dst_tag = edge.get("dst_tag", "?")
            links_list.append(f"{src_idx}({src_tag}) --[{relation}]--> {dst_idx}({dst_tag})")

        self.full_graph_str = json.dumps({"nodes": nodes, "links": links_list})
        self.collapsed_graph_str = json.dumps({
            "nodes": {"room": [{"id": r, "type": "room"} for r in room_labels]},
            "links": []})

    def refresh_from_react(self, coords_only=False):
        """Sync scene graph from REACT worker.
        coords_only=True: only update raw_data (skip expensive json.dumps rebuild)."""
        if self.react_bridge is None or self.react_bridge.current_sg is None: return
        try:
            self.raw_data = {k: dict(v) for k, v in self.react_bridge.current_sg.items()}
        except RuntimeError:
            return
        if not coords_only:
            self._rebuild_graph_strings(self.raw_data)
            if self.current_graph_state != self.collapsed_graph_str:
                self.current_graph_state = self.full_graph_str

    def _build_system_prompt(self):
        # Build edge relation summary for prompt
        edge_summary = ""
        if self.edges:
            edge_lines = []
            for e in self.edges:
                edge_lines.append(f"  - {e['src_tag']}(id:{e['src_index']}) {e['object_relation']} {e['dst_tag']}(id:{e['dst_index']})")
            edge_summary = "Known spatial relations:\n" + "\n".join(edge_lines)

        room_list = sorted(set(self.room_map.values())) if self.room_map else ["main_area"]

        return f"""You are a SayPlan graph planning agent controlling a mobile manipulator robot (Jackal base + Franka arm).

You operate in TWO PHASES:
1. EXPLORING: Expand room nodes to discover objects inside. You MUST expand a room before you can plan with its objects.
2. PLANNING: Once you see the objects, generate a plan using the available actions.

Available rooms: {room_list}

{edge_summary}

Use spatial relations to understand object context:
- "a on b" means a is resting on b's surface (pick a from b, or place onto b)
- "a in b" means a is inside b (may need to open b first)
- "a supporting b" means a holds b up
- "a attached to b" means a is fixed to b (cannot be picked)
- "a next to b" means a and b are side by side

Environment Functions:
- goto(<asset_id>): Move the robot base to an asset.
- pick(<asset_id>): Pick up the asset with the robot arm. Must goto() first.
- place(<asset_id>): Place the held object at/on target asset. Must goto() first.
- open(<asset_id>): Open a door/drawer/refrigerator. Must goto() first.
- close(<asset_id>): Close a door/drawer/refrigerator. Must goto() first.
- done(): Task completed.
- expand_node(<room_id>): Reveal objects inside a room. Must do this FIRST.
- contract_node(<room_id>): Hide objects in a room.

IMPORTANT RULES:
- Check your CURRENT ROBOT STATE. If already holding the object, just goto destination and place.
- Always goto() BEFORE pick/place/open/close.
- Use open/close for interactive furniture: refrigerator, cabinet, drawer, door, oven, etc.
- If an object is "in" a container, open the container first before picking.
- If an object is "attached to" something, it CANNOT be picked up.
- Use edge relations to decide if you need open() before pick().

Examples:
- "move plate to counter" (plate is on desk): ["goto(13)", "pick(13)", "goto(9)", "place(9)", "done()"]
- "put bowl in refrigerator" (bowl on cabinet): ["goto(14)", "pick(14)", "goto(11)", "open(11)", "place(11)", "close(11)", "done()"]

Output Format (Respond ONLY with valid, raw JSON. Do not include markdown code blocks like ```json):
{{
    "chain_of_thought": "Reasoning using spatial relations and room structure...",
    "mode": "exploring" OR "planning",
    "command": {{
        "command_name": "expand_node" OR "contract_node" OR "plan",
        "node_name": "room_id_for_expand",
        "plan": ["goto(13)", "pick(13)", "goto(9)", "place(9)", "done()"]
    }}
}}

The scene graph is LIVE and updated by vision. Objects may have moved, appeared, or disappeared.
NOTE: Start with 'Collapsed Graph'. You MUST expand_node a room first before planning.
"""

    def _call_with_retry(self, prompt, max_retries=5):
        """LLM API 호출 (retry 로직은 llm_wrapper에 내장)."""
        return self.model.generate(prompt, max_retries=max_retries)

    def process_turn(self, user_instruction, simulator_feedback=""):
        self.refresh_from_react()
        change_summary = self.react_bridge.get_change_summary() if self.react_bridge else ""
        
        # Format action history for prompt
        history_str = "None"
        if self.action_history:
            history_str = "\n".join([f"User: {h['user_instruction']} -> Plan: {h['plan']}" for h in self.action_history[-5:]])

        holding_str = "Nothing"
        if self.holding_object:
            holding_name = self.get_obj_name_by_id(self.holding_object)
            holding_str = f"'{holding_name}' (ID: {self.holding_object})"
            
        dynamic_input = f"""
[Current Robot State]: Holding {holding_str}
[Action History]:
{history_str}
[User Instruction]: {user_instruction}
[Current Graph State]: {self.current_graph_state}
[Scene Changes]: {change_summary}
[Memory/History]: {str(self.memory)}
[Simulator Feedback]: {simulator_feedback}
"""
        full_prompt = self.system_prompt + "\n" + "-" * 20 + "\n" + dynamic_input
        try:
            response = self._call_with_retry(full_prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            if "{" in text: text = text[text.find("{"):text.rfind("}") + 1]
            try:
                parsed_resp = json.loads(text)
            except json.JSONDecodeError:
                print(f"[SAYPLAN] JSON parse failed, attempting repair...")
                parsed_resp = _repair_json(text)
            
            # Record plan into history if planning mode
            if parsed_resp.get("mode") == "planning":
                plan = parsed_resp.get("command", {}).get("plan", [])
                if plan:
                    self.action_history.append({
                        "user_instruction": user_instruction,
                        "plan": plan
                    })
            return parsed_resp
        except Exception as e:
            return {"mode": "error", "reasoning": str(e), "command": {}}

    def execute_graph_api(self, command_name, node_name):
        room_labels = sorted(set(self.room_map.values())) if self.room_map else ["main_area"]
        if command_name == "expand_node":
            # Accept any room name or "main_area" as expand target
            if node_name in room_labels or node_name == "main_area":
                self.current_graph_state = self.full_graph_str
                if node_name not in self.memory:
                    self.memory.append(node_name)
                return f"Graph Expanded (room: {node_name})."
        elif command_name == "contract_node":
            self.current_graph_state = self.collapsed_graph_str
            return "Graph Contracted."
        return "No graph operation needed."

    def get_coords_by_id(self, asset_id):
        for info in self.raw_data.values():
            if str(info.get("id")) == str(asset_id):
                return np.array(info.get("bbox_center", info.get("center")))
        return None

    def get_obj_name_by_id(self, asset_id):
        for info in self.raw_data.values():
            if str(info.get("id")) == str(asset_id):
                return info.get("object_tag", "Unknown")
        return "Unknown"

    def get_obj_size_by_id(self, asset_id):
        for info in self.raw_data.values():
            if str(info.get("id")) == str(asset_id):
                ext = info.get("bbox_extent", [0.5, 0.5, 0.5])
                return max(ext[0], ext[1])
        return 0.5

    def get_obj_surface_z(self, asset_id):
        for info in self.raw_data.values():
            if str(info.get("id")) == str(asset_id):
                center = info.get("bbox_center", info.get("center", [0, 0, 1.0]))
                ext = info.get("bbox_extent", [0.5, 0.5, 0.5])
                return center[2] + ext[2] / 2.0
        return 1.0

    def verify_visual(self, image_np, target_name):
        return "YES"