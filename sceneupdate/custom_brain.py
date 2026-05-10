"""
CustomBrain - Custom task planner (based on PRED framework)

Started as a copy of PREDBrain. Add custom ideas and mechanisms here.
Base mechanisms inherited from PRED:
  - DTA (Dynamic Target Adaptation): Find substitute when target missing/absent
  - APM (Attribute-Driven Plan Modification): Modify plan based on object attributes
  - ASR (Action Skipping by Relationship): Skip actions if goal already met
"""
import os
import re
import json
import numpy as np

from config import EDGE_JSON_FILE, HIERARCHY_JSON_FILE, SURFACE_JSON_FILE, ROOMS_DIR, ROOMS_INDEX_FILE
from surface_analyzer_v2 import SurfaceAnalyzer  # V2: grid-based (test)


def _repair_json(text: str) -> dict:
    """Try to parse JSON with common LLM output fixes."""
    # First try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Remove trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Try removing single-line // comments
    fixed = re.sub(r'//[^\n]*', '', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Try replacing single quotes with double quotes
    fixed2 = text.replace("'", '"')
    fixed2 = re.sub(r',\s*([}\]])', r'\1', fixed2)
    return json.loads(fixed2)


class CustomBrain:
    def __init__(self, json_path, react_bridge=None, llm=None):
        self.raw_data = {}           # objects currently visible to LLM (expanded rooms only)
        self.full_graph_str = ""
        self.collapsed_graph_str = ""
        self.current_graph_state = ""
        self.memory = []
        self.action_history = []
        self.holding_object = None
        self.react_bridge = react_bridge

        # Load hierarchy (room structure) and edge relations
        self.hierarchy = {}
        self.edges = []
        self.room_map = {}  # obj_id → room_instance_name
        self._edge_idx_to_id = {}
        self._load_hierarchy_and_edges()

        # Per-room lazy loading
        self.room_index = []         # list of room summary dicts from _index.json
        self.expanded_rooms = {}     # instance_name → list of object dicts
        self._load_room_index()

        self.surface_analyzer = SurfaceAnalyzer()
        if os.path.exists(SURFACE_JSON_FILE):
            self.surface_analyzer.load_from_json(SURFACE_JSON_FILE)
            print(f"[CUSTOM] Loaded surface JSON: {SURFACE_JSON_FILE}")

        self._pending_edge_objects = set()

        # Start with collapsed view (no objects loaded yet)
        self._rebuild_graph_strings(self.raw_data)
        self.model = llm
        self.system_prompt = self._build_system_prompt()

    # ================================================================
    # Scene graph & edge loading (same as SayPlanBrain)
    # ================================================================

    def _load_hierarchy_and_edges(self):
        """Load hierarchy (room assignments) and edge relations from Gemini v2 data."""
        # Hierarchy → room_map: index → room_label
        if os.path.exists(HIERARCHY_JSON_FILE):
            with open(HIERARCHY_JSON_FILE, 'r') as f:
                self.hierarchy = json.load(f)
            # Support both old list format and new building/dict format
            top = self.hierarchy
            if "building" in top:
                # New format: {"building": {"children": {floor_id: {children: {room_id: {children: {obj_key: obj}}}}}}}
                floors = top["building"].get("children", {})
                floors_iter = floors.values() if isinstance(floors, dict) else floors
                for floor in floors_iter:
                    rooms = floor.get("children", {})
                    rooms_iter = rooms.values() if isinstance(rooms, dict) else rooms
                    for room in rooms_iter:
                        room_label = room.get("name", room.get("room_label", "unknown_room"))
                        objs = room.get("children", {})
                        objs_iter = objs.values() if isinstance(objs, dict) else objs
                        for obj in objs_iter:
                            if obj.get("type") == "object":
                                obj_id = obj.get("id", obj.get("index"))
                                if obj_id is not None:
                                    self.room_map[obj_id] = room_label
            else:
                # Old format: {"children": [floors...]}
                for floor in top.get("children", []):
                    for room in floor.get("children", []):
                        room_label = room.get("room_label", "unknown_room")
                        for obj in room.get("children", []):
                            if obj.get("type") == "object":
                                self.room_map[obj["index"]] = room_label
            print(f"[CUSTOM] Loaded hierarchy: {len(self.room_map)} objects in rooms: "
                  f"{set(self.room_map.values())}")

        # Edges → object relations
        if os.path.exists(EDGE_JSON_FILE):
            try:
                with open(EDGE_JSON_FILE, 'r') as f:
                    data = json.load(f)
                self.edges = data.get("edges", data if isinstance(data, list) else [])
                # detection_id → obj_id fallback (이전 포맷 호환)
                for edge in self.edges:
                    if "src_obj_id" not in edge and "src_detection_id" in edge:
                        edge["src_obj_id"] = edge["src_detection_id"]
                    if "dst_obj_id" not in edge and "dst_detection_id" in edge:
                        edge["dst_obj_id"] = edge["dst_detection_id"]
                print(f"[CUSTOM] Loaded {len(self.edges)} edge relations")
            except Exception as e:
                print(f"[CUSTOM] Edge load error: {e}")

    def _load_room_index(self):
        """Load _index.json for room-level summary."""
        if not os.path.exists(ROOMS_INDEX_FILE):
            print(f"[CUSTOM] No room index found at {ROOMS_INDEX_FILE}")
            return
        with open(ROOMS_INDEX_FILE) as f:
            data = json.load(f)
        self.room_index = data.get("rooms", [])
        print(f"[CUSTOM] Room index loaded: {[r['instance_name'] for r in self.room_index]}")

    def _expand_room(self, instance_name):
        """Load a room file and merge its objects into raw_data. Returns object count."""
        room_file = os.path.join(ROOMS_DIR, f"{instance_name}.json")
        if not os.path.exists(room_file):
            # Try matching by room_name prefix
            for r in self.room_index:
                if r["room_name"] == instance_name or r["instance_name"] == instance_name:
                    room_file = os.path.join(ROOMS_DIR, r["file"])
                    instance_name = r["instance_name"]
                    break
        if not os.path.exists(room_file):
            return 0
        if instance_name in self.expanded_rooms:
            return len(self.expanded_rooms[instance_name])
        with open(room_file) as f:
            room_data = json.load(f)
        objects = room_data.get("objects", [])
        self.expanded_rooms[instance_name] = objects
        # Merge into raw_data (obj_json format)
        for obj in objects:
            key = f"object_{obj['id']}"
            self.raw_data[key] = {
                "id":            obj["id"],
                "object_tag":    obj["object_tag"],
                "object_caption": obj.get("object_caption"),
                "bbox_center":   obj["bbox_center"],
                "bbox_extent":   obj["bbox_extent"],
                "bbox_volume":   obj.get("bbox_volume", 0.0),
            }
        self._rebuild_graph_strings(self.raw_data)
        return len(objects)

    def _build_current_graph_str(self):
        """Build graph string: expanded rooms show objects, others show summary."""
        expanded_names = set(self.expanded_rooms.keys())
        rooms_view = []
        for r in self.room_index:
            iname = r["instance_name"]
            if iname in expanded_names:
                rooms_view.append({
                    "id": iname, "type": "room", "status": "expanded",
                    "n_objects": len(self.expanded_rooms[iname]),
                    "adjacent_rooms": r.get("adjacent_rooms", []),
                })
            else:
                rooms_view.append({
                    "id": iname, "type": "room", "status": "collapsed",
                    "n_objects": r["n_objects"],
                    "adjacent_rooms": r.get("adjacent_rooms", []),
                })
        # Build asset nodes from expanded rooms only
        asset_nodes = []
        links_list = []
        room_labels = sorted(set(self.room_map.values())) if self.room_map else []
        for key, info in self.raw_data.items():
            obj_id = info.get("id")
            tag = info.get("object_tag", "unknown")
            status = info.get("_status", "present")
            room = self.room_map.get(obj_id, room_labels[0] if room_labels else "unknown")
            asset_nodes.append({
                "id": str(obj_id), "name": tag, "type": "asset",
                "room": room, "state": status,
                "position": info.get("bbox_center", [0, 0, 0]),
            })
            links_list.append(f"{room} <-> {obj_id}")
        return json.dumps({
            "rooms": rooms_view,
            "objects": asset_nodes,
            "links": links_list,
        })

    def _build_edge_idx_map(self):
        """Build index→id map from raw_data for edge lookups."""
        self._edge_idx_to_id = {}
        for key in self.raw_data:
            try:
                pos_idx = int(key.split('_')[1]) - 1
                self._edge_idx_to_id[pos_idx] = self.raw_data[key]["id"]
            except (IndexError, ValueError):
                pass

    def _edge_idx_to_obj_id(self, edge_idx):
        return self._edge_idx_to_id.get(edge_idx)

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

        # Track REACT-added objects that need LLM edge generation
        edge_obj_ids = set()
        for edge in self.edges:
            edge_obj_ids.add(edge.get("src_obj_id"))
            edge_obj_ids.add(edge.get("dst_obj_id"))
            edge_obj_ids.add(edge.get("src_index"))
            edge_obj_ids.add(edge.get("dst_index"))
        for key, info in data.items():
            if not info.get("_is_new"):
                continue
            obj_id = info.get("id")
            if obj_id in edge_obj_ids:
                continue
            # Queue for LLM edge generation (will be done during idle)
            self._pending_edge_objects.add(obj_id)

        self.full_graph_str = json.dumps({"nodes": nodes, "links": links_list})
        # Collapsed view: show room index if available, else just room labels
        if self.room_index:
            self.collapsed_graph_str = json.dumps({
                "rooms": [{"id": r["instance_name"], "type": "room",
                           "status": "collapsed", "n_objects": r["n_objects"],
                           "adjacent_rooms": r.get("adjacent_rooms", [])}
                          for r in self.room_index],
                "objects": []})
        else:
            self.collapsed_graph_str = json.dumps({
                "nodes": {"room": [{"id": r, "type": "room"} for r in room_labels]},
                "links": []})
        # Initial state is always collapsed
        if not self.current_graph_state:
            self.current_graph_state = self.collapsed_graph_str

        # Surface analyzer is updated via idle_update() and saved to JSON.
        # Only load from JSON here (no real-time computation).
        if os.path.exists(SURFACE_JSON_FILE):
            self.surface_analyzer.load_from_json(SURFACE_JSON_FILE)

    def check_false_positive_reverts(self):
        """Check if any REACT-added objects have been removed (false positives).

        Compares brain's raw_data with react_bridge's current_sg.
        If a _is_new object disappeared from react_bridge, remove it from raw_data
        and rebuild graph strings.

        Returns list of reverted object tags, or empty list.
        """
        if self.react_bridge is None or self.react_bridge.current_sg is None:
            return []

        bridge_ids = set()
        try:
            for obj in self.react_bridge.current_sg.values():
                bridge_ids.add(str(obj.get("id")))
        except RuntimeError:
            return []

        reverted = []
        keys_to_remove = []
        for key, obj in self.raw_data.items():
            if obj.get("_is_new") and str(obj.get("id")) not in bridge_ids:
                tag = obj.get("object_tag", "unknown")
                print(f"[CUSTOM] False positive reverted: '{tag}'(id:{obj.get('id')})")
                reverted.append(tag)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.raw_data[key]

        if reverted:
            self._rebuild_graph_strings(self.raw_data)
            if self.expanded_rooms:
                self.current_graph_state = self._build_current_graph_str()

        return reverted

    def refresh_from_react(self, coords_only=False):
        """Sync scene graph from REACT worker.
        coords_only=True: only update coords (skip expensive json.dumps rebuild).
        Preserves manual overrides (_status, _manual_pos) so pick/place changes
        aren't overwritten by stale REACT detections."""
        if self.react_bridge is None or self.react_bridge.current_sg is None: return

        # Save manual overrides before merging
        manual_overrides = {}
        for key, obj in self.raw_data.items():
            obj_id = str(obj.get("id"))
            overrides = {}
            if obj.get("_status") == "absent":
                overrides["_status"] = "absent"
            if obj.get("_manual_pos"):
                overrides["_manual_pos"] = True
                overrides["bbox_center"] = obj.get("bbox_center")
            if overrides:
                manual_overrides[obj_id] = overrides

        # Shallow copy from react
        try:
            self.raw_data = {k: dict(v) for k, v in self.react_bridge.current_sg.items()}
        except RuntimeError:
            return

        # Re-apply manual overrides (position, status)
        for key, obj in self.raw_data.items():
            obj_id = str(obj.get("id"))
            if obj_id in manual_overrides:
                ov = manual_overrides[obj_id]
                if "_status" in ov:
                    obj["_status"] = ov["_status"]
                if ov.get("_manual_pos") and "bbox_center" in ov:
                    obj["bbox_center"] = ov["bbox_center"]
                    obj["_manual_pos"] = True

        if not coords_only:
            self._rebuild_graph_strings(self.raw_data)
            if self.expanded_rooms:
                self.current_graph_state = self._build_current_graph_str()
        else:
            # In coords_only mode, just reload surface JSON (updated by idle_update)
            if os.path.exists(SURFACE_JSON_FILE):
                self.surface_analyzer.load_from_json(SURFACE_JSON_FILE)

    # ================================================================
    # PRED Core: ASR - Action Skipping by Relationship
    # ================================================================

    def _get_object_relationships(self, obj_id):
        """Get spatial relationships for an object from edge data."""
        rels = {"on": [], "in": [], "supports": [], "attached": [], "next_to": []}
        obj_id_int = int(obj_id) if str(obj_id).isdigit() else None
        if obj_id_int is None:
            return rels
        for edge in self.edges:
            src_idx = edge.get("src_index")
            dst_idx = edge.get("dst_index")
            relation = edge.get("object_relation", "")
            if src_idx == obj_id_int:
                if "on" in relation and "a" in relation.split()[0]:
                    rels["on"].append(dst_idx)
                elif "in" in relation and "a" in relation.split()[0]:
                    rels["in"].append(dst_idx)
                elif "supporting" in relation:
                    rels["supports"].append(dst_idx)
                elif "attached" in relation:
                    rels["attached"].append(dst_idx)
                elif "next to" in relation:
                    rels["next_to"].append(dst_idx)
            elif dst_idx == obj_id_int:
                if "on" in relation and "b" in relation.split()[0]:
                    rels["on"].append(src_idx)
                elif "b on a" in relation:
                    rels["on"].append(src_idx)  # b is on a, so src is on dst
                elif "in" in relation and "b" in relation.split()[0]:
                    rels["in"].append(src_idx)
                elif "supporting" in relation and "b" in relation.split()[0]:
                    rels["supports"].append(src_idx)
                elif "next to" in relation:
                    rels["next_to"].append(src_idx)
        return rels

    def _check_asr(self, obj_id, dest_id):
        """ASR: Check if object is already at/on/in destination.

        Returns (should_skip, reason_str).
        """
        if obj_id is None or dest_id is None:
            return False, ""

        rels = self._get_object_relationships(int(obj_id))
        dest_id_int = int(dest_id)

        # Object is already ON destination
        if dest_id_int in rels["on"]:
            obj_name = self.get_obj_name_by_id(obj_id)
            dest_name = self.get_obj_name_by_id(dest_id)
            reason = (f"ASR: '{obj_name}'(id={obj_id}) is already ON "
                      f"'{dest_name}'(id={dest_id}). Skip pick/place.")
            print(f"[CUSTOM] {reason}")
            return True, reason

        # Object is already IN destination
        if dest_id_int in rels["in"]:
            obj_name = self.get_obj_name_by_id(obj_id)
            dest_name = self.get_obj_name_by_id(dest_id)
            reason = (f"ASR: '{obj_name}'(id={obj_id}) is already IN "
                      f"'{dest_name}'(id={dest_id}). Skip pick/place.")
            print(f"[CUSTOM] {reason}")
            return True, reason

        # Check proximity: if object and destination are very close
        obj_pos = self.get_coords_by_id(obj_id)
        dest_pos = self.get_coords_by_id(dest_id)
        if obj_pos is not None and dest_pos is not None:
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(dest_pos[:2]))
            dest_size = self.get_obj_size_by_id(dest_id)
            if dist < dest_size * 0.5:
                obj_name = self.get_obj_name_by_id(obj_id)
                dest_name = self.get_obj_name_by_id(dest_id)
                reason = (f"ASR: '{obj_name}' is already very close to "
                          f"'{dest_name}' (dist={dist:.2f}m). Skip pick/place.")
                print(f"[CUSTOM] {reason}")
                return True, reason

        return False, ""

    # ================================================================
    # PRED Core: DTA - Dynamic Target Adaptation
    # ================================================================

    def _find_object_by_name(self, name_query):
        """Find object IDs matching a name query (case-insensitive partial match)."""
        query_lower = name_query.lower().strip()
        matches = []
        for info in self.raw_data.values():
            tag = info.get("object_tag", "").lower()
            status = info.get("_status", "present")
            if query_lower in tag and status == "present":
                matches.append({
                    "id": info["id"],
                    "tag": info.get("object_tag"),
                    "pos": info.get("bbox_center", info.get("center")),
                    "size": max(info.get("bbox_extent", [0.5, 0.5, 0.5])[:2]),
                })
        return matches

    def _find_substitute(self, target_id, category_hint=None):
        """DTA: Find a substitute object when target is missing/absent.

        Looks for objects with similar tags or in the same category.
        Returns (substitute_id, reason) or (None, reason).
        """
        target_name = self.get_obj_name_by_id(target_id)
        target_status = None
        for info in self.raw_data.values():
            if str(info.get("id")) == str(target_id):
                target_status = info.get("_status", "present")
                break

        if target_status == "present":
            return None, ""  # No substitution needed

        print(f"[CUSTOM-DTA] Target '{target_name}'(id={target_id}) "
              f"status='{target_status}', searching substitutes...")

        # Category mapping for common household objects
        _CATEGORY_MAP = {
            "cup": ["mug", "glass", "cup"],
            "mug": ["cup", "glass", "mug"],
            "glass": ["cup", "mug", "glass"],
            "plate": ["dish", "tray", "plate"],
            "dish": ["plate", "tray", "dish"],
            "bowl": ["plate", "dish", "bowl"],
            "knife": ["cutter", "blade", "knife"],
            "spoon": ["fork", "ladle", "spoon"],
            "fork": ["spoon", "fork"],
            "chair": ["stool", "seat", "chair"],
            "stool": ["chair", "seat", "stool"],
            "table": ["desk", "counter", "table"],
            "desk": ["table", "counter", "desk"],
        }

        # Build search terms
        search_terms = [target_name.lower()]
        for key, synonyms in _CATEGORY_MAP.items():
            if target_name.lower() in synonyms or key == target_name.lower():
                search_terms = synonyms
                break

        if category_hint:
            search_terms.append(category_hint.lower())

        # Search for substitutes
        candidates = []
        for term in search_terms:
            for info in self.raw_data.values():
                if str(info.get("id")) == str(target_id):
                    continue  # Skip the original
                tag = info.get("object_tag", "").lower()
                status = info.get("_status", "present")
                if term in tag and status == "present":
                    candidates.append({
                        "id": info["id"],
                        "tag": info.get("object_tag"),
                        "pos": info.get("bbox_center", info.get("center")),
                    })

        if not candidates:
            reason = (f"DTA: '{target_name}'(id={target_id}) is {target_status} "
                      f"and no substitute found. Proceeding with active search...")
            print(f"[CUSTOM] {reason}")
            return target_id, reason

        # Pick the first match (could be enhanced with distance-based ranking)
        sub = candidates[0]
        reason = (f"DTA: '{target_name}'(id={target_id}) is {target_status}. "
                  f"Substituting with '{sub['tag']}'(id={sub['id']}).")
        print(f"[CUSTOM] {reason}")
        return sub["id"], reason

    # ================================================================
    # PRED Core: APM - Attribute-Driven Plan Modification
    # ================================================================

    def _check_apm(self, plan_actions):
        """APM: Modify plan based on object attributes.

        Checks:
        1. If picking from inside a container (refrigerator, cabinet),
           insert open() before pick() and close() after.
        2. If placing into a container, insert open/close around place.
        3. If object is too large to pick (size > threshold), warn.

        Returns (modified_plan, modifications_log).
        """
        if not plan_actions:
            return plan_actions, []

        modifications = []
        new_plan = []
        _CONTAINER_TAGS = {"refrigerator", "cabinet", "drawer", "oven",
                           "microwave", "dishwasher", "washer", "closet"}

        i = 0
        while i < len(plan_actions):
            action = plan_actions[i]
            action_str = action if isinstance(action, str) else str(action)

            # Check pick() actions
            if "pick(" in action_str:
                pick_id = action_str.split("(")[1].split(")")[0]
                rels = self._get_object_relationships(pick_id)

                # Check if object is IN a container
                for container_id in rels["in"]:
                    container_name = self.get_obj_name_by_id(container_id)
                    container_tag = container_name.lower()
                    if any(ct in container_tag for ct in _CONTAINER_TAGS):
                        # Check if there's already an open() for this container
                        plan_so_far = " ".join(new_plan)
                        if f"open({container_id})" not in plan_so_far:
                            mod = (f"APM: '{self.get_obj_name_by_id(pick_id)}' "
                                   f"is IN '{container_name}'. "
                                   f"Inserting open({container_id}) before pick.")
                            modifications.append(mod)
                            print(f"[CUSTOM] {mod}")
                            new_plan.append(f"goto({container_id})")
                            new_plan.append(f"open({container_id})")
                        break

                new_plan.append(action_str)

            # Check place() actions
            elif "place(" in action_str:
                place_id = action_str.split("(")[1].split(")")[0]
                dest_tag = self.get_obj_name_by_id(place_id).lower()

                # Only treat as container if it's NOT being used as a surface
                # (i.e. SurfaceAnalyzer tracks it as a surface with objects/free space)
                is_surface = int(place_id) in self.surface_analyzer.surfaces
                is_container_tag = any(ct in dest_tag for ct in _CONTAINER_TAGS)

                if is_container_tag and not is_surface:
                    plan_so_far = " ".join(new_plan)
                    if f"open({place_id})" not in plan_so_far:
                        mod = (f"APM: Destination '{self.get_obj_name_by_id(place_id)}' "
                               f"is a container. Inserting open before place.")
                        modifications.append(mod)
                        print(f"[CUSTOM] {mod}")
                        new_plan.append(f"open({place_id})")

                new_plan.append(action_str)

                # Add close() after placing in container
                if is_container_tag and not is_surface:
                    mod = f"APM: Inserting close({place_id}) after place."
                    modifications.append(mod)
                    print(f"[CUSTOM] {mod}")
                    new_plan.append(f"close({place_id})")

            else:
                new_plan.append(action_str)

            i += 1

        # Check for unpickable objects (too large)
        for action_str in new_plan:
            if "pick(" in action_str:
                pick_id = action_str.split("(")[1].split(")")[0]
                size = self.get_obj_size_by_id(pick_id)
                if size > 1.0:
                    obj_name = self.get_obj_name_by_id(pick_id)
                    mod = (f"APM WARNING: '{obj_name}'(id={pick_id}) "
                           f"is very large (size={size:.2f}m). "
                           f"Pick may fail.")
                    modifications.append(mod)
                    print(f"[CUSTOM] {mod}")

        return new_plan, modifications

    # ================================================================
    # PRED Core: DTA - Dynamic Search Target Generation
    # ================================================================

    def generate_search_targets(self, target_name, num_targets=3):
        """Ask the LLM to guess likely locations for a missing object based on the scene graph."""
        # target_name is already the name of the missing object (string)
        
        # Build list of available furniture/surfaces
        surfaces = []
        for info in self.raw_data.values():
            if info.get("_status", "present") == "present":
                tag = info.get("object_tag", "").lower()
                # Typical surfaces/containers
                if any(k in tag for k in ["table", "desk", "counter", "cabinet", "shelf", "refrigerator", "drawer", "chair"]):
                    surfaces.append(f"'{tag}'(id={info['id']})")
        
        surfaces_str = ", ".join(surfaces[:15]) if surfaces else "Any available surface"
        
        prompt = f"""The robot is looking for '{target_name}' which is MISSING.
Available surfaces: {surfaces_str}

Which {num_targets} locations most likely have '{target_name}'?
Respond as JSON: {{"ids": ["12", "45", "7"]}}
"""
        try:
            resp = self.model.generate(prompt)
            text = resp.text.replace("```json", "").replace("```", "").strip()
            if "{" in text:
                text = text[text.find("{"):text.rfind("}") + 1]
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    guessed_ids = parsed.get("ids", list(parsed.values())[0] if parsed else [])
                else:
                    guessed_ids = parsed
            elif "[" in text:
                text = text[text.find("["):text.rfind("]") + 1]
                guessed_ids = json.loads(text)
            else:
                guessed_ids = []
            if not isinstance(guessed_ids, list):
                guessed_ids = [guessed_ids] if guessed_ids else []
            # Ensure string IDs
            guessed_ids = [str(x) for x in guessed_ids]
            print(f"[CUSTOM] LLM guessed likely locations for '{target_name}': {guessed_ids}")
            return guessed_ids[:num_targets]
        except Exception as e:
            print(f"[CUSTOM] Error guessing search locations: {e}")
            # Fallback: pick some random surfaces if LLM fails
            fallback = []
            for info in self.raw_data.values():
                tag = info.get("object_tag", "").lower()
                if "table" in tag or "counter" in tag or "desk" in tag:
                    fallback.append(str(info["id"]))
            return fallback[:num_targets]

    # ================================================================
    # PRED Post-LLM: Revise generated plan
    # ================================================================

    def _revise_plan(self, plan_actions, user_instruction):
        """Post-LLM plan revision using PRED mechanisms.

        1. ASR: Skip redundant pick/place if already at destination
        2. APM: Insert open/close for containers
        3. DTA: Substitute missing targets

        Returns (revised_plan, revision_log).
        """
        if not plan_actions:
            return plan_actions, []

        revision_log = []

        # --- Pass 1: DTA - substitute missing/absent targets ---
        dta_plan = []
        for action_str in plan_actions:
            if "(" in action_str and "done" not in action_str:
                target_id = action_str.split("(")[1].split(")")[0]
                action_type = action_str.split("(")[0].strip()

                sub_id, dta_reason = self._find_substitute(target_id)
                if sub_id is not None:
                    revision_log.append(dta_reason)
                    dta_plan.append(f"{action_type}({sub_id})")
                elif dta_reason:
                    revision_log.append(dta_reason)
                    dta_plan.append(action_str)
                else:
                    dta_plan.append(action_str)
            else:
                dta_plan.append(action_str)

        # --- Pass 2: ASR - skip if object already at destination ---
        # Detect pick-place pairs: pick(A) ... place(B) means "move A to B"
        asr_plan = list(dta_plan)
        pick_target = None
        pick_idx = None
        skip_indices = set()

        for i, action_str in enumerate(asr_plan):
            if "pick(" in action_str:
                pick_target = action_str.split("(")[1].split(")")[0]
                pick_idx = i
            elif "place(" in action_str and pick_target is not None and pick_idx is not None:
                dest_id = action_str.split("(")[1].split(")")[0]
                should_skip, asr_reason = self._check_asr(pick_target, dest_id)
                if should_skip:
                    revision_log.append(asr_reason)
                    # Mark pick, place, and surrounding goto for skip
                    skip_indices.add(pick_idx)
                    skip_indices.add(i)
                    # Also skip goto before pick and goto before place
                    if pick_idx > 0 and "goto(" in asr_plan[pick_idx - 1]:
                        skip_indices.add(pick_idx - 1)
                    if i > 0 and "goto(" in asr_plan[i - 1]:
                        skip_indices.add(i - 1)
                pick_target = None
                pick_idx = None

        if skip_indices:
            asr_plan = [a for idx, a in enumerate(asr_plan)
                        if idx not in skip_indices]
            # If everything was skipped, just return done
            remaining = [a for a in asr_plan if "done" not in a]
            if not remaining:
                asr_plan = ["done()"]

        # --- Pass 3: APM - modify based on attributes ---
        final_plan, apm_mods = self._check_apm(asr_plan)
        revision_log.extend(apm_mods)

        if revision_log:
            print(f"[CUSTOM] Plan revised with {len(revision_log)} modifications:")
            for r in revision_log:
                print(f"  - {r}")

        return final_plan, revision_log

    # ================================================================
    # System prompt & LLM interaction
    # ================================================================

    def _build_system_prompt(self):
        edge_summary = ""
        if self.edges:
            edge_lines = [f"  - {e['src_tag']}(id:{e['src_index']}) {e['object_relation']} {e['dst_tag']}(id:{e['dst_index']})"
                          for e in self.edges]
            edge_summary = "Known spatial relations:\n" + "\n".join(edge_lines)

        room_summary = ""
        if self.room_index:
            lines = []
            for r in self.room_index:
                adj = r.get("adjacent_rooms", [])
                adj_str = f", adjacent: {adj}" if adj else ""
                lines.append(f"  - {r['instance_name']} ({r['n_objects']} objects{adj_str})")
            room_summary = "Available rooms:\n" + "\n".join(lines)
        else:
            room_names = sorted(set(self.room_map.values())) if self.room_map else ["main_area"]
            room_summary = f"Available rooms: {room_names}"

        first_room = self.room_index[0]["instance_name"] if self.room_index else "main_area"

        return f"""You are a robot task planner controlling a mobile manipulator (Fetch robot).

{room_summary}

{edge_summary}

Scene graph starts COLLAPSED. You must expand rooms to see objects inside.
Strategy:
1. Read the task. Decide which room(s) most likely contain the relevant objects.
2. expand_node that room. Check if the needed objects are there.
3. If not found, expand adjacent rooms or other candidates one by one.
4. Once you have found all needed objects, output a plan.

Spatial relations:
- "a on b" = a is on b's surface
- "a in b" = a is inside b (open b first)
- "a next to b" = a and b are side by side

Actions:
- goto(ID): Move to object. MUST do before pick/place/open/close.
- pick(ID): Pick up object.
- place(ID): Place held object onto surface ID. System finds free space automatically.
- open(ID): Open container (refrigerator/cabinet/drawer).
- close(ID): Close container.
- done(): Task complete.
- expand_node(ROOM): Load objects in a room. Use instance_name (e.g. "kitchen_1").

Rules:
1. ALWAYS expand_node before planning. Never plan from a collapsed graph.
2. Always goto() BEFORE pick/place/open/close.
3. If already holding an object, skip pick — just goto destination and place.
4. If object is "in" a container, open container before picking.
5. Choose specific object IDs from the graph. If duplicates exist, pick the best one by ID.
6. The system auto-corrects: skips redundant actions, substitutes missing targets.
7. "move A next to B" requires pick(A) then place on B's surface.
8. NEVER return done() without any actions when asked to move/place something.

You MUST respond with valid JSON in one of these formats:

FORMAT 1 - Exploring (need to see more objects):
{{"chain_of_thought": "Task involves X. That is most likely in kitchen. Expanding kitchen_1.", "mode": "exploring", "command": {{"command_name": "expand_node", "node_name": "{first_room}"}}}}

FORMAT 2 - Planning (objects found, ready to act):
{{"chain_of_thought": "Found object IDs. Step by step plan...", "mode": "planning", "command": {{"command_name": "plan", "plan": ["goto(13)", "pick(13)", "goto(9)", "place(9)", "done()"]}}}}

EXAMPLES:

User says "bring me the book" (graph collapsed):
{{"chain_of_thought": "Books are typically in a library or bedroom. I'll check library_1 first.", "mode": "exploring", "command": {{"command_name": "expand_node", "node_name": "library_1"}}}}

After expanding library_1, book id=55 found:
{{"chain_of_thought": "Book id=55 is in library_1. I'll go pick it up.", "mode": "planning", "command": {{"command_name": "plan", "plan": ["goto(55)", "pick(55)", "done()"]}}}}

User says "put the cup on the kitchen counter" (graph collapsed):
{{"chain_of_thought": "Cup and counter are likely in kitchen. Expanding kitchen_1.", "mode": "exploring", "command": {{"command_name": "expand_node", "node_name": "kitchen_1"}}}}

After expanding kitchen_1, cup not found there, but bedroom_1 is adjacent:
{{"chain_of_thought": "Cup not in kitchen. Checking bedroom_1 next.", "mode": "exploring", "command": {{"command_name": "expand_node", "node_name": "bedroom_1"}}}}

User says "get the milk from the refrigerator" (fridge id=20, milk id=15 inside):
{{"chain_of_thought": "Milk is inside refrigerator. Open fridge, pick milk, close fridge.", "mode": "planning", "command": {{"command_name": "plan", "plan": ["goto(20)", "open(20)", "goto(15)", "pick(15)", "goto(20)", "close(20)", "done()"]}}}}
"""

    def _build_relationship_context(self):
        """Build edge relation summary string for LLM context."""
        if not self.edges:
            return "No spatial relations available."
        lines = []
        for e in self.edges:
            lines.append(f"  - {e['src_tag']}(id:{e['src_index']}) {e['object_relation']} {e['dst_tag']}(id:{e['dst_index']})")
        return "\n".join(lines)

    def _extract_target_objects(self, user_instruction):
        """Use LLM to extract object names from the user instruction.

        Returns list of object name strings mentioned in the instruction.
        """
        # Build available object list for context
        available = []
        for info in self.raw_data.values():
            tag = info.get("object_tag", "unknown")
            if tag not in available:
                available.append(tag)
        available_str = ", ".join(available[:40])

        prompt = f"""Extract object names from this instruction. Match to available objects when possible.

Available: [{available_str}]
Instruction: "{user_instruction}"

Respond with a JSON list of object names.
Example input: "move the cup to the table" → {{"objects": ["cup", "table"]}}
Example input: "put apple on plate" → {{"objects": ["apple", "plate"]}}
"""
        try:
            resp = self.model.generate(prompt)
            text = resp.text.replace("```json", "").replace("```", "").strip()
            if "{" in text:
                text = text[text.find("{"):text.rfind("}") + 1]
                parsed = json.loads(text)
                # Support both {"objects": [...]} and bare [...]
                if isinstance(parsed, dict):
                    result = parsed.get("objects", list(parsed.values())[0] if parsed else [])
                else:
                    result = parsed
            elif "[" in text:
                text = text[text.find("["):text.rfind("]") + 1]
                result = json.loads(text)
            else:
                result = []
            if not isinstance(result, list):
                result = [result] if result else []
            print(f"[CUSTOM] Extracted objects from instruction: {result}")
            return result
        except Exception as e:
            print(f"[CUSTOM] Object extraction error: {e}")
            return []

    def _check_missing_objects(self, object_names):
        """Check which extracted object names are missing from the scene graph.

        Returns list of missing object name strings.
        """
        present_tags = set()
        for info in self.raw_data.values():
            tag = info.get("object_tag", "").lower()
            status = info.get("_status", "present")
            if status == "present":
                present_tags.add(tag)

        missing = []
        for name in object_names:
            name_lower = name.lower().strip()
            # Check exact match or partial match
            found = False
            for tag in present_tags:
                if name_lower in tag or tag in name_lower:
                    found = True
                    break
            if not found:
                missing.append(name)

        return missing

    def process_turn(self, user_instruction, simulator_feedback=""):
        self.refresh_from_react()
        change_summary = self.react_bridge.get_change_summary() if self.react_bridge else ""

        # --- PRED Pre-check: detect missing objects BEFORE LLM planning ---
        # Only do this when we're not already replanning from failure
        # and graph is expanded (not collapsed)
        is_replanning = simulator_feedback and (
            "fail" in simulator_feedback.lower() or
            "error" in simulator_feedback.lower() or
            "found" in simulator_feedback.lower() or
            "target" in simulator_feedback.lower())

        if (not is_replanning
            and bool(self.expanded_rooms)
            and not simulator_feedback.startswith("FOUND_AFTER_SEARCH:")
            and not simulator_feedback.startswith("FOUND_NEW_RELEVANT_OBJECT:")):
            extracted = self._extract_target_objects(user_instruction)
            if extracted:
                missing = self._check_missing_objects(extracted)
                self.missing_objects = missing
                if missing:
                    should_search = False
                    if len(missing) == len(extracted):
                        should_search = True
                    else:
                        # Check dependency: can we proceed without the missing object?
                        available = [o for o in extracted if o not in missing]
                        check_prompt = f"""Instruction: "{user_instruction}"
Missing objects: {missing}
Available objects: {available}
Is the task BLOCKED without the missing objects, or can we PROCEED with available ones?
Respond as JSON: {{"answer": "PROCEED"}} or {{"answer": "BLOCKED"}}"""
                        try:
                            resp = self.model.generate(check_prompt)
                            resp_text = resp.text.upper()
                            if "BLOCKED" in resp_text:
                                should_search = True
                            else:
                                print(f"[CUSTOM] Some objects missing ({missing}) but can PROCEED with available: {available}")
                        except Exception as e:
                            print(f"[CUSTOM] Dependency check error: {e}")
                            should_search = True

                    if should_search:
                        missing_name = missing[0]  # Handle first missing object
                        print(f"[CUSTOM] Task blocked. Entering search mode for '{missing_name}'.")
                        search_targets = self.generate_search_targets(
                            missing_name, num_targets=3)
                        return {
                            "mode": "searching",
                            "chain_of_thought": (
                                f"Task blocked because '{missing_name}' is missing. "
                                f"Initiating active search at likely locations."),
                            "command": {},
                            "missing_object": missing_name,
                            "search_targets": search_targets,
                            "original_instruction": user_instruction,
                        }

        history_str = "None"
        if self.action_history:
            history_str = "\n".join(
                [f"User: {h['user_instruction']} -> Plan: {h['plan']}"
                 + (f" [REVISED: {h.get('revisions', '')}]" if h.get('revisions') else "")
                 for h in self.action_history[-5:]])

        holding_str = "Nothing"
        if self.holding_object:
            holding_name = self.get_obj_name_by_id(self.holding_object)
            holding_str = f"'{holding_name}' (ID: {self.holding_object})"

        # PRED replanning on failure
        if is_replanning:
            replanning_block = f"""
PRED REPLANNING REQUIRED: '{simulator_feedback}'
Analyze the failure:
- Is the target unreachable/missing? (DTA will auto-substitute if possible)
- Does the goal relationship already exist? (ASR will auto-skip if so)
- Are container interactions needed? (APM will auto-insert open/close)
Generate a NEW plan to accomplish: '{user_instruction}'.
"""
        else:
            replanning_block = ""

        surface_info = self.surface_analyzer.get_graph_summary()

        # Build concise object list for small models
        obj_list_lines = []
        for info in self.raw_data.values():
            oid = info.get("id")
            tag = info.get("object_tag", "unknown")
            status = info.get("_status", "present")
            is_new = " [NEW]" if info.get("_is_new") else ""
            pos = info.get("bbox_center", info.get("center", [0, 0, 0]))
            obj_list_lines.append(
                f"  id={oid}: {tag} ({status}{is_new}) pos=({pos[0]:.1f},{pos[1]:.1f})")
        obj_list_str = "\n".join(obj_list_lines)

        dynamic_input = f"""
[Current Robot State]: Holding {holding_str}
[Action History]:
{history_str}
[User Instruction]: {user_instruction}
[Available Objects]:
{obj_list_str}
[Current Graph State]: {self.current_graph_state}
[Surface Free Spaces]:
{surface_info if surface_info else "No surfaces detected."}
[Scene Changes]: {change_summary}
[Memory/History]: {str(self.memory)}
[Simulator Feedback]: {simulator_feedback}
{replanning_block}
"""
        full_prompt = self.system_prompt + "\n" + "-" * 20 + "\n" + dynamic_input
        try:
            response = self.model.generate(full_prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            if "{" in text: text = text[text.find("{"):text.rfind("}") + 1]
            try:
                parsed_resp = json.loads(text)
            except json.JSONDecodeError:
                print(f"[CUSTOM] JSON parse failed, attempting repair...")
                parsed_resp = _repair_json(text)

            # --- PRED post-processing: revise the plan ---
            if parsed_resp.get("mode") == "planning":
                original_plan = parsed_resp.get("command", {}).get("plan", [])
                if original_plan:
                    revised_plan, revision_log = self._revise_plan(
                        original_plan, user_instruction)

                    if revision_log:
                        parsed_resp["command"]["plan"] = revised_plan
                        parsed_resp["pred_revisions"] = revision_log
                        parsed_resp["original_plan"] = original_plan
                        print(f"[CUSTOM] Original: {original_plan}")
                        print(f"[CUSTOM] Revised:  {revised_plan}")

                    self.action_history.append({
                        "user_instruction": user_instruction,
                        "plan": revised_plan,
                        "revisions": revision_log if revision_log else None,
                    })
                else:
                    self.action_history.append({
                        "user_instruction": user_instruction,
                        "plan": original_plan,
                    })

            return parsed_resp
        except Exception as e:
            return {"mode": "error", "reasoning": str(e), "command": {}}

    def execute_graph_api(self, command_name, node_name):
        if command_name == "expand_node":
            n = self._expand_room(node_name)
            if n > 0:
                self.current_graph_state = self._build_current_graph_str()
                if node_name not in self.memory:
                    self.memory.append(node_name)
                # Report which rooms are still unexplored
                unexplored = [r["instance_name"] for r in self.room_index
                              if r["instance_name"] not in self.expanded_rooms]
                return (f"Expanded '{node_name}': {n} objects loaded. "
                        f"Unexplored rooms: {unexplored}")
            else:
                return f"Room '{node_name}' not found. Available: {[r['instance_name'] for r in self.room_index]}"
        elif command_name == "contract_node":
            self.current_graph_state = self.collapsed_graph_str
            return "Graph Contracted (room summaries only)."
        return "No graph operation needed."

    # ================================================================
    # Object query helpers (same interface as SayPlanBrain)
    # ================================================================

    def get_coords_by_id(self, asset_id):
        for info in self.raw_data.values():
            if str(info.get("id")) == str(asset_id):
                return np.array(info.get("bbox_center", info.get("center")))
        return None

    def get_obj_name_by_id(self, asset_id):
        for info in self.raw_data.values():
            if str(info.get("id")) == str(asset_id):
                return info.get("object_tag", "Unknown")
        return str(asset_id)

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
                # Pull down slightly — bbox overestimates height
                return center[2] + ext[2] / 2.0 - 0.02
        return 1.0

    def get_placement_pos(self, surface_id, near_obj_id=None):
        """Find the best free-space placement position on a surface.

        Uses SurfaceAnalyzer to find the optimal position that avoids
        existing objects. If near_obj_id is given, prefers spots closest
        to that object ("next to" semantics).

        Returns [x, y, z] placement coordinate.
        """
        try:
            sid = int(surface_id)
        except (ValueError, TypeError):
            print(f"[CUSTOM] get_placement_pos: invalid surface_id '{surface_id}'")
            return None

        # Load surface analysis from JSON (updated during idle_update)
        if os.path.exists(SURFACE_JSON_FILE):
            self.surface_analyzer.load_from_json(SURFACE_JSON_FILE)
        else:
            # Fallback: compute if JSON doesn't exist yet
            react_sg = None
            if self.react_bridge and hasattr(self.react_bridge, 'current_sg'):
                react_sg = self.react_bridge.current_sg
            self.surface_analyzer.update(self.raw_data, self.edges, react_sg)
            self.surface_analyzer.save_to_json(SURFACE_JSON_FILE)

        # If target is NOT a surface (e.g. plate, bowl), find the surface
        # it sits on and place there instead (near the target object)
        if sid not in self.surface_analyzer.surfaces:
            parent_surface_id = self._find_parent_surface(sid)
            if parent_surface_id is not None:
                tag = self.get_obj_name_by_id(sid)
                parent_tag = self.get_obj_name_by_id(parent_surface_id)
                print(f"[CUSTOM] place target '{tag}'(id:{sid}) is not a surface, "
                      f"using parent surface '{parent_tag}'(id:{parent_surface_id})")
                near_obj_id = sid  # place near the target object
                sid = parent_surface_id
            else:
                print(f"[CUSTOM] place target id:{sid} is not a surface and no parent found")

        # Get held object size for fitting check (shrink like surface analyzer)
        obj_w, obj_d = 0.15, 0.15  # default small object
        if self.holding_object is not None:
            for info in self.raw_data.values():
                if str(info.get("id")) == str(self.holding_object):
                    ext = info.get("bbox_extent", [0.15, 0.15, 0.1])
                    obj_w, obj_d = ext[0] * 0.80, ext[1] * 0.80
                    break

        # If no explicit near_obj_id, auto-detect: if there are objects on
        # this surface, place near the first one (avoid collisions)
        if near_obj_id is None:
            surf = self.surface_analyzer.surfaces.get(sid)
            if surf and surf.objects_on:
                near_obj_id = surf.objects_on[0]["obj_id"]
                print(f"[CUSTOM] Auto near_obj: {surf.objects_on[0]['tag']}(id:{near_obj_id})")

        # Try SurfaceAnalyzer for optimal placement
        placement = self.surface_analyzer.find_best_placement(
            sid, obj_w, obj_d, near_obj_id=near_obj_id)
        if placement is not None:
            pos = placement["center"]
            print(f"[CUSTOM] Placement on surface {sid}: "
                  f"free-space ({placement['direction']}) at "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                  f"avail={placement['available_size']}")
            hold_tag = self.get_obj_name_by_id(self.holding_object) if self.holding_object else "obj"
            self.surface_analyzer._last_placement = {
                "x": pos[0], "y": pos[1], "z": pos[2], "tag": hold_tag}
            return pos

        # Fallback: retry with reduced margins
        placement = self.surface_analyzer.find_best_placement(
            sid, obj_w, obj_d, near_obj_id=near_obj_id,
            edge_margin=0.01, obj_clearance=0.02)
        if placement is not None:
            pos = placement["center"]
            print(f"[CUSTOM] Placement fallback (reduced margin): "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            hold_tag = self.get_obj_name_by_id(self.holding_object) if self.holding_object else "obj"
            self.surface_analyzer._last_placement = {
                "x": pos[0], "y": pos[1], "z": pos[2], "tag": hold_tag}
            return pos

        # Last resort: offset from surface center away from existing objects
        center = self.get_coords_by_id(surface_id)
        surface_z = self.get_obj_surface_z(surface_id)
        if center is not None:
            surf = self.surface_analyzer.surfaces.get(sid)
            if surf and surf.objects_on:
                # Push placement to opposite side of existing objects
                obj_cx = surf.objects_on[0]["cx"]
                obj_cy = surf.objects_on[0]["cy"]
                dx = float(center[0]) - obj_cx
                dy = float(center[1]) - obj_cy
                offset = max(obj_w, obj_d) * 0.6
                norm = (dx**2 + dy**2)**0.5
                if norm > 0.01:
                    fx = float(center[0]) + dx / norm * offset
                    fy = float(center[1]) + dy / norm * offset
                else:
                    fx = float(center[0]) + offset
                    fy = float(center[1])
                # Clamp to surface bounds
                if surf:
                    fx = max(surf.x_min + obj_w/2, min(surf.x_max - obj_w/2, fx))
                    fy = max(surf.y_min + obj_d/2, min(surf.y_max - obj_d/2, fy))
                print(f"[CUSTOM] Placement fallback: offset from objects at "
                      f"({fx:.3f}, {fy:.3f}, {surface_z:.3f})")
                return [fx, fy, float(surface_z)]
            print(f"[CUSTOM] Placement fallback: surface {sid} center")
            return [float(center[0]), float(center[1]), float(surface_z)]

        return None

    def _find_parent_surface(self, obj_id):
        """Find the surface that an object is sitting on, using edges and proximity."""
        obj_id_int = int(obj_id) if str(obj_id).isdigit() else None
        if obj_id_int is None:
            return None

        # 1) Check edges: "a on b" relation
        for edge in self.edges:
            src_idx = edge.get("src_index")
            dst_idx = edge.get("dst_index")
            relation = edge.get("object_relation", "")
            if relation == "a on b" and src_idx == obj_id_int:
                if dst_idx in self.surface_analyzer.surfaces:
                    return dst_idx
            elif relation == "b on a" and dst_idx == obj_id_int:
                if src_idx in self.surface_analyzer.surfaces:
                    return src_idx

        # 2) Proximity: find closest surface below/near the object
        obj_pos = self.get_coords_by_id(obj_id)
        if obj_pos is None:
            return None
        best_sid, best_dist = None, float("inf")
        for sid, surf in self.surface_analyzer.surfaces.items():
            dist = float(np.linalg.norm(
                np.array(obj_pos[:2]) - np.array(surf.center[:2])))
            # Object should be above the surface and within its XY footprint
            if dist < best_dist:
                best_dist = dist
                best_sid = sid
        if best_sid is not None and best_dist < 1.5:
            return best_sid
        return None

    # ================================================================
    # LLM-based edge generation for REACT-added objects
    # ================================================================

    def _compute_spatial_hints(self, obj1_info, obj2_info):
        """Compute spatial hints between two objects for LLM edge classification."""
        c1 = np.array(obj1_info.get("bbox_center", obj1_info.get("center", [0, 0, 0])))
        c2 = np.array(obj2_info.get("bbox_center", obj2_info.get("center", [0, 0, 0])))
        e1 = np.array(obj1_info.get("bbox_extent", [0.1, 0.1, 0.1]))
        e2 = np.array(obj2_info.get("bbox_extent", [0.1, 0.1, 0.1]))

        vertical_diff = float(c1[2] - c2[2])
        horizontal_dist = float(np.linalg.norm(c1[:2] - c2[:2]))

        # Bbox overlap (3D IoU approximation)
        overlap_per_axis = []
        for i in range(3):
            lo = max(c1[i] - e1[i]/2, c2[i] - e2[i]/2)
            hi = min(c1[i] + e1[i]/2, c2[i] + e2[i]/2)
            overlap_per_axis.append(max(0, hi - lo))
        overlap_vol = overlap_per_axis[0] * overlap_per_axis[1] * overlap_per_axis[2]
        vol1 = float(np.prod(e1))
        vol2 = float(np.prod(e2))
        min_vol = min(vol1, vol2) if min(vol1, vol2) > 0 else 0.001
        bbox_overlap = overlap_vol / min_vol

        size_ratio = vol1 / vol2 if vol2 > 0 else 999.0

        # "on" bias: object1 bottom Z vs object2 top Z
        obj1_bottom = c1[2] - e1[2] / 2.0
        obj2_top = c2[2] + e2[2] / 2.0
        on_gap = abs(obj1_bottom - obj2_top)  # small = likely "on"

        return {
            "vertical_diff": round(vertical_diff, 3),
            "horizontal_dist": round(horizontal_dist, 3),
            "bbox_overlap_score": round(min(bbox_overlap, 1.0), 3),
            "size_ratio": round(size_ratio, 3),
            "on_gap": round(on_gap, 3),
        }

    def _generate_edge_with_llm(self, obj1_info, obj2_info):
        """Use LLM to classify spatial relation between two objects.
        Biased toward 'on' over 'in' for overlapping objects."""
        if self.model is None:
            return None

        hints = self._compute_spatial_hints(obj1_info, obj2_info)
        tag1 = obj1_info.get("object_tag", "unknown")
        tag2 = obj2_info.get("object_tag", "unknown")
        e1 = obj1_info.get("bbox_extent", [0.1, 0.1, 0.1])
        e2 = obj2_info.get("bbox_extent", [0.1, 0.1, 0.1])
        c1 = obj1_info.get("bbox_center", obj1_info.get("center", [0, 0, 0]))
        c2 = obj2_info.get("bbox_center", obj2_info.get("center", [0, 0, 0]))

        prompt = f"""You are analyzing spatial relationships between two 3D objects in an indoor scene.

Object 1: "{tag1}"
  bbox_center: {[round(x,3) for x in c1]}
  bbox_extent: {[round(x,3) for x in e1]}

Object 2: "{tag2}"
  bbox_center: {[round(x,3) for x in c2]}
  bbox_extent: {[round(x,3) for x in e2]}

Spatial hints:
  vertical_diff: {hints['vertical_diff']} (positive = object1 is higher)
  horizontal_dist: {hints['horizontal_dist']}m
  bbox_overlap_score: {hints['bbox_overlap_score']}
  size_ratio: {hints['size_ratio']} (vol1/vol2)
  on_gap: {hints['on_gap']}m (gap between obj1 bottom and obj2 top, small = likely resting on)

IMPORTANT RULES for "on" vs "in":
- "a on b" means object1 RESTS ON TOP of object2's surface. This is the MOST COMMON relation for small objects on furniture.
- If on_gap < 0.15m and object2 is furniture (desk, table, cabinet, counter, shelf), STRONGLY prefer "a on b".
- "a in b" means object1 is FULLY ENCLOSED inside object2 (like clothes in a closed drawer, food in refrigerator).
- bbox_overlap happens naturally when objects rest on surfaces — overlap alone does NOT mean "in".
- Only use "in" when the smaller object is clearly INSIDE a container (refrigerator, drawer, box, closet).
- "next to" is for objects at similar heights, side by side, not stacked.

Choose the BEST relationship:
1. "a on b" — object1 rests on surface of object2
2. "b on a" — object2 rests on surface of object1
3. "a in b" — object1 enclosed inside object2
4. "b in a" — object2 enclosed inside object1
5. "a next to b" — side by side
6. "a attached to b" — physically fixed
7. "b attached to a"
8. "none of these"

----
Output JSON with exactly two keys:
{{"object_relation": "one of the above", "reason": "brief explanation"}}"""

        try:
            response = self.model.generate(prompt, max_retries=3)
            text = response.text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            if "{" in text:
                text = text[text.find("{"):text.rfind("}") + 1]
            result = json.loads(text)
            relation = result.get("object_relation", "none of these")
            reason = result.get("reason", "")
            print(f"[CUSTOM] LLM edge: {tag1} --[{relation}]--> {tag2} ({reason})")
            return relation
        except Exception as e:
            print(f"[CUSTOM] LLM edge generation failed: {e}")
            return None

    def _find_nearby_objects(self, obj_id, max_dist=2.0, max_count=3):
        """Find nearby objects for edge generation."""
        obj_info = None
        for info in self.raw_data.values():
            if info.get("id") == obj_id:
                obj_info = info
                break
        if obj_info is None:
            return []

        pos = np.array(obj_info.get("bbox_center", obj_info.get("center", [0, 0, 0])))
        candidates = []
        for info in self.raw_data.values():
            other_id = info.get("id")
            if other_id == obj_id:
                continue
            if info.get("_status", "present") != "present":
                continue
            other_pos = np.array(info.get("bbox_center", info.get("center", [0, 0, 0])))
            dist = float(np.linalg.norm(pos[:2] - other_pos[:2]))
            if dist < max_dist:
                candidates.append((dist, info))

        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:max_count]]

    def generate_edges_for_new_objects(self):
        """Generate edges for REACT-added objects using LLM.
        Called during idle (robot not executing task).
        Returns number of new edges created."""
        if not self._pending_edge_objects:
            return 0

        new_edge_count = 0
        processed = set()

        for obj_id in list(self._pending_edge_objects):
            obj_info = None
            for info in self.raw_data.values():
                if info.get("id") == obj_id:
                    obj_info = info
                    break
            if obj_info is None:
                processed.add(obj_id)
                continue

            tag = obj_info.get("object_tag", "unknown")
            nearby = self._find_nearby_objects(obj_id)

            for other_info in nearby:
                other_id = other_info.get("id")
                other_tag = other_info.get("object_tag", "unknown")

                relation = self._generate_edge_with_llm(obj_info, other_info)
                if relation and relation != "none of these":
                    new_edge = {
                        "src_obj_id": obj_id,
                        "dst_obj_id": other_id,
                        "src_index": obj_id,
                        "dst_index": other_id,
                        "src_tag": tag,
                        "dst_tag": other_tag,
                        "object_relation": relation,
                        "reason": "LLM-generated for REACT-added object",
                        "_is_generated": True,
                    }
                    self.edges.append(new_edge)
                    new_edge_count += 1
                    print(f"[CUSTOM] New edge: {tag}(id:{obj_id}) --[{relation}]--> "
                          f"{other_tag}(id:{other_id})")

            processed.add(obj_id)

        self._pending_edge_objects -= processed

        if new_edge_count > 0:
            # Save updated edges to JSON
            self._save_edges_json()
            print(f"[CUSTOM] Generated {new_edge_count} new edges for REACT objects")

        return new_edge_count

    def _save_edges_json(self):
        """Save current edges to edge JSON file."""
        try:
            output = {"edges": self.edges}
            with open(EDGE_JSON_FILE, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"[CUSTOM] Saved {len(self.edges)} edges to {EDGE_JSON_FILE}")
        except Exception as e:
            print(f"[CUSTOM] Failed to save edges: {e}")

    def idle_update(self):
        """Called when robot is idle (not executing a task).
        1. Generate edges for new REACT objects via LLM
        2. Update surface analyzer and save to JSON (only if changes exist)
        3. Rebuild graph strings with new edges
        """
        if not self.raw_data:
            return 0

        self.refresh_from_react()

        # 1. Generate edges for pending new objects
        n_edges = self.generate_edges_for_new_objects()

        # 2. Only recompute + save surface JSON if edges changed or first time
        if n_edges > 0 or not os.path.exists(SURFACE_JSON_FILE):
            react_sg = None
            if self.react_bridge and hasattr(self.react_bridge, 'current_sg'):
                react_sg = self.react_bridge.current_sg
            self.surface_analyzer.update(self.raw_data, self.edges, react_sg)
            self.surface_analyzer.save_to_json(SURFACE_JSON_FILE)

        # 3. Rebuild graph strings so LLM sees updated edges
        if n_edges > 0:
            self._rebuild_graph_strings(self.raw_data)
            if self.expanded_rooms:
                self.current_graph_state = self._build_current_graph_str()

        return n_edges

    def get_nav_approach_info(self, target_id):
        """Determine approach position for navigation. Direct coordinate-based, no edge lookup."""
        if not target_id:
            return None

        # Find target object info by ID or name
        target_info = None
        target_id_str = str(target_id)
        if target_id_str.isdigit():
            for info in self.raw_data.values():
                if str(info.get("id")) == target_id_str:
                    target_info = info
                    break
        else:
            for info in self.raw_data.values():
                if info.get("object_tag", "").lower() == target_id_str.lower() and info.get("_status", "present") == "present":
                    target_info = info
                    break

        if target_info is None:
            return None

        target_pos = np.array(target_info.get("bbox_center", target_info.get("center")))
        ext = target_info.get("bbox_extent", [0.5, 0.5, 0.5])
        target_size = max(ext[0], ext[1])

        arrival_dist = target_size / 2.0 + 0.5
        arrival_dist = max(arrival_dist, 0.6)
        return {
            "nav_pos": target_pos, "face_pos": target_pos,
            "arrival_dist": arrival_dist,
            "parent_name": None, "parent_id": None,
        }

    def verify_visual(self, image_np, target_name):
        return "YES"
