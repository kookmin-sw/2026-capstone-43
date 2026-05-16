import numpy as np
import json
import os
from semantic_utils import (
    DEFAULT_SEMANTIC_MAP_PATH,
    load_semantic_id_to_type,
    object_type_from_handle,
    resolve_object_semantics_from_handle,
)

def _as_sim(sim_or_env):
    return sim_or_env._sim if hasattr(sim_or_env, "_sim") else sim_or_env

def get_divided_positions(sim_or_env, divisions=3, island_index=None, snap_to_navmesh=True):
    sim = _as_sim(sim_or_env)
    pf = sim.pathfinder # habitat-sim의 맵의 이동 가능 구역 NavMesh를 관리하고 경로를 탐색하는 모듈
    if not pf.is_loaded:
        raise RuntimeError("NavMesh is not loaded in the simulator.")
    
    # island는 서로 연결되어 있어 어디든 끊기지 않고 이동할 수 있는 하나의 독립된 영역을 의미한다.
    # 같은 floor/연결영역(island)만 쓰기
    if island_index is None:
        if (hasattr(sim, "articulated_agent")): # Habitat-Lab Rearrangement
            agent_pos = np.array(sim.articulated_agent.base_pos, dtype=np.float32)
        else: # habitat_sim.Simulator direct
            agent_pos = np.array(sim.get_agent(0).get_state().position, dtype=np.float32)
        island_index = int(pf.get_island(agent_pos))

    # GetReachablePositions 대체: navmesh 정점들
    # 현재 island를 구성하는 모든 꼭짓점(vertices) 좌표를 가져옴
    verts = np.asarray(pf.build_navmesh_vertices(island_index), dtype=np.float32)
    if verts.size == 0:
        raise RuntimeError(f"No navmesh vertices on island {island_index}.")
    
    min_x, max_x = float(verts[:, 0].min()), float(verts[:, 0].max())
    min_z, max_z = float(verts[:, 2].min()), float(verts[:, 2].max())
    y_ref = float(np.median(verts[:, 1])) # y값은 대체로 일정하므로 중앙값 사용

    x_interval = (max_x - min_x) / divisions
    z_interval = (max_z - min_z) / divisions

    centers = []
    for i in range(divisions):
        for j in range(divisions):
            x = min_x + (i + 0.5) * x_interval
            z = min_z + (j + 0.5) * z_interval
            p = np.array([x, y_ref, z], dtype=np.float32)

            # center가 비가용 영역에 떨어지면 navmesh로 스냅
            # 단순히 사각형으로 나눈 격자의 중심점은 실제 맵에서는 벽 속이거나 낭떠러지, 혹은 갈 수 없는 장애물 위일 가능성
            # pf.snap_point()는 이 계산된 중심점에서 가장 가까운 실제 이동 가능 구역(NavMesh)의 좌표로 위치를 보정(Snap)
            # 이를 통해 최종적으로 반환되는 center 좌표들은 에이전트가 실제로 이동할 수 있는 유효한 좌표임이 보장됨
            if snap_to_navmesh:
                s = np.array(pf.snap_point(p, island_index), dtype=np.float32)
                if np.all(np.isfinite(s)):
                    centers.append((float(s[0]), float(s[1]), float(s[2])))
                    continue
            centers.append((float(x), y_ref, float(z)))
    
    return centers

def get_static_objects_in_regions(sim_or_env, centers, grid_size=0.25, semantic_mapping_path=DEFAULT_SEMANTIC_MAP_PATH):
    sim = _as_sim(sim_or_env)
    regions = {center: [] for center in centers}
    if not centers:
        return regions
    
    # sem_id_to_type = load_semantic_id_to_type(semantic_mapping_path) # 시뮬레이터에서 객체의 semantic_id를 실제 객체 유형으로 변환하기 위한 매핑 정보 로드

    # 맵 안의 모든 사물 가져오기 (oracle 방식)
    # 시뮬레이터에서 두 가지 종류의 사물을 모두 가져와서 하나로 합치기
        ## - Rigid Object: 형태가 변하지 않는 일반적인 물체
        ## - Articulated Object: 관절이 있는 물체(예: 문, 서랍 등)
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    objs = list(rom.get_objects_by_handle_substring().values()) + \
           list(aom.get_objects_by_handle_substring().values())
    
    for obj in objs:
        if not getattr(obj, "is_alive", True): # pickupable이 False인 객체는 static object로 간주
            continue

        # AI2THOR의 'not pickupable' 대체: DYNAMIC 제외(static/kinematic만 사용)
            ## - 움직이는 물체 걸러내기 위해 객체의 motion_type 속성을 확인하여 DYNAMIC이 포함된 경우 해당 객체는 static object가 아니라고 판단하여 건너뛰도록 함
        motion = str(getattr(obj, "motion_type", "")).upper()
        if "DYNAMIC" in motion:
            continue

        pos = np.array(obj.translation, dtype=np.float32)

        raw_handle = str(getattr(obj, "handle", "")) # ex) FloorPlan1_physics-Drawer_814ccbab_:0000
        if "fetch" in raw_handle.lower() or "hab_fetch" in raw_handle.lower() or "hab_suction" in raw_handle.lower(): # fetch 로봇 자체 제외. 지금은 articulated object 전체를 다 읽어서, 로봇 본체가 맵에 섞일 수 있음
            continue
        runtime_semantic_id = None
        try:
            runtime_semantic_id = int(getattr(obj, "semantic_id"))
        except Exception:
            runtime_semantic_id = None

        resolved_sem = resolve_object_semantics_from_handle(
            handle=raw_handle,
            template_class=str(getattr(obj, "template_class", "")),
        )
        fallback_type = object_type_from_handle(raw_handle) # handle 기반 fallback
        object_type = str(resolved_sem.get("object_type") or fallback_type)

        """
        ## 이 코드는 Habitat-Sim 전용. Habitat-Lab에선 semantic_id가 씬 데이터셋이 제공하는 값이랑 다르기 때문에 이렇게 못 함.
        # semantic_id = None
        # try:
        #     semantic_id = int(getattr(obj, "semantic_id"))
        # except Exception:
        #     semantic_id = None

        # raw_handle = str(getattr(obj, "handle", ""))  # ex) FloorPlan1_physics-Drawer_814ccbab_:0000
        # # 1순위: semantic_id -> canonical object type
        # mapped_type = sem_id_to_type.get(semantic_id, None)
        # # 2순위: handle에서 추출한 이름
        # fallback_type = object_type_from_handle(getattr(obj, "handle", ""), getattr(obj, "template_class", ""))
        # # 최종 object type 결정: 매핑된 유형이 유효하면 사용, 그렇지 않으면 handle에서 추출한 이름 사용
        # object_type = mapped_type if (mapped_type and mapped_type != "Undefined") else fallback_type
        """

        # 사물 정보 기록 및 구역 할당
        obj_record = {
            "objectType_raw": raw_handle, # 원래 handle 정보, ex) "FloorPlan1_physics-Drawer_814ccbab_:0000"
            "runtimeSemanticId": runtime_semantic_id, # Habitat-Lab runtime semantic_id (재할당될 수 있음)
            "objectTemplateName": resolved_sem.get("template_name", ""), # handle에서 추출된 템플릿 이름(대개 해시 id)
            "datasetObjectId": resolved_sem.get("dataset_object_id", ""), # metadata/objects.json 조회 키가 되는 id
            "objectName": resolved_sem.get("object_name", ""), # metadata/objects.json의 사람이 읽는 이름
            "datasetSemanticId": resolved_sem.get("dataset_semantic_id"), # dataset 원본 semantic_id
            "datasetSemanticClass": resolved_sem.get("dataset_semantic_class", ""), # dataset semantic class
            "objectTypeSource": resolved_sem.get("source", "legacy_handle"), # objectType 출처
            # "semantic_id": semantic_id, # 오브젝트의 semantic_id (인스턴스 아이디랑은 다름, 얘는 똑같은 종류의 객체는 같은 semantic_id 가짐)
            "objectType": object_type, # 최종적으로 결정된 객체 유형, ex) "Drawer"
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}, # 객체의 3D 위치 정보
            "objectId": str(getattr(obj, "object_id", "")), # 객체의 고유 식별자, 시뮬레이터 내부에서 객체를 구분하는 데 사용되는 ID (근데 런타임마다 달라질 수 있음)
        }

        closest_center = min(
            centers,
            key=lambda c: np.linalg.norm([obj_record["position"]["x"] - c[0], 
                                          obj_record["position"]["z"] - c[2]])
        )
        regions[closest_center].append(obj_record)
    
    return regions
    
def extract_regions_from_json(filename='/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/longterm_memory.json'):
    """
    regions.json 파일의 내용을 읽어서 각 구역에 어떤 객체들이 있는지 문장 형태로 변환하는 함수

    * 예시 입력 (regions.json):
    {
        "(1.00, 0.00, 2.00)": [
            {"objectType": "Cabinet"},
            {"objectType": "Drawer"}
        ]
    }

    ->

    * 예시 출력:
    ["center (1.00, 0.00, 2.00) has {Cabinet, Drawer}"]
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    sentences = []
    for center, objects in data.items():
        object_types =[obj['objectType'] for obj in objects]
        sentence = f"center {center} has {{{', '.join(object_types)}}}"
        sentences.append(sentence)
    
    return sentences
