import json
import os
import numpy as np
import habitat_sim
import habitat_sim.geo
import re
from semantic_utils import (
    DEFAULT_SEMANTIC_MAP_PATH,
    load_semantic_id_to_type,
    object_type_from_handle,
    resolve_object_semantics_from_handle,
)

def _as_sim(sim_or_env):
    return sim_or_env._sim if hasattr(sim_or_env, "_sim") else sim_or_env


def _resolve_object_metadata(raw_handle: str, obj) -> dict:
    """
    resolve_object_semantics_from_handle을 먼저 호출하고 없으면 object_type_from_handle로 fallback
    """
    resolved = resolve_object_semantics_from_handle(
        handle=raw_handle,
        template_class=str(getattr(obj, "template_class", "")),
    )
    fallback_type = object_type_from_handle(raw_handle)
    object_type = str(resolved.get("object_type") or fallback_type)
    return {
        "objectType": object_type,
        "objectName": str(resolved.get("object_name") or ""),
    }


def first_map(
        sim, 
        output_path="/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json",
        # semantic_mapping_path=DEFAULT_SEMANTIC_MAP_PATH
):  
    sim = _as_sim(sim)
    # sem_id_to_type = load_semantic_id_to_type(semantic_mapping_path) # 시뮬레이터에서 객체의 semantic_id를 실제 객체 유형으로 변환하기 위한 매핑 정보 로드

    # 맵 안의 모든 사물 가져오기 (oracle 방식)
    # 시뮬레이터에서 두 가지 종류의 사물을 모두 가져와서 하나로 합치기
        ## - Rigid Object: 형태가 변하지 않는 일반적인 물체
        ## - Articulated Object: 관절이 있는 물체(예: 문, 서랍 등)
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    objs = list(rom.get_objects_by_handle_substring().values()) + \
           list(aom.get_objects_by_handle_substring().values())
    
    objects_locations = []
    for obj in objs:
        if not getattr(obj, "is_alive", True): # pickupable이 False인 객체는 static object로 간주
            continue

        pos = np.array(obj.translation, dtype=np.float32)

        raw_handle = str(getattr(obj, "handle", "")) # ex) FloorPlan1_physics-Drawer_814ccbab_:0000
        if "fetch" in raw_handle.lower() or "hab_fetch" in raw_handle.lower() or "hab_suction" in raw_handle.lower(): # fetch 로봇 자체 제외. 지금은 articulated object 전체를 다 읽어서, 로봇 본체가 맵에 섞일 수 있음
            continue
        resolved_meta = _resolve_object_metadata(raw_handle, obj)

        """
        # 이건 Habitat-Sim에서 유효한 코드 
        # (Habitat-Lab에선 semantic_id가 scene dataset이 제공하는 거랑 다르게, 
        # 런타임에 object_id + object_ids_start로 재할당하기 때문에 이 로직 쓰면 논리적 오류..) 그래서 일단 주석처리.. ㅠㅠ
        # try:
        #     semantic_id = int(getattr(obj, "semantic_id"))
        # except Exception:
        #     semantic_id = None

        # raw_handle = str(getattr(obj, "handle", "")) # ex) FloorPlan1_physics-Drawer_814ccbab_:0000
        # lh = raw_handle.lower()
        # # 로봇/헬퍼 객체 제외
        # if "fetch" in lh or "hab_fetch" in lh or "hab_suction" in lh:
        #     continue

        # # 1순위: semantic_id -> canonical object type
        # mapped_type = sem_id_to_type.get(semantic_id)
        # # 2순위: handle에서 추출한 이름
        # fallback_type = object_type_from_handle(raw_handle)
        # # 최종 object type 결정: 매핑된 유형이 유효하면 사용, 그렇지 않으면 handle에서 추출한 이름 사용
        # object_type = mapped_type if (mapped_type and mapped_type != "Undefined") else fallback_type
        """
        
        # karma 형식 맞춰서 저장하기 위해 objectId도 포함해서 저장
        obj_info = {
            "objectType": resolved_meta["objectType"],
            "objectName": resolved_meta["objectName"],
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "objectId": str(getattr(obj, "object_id", ""))  # objectId는 시뮬레이터 내부에서 객체를 구분하는 데 사용되는 ID로, 런타임마다 달라질 수 있음
        }
        objects_locations.append(obj_info)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(objects_locations, f, indent=4, ensure_ascii=False)

    print(f"물체 위치 정보 저장 완료: {output_path} (count={len(objects_locations)})")


def _vec3_to_xyz_dict(v):
    # v가 numpy 배열/리스트/Vector3 형태일 때, [0], [1], [2]를 꺼내
    # JSON 저장에 바로 쓰기 좋은 {"x": ..., "y": ..., "z": ...} 형태로 변환
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}

def _range3d_min_max(bb):
    # Habitat-sim의 Range3D는 버전에 따라
    # bb.min / bb.max가 property일 수도 있고
    # bb.min() / bb.max() 메서도 일 수도 있음
    bb_min = getattr(bb, "min", None) # miin 쪽 값(또는 함수) 가져오기
    bb_max = getattr(bb, "max", None) # max 쪽 값(또는 함수) 가져오기

    # bb_min이 함수면 실제 값을 얻기 위해 호출
    if callable(bb_min):
        bb_min = bb_min()
    
    # bb_max이 함수면 실제 값을 얻기 위해 호출
    if callable(bb_max):
        bb_max = bb_max()
    
    # 이후 계산 편의를 위해 np.float32 배열로 통일해서 반환
    return np.array(bb_min, dtype=np.float32), np.array(bb_max, dtype=np.float32)

def _build_axis_aligned_bounding_box(obj):
    """
    Habitat-Sim 객체(obj)에서 AABB(Axis-Aligned Bounding Box)를 꺼내
    AI2-THOR metadata['axisAlignedBoundingBox']와 유사한 JSON 형태로 변환:
    {
      "center": {"x","y","z"},
      "size": {"x","y","z"},
      "cornerPoints": [[x,y,z] * 8]
    }
    """

    # 어떤 객체는 aabb 속성이 없을 수 있으므로 예외 처리
    # 이 경우 객체 위치를 중심으로 한 "크기 0 박스"를 fallback으로 생성
    if not hasattr(obj, "aabb"):
        p = np.array(obj.translation, dtype=np.float32)  # 객체 월드 위치
        p_list = [float(p[0]), float(p[1]), float(p[2])]  # JSON용 실수 리스트로 변환
        return {
            "center": {"x": p_list[0], "y": p_list[1], "z": p_list[2]},  # 중심=현재 위치
            "size": {"x": 0.0, "y": 0.0, "z": 0.0},                       # 박스 크기 0
            "cornerPoints": [p_list[:] for _ in range(8)],                # 꼭짓점 8개 모두 동일
        }
    
    # Habitat-Sim의 obj.aabb는 보통 "local coordinate AABB"
    # 원하는 건 world coordinate AABB이므로, transformation 적용
    try:
        bb_world = habitat_sim.geo.get_transformed_bb(obj.aabb, obj.transformation)
    except Exception:
        # 변환 실패하면 최소한 local AABB라도 사용
        bb_world = obj.aabb

    # 월드 AABB에서 min/max 좌표 (각 축 최소/최대) 획득
    bb_min, bb_max = _range3d_min_max(bb_world)

    # 중심점: (min + max) / 2
    center = (bb_min + bb_max) * 0.5

    # 크기: max - min, 음수 방지 위해 np.maximum으로 0보다 작은 값은 0으로 보정
    size = np.maximum(bb_max - bb_min, 0.0)

    # 이후 corner 계산을 위해 scalar(float)로 변환
    min_x, min_y, min_z = float(bb_min[0]), float(bb_min[1]), float(bb_min[2])
    max_x, max_y, max_z = float(bb_max[0]), float(bb_max[1]), float(bb_max[2])

    # 축 정렬 박스(AABB)의 8개 꼭짓점 구성
    # (AI2-THOR 샘플과 유사하게 x=max 4개 뒤 x=min 4개 순서)
    corner_points = [
        [max_x, max_y, max_z],
        [max_x, max_y, min_z],
        [max_x, min_y, max_z],
        [max_x, min_y, min_z],
        [min_x, max_y, max_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [min_x, min_y, min_z],
    ]

    # 최종 AI2-THOR 유사 포맷으로 반환
    return {
        "center": _vec3_to_xyz_dict(center),  # 중심점
        "size": _vec3_to_xyz_dict(size),      # 박스 크기
        "cornerPoints": corner_points,        # 꼭짓점 8개
    }
    
def first_map_for_next_time(
    sim,
    output_path="/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations.json",
    # semantic_mapping_path=DEFAULT_SEMANTIC_MAP_PATH,
):
    sim = _as_sim(sim)
    # sem_id_to_type = load_semantic_id_to_type(semantic_mapping_path)

    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    objs = list(rom.get_objects_by_handle_substring().values()) + \
           list(aom.get_objects_by_handle_substring().values())

    objects_locations = []
    for obj in objs:
        if not getattr(obj, "is_alive", True):
            continue

        pos = np.array(obj.translation, dtype=np.float32)

        raw_handle = str(getattr(obj, "handle", "")) # ex) FloorPlan1_physics-Drawer_814ccbab_:0000
        if "fetch" in raw_handle.lower() or "hab_fetch" in raw_handle.lower() or "hab_suction" in raw_handle.lower(): # fetch 로봇 자체 제외. 지금은 articulated object 전체를 다 읽어서, 로봇 본체가 맵에 섞일 수 있음
            continue
        resolved_meta = _resolve_object_metadata(raw_handle, obj)

        """
        ## 얘도 위에서 설명한 거랑 마찬가지.. 
        # try:
        #     semantic_id = int(getattr(obj, "semantic_id"))
        # except Exception:
        #     semantic_id = None

        # raw_handle = str(getattr(obj, "handle", ""))
        # mapped_type = sem_id_to_type.get(semantic_id)
        # fallback_type = object_type_from_handle(raw_handle)
        # object_type = mapped_type if (mapped_type and mapped_type != "Undefined") else fallback_type
        """

        obj_info = {
            "objectType": resolved_meta["objectType"],
            "objectName": resolved_meta["objectName"],
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "objectId": str(getattr(obj, "object_id", "")),
            "axisAlignedBoundingBox": _build_axis_aligned_bounding_box(obj),
        }
        objects_locations.append(obj_info)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(objects_locations, f, indent=4, ensure_ascii=False)

    print(f"물체 위치 + AABB 저장 완료: {output_path} (count={len(objects_locations)})")


def second_map(
        sim,
        output_path="/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations2.json",
        # semantic_mapping_path=DEFAULT_SEMANTIC_MAP_PATH
):
    # ai2thor 시뮬레이터에서 karma 구현은 second_map의 인자로 c.step 후의 event가 들어오는데, 
    # habitat-sim에서는 시뮬레이터 객체(sim)를 직접 받아서 그 안에서 현재 상태의 객체 정보들을 꺼내오는 방식으로 구현해야 함
    # 그래서 인자로 event 대신 sim을 받도록 수정함
    # 그리고 사실 함수 내용 자체는 first_map과 동일 

    sim = _as_sim(sim)
    # sem_id_to_type = load_semantic_id_to_type(semantic_mapping_path) # 시뮬레이터에서 객체의 semantic_id를 실제 객체 유형으로 변환하기 위한 매핑 정보 로드

    # 맵 안의 모든 사물 가져오기 (oracle 방식)
    # 시뮬레이터에서 두 가지 종류의 사물을 모두 가져와서 하나로 합치기
        ## - Rigid Object: 형태가 변하지 않는 일반적인 물체
        ## - Articulated Object: 관절이 있는 물체(예: 문, 서랍 등)
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    objs = list(rom.get_objects_by_handle_substring().values()) + \
           list(aom.get_objects_by_handle_substring().values())
    
    objects_locations = []
    for obj in objs:
        if not getattr(obj, "is_alive", True): # pickupable이 False인 객체는 static object로 간주
            continue

        pos = np.array(obj.translation, dtype=np.float32)

        raw_handle = str(getattr(obj, "handle", "")) # ex) FloorPlan1_physics-Drawer_814ccbab_:0000
        if "fetch" in raw_handle.lower() or "hab_fetch" in raw_handle.lower() or "hab_suction" in raw_handle.lower(): # fetch 로봇 자체 제외. 지금은 articulated object 전체를 다 읽어서, 로봇 본체가 맵에 섞일 수 있음
            continue
        resolved_meta = _resolve_object_metadata(raw_handle, obj)

        """
        ## 얘도 위에서 설명한거랑 마찬가지
        # try:
        #     semantic_id = int(getattr(obj, "semantic_id"))
        # except Exception:
        #     semantic_id = None

        # raw_handle = str(getattr(obj, "handle", "")) # ex) FloorPlan1_physics-Drawer_814ccbab_:0000

        # # 1순위: semantic_id -> canonical object type
        # mapped_type = sem_id_to_type.get(semantic_id)
        # # 2순위: handle에서 추출한 이름
        # fallback_type = object_type_from_handle(raw_handle)
        # # 최종 object type 결정: 매핑된 유형이 유효하면 사용, 그렇지 않으면 handle에서 추출한 이름 사용
        # object_type = mapped_type if (mapped_type and mapped_type != "Undefined") else fallback_type
        """
    
        # karma 형식 맞춰서 저장하기 위해 objectId도 포함해서 저장
        obj_info = {
            "objectType": resolved_meta["objectType"],
            "objectName": resolved_meta["objectName"],
            "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "objectId": str(getattr(obj, "object_id", ""))  # objectId는 시뮬레이터 내부에서 객체를 구분하는 데 사용되는 ID로, 런타임마다 달라질 수 있음
        }
        objects_locations.append(obj_info)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(objects_locations, f, indent=4, ensure_ascii=False)

    print(f"물체 위치 정보 저장 완료: {output_path} (count={len(objects_locations)})")
