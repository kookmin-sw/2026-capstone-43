# from queue import Queue
import os
import re
import numpy as np
import habitat_sim
import json
from functools import lru_cache
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from semantic_utils import (
    object_type_from_handle,
)

# ---------------------------------------- HSG + ConceptGraph 검색 utils ----------------------------------------
"""
1) planner가 넘긴 목표 해석
    - ex: "Apple"
    - ex: {"object": "Apple", "room": "kitchen", "floor": 1}

2) hierarchical 3D scene graph (HSG)를 사용해서 floor -> room -> object 순서로 검색 범위 좁히기

3) 최종적으로 선택된 HSG object의 source_id를 사용해서 ConceptGraph obj_json에서 원본 object 정보 다시 가져옴

4) navigation용 좌표는 ConceptGraph의 bbox_center([x, y, z]) 우선 사용

역할 정리:
- HSG: 계층 구조 기반 검색 (floor / room / object 좁히기)
- ConceptGraph: 최종 object geometry / center / caption / extent 등 상세 정보
- Simulator live object: 나중에 Pickup/Put에서 실제 runtime object id 잡을 때 사용 
"""

# HSG가 저장된 루트 디렉토리
DEFAULT_HSG_ROOT = "/home/yuchaehee/long_term_memory_project/my_local_data/hierarchy_scene_graphs"
# 현재 사용 중인 HSG dataset 이름
DEFAULT_HSG_DATASET = "replica_hov"
# ConceptGraph obj_json이 저장된 루트 디렉토리
DEFAULT_CONCEPTGRAPH_ROOT = (
    "/home/yuchaehee/long_term_memory_project/my_local_data/hssd"
)

# 현재 사용 중인 ConceptGraph 결과 이름
DEFAULT_CONCEPTGRAPH_EXPERIMENT = os.environ.get(
    "KARMA_CONCEPTGRAPH_EXPERIMENT",
    "r_mapping_stride5",
)

# object label canonicalization에 사용할 내부 vocab 파일
# - 현재 실행 코드가 이 json을 직접 쓰고 있지는 않지만,
#   AI2-THOR / KARMA 계열에서 어떤 이름을 대표 이름으로 쓰는지 알려주는
#   기준표 역할을 한다.
# - 아래 canonicalization 레이어는 이 vocab을 읽어서
#   "AlarmClock" -> "alarm clock", "TissueBox" -> "tissue box" 같은
#   formatting mismatch를 먼저 흡수한다.
DEFAULT_OBJECT_VOCAB_PATH = (
    "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/resources/object_vocab.json"
)

def normalize_navigation_target(dest_obj) -> Dict[str, Any]:
    """
    planner / task_functions가 넘긴 목표를 통일된 dict 형식으로 정규화한다.

    지원 입력 형태:
    1) 문자열
       - 예: "Apple"
       - 이 경우 object만 주어진 것으로 해석하고 room / room_instance / floor는 None으로 둔다.

    2) dict
       - 예: {"object": "Apple", "room": "kitchen", "floor": 1}
       - 예: {"object": "Pillow", "room_instance": "bathroom_2", "floor": 1}

    반환 형식:
        {
            "object": ...,
            "room": ...,
            "room_instance": ...,
            "floor": ...,
        }

    주의:
    - 이 함수는 전체 instruction 문장을 파싱하지 않는다.
    - planner가 이미 만들어 준 dest_obj만 해석한다.
    """
    # 기존 planner와의 backward compatibility:
    # 문자열 하나만 들어와도 object-only query로 계속 동작하게 한다.
    if isinstance(dest_obj, str):
        return {
            "object": dest_obj,
            "room": None,
            "room_instance": None,
            "floor": None,
        }

    # dict를 받는 경우에는 여러 key alias를 유연하게 허용한다.
    # 이후 planner 예시를 바꾸더라도 여기서 받아줄 수 있게 하기 위함이다.
    if isinstance(dest_obj, dict):
        target = {
            "object": dest_obj.get("object")
            or dest_obj.get("object_name")
            or dest_obj.get("name")
            or dest_obj.get("dest_obj"),
            "room": dest_obj.get("room")
            or dest_obj.get("room_name"),
            "room_instance": dest_obj.get("room_instance")
            or dest_obj.get("instance_name")
            or dest_obj.get("roomInstance"),
            "floor": dest_obj.get("floor")
            or dest_obj.get("floor_number")
            or dest_obj.get("floor_name"),
        }

        # object는 검색의 최소 단위이므로 반드시 있어야 한다.
        if not target["object"]:
            raise ValueError(f"Target object is missing: {dest_obj}")

        return target

    raise ValueError(f"Unsupported target format: {dest_obj}")


# ========================================= [MOD: strict navigation target helpers] =========================================
def require_fully_specified_navigation_target(dest_obj) -> Dict[str, Any]:
    """
    GoToObject 전용 strict validator.

    왜 필요한가:
    - 이제 GoToObject는 "정확히 어디로 가야 하는지 아는 경우"에만 쓰고,
      floor / room 정보가 덜 주어진 경우는 Explore 계열이 맡도록 역할을 분리하려고 한다.

    통과 조건:
    - object는 반드시 있어야 함
    - floor는 반드시 있어야 함
    - room 또는 room_instance 중 하나는 반드시 있어야 함

    반환:
        normalize_navigation_target(...)로 정규화된 spec dict

    실패 시:
        ValueError를 발생시켜 호출부가 "Explore를 써야 하는 상황"임을 명확히 알 수 있게 한다.
    """
    spec = normalize_navigation_target(dest_obj)

    has_object = bool(spec.get("object"))
    has_floor = spec.get("floor") is not None
    has_room = bool(spec.get("room") or spec.get("room_instance"))

    if has_object and has_floor and has_room:
        return spec

    raise ValueError(
        "GoToObject requires a fully specified target: "
        "object + floor + room/room_instance. "
        f"Received={spec}. Use Explore for underspecified targets."
    )


def resolve_runtime_object_from_target_context(
    sim,
    target_context: Dict[str, Any],
    agent_idx: int = 0,
    max_xz_dist: float = 1.75,
    allow_nearest_fallback: bool = False,
) -> Dict[str, Any]:
    """
    graph / ConceptGraph 기준 target context를 실제 Habitat runtime object로 연결한다.

    이 함수의 목적:
    - HSG / ConceptGraph는 "semantic memory + reference geometry" 역할
    - Habitat runtime object는 "실제로 snap_to_obj / receptacle placement에 쓰는 실행 대상" 역할
    - 따라서 PickupObject / PutObject 직전에 "memory target -> runtime object id" 연결이 필요하다.

    입력으로 기대하는 target_context 예시:
        {
            "query": {"object": "Apple", "room": "kitchen", "floor": 1},
            "graph_object_name": "apple",
            "conceptgraph_object_tag": "apple",
            "conceptgraph_bbox_center": [-1.2, 0.9, 1.57],
            ...
        }

    매칭 방식:
    1) target_context 안의 reference xyz를 기준으로 주변 sim object 후보를 수집
    2) object_type / handle이 target label과 얼마나 잘 맞는지 text score 계산
    3) text score가 높은 후보를 우선, 거리가 가까운 후보를 차선으로 선택

    주의:
    - 이 함수는 의도적으로 deterministic 하게 만든다.
    - planner LLM이 runtime object id를 직접 고르지 않도록 하는 것이 목적이다.
    """
    if not isinstance(target_context, dict) or not target_context:
        raise ValueError(f"Invalid target_context: {target_context}")

    # ------------------------------------------------------------------
    # 1) reference xyz 결정
    # ------------------------------------------------------------------
    # 가장 우선시하는 기준은 ConceptGraph bbox_center다.
    # 이 값이 가장 "원래 target object가 있어야 할 위치"에 가깝기 때문이다.
    #
    # 다만 상황에 따라 호출부가 nav_xyz만 갖고 있을 수도 있으므로,
    # reference_xyz / nav_xyz도 순서대로 fallback 후보에 넣는다.
    ref_xyz = None
    for key in ("conceptgraph_bbox_center", "reference_xyz", "nav_xyz"):
        value = target_context.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (3,) and np.all(np.isfinite(arr)):
            ref_xyz = arr
            break

    if ref_xyz is None:
        raise ValueError(
            "target_context does not contain a valid reference xyz. "
            f"context_keys={list(target_context.keys())}"
        )

    # ------------------------------------------------------------------
    # 2) target label 후보 결정
    # ------------------------------------------------------------------
    # 어떤 문자열을 sim object_type과 비교할지 정한다.
    # 일반적으로는 ConceptGraph object_tag가 가장 좋고,
    # 없으면 graph object name, 그마저 없으면 query.object를 사용한다.
    query = target_context.get("query", {}) if isinstance(target_context.get("query"), dict) else {}
    label_candidates = [
        target_context.get("conceptgraph_object_tag"),
        target_context.get("graph_object_name"),
        query.get("object"),
    ]
    label_candidates = [str(x) for x in label_candidates if x]
    desired_label = label_candidates[0] if label_candidates else ""

    if not desired_label:
        raise ValueError(
            "target_context does not contain a usable target label. "
            f"context={target_context}"
        )

    cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_idx)

    # ------------------------------------------------------------------
    # 3) 현재 sim 안의 runtime object 후보 수집
    # ------------------------------------------------------------------
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    sim_objs = list(rom.get_objects_by_handle_substring().values()) + \
               list(aom.get_objects_by_handle_substring().values())

    typed_candidates: List[Dict[str, Any]] = []
    loose_candidates: List[Dict[str, Any]] = []

    for obj in sim_objs:
        if not getattr(obj, "is_alive", True):
            continue

        pos = np.asarray(getattr(obj, "translation", np.zeros(3)), dtype=np.float32)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            continue

        dist_xz = distance_xz(pos, ref_xyz)
        if np.isfinite(max_xz_dist) and dist_xz > float(max_xz_dist):
            continue

        handle = str(getattr(obj, "handle", ""))
        object_id = str(getattr(obj, "object_id", ""))
        template_class = str(getattr(obj, "template_class", ""))
        object_type = object_type_from_handle(handle, template_class)

        # object_type은 semantic label이므로 alias-aware semantic match를 사용한다.
        # 예:
        # - desired_label="Mug"
        # - runtime object_type="cup"
        # 이 경우 canonical alias 덕분에 여전히 높은 점수를 받을 수 있다.
        #
        # handle은 긴 런타임 문자열이므로 free-text match로 비교한다.
        text_score = max(
            score_object_label_match(desired_label, object_type),
            score_object_query_against_text(desired_label, handle),
        )

        candidate = {
            "runtime_object_id": object_id,
            "runtime_handle": handle,
            "runtime_object_type": object_type,
            "runtime_position": pos.astype(np.float32).tolist(),
            "reference_dist_xz": float(dist_xz),
            "agent_dist_xz": float(distance_xz(cur_pos, pos)),
            "text_score": float(text_score),
            "match_source": "text_and_geometry",
        }

        loose_candidates.append(candidate)
        if text_score >= 0:
            typed_candidates.append(candidate)

    # ------------------------------------------------------------------
    # 4) 가장 좋은 후보 선택
    # ------------------------------------------------------------------
    # 1순위: text score 높은 후보
    # 2순위: reference xyz에 더 가까운 후보
    # 3순위: 현재 agent에도 가까운 후보
    if typed_candidates:
        typed_candidates.sort(
            key=lambda x: (
                -x["text_score"],
                x["reference_dist_xz"],
                x["agent_dist_xz"],
            )
        )
        return typed_candidates[0]

    # 필요할 때만 "가까운 물체" fallback을 허용한다.
    # 기본은 False로 두는 이유:
    # - pickup/put은 잘못된 target을 집거나 놓는 리스크가 크기 때문이다.
    if allow_nearest_fallback and loose_candidates:
        loose_candidates.sort(
            key=lambda x: (
                x["reference_dist_xz"],
                x["agent_dist_xz"],
            )
        )
        best = dict(loose_candidates[0])
        best["match_source"] = "nearest_geometry_fallback"
        return best

    raise ValueError(
        "No runtime sim object matched the memory target context. "
        f"desired_label={desired_label}, ref_xyz={ref_xyz.tolist()}, max_xz_dist={max_xz_dist}"
    )


def _scene_key_from_scene_id(scene_id: str) -> str:
    """
    scene_id에서 실제 폴더 이름으로 쓰이는 scene key를 만든다.

    예:
        "108736872_177263607.scene_instance"
        -> "108736872_177263607"

    이유:
    - HSG 폴더와 ConceptGraph 폴더는 scene_instance 확장자 없는 이름을 사용한다.
    """
    raw = str(scene_id).strip()
    return Path(raw).stem


def _normalize_text(text: Optional[str]) -> str:
    """
    문자열 비교 전에 표준화한다.

    하는 일:
    - None 방지
    - 소문자화
    - underscore / hyphen 을 공백으로 통일
    - 연속 공백을 하나로 축소

    예:
        "bathroom_2" -> "bathroom 2"
        "Living-Room" -> "living room"
    """
    s = str(text or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


# ========================================= [MOD: object label canonicalization helpers] =========================================
# 이 블록의 목적:
# - planner / rigid_objs / sim runtime object / ConceptGraph / HSG가
#   서로 다른 이름 공간(label space)을 쓰더라도,
#   검색과 매칭을 조금 더 안정적으로 만들기 위함이다.
#
# 대표적인 문제:
# - planner / runtime 쪽은 "Mug"를 기대
# - ConceptGraph / HSG는 "cup"으로 라벨링
# - 그러면 기존 exact / substring 기반 로직만으로는
#   GoToObject target lookup이 실패할 수 있다.
#
# 따라서 아래 레이어는:
# 1) "AlarmClock" vs "alarm clock" 같은 formatting 차이
# 2) "cup" vs "mug" 같은 cross-dataset alias 차이
# 를 흡수할 수 있는 internal canonical label space를 만든다.
#
# 설계 원칙:
# - 비슷하다고 해서 전부 큰 그룹으로 뭉개지 않는다.
# - 조작/검색에 자주 등장하는 actionable object 위주로만 보수적으로 묶는다.
# - 예를 들어 cup <-> mug 는 묶지만,
#   bowl <-> plate <-> cup 같은 다른 물체까지 한 그룹으로 합치지는 않는다.
#
# canonical label은 내부적으로 모두 lower-case phrase로 관리한다.
# 예:
# - "AlarmClock" -> "alarm clock"
# - "Mug"        -> "mug"
# - "TissueBox"  -> "tissue box"
#
# 이후 object query / runtime object match / pickup / put context match는
# 모두 이 canonicalization 규칙을 공유하게 된다.
MANUAL_OBJECT_CANONICAL_GROUPS: Dict[str, List[str]] = {
    # ConceptGraph/HSG가 "cup"이라고 뽑았지만,
    # 실제 조작 대상이나 planner 표현은 "Mug"인 경우가 자주 발생한다.
    "mug": [
        "mug",
        "cup",
        "coffee cup",
        "tea cup",
        "teacup",
    ],

    # HSSD / ScanNet 계열은 "couch"를 자주 쓰고,
    # AI2-THOR / KARMA 계열은 "Sofa"를 더 자주 쓴다.
    "sofa": [
        "sofa",
        "couch",
    ],

    # dataset마다 "plant" / "potted plant" / "HousePlant"로 섞이는 경우를 흡수
    "house plant": [
        "house plant",
        "houseplant",
        "plant",
        "potted plant",
    ],

    # 쓰레기통 계열은 dataset마다 이름이 심하게 흔들린다.
    "garbage can": [
        "garbage can",
        "trash can",
        "trash bin",
        "recycling bin",
        "bin",
    ],

    # ConceptGraph class list는 "coffee maker"를 쓰고,
    # AI2-THOR/KARMA 쪽은 "CoffeeMachine"을 쓰는 경우가 있다.
    "coffee machine": [
        "coffee machine",
        "coffee maker",
    ],

    # TV 계열
    "television": [
        "television",
        "tv",
    ],

    # formatting 차이가 큰 것들
    "alarm clock": [
        "alarm clock",
        "alarmclock",
    ],
    "light switch": [
        "light switch",
        "lightswitch",
    ],
    "tissue box": [
        "tissue box",
        "tissuebox",
    ],
    "tv stand": [
        "tv stand",
        "tvstand",
    ],

    # 냉장고도 fridge / refrigerator 로 자주 섞인다.
    "fridge": [
        "fridge",
        "refrigerator",
    ],

    # storage bin이랑 box도 섞이는 경우가 있다.
    "storage bin": [
        "storage bin",
        "storage box",
        "box",
    ],
}


def _camel_or_pascal_to_phrase(text: Optional[str]) -> str:
    """
    CamelCase / PascalCase / underscore 이름을 사람이 읽는 phrase로 풀어준다.

    예:
    - "AlarmClock" -> "alarm clock"
    - "Tissue_Box" -> "tissue box"
    - "TVStand"    -> "tv stand"

    이 함수는 object_vocab.json 같은 레거시 vocab을
    현재 canonical phrase 체계로 옮길 때 사용한다.
    """
    s = str(text or "").strip()
    if not s:
        return ""

    # CamelCase 경계에 공백을 넣는다.
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    # 약어 뒤에 오는 일반 단어 경계도 분리한다.
    # 예: TVStand -> TV Stand
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return _normalize_text(s)


@lru_cache(maxsize=1)
def _build_object_label_alias_index(
    object_vocab_path: str = DEFAULT_OBJECT_VOCAB_PATH,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    object label canonicalization에 필요한 alias index를 한 번만 구성해서 캐시한다.

    구성 방식:
    1) object_vocab.json을 읽어서 formatting alias(CamelCase, underscore 등) 자동 생성
    2) MANUAL_OBJECT_CANONICAL_GROUPS를 덮어써서 cross-dataset alias 추가

    반환 구조:
        {
            "alias_to_canonical": {
                "cup": "mug",
                "mug": "mug",
                "alarmclock": "alarm clock",
                ...
            },
            "canonical_to_aliases": {
                "mug": {"mug", "cup", "coffee cup", ...},
                ...
            }
        }
    """
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_aliases: Dict[str, Set[str]] = {}

    def _register_alias(canonical_label: str, alias_label: str) -> None:
        """
        canonical label 하나와 alias 하나를 index에 등록한다.

        later registration이 earlier registration을 덮어쓴다.
        즉 manual alias가 object_vocab 기반 자동 별칭보다 우선한다.
        """
        canonical_key = _normalize_text(canonical_label)
        alias_key = _normalize_text(alias_label)

        if not canonical_key or not alias_key:
            return

        alias_to_canonical[alias_key] = canonical_key
        canonical_to_aliases.setdefault(canonical_key, set()).add(alias_key)
        canonical_to_aliases[canonical_key].add(canonical_key)

    # 1) object_vocab.json 기반 formatting alias 자동 생성
    try:
        vocab_entries = json.loads(Path(object_vocab_path).read_text(encoding="utf-8"))
    except Exception:
        vocab_entries = []

    if isinstance(vocab_entries, list):
        for raw_name in vocab_entries:
            if not isinstance(raw_name, str):
                continue

            canonical = _camel_or_pascal_to_phrase(raw_name)
            if not canonical:
                continue

            _register_alias(canonical, canonical)
            _register_alias(canonical, raw_name)
            _register_alias(canonical, raw_name.lower())
            _register_alias(canonical, raw_name.replace("_", " "))
            _register_alias(canonical, raw_name.replace("_", ""))
            _register_alias(canonical, canonical.replace(" ", ""))

    # 2) cross-dataset semantic alias 수동 등록
    for canonical, aliases in MANUAL_OBJECT_CANONICAL_GROUPS.items():
        _register_alias(canonical, canonical)
        for alias in aliases:
            _register_alias(canonical, alias)

    return {
        "alias_to_canonical": alias_to_canonical,
        "canonical_to_aliases": canonical_to_aliases,
    }


def canonicalize_object_label(label: Optional[str]) -> str:
    """
    object label 하나를 internal canonical phrase로 변환한다.

    예:
    - "Mug"         -> "mug"
    - "cup"         -> "mug"
    - "AlarmClock"  -> "alarm clock"
    - "tissue_box"  -> "tissue box"

    alias 사전에 없으면 단순 정규화 결과를 그대로 반환한다.
    """
    normalized = _normalize_text(label)
    if not normalized:
        return ""

    alias_index = _build_object_label_alias_index()
    return alias_index["alias_to_canonical"].get(normalized, normalized)


def expand_object_query_terms(label: Optional[str]) -> List[str]:
    """
    object query 하나를 여러 검색 표현으로 확장한다.

    예:
    - query="Mug"
      -> ["mug", "cup", "coffee cup", "tea cup", ...]

    query_object(...)나 runtime object matching이 exact string 하나만 보면
    cross-dataset label mismatch를 놓치기 쉬우므로,
    canonical label과 alias 집합으로 검색어를 확장해준다.
    """
    normalized = _normalize_text(label)
    if not normalized:
        return []

    alias_index = _build_object_label_alias_index()
    canonical = canonicalize_object_label(normalized)
    alias_terms = alias_index["canonical_to_aliases"].get(canonical, set())

    ordered_terms: List[str] = []
    for term in [normalized, canonical, *sorted(alias_terms)]:
        term = _normalize_text(term)
        if term and term not in ordered_terms:
            ordered_terms.append(term)

    return ordered_terms


def score_object_query_against_text(
    object_query: Optional[str],
    candidate_text: Optional[str],
) -> int:
    """
    object query를 자유 텍스트(candidate_text)와 비교할 때 사용하는 점수 함수.

    사용 예:
    - object caption 비교
    - runtime handle 문자열 비교

    방법:
    - object query를 alias까지 확장한 뒤
    - 각 alias를 candidate_text에 대해 _score_text_match(...)로 평가
    - 가장 높은 점수를 사용
    """
    if not object_query or not candidate_text:
        return -1

    best_score = -1
    for query_term in expand_object_query_terms(object_query):
        best_score = max(best_score, _score_text_match(query_term, candidate_text))
    return best_score


def score_object_label_match(
    query_label: Optional[str],
    candidate_label: Optional[str],
) -> int:
    """
    object label끼리 semantic-aware match score를 계산한다.

    일반 텍스트 매칭보다 object-specific 한 이유:
    - Cup/Mug처럼 dataset마다 대표 이름이 다를 수 있고
    - AlarmClock/alarm clock처럼 formatting 차이도 흔하기 때문이다.

    점수 의도:
    - 완전 동일 문자열: 가장 높음
    - canonical label 동일(예: mug vs cup): 매우 높음
    - alias 확장 후 부분 일치: 그보다 낮음
    - 전혀 안 맞음: -1
    """
    q_norm = _normalize_text(query_label)
    c_norm = _normalize_text(candidate_label)

    if not q_norm or not c_norm:
        return -1

    if q_norm == c_norm:
        return 120

    q_canonical = canonicalize_object_label(q_norm)
    c_canonical = canonicalize_object_label(c_norm)

    if q_canonical and c_canonical and q_canonical == c_canonical:
        return 105

    best_score = -1
    candidate_terms = expand_object_query_terms(candidate_label)
    for query_term in expand_object_query_terms(query_label):
        best_score = max(best_score, _score_text_match(query_term, c_norm))
        for candidate_term in candidate_terms:
            best_score = max(best_score, _score_text_match(query_term, candidate_term))

    return best_score


def object_labels_semantically_match(
    query_label: Optional[str],
    candidate_label: Optional[str],
    min_score: int = 90,
) -> bool:
    """
    두 object label이 semantic하게 같은 물체를 가리키는지 bool로 판단한다.

    기본 threshold=90 의미:
    - cup <-> mug 같은 canonical alias는 True
    - alarm clock <-> AlarmClock 같은 formatting 차이도 True
    - table <-> dining table 같은 느슨한 상하위 관계는 기본적으로 False
    """
    return score_object_label_match(query_label, candidate_label) >= int(min_score)


def _safe_int(value) -> Optional[int]:
    """
    int 변환이 가능한 값이면 int로 바꾸고,
    실패하면 None을 반환한다.

    source_id, floor 번호 등을 다룰 때 자주 사용한다.
    """
    try:
        return int(value)
    except Exception:
        return None


def _parse_floor_number(value) -> Optional[int]:
    """
    floor 필드 값만 받아서 층 번호를 추출한다.

    주의:
    - 이 함수는 전체 instruction 문장을 파싱하는 함수가 아니다.
    - planner가 이미 분리해 둔 target_spec["floor"] 값에만 사용한다.

    예:
        1          -> 1
        "1층"      -> 1
        "floor 1"  -> 1
        "floor_0"  -> 0
    """
    if isinstance(value, int):
        return value

    m = re.search(r"-?\d+", str(value))
    if m is None:
        return None
    return int(m.group(0))


def _score_text_match(query: Optional[str], candidate: Optional[str]) -> int:
    """
    query와 candidate 사이의 단순 텍스트 매칭 점수 계산

    점수 규칙:
    - exact normalized match -> 100
    - exact compact match(공백 제거 후 일치) -> 95
    - query가 candidate 안에 포함 -> 85
    - candidate가 query 안에 포함
    - token overlap 있음 -> 40 + 겹친 token 수

    label / caption 기반 1차 검색용
    """
    q = _normalize_text(query)
    c = _normalize_text(candidate)

    if not q or not c:
        return -1
    
    # 완전 일치
    if q == c :
        return 100
    
    # 공백 제거 후 일치
    # ex:
    # - "bathroom2" vs "bathroom 2"
    # - "bed room" vs "bedroom"
    q_compact = q.replace(" ", "")
    c_compact = c.replace(" ", "")
    if q_compact == c_compact:
        return 95
    
    # 부분 포함
    if q in c or q_compact in c_compact:
        return 85
    if c in q or c_compact in q_compact:
        return 75

    # token 단위 overlap
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    overlap = len(q_tokens & c_tokens)
    if overlap > 0:
        return 40 + overlap

    return -1

def _mean_xz_from_vertices(vertices) -> Optional[np.ndarray]:
    """
    HSG room / object vertices의 평균 xz 중심 구후ㅏ기
    - 현재 생성된 HSG object vertices는 사실상 2D (x, z)로 보는 것이 적절
    - 일부 경우 3D 점일 수도 있으므로 둘 다 처리

    결과: 
        np.ndarray([x, z], dtype=float32)
        또는 계산 불가 시 None
    """
    arr = np.asarray(vertices, dtype=np.float32)

    # 점이 하나도 없으면 중심 계산 불가
    if arr.size == 0:
        return None
    
    # (N, D) 형태가 아니면 예상한 구조가 아님
    if arr.ndim != 2:
        return None
    
    # 3차원 이상이면 x, z만 사용
    if arr.shape[1] >= 3:
        arr = arr[: ,[0, 2]]

    return np.mean(arr[:, :2], axis=0).astype(np.float32)

def _build_hsg_graph_dir(
        scene_id: str,
        dataset_name: str=DEFAULT_HSG_DATASET,
        hsg_root: str=DEFAULT_HSG_ROOT,
) -> Path:
    """
    scene_id에 대응하는 HSG graph 디렉토리 경로 생성
    """
    scene_key = _scene_key_from_scene_id(scene_id)
    return Path(hsg_root) / dataset_name / scene_key / "graph"

def _build_conceptgraph_obj_json_path(
        scene_id: str,
        conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
        experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Path:
    """
    scene_id에 대응하는 ConceptGraph obj_json 경로 만들기

    ex)
        scene_id = "108736872_177263607.scene_instance"
        ->
        /.../hssd/108736872_177263607/exps/r_mapping_stride10/obj_json_r_mapping_stride10.json
    """
    env_override = os.environ.get("KARMA_CONCEPTGRAPH_OBJ_JSON_PATH", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()

    scene_key = _scene_key_from_scene_id(scene_id)
    return(
        Path(conceptgraph_root)
        / scene_key
        / "exps"
        / experiment_name
        / f"obj_json_{experiment_name}.json"
    )

@lru_cache(maxsize=8)
def load_hierarchical_graph_index(
    scene_id: str,
    dataset_name: str = DEFAULT_HSG_DATASET,
    hsg_root: str = DEFAULT_HSG_ROOT,
) -> Dict[str, Any]:
    """
    HSG export json들을 읽어서 index 형태로 캐시한다.

    캐시 이유:
    - GoToObject가 여러 번 호출될 수 있는데
      매번 floors / rooms / objects json을 전부 다시 읽으면 비효율적이다.

    반환 예:
        {
            "scene_id": ...,
            "scene_key": ...,
            "graph_dir": ...,
            "floors_by_id": {...},
            "rooms_by_id": {...},
            "objects_by_id": {...},
            "ordered_floors": [...],
        }
    """
    graph_dir = _build_hsg_graph_dir(
        scene_id=scene_id,
        dataset_name=dataset_name,
        hsg_root=hsg_root,
    )

    if not graph_dir.exists():
        raise FileNotFoundError(f"HSG graph dir not found: {graph_dir}")

    floors_dir = graph_dir / "floors"
    rooms_dir = graph_dir / "rooms"
    objects_dir = graph_dir / "objects"

    floors_by_id: Dict[str, Dict[str, Any]] = {}
    rooms_by_id: Dict[str, Dict[str, Any]] = {}
    objects_by_id: Dict[str, Dict[str, Any]] = {}

    # -------------------- floors 로드 --------------------
    for path in sorted(floors_dir.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        floor_id = str(meta["floor_id"])

        floors_by_id[floor_id] = {
            "floor_id": floor_id,
            "name": meta.get("name", floor_id),
            "rooms": [str(x) for x in meta.get("rooms", [])],
            "floor_zero_level": meta.get("floor_zero_level"),
            "floor_height": meta.get("floor_height"),
            "json_path": str(path),
        }

    # -------------------- rooms 로드 --------------------
    for path in sorted(rooms_dir.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        room_id = str(meta["room_id"])

        rooms_by_id[room_id] = {
            "room_id": room_id,
            "floor_id": str(meta["floor_id"]),
            "name": meta.get("name", room_id),                  # room type ex) bathroom
            "instance_name": meta.get("instance_name"),         # room instance ex) bathroom_2
            "objects": [str(x) for x in meta.get("objects", [])],
            "vertices": meta.get("vertices", []),
            "json_path": str(path),
        }

    # -------------------- objects 로드 --------------------
    for path in sorted(objects_dir.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        object_id = str(meta["object_id"])
        room_id = str(meta["room_id"])

        # object json에는 floor_id가 직접 없으므로 parent room에서 복원한다.
        parent_room = rooms_by_id.get(room_id)
        floor_id = None if parent_room is None else parent_room["floor_id"]

        objects_by_id[object_id] = {
            "object_id": object_id,
            "room_id": room_id,
            "floor_id": floor_id,
            "name": meta.get("name", object_id),
            "caption": meta.get("caption"),
            "source_id": meta.get("source_id"),
            "vertices": meta.get("vertices", []),
            "json_path": str(path),
        }

    # 사람이 말하는 "1층 / 2층" 순서를 맞추기 위해 floor_zero_level 기준으로 정렬한다.
    ordered_floors = sorted(
        floors_by_id.values(),
        key=lambda item: float(item.get("floor_zero_level") or 0.0),
    )

    return {
        "scene_id": scene_id,
        "scene_key": _scene_key_from_scene_id(scene_id),
        "graph_dir": str(graph_dir),
        "floors_by_id": floors_by_id,
        "rooms_by_id": rooms_by_id,
        "objects_by_id": objects_by_id,
        "ordered_floors": ordered_floors,
    }

@lru_cache(maxsize=8)
def load_conceptgraph_index(
    scene_id: str,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Dict[str, Any]:
    """
    ConceptGraph obj_json을 읽어서 source_id(id) 기반 index로 캐시한다.

    반환 예:
        {
            "obj_json_path": "...",
            "objects_by_source_id": {
                441: {
                    "object_key": "object_64",
                    "id": 441,
                    "object_tag": "apple",
                    "object_caption": "...",
                    "bbox_center": [...],
                    "bbox_extent": [...],
                    "bbox_volume": 0.02,
                },
                ...
            }
        }
    """
    obj_json_path = _build_conceptgraph_obj_json_path(
        scene_id=scene_id,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )

    if not obj_json_path.exists():
        raise FileNotFoundError(f"ConceptGraph obj_json not found: {obj_json_path}")

    raw = json.loads(obj_json_path.read_text(encoding="utf-8"))
    objects_by_source_id: Dict[int, Dict[str, Any]] = {}

    for object_key, entry in raw.items():
        source_id = _safe_int(entry.get("id"))
        if source_id is None:
            continue

        objects_by_source_id[source_id] = {
            "object_key": object_key,
            **entry,
        }

    return {
        "obj_json_path": str(obj_json_path),
        "objects_by_source_id": objects_by_source_id,
    }

def query_floor(
        graph_index: Dict[str, Any],
        floor_query
) -> Optional[Dict[str, Any]]:
    """
    floor query를 해석해서 target floor 하나 반환

    입력 예시:
    - 1
    - "1"
    - "1층"
    - "floor 1"
    - 내부 floor_id 직접 지정: "0"

    우선 순위
    - 1. 내부 floor_id exact match
    - 2. 사람이 말한 층 번호(1층, 2층, ...) 해석
    - 3. floor name 기반 fallback
    """
    # floor 제약이 없으면 global search
    if floor_query is None:
        return None
    
    floors_by_id = graph_index["floors_by_id"]
    ordered_floors = graph_index["ordered_floors"]

    # 1) 내부 floor_id exact match
    #    예: "0" 이면 floors_by_id["0"] 바로 사용
    floor_query_str = str(floor_query).strip()
    if floor_query_str in floors_by_id:
        return floors_by_id[floor_query_str]

    # 2) 사람이 말한 층 번호 해석
    #    예: 1층 -> ordered_floors[0]
    floor_number = _parse_floor_number(floor_query)
    if floor_number is not None and 1 <= floor_number <= len(ordered_floors):
        return ordered_floors[floor_number - 1]

    # 3) 이름 기반 fallback
    scored = []
    for floor in ordered_floors:
        score = _score_text_match(floor_query, floor.get("name"))
        if score >= 0:
            scored.append((score, floor))

    if not scored:
        raise ValueError(f"Floor not found for query: {floor_query}")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def query_room(
        graph_index: Dict[str, Any],
        room_query: Optional[str],
        floor_meta: Optional[Dict[str, Any]] = None,
        room_instance_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    room 검색 수행 후 room 후보 리스트 반환
    - 같은 타입의 방 여러 개 있을 수 있어서 리스트 반환
        -- ex) bathroom_1, bathroom_2, bathroom_3
    - 따라서 만약 roomtype만 중진 경우에는 여러 room을 후보로 유지한 뒤
        object 검색 단계에서 최종적으로 좁히는 것이 안전

    room 검색 규칙:
    - 1. room_instance_query가 있으면 instance_name exact/near-exact match 우선
        -- ex: "bathroom_2"
        -- 이 경우 보통 방 1개로 확정
    - 2. room_query가 generc room type이면 type match 우선
        -- ex) "bathroom"
        -- 같은 floor 안의 bathroom_* 들 모두 반환

    - 3. 그래도 못 찾으면 name / instance_name 둘 다 대상으로 점수 기반 fallback
    """
    rooms_by_id = graph_index["rooms_by_id"]

    # floor 지정되었으면 그 floor 안의 room만 검색 후보로 사용
    if floor_meta is not None:
        candidate_rooms = [
            rooms_by_id[room_id] for room_id in floor_meta.get("rooms", []) if room_id in rooms_by_id
        ]
    else:
        candidate_rooms = list(rooms_by_id.values())

    if not candidate_rooms:
        raise ValueError("No candidate rooms available.")
    
    # 1. room instance 검색
    if room_instance_query:
        scored = []
        for room in candidate_rooms:
            score = _score_text_match(
                room_instance_query,
                room.get("instance_name"),
            )
            if score >= 95:
                scored.append((score, room))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score = scored[0][0]
            return [room for score, room in scored if score == best_score]

        raise ValueError(f"Room instance not found: {room_instance_query}")
    
    # 2. room query 없는 경우
    # room 제약이 없으면 현재 후보 room 전체를 그대로 반환
    if room_query is None:
        return candidate_rooms
    
    # 3. generic room type exact match
    # 만약 room_query="bathroom" 인 경우 floor 내부의 bathroom_1, bathroom_2, bathroom_3 을 모두 반환
    exact_type_matches = [
        room for room in candidate_rooms
        if _normalize_text(room.get("name")) == _normalize_text(room_query)
    ]
    if exact_type_matches:
        return exact_type_matches
    
    # 4. instance_name exact match
    # room_query 하나만 들어왔더라도 그 값이 bathroom_2 같은 인스턴스 이름일 수 있음
    exact_instance_matches = [
        room for room in candidate_rooms
        if _normalize_text(room.get("instance_name")) == _normalize_text(room_query)
    ]
    if exact_instance_matches:
        return exact_instance_matches
    
    # 5. fallback
    scored = []
    for room in candidate_rooms:
        score = max(
            _score_text_match(room_query, room.get("name")),
            _score_text_match(room_query, room.get("instance_name")),
            _score_text_match(room_query, room.get("room_id")),
        )
        if score >= 0:
            scored.append((score, room))

    if not scored:
        floor_label = None if floor_meta is None else floor_meta.get("floor_id")
        raise ValueError(
            f"Room not found for query={room_query}, floor={floor_label}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    return [room for score, room in scored if score == best_score]

def query_object(
    graph_index: Dict[str, Any],
    object_query: str,
    room_candidates: List[Dict[str, Any]],
    sim=None,
    agent_idx: int = 0,
) -> Dict[str, Any]:
    """
    object query를 해석해서 최종 object 하나를 선택한다.

    검색 범위:
    - room_candidates 아래에 속한 object들만 검색한다.
    - 즉, floor / room 단계에서 이미 후보가 상당히 좁혀진 상태라고 본다.

    점수 기준:
    1) object name
    2) object caption
    3) 점수가 같으면 현재 agent와의 xz 거리로 tie-break

    주의:
    - 여기서는 HSG 기준으로 object를 우선 선택한다.
    - 최종 상세 정보와 정확 좌표는 이후 source_id로 ConceptGraph에서 가져온다.
    """
    if not object_query:
        raise ValueError("object_query is empty")

    if not room_candidates:
        raise ValueError("room_candidates is empty")

    objects_by_id = graph_index["objects_by_id"]

    # room 후보들 아래의 object 후보를 모은다.
    candidate_objects: List[Dict[str, Any]] = []
    for room in room_candidates:
        for object_id in room.get("objects", []):
            if object_id not in objects_by_id:
                continue
            candidate_objects.append(objects_by_id[object_id])

    if not candidate_objects:
        room_ids = [room.get("room_id") for room in room_candidates]
        raise ValueError(f"No objects found under candidate rooms: {room_ids}")

    # tie-break용 agent xz 위치
    agent_xz = None
    if sim is not None:
        cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_idx)
        agent_xz = np.asarray([cur_pos[0], cur_pos[2]], dtype=np.float32)

    scored = []
    for obj in candidate_objects:
        score = max(
            # object name은 label이므로 alias-aware semantic match를 적용한다.
            # 예:
            # - query="Mug"
            # - obj["name"]="cup"
            # 이 경우 canonical alias(mug <-> cup) 덕분에 매칭될 수 있다.
            score_object_label_match(object_query, obj.get("name")),

            # caption은 자유 문장이므로 확장 query term을 사용한 text match를 적용한다.
            # 예:
            # - query="Mug"
            # - caption="a white cup on the counter"
            score_object_query_against_text(object_query, obj.get("caption")),
        )

        if score < 0:
            continue

        # HSG object vertices 평균 xz 중심을 tie-break에 사용
        center_xz = _mean_xz_from_vertices(obj.get("vertices", []))
        if agent_xz is not None and center_xz is not None:
            dist_xz = float(np.linalg.norm(center_xz - agent_xz))
        else:
            dist_xz = float("inf")

        scored.append((score, dist_xz, obj))

    if not scored:
        room_ids = [room.get("room_id") for room in room_candidates]
        raise ValueError(
            f"Object not found for query={object_query}, rooms={room_ids}"
        )

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_dist_xz, best_obj = scored[0]

    print(
        f"[HSG][object] matched -> "
        f"object_id={best_obj['object_id']}, "
        f"name={best_obj.get('name')}, "
        f"room_id={best_obj.get('room_id')}, "
        f"source_id={best_obj.get('source_id')}, "
        f"score={best_score}, "
        f"dist_xz={best_dist_xz:.3f}"
    )

    return best_obj

def get_conceptgraph_object_by_source_id(
    scene_id: str,
    source_id,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Dict[str, Any]:
    """
    HSG object의 source_id를 이용해서
    ConceptGraph obj_json 안의 원본 object entry를 가져온다.

    예:
        HSG object source_id = 441
        -> ConceptGraph id == 441 인 object를 찾는다.
    """
    cg_index = load_conceptgraph_index(
        scene_id=scene_id,
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )

    source_id_int = _safe_int(source_id)
    if source_id_int is None:
        raise ValueError(f"Invalid source_id: {source_id}")

    objects_by_source_id = cg_index["objects_by_source_id"]

    if source_id_int not in objects_by_source_id:
        raise KeyError(
            f"ConceptGraph object not found for source_id={source_id_int}"
        )

    return objects_by_source_id[source_id_int]

def _build_nav_xyz_from_hsg_fallback(
    sim,
    graph_object: Dict[str, Any],
    graph_floor: Optional[Dict[str, Any]],
    agent_idx: int = 0,
) -> np.ndarray:
    """
    ConceptGraph bbox_center를 사용할 수 없을 때의 fallback 좌표를 만든다.

    방식:
    - HSG object vertices의 평균 xz 중심을 사용
    - y는 현재 로봇 높이 또는 floor_zero_level을 사용

    주의:
    - 이 좌표는 fallback이다.
    - 정상적인 경우에는 ConceptGraph bbox_center가 우선되어야 한다.
    """
    center_xz = _mean_xz_from_vertices(graph_object.get("vertices", []))
    if center_xz is None:
        raise ValueError(
            f"Cannot build fallback nav xyz from HSG object: {graph_object.get('object_id')}"
        )

    cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_idx)

    # 기본 y는 현재 agent 높이를 사용
    y = float(cur_pos[1])

    # floor 정보가 있고 floor_zero_level이 있으면 그 값을 우선할 수도 있다.
    # 다만 scene 세팅마다 navmesh 높이와 약간 차이 날 수 있으므로,
    # 필요하면 이후 project_goal_to_navmesh에서 다시 보정된다.
    if graph_floor is not None and graph_floor.get("floor_zero_level") is not None:
        y = float(graph_floor["floor_zero_level"])

    return np.asarray([center_xz[0], y, center_xz[1]], dtype=np.float32)

def resolve_hierarchical_target_with_conceptgraph(
    sim,
    scene_id: str,
    dest_obj,
    agent_idx: int = 0,
    dataset_name: str = DEFAULT_HSG_DATASET,
    hsg_root: str = DEFAULT_HSG_ROOT,
    conceptgraph_root: str = DEFAULT_CONCEPTGRAPH_ROOT,
    experiment_name: str = DEFAULT_CONCEPTGRAPH_EXPERIMENT,
) -> Dict[str, Any]:
    """
    이번 구조의 핵심 resolver.

    처리 순서:
    1) dest_obj를 표준 target spec으로 정규화
    2) HSG index 로드
    3) floor 검색
    4) room / room_instance 검색
    5) object 검색
    6) 선택된 HSG object의 source_id로 ConceptGraph object 조회
    7) bbox_center를 최종 nav target으로 사용
       - bbox_center가 이상하면 HSG vertices 기반 fallback 사용

    반환:
        {
            "query": ...,
            "graph_index": ...,
            "graph_floor": ...,
            "graph_room_candidates": ...,
            "graph_object": ...,
            "conceptgraph_object": ...,
            "nav_xyz": np.ndarray([x, y, z], dtype=float32),
            "nav_source": "conceptgraph_bbox_center" or "hsg_vertices_fallback",
        }
    """
    # 1) 목표 spec 정규화
    target_spec = normalize_navigation_target(dest_obj)

    # 2) HSG index 로드
    graph_index = load_hierarchical_graph_index(
        scene_id=scene_id,
        dataset_name=dataset_name,
        hsg_root=hsg_root,
    )

    # 3) floor 검색
    graph_floor = query_floor(
        graph_index=graph_index,
        floor_query=target_spec.get("floor"),
    )

    # 4) room / room_instance 검색
    graph_room_candidates = query_room(
        graph_index=graph_index,
        room_query=target_spec.get("room"),
        floor_meta=graph_floor,
        room_instance_query=target_spec.get("room_instance"),
    )

    # 5) object 검색
    graph_object = query_object(
        graph_index=graph_index,
        object_query=target_spec["object"],
        room_candidates=graph_room_candidates,
        sim=sim,
        agent_idx=agent_idx,
    )

    # 6) source_id -> ConceptGraph object 조회
    conceptgraph_object = get_conceptgraph_object_by_source_id(
        scene_id=scene_id,
        source_id=graph_object.get("source_id"),
        conceptgraph_root=conceptgraph_root,
        experiment_name=experiment_name,
    )

    # 7) 최종 nav_xyz 결정
    #    우선순위는 ConceptGraph bbox_center -> HSG fallback
    nav_source = "conceptgraph_bbox_center"
    bbox_center = conceptgraph_object.get("bbox_center")
    nav_xyz = np.asarray(bbox_center, dtype=np.float32)

    if nav_xyz.shape != (3,) or not np.all(np.isfinite(nav_xyz)):
        nav_source = "hsg_vertices_fallback"
        nav_xyz = _build_nav_xyz_from_hsg_fallback(
            sim=sim,
            graph_object=graph_object,
            graph_floor=graph_floor,
            agent_idx=agent_idx,
        )

    # 디버그 로그
    print("[HSG][resolve] query =", target_spec)
    print(
        "[HSG][resolve] floor =",
        None if graph_floor is None else graph_floor.get("floor_id"),
    )
    print(
        "[HSG][resolve] rooms =",
        [room.get("room_id") for room in graph_room_candidates],
    )
    print(
        "[HSG][resolve] graph_object =",
        {
            "object_id": graph_object.get("object_id"),
            "name": graph_object.get("name"),
            "source_id": graph_object.get("source_id"),
            "room_id": graph_object.get("room_id"),
        },
    )
    print(
        "[CG][resolve] conceptgraph_object =",
        {
            "object_key": conceptgraph_object.get("object_key"),
            "id": conceptgraph_object.get("id"),
            "object_tag": conceptgraph_object.get("object_tag"),
            "bbox_center": conceptgraph_object.get("bbox_center"),
            "bbox_extent": conceptgraph_object.get("bbox_extent"),
            "bbox_volume": conceptgraph_object.get("bbox_volume"),
        },
    )
    print("[resolve] nav_source =", nav_source)
    print("[resolve] nav_xyz =", nav_xyz.tolist())

    return {
        "query": target_spec,
        "graph_index": graph_index,
        "graph_floor": graph_floor,
        "graph_room_candidates": graph_room_candidates,
        "graph_object": graph_object,
        "conceptgraph_object": conceptgraph_object,
        "nav_xyz": nav_xyz,
        "nav_source": nav_source,
    }

# ----------------------------------------------------------------------------------------------

def wrap_to_pi(x: float) -> float:
    """
    입력 각도(라디안)을 [-pi, pi) 범위로 정규화
    - 각도는 2*pi 주기로 동일한 방향을 의미함로, 비교/제어 시 범위를 고정해두면 안정적이라고 함
    - ex) 3.5 * pi 같은 값을 그대로 쓰면 회전 오차 계산이 꼬일 수 있어, 표준 범위로 잡아두기

    수식 설명:
    - x + pi로 범위를 오른쪽으로 이동
    - % (2*pi)로 2*pi 주기 내로 접기
    - 다시 -pi 해서 원래 중심으로 복원
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def get_fetch_base_pose(sim, agent_idx: int = 0):
    """
    Habitat-Sim의 articulated agent(Fetch) 기준 베이스 position/yaw 읽기

    Args:
        agent_idx: 멀티 에이전트 환경일 때 가져올 에이전트 인덱스(기본 0번)
    Returns:
        pos: np.ndarray([x, y, z], dtype=float32)
            - 월드 좌표계에서의 로봇 베이스 위치
        yaw: float (rad)
            - 로봇 베이스의 heading (yaw 각도, 라디안)
    """
    # 해당 agent의 articulated_agent 핸들 접근
    art = sim.get_agent_data(agent_idx).articulated_agent
    # base_pos를 numpy 배열로 변환 (나중에 거리 계산할 때 씀)
    pos = np.array(art.base_pos, dtype=np.float32)
    # base_rot을 float으로 캐스팅 (단위: rad)
    yaw = float(art.base_rot) # rad
    return pos, yaw

def find_target_object_position(
    sim,
    dest_obj_regex: str,
    objects_json_path: str = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json",
    agent_idx: int = 0,
):
    """
    objects_locations1.json에서 정규식으로 대상 물체를 찾고,
    현재 로봇 기준으로 '가장 가까운 후보'의 3D 위치 반환

    동작 순서:
    1) JSON 파일에서 물체 목록 읽음
    2) 'dest_obj_regex'를 objectType/objectId에 대해 매칭
    3) 매칭된 후보들에 대해 현재 로봇 위치와의 평면거리(x-z) 계산
    4) 가장 가까운 후보 1개 선택 후 그 위치 반환

    Args:
        dest_obj_regex: 찾고 싶은 물체 이름 패턴(정규식) ex: "Mug", "Apple.*", "FloorPlan1_physics-Drawer_.*"
        objects_json_path: 물체 맵 JSON 파일 경로

    Returns:
        np.ndarray([x, y, z], dtype=float32): 선택된 대상 물체의 월드 좌표
    
    Raises:
        ValueError: 정규식에 매칭되는 후보가 하나도 없을 때.
    """
    # 맵 상 존재하는 오브젝트들의 정보 가져와 (기존 만들어 놨떤 objects_locations1.json)
    objs = read_json_file(objects_json_path)

    # 대소문자 무시 정규식 컴파일
    pat = re.compile(dest_obj_regex, flags=re.IGNORECASE)
    # 현재 로봇 베이스 위치 획득 (거리 계산 기준점)
    cur_pos, _ = get_fetch_base_pose(sim, agent_idx)

    # (거리, 위치, objectType, objectId)를 담아 후보 목록 구성
    candidates = []
    for o in objs:
        obj_type = str(o.get("objectType", ""))
        obj_id = str(o.get("objectId", ""))

        # objectType 또는 objectId 중 하나라도 정규식 매칭되면 후보로 채택
        if pat.search(obj_type) or pat.search(obj_id):
            p = o["position"]
            pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)

            # 평면 거리만 사용(x-z): 내비게이션 관점에서 높이(y)는 보통 제외
            d = float(np.linalg.norm((pos - cur_pos)[[0, 2]]))
            candidates.append((d, pos, obj_type, obj_id))

    # 후보가 없으면 호출부에서 처리 가능하도록 명시적으로 예외 발생
    if not candidates:
        raise ValueError(
            f"Destination object regex '{dest_obj_regex}' not found in {objects_json_path}"
        )
    
    # 가장 가까운 후보가 앞에 오도록 정렬
    candidates.sort(key=lambda x:x[0])
    best = candidates[0]

    # 디버깅 로그: 어떤 물체가 선택됐는지 확인용
    print(
        f"[GoToObject] matched -> type={best[2]}, objectId={best[3]}, dist={best[0]:.3f}"
    )
    # 선택된 후보의 3D 위치 반환
    return best[1]

def teleport_fetch_base(sim, env, empty_action: str, pos_xyz, yaw_rad=None, agent_idx: int = 0):
    """
    Fetch base를 월드 좌표로 순간이동.
    """
    art = sim.get_agent_data(agent_idx).articulated_agent
    art.base_pos = np.array(pos_xyz, dtype=np.float32)
    if yaw_rad is not None:
        art.base_rot = float(yaw_rad)

    # 내부 상태 동기화
    try:
        art.update()
    except Exception:
        pass

    # 관측 갱신
    return env.step({"action": empty_action, "action_args": {}})

def debug_teleport_to_object(
    sim,
    env,
    empty_action: str,
    obs,
    dest_obj_regex: str,
    stand_off: float = 0.7,
    exact: bool = False,
    objects_json_path: str = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/objects_locations1.json",
    agent_idx: int = 0,
    obs_by_suffix_fn=None,
    save_rgb_fn=None,
):
    """
    dest_obj_regex로 찾은 물체 근처(또는 정확 좌표)로 텔레포트해서
    좌표/월드 정합 확인하는 디버그 함수.
    """
    target = find_target_object_position(
        sim,
        dest_obj_regex,
        objects_json_path=objects_json_path,
        agent_idx=agent_idx,
    )
    pf = sim.pathfinder
    cur_pos, _ = get_fetch_base_pose(sim, agent_idx)

    if exact:
        candidate = np.array(target, dtype=np.float32)
    else:
        # 물체 중심에 박히지 않게 약간 떨어진 위치로 배치
        v = np.array([cur_pos[0] - target[0], 0.0, cur_pos[2] - target[2]], dtype=np.float32)
        n = float(np.linalg.norm(v[[0, 2]]))
        if n < 1e-6:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            n = 1.0
        v /= n
        candidate = np.array(target, dtype=np.float32) + v * float(stand_off)
        candidate[1] = cur_pos[1]

    tp = np.array(pf.snap_point(candidate), dtype=np.float32)
    if not np.all(np.isfinite(tp)):
        tp = np.array(pf.snap_point(target), dtype=np.float32)
    if not np.all(np.isfinite(tp)):
        raise RuntimeError("Teleport failed: no navigable point near target.")

    # 물체를 바라보도록 yaw 설정
    yaw = float(np.arctan2(target[2] - tp[2], target[0] - tp[0]))
    obs = teleport_fetch_base(
        sim,
        env,
        empty_action,
        tp,
        yaw,
        agent_idx=agent_idx,
    )

    dist = float(np.linalg.norm((target - tp)[[0, 2]]))
    print(f"[teleport] regex={dest_obj_regex}")
    print(f"[teleport] target={target}, teleported={tp}, dist_to_target={dist:.3f}m")

    # 확인용 이미지 저장
    out_dir = "/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/logs/teleport_check"
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", dest_obj_regex)
    if obs_by_suffix_fn is not None and save_rgb_fn is not None:
        head_rgb = obs_by_suffix_fn(obs, "head_rgb")
        save_rgb_fn(head_rgb, out_dir, f"teleport_{safe}.png")

    return obs


def distance_xz(a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a_xyz) - np.asarray(b_xyz))[[0, 2]]))


def is_finite_point(p: np.ndarray) -> bool:
    q = np.asarray(p, dtype=np.float32)
    return q.shape[0] == 3 and bool(np.all(np.isfinite(q)))


def _pathfinder_num_islands(pathfinder) -> int:
    try:
        return int(pathfinder.num_islands)
    except Exception:
        return -1


def _is_valid_island_index(pathfinder, island_index: int) -> bool:
    n_islands = _pathfinder_num_islands(pathfinder)
    if n_islands <= 0:
        return False
    return 0 <= int(island_index) < n_islands


def _refresh_rearrange_largest_island_idx(sim, pathfinder, fallback_pos: np.ndarray) -> int:
    """
    RearrangeSim이 캐시한 largest island index가 navmesh 재계산 이후 stale 될 수 있어
    safe_snap_point 호출 전 유효성을 확인/보정한다.
    """
    largest_idx = None
    for attr in ("largest_island_idx", "_largest_indoor_island_idx"):
        try:
            largest_idx = int(getattr(sim, attr))
            break
        except Exception:
            continue

    if largest_idx is not None and _is_valid_island_index(pathfinder, largest_idx):
        return largest_idx

    try:
        recovered_idx = int(pathfinder.get_island(np.asarray(fallback_pos, dtype=np.float32)))
    except Exception:
        return -1

    if not _is_valid_island_index(pathfinder, recovered_idx):
        return -1

    if hasattr(sim, "_largest_indoor_island_idx"):
        try:
            sim._largest_indoor_island_idx = recovered_idx
        except Exception:
            pass
    return recovered_idx


def try_shortest_path(sim, start_xyz: np.ndarray, goal_xyz: np.ndarray):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start_xyz, dtype=np.float32)
    path.requested_end = np.asarray(goal_xyz, dtype=np.float32)
    found = bool(sim.pathfinder.find_path(path))
    if not found:
        return False, float("inf")
    return True, float(path.geodesic_distance)


def project_goal_to_navmesh(sim, raw_target_xyz: np.ndarray, agent_idx: int = 0):
    pf = sim.pathfinder
    cur_pos, _ = get_fetch_base_pose(sim, agent_idx)
    cur_pos = np.asarray(cur_pos, dtype=np.float32)
    raw_target = np.asarray(raw_target_xyz, dtype=np.float32)

    candidates = []
    try:
        island = int(pf.get_island(cur_pos))
    except Exception:
        island = -1

    # 1) object xz + current y를 같은 island 기준으로 snap
    c1 = np.array([raw_target[0], cur_pos[1], raw_target[2]], dtype=np.float32)
    try:
        s1 = np.array(pf.snap_point(c1, island), dtype=np.float32)
        if is_finite_point(s1):
            candidates.append(("snap_same_island_xy", s1))
    except Exception:
        pass

    # 2) raw target 자체 snap
    try:
        s2 = np.array(pf.snap_point(raw_target), dtype=np.float32)
        if is_finite_point(s2):
            candidates.append(("snap_raw", s2))
    except Exception:
        pass

    # 3) RearrangeSim helper 사용
    if hasattr(sim, "safe_snap_point"):
        safe_island_idx = _refresh_rearrange_largest_island_idx(sim, pf, cur_pos)
        if _is_valid_island_index(pf, safe_island_idx):
            try:
                s3 = np.array(sim.safe_snap_point(raw_target), dtype=np.float32)
                if is_finite_point(s3):
                    candidates.append(("safe_snap_point", s3))
            except Exception:
                pass

    if not candidates:
        raise RuntimeError(f"Cannot project target to navmesh. raw_target={raw_target.tolist()}")

    best_name = None
    best_goal = None
    best_geo = float("inf")
    best_obj_dist = float("inf")
    found_any_path = False

    for name, goal in candidates:
        found, geo = try_shortest_path(sim, cur_pos, goal)
        obj_dist = distance_xz(raw_target, goal)
        if found:
            found_any_path = True
            if (geo < best_geo) or (np.isclose(geo, best_geo) and obj_dist < best_obj_dist):
                best_name = name
                best_goal = goal
                best_geo = geo
                best_obj_dist = obj_dist

    if found_any_path:
        return best_goal, best_name, best_geo, best_obj_dist

    # 경로가 없으면 object와 더 가까운 snap 점을 fallback
    candidates.sort(key=lambda x: distance_xz(raw_target, x[1]))
    name, goal = candidates[0]
    return goal, f"{name}_no_path_fallback", float("inf"), distance_xz(raw_target, goal)


def sync_follower_agent_state(sim, agent_id: int = 0) -> bool:
    """
    ShortestPathFollower가 참고하는 내부 sim agent pose를
    현재 articulated Fetch base pose와 맞춰준다.
    """
    try:
        st = sim.get_agent_state(agent_id)  # RearrangeSim에서는 articulated base 기준 state
        sim.set_agent_state(
            position=np.asarray(st.position, dtype=np.float32),
            rotation=st.rotation,
            agent_id=agent_id,
            reset_sensors=False,
        )
        return True
    except Exception as e:
        print(f"[spf] warn: failed to sync follower agent state: {e}")
        return False


def dest_obj_to_xyz(sim, dest_obj, agent_id: int = 0) -> np.ndarray:
    """
    dest_obj를 [x, y, z]로 정규화
    - tuple/list 길이 2면 (x, y)로 간주하고 y는 현재 로봇 y 사용
    - tuple/list 길이 3 이상이면 앞 3개 사용
    - dict면 x/y/z 키 사용
    """
    cur_pos, _ = get_fetch_base_pose(sim, agent_idx=agent_id)

    if isinstance(dest_obj, dict):
        x = float(dest_obj.get("x", cur_pos[0]))
        y = float(dest_obj.get("y", cur_pos[1]))
        z = float(dest_obj.get("z", cur_pos[2]))
        return np.array([x, y, z], dtype=np.float32)
    
    if isinstance(dest_obj, (list, tuple)):
        if len(dest_obj) == 2:
            x, z = float(dest_obj[0]), float(dest_obj[1])
            return np.array([x, float(cur_pos[1]), z], dtype=np.float32)
        if len(dest_obj) >= 3:
            x, y, z = float(dest_obj[0]), float(dest_obj[1]), float(dest_obj[2])
            return np.array([x, y, z], dtype=np.float32)
        
    raise ValueError(f"Unsupported dest_obj format: {dest_obj}")


def drop_pending_nav_actions_for_agent(action_queue, agent_id: int):
    """
    특정 agent의 이동 관련 pending 액션만 제거

    action_queue에 있는 모든 행동을 하나씩 보면서, 
    지정된 에이전트(agent_id)가 하려던 이동 행동(nav_names)은 버리고 
    나머지만 모아서 원본 대기열을 새롭게 업데이트
    """
    nav_names = {"ObjectNavExpertAction", "MoveAhead", "MoveBack", "RotateLeft", "RotateRight"}
    kept = []
    for a in action_queue:
        if int(a.get("agent_id", -999)) == int(agent_id) and a.get("action") in nav_names:
            continue
        kept.append(a)
    action_queue[:] = kept


def find_target_object_position_live(
        sim,
        dest_obj_regex: str,
        agent_idx: int = 0,
):
    """
    현재 시뮬레이터(sim) 상태를 직접 조회해서, 정규식(dest_obj_regex)에 매칭되는
    객체들 중 현재 로봇(agent_idx) 기준으로 가장 가까운 객체의 위치를 반환

    매칭 대상 필드:
    1) object_type (handle 기반으로 추출한 타입 문자열)
    2) handle      (예: FloorPlan1_physics-Drawer_814ccbab_:0000)
    3) object_id   (Habitat 내부 런타임 object id)

    Returns:
        best_pos (np.ndarray): [x, y, z] float32
        best_meta (dict): distance/objectType/handle/objectId

    Raises:
        ValueError: 매칭되는 객체가 하나도 없을 때
    """
    # -----------------------------
    # 0) 정규식 컴파일 (대소문자 무시)
    # -----------------------------
    # dest_obj_regex가 "sink", "Sink", "SINK" 등 어떤 케이스로 들어와도 찾기 쉽게 IGNORECASE 사용
    pat = re.compile(dest_obj_regex, flags=re.IGNORECASE)

    # --------------------------------------------------------
    # 1) 현재 로봇 위치 가져오기 (거리 계산 기준점)
    # --------------------------------------------------------
    # get_fetch_base_pose(sim, agent_idx) -> (pos, yaw)
    # 여기서는 거리 계산에 pos만 사용
    cur_pos, _ = get_fetch_base_pose(sim, agent_idx)
    
    # --------------------------------------------------------
    # 2) 시뮬레이터에 존재하는 객체 목록 수집
    # --------------------------------------------------------
    # rigid object + articulated object 둘 다 합쳐서 전체 후보군 구성
    rom = sim.get_rigid_object_manager()
    aom = sim.get_articulated_object_manager()
    objs = list(rom.get_objects_by_handle_substring().values()) + \
           list(aom.get_objects_by_handle_substring().values())
    
    # 후보: (distance_xz, pos, object_type, handle, object_id)
    candidates = []

    # --------------------------------------------------------
    # 3) 객체별로 매칭 검사 + 거리 계산
    # --------------------------------------------------------
    for obj in objs:
        # 이미 제거되었거나 비활성 상태 객체는 스킵
        if not getattr(obj, "is_alive", True):
            continue

        # 월드 좌표계 위치 (x, y, z)
        pos = np.array(obj.translation, dtype=np.float32)

        # 원본 식별 문자열들
        handle = str(getattr(obj, "handle", ""))
        object_id = str(getattr(obj, "object_id", ""))
        template_class = str(getattr(obj, "template_class", ""))

        # handle/tamplate_class 기반 fallback 타입만 사용
        fallback_type = object_type_from_handle(handle, template_class)
        object_type = fallback_type

        # 세 필드 중 하나라도 정규식 매칭되면 후보로 채택
        if pat.search(object_type) or pat.search(handle) or pat.search(object_id):
            # 내비게이션 관점에서는 보통 높이(y) 차이보다 평면(xz) 거리가 중요
            d = distance_xz(cur_pos, pos)
            candidates.append((d, pos, object_type, handle, object_id))

    # --------------------------------------------------------
    # 4) 매칭 실패 처리
    # --------------------------------------------------------
    if not candidates:
        raise ValueError(f"[live] object regex not found: {dest_obj_regex}")
    
    # --------------------------------------------------------
    # 5) 가장 가까운 후보 선택
    # --------------------------------------------------------
    # distance_xz 기준 오름차순 정렬 후 첫 번째 사용
    candidates.sort(key=lambda x:x[0])
    best = candidates[0]
    best_pos = best[1]
    best_meta = {
        "distance": float(best[0]),
        "objectType": best[2],
        "handle": best[3],
        "objectId": best[4],
    }
    # 디버그 로그 (어떤 객체가 선택됐는지 확인용)
    print(
        f"[live] matched -> type={best_meta['objectType']}, "
        f"handle={best_meta['handle']}, objectId={best_meta['objectId']}, "
        f"dist={best_meta['distance']:.3f}"
    )
    return best_pos, best_meta
