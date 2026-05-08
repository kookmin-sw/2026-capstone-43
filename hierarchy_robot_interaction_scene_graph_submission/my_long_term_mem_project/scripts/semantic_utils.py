import csv
import json
import os
import re
from typing import Dict, Optional

DEFAULT_SEMANTIC_MAP_PATH = (
    "/home/yuchaehee/long_term_memory_project/scene_data/scene_datasets/"
    "ai2thorhab/ai2thor-hab/ai2thor-hab/configs/object_semantic_id_mapping.json"
)
DEFAULT_HSSD_DATASET_DIR = (
    "/home/yuchaehee/long_term_memory_project/scene_data/scene_datasets/hssd-hab"
)

_SEM_ID_TO_TYPE_CACHE: Dict[str, Dict[int, str]] = {}
_HSSD_LEXICON_CACHE: Dict[str, Dict[int, str]] = {}
_HSSD_CONDENSED_CACHE: Dict[str, Dict[str, str]] = {}
_HSSD_FPMODEL_CACHE: Dict[str, Dict[str, str]] = {}
_HSSD_OBJECT_NAME_CACHE: Dict[str, Dict[str, str]] = {}
_HSSD_RESOLUTION_CACHE: Dict[tuple, Dict[str, object]] = {}


def load_semantic_id_to_type(mapping_path=DEFAULT_SEMANTIC_MAP_PATH):
    # semantic_id(숫자) -> objectType(문자열) 매핑 정보를 파일에서 읽어서 반환하는 함수
    if mapping_path in _SEM_ID_TO_TYPE_CACHE:
        return _SEM_ID_TO_TYPE_CACHE[mapping_path]

    if not mapping_path or not os.path.exists(mapping_path):
        _SEM_ID_TO_TYPE_CACHE[mapping_path] = {}
        return _SEM_ID_TO_TYPE_CACHE[mapping_path]

    with open(mapping_path, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)

    id_to_name = {}
    for name, sid in name_to_id.items():
        try:
            id_to_name[int(sid)] = str(name)
        except Exception:
            continue

    _SEM_ID_TO_TYPE_CACHE[mapping_path] = id_to_name
    return id_to_name


def _legacy_object_type_from_handle(handle: str, template_class: str = "") -> str:
    # 기존 AI2THOR 스타일 handle 정리 fallback
    s = os.path.basename(str(handle)).split(".")[0]
    s = re.sub(r"_?:\d+$", "", s)
    if "-" in s:
        s = s.split("-")[-1]
    s = re.sub(r"_[0-9a-f]{8,}$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_\d+$", "", s)
    s = s.strip("_")
    return s or template_class or "UnknownObject"


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _normalize_category_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    return s


def _default_scene_dataset_dir(scene_dataset_dir: Optional[str]) -> str:
    if scene_dataset_dir and str(scene_dataset_dir).strip():
        return str(scene_dataset_dir).strip()
    env_dir = os.environ.get("KARMA_SCENE_DATASET_DIR", "").strip()
    if env_dir:
        return env_dir
    return DEFAULT_HSSD_DATASET_DIR


def _ycb_template_to_names(template_name: str):
    """
    ycb objects의 object_type, object_name 구하는 로직
    """
    t = str(template_name or "").strip().lower()
    m = re.fullmatch(r"(\d{3})_([a-z0-9_]+)", t) # ex) 002_master_chef_can => [002, master_chef_can]
    if not m:
        return "", ""
    raw = m.group(2) # ex) master_chef_can
    object_type = raw # ex) master_chef_can
    object_name = " ".join(w.capitalize() for w in raw.split("_")) # Master Chef Can
    return object_type, object_name # ex) master_chef_can, Master Chef Can


def _extract_template_name_from_handle(handle: str) -> str:
    """
    handle에서 template/ID 추출
    - handle (..._:0000)을 정리해 tamplate 이름을 뽑음
    """
    raw = str(handle or "").strip().replace("\\", "/")
    if not raw:
        return ""

    # runtime suffix 제거: ":0000", "_:0000"
    raw = re.sub(r"_?:\d+$", "", raw)
    base = os.path.basename(raw)

    # 확장자 제거
    for suffix in (
        ".object_config.json",
        ".ao_config.json",
        ".scene_instance",
        ".json",
        ".glb",
        ".urdf",
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)]

    part_m = re.search(r"([0-9a-f]{40}_part_\d+)", base, flags=re.IGNORECASE)
    if part_m:
        return part_m.group(1).lower()

    hash_m = re.search(r"([0-9a-f]{40})", base, flags=re.IGNORECASE)
    if hash_m:
        return hash_m.group(1).lower()

    opening_m = re.search(r"(\d{2,5}-\d+)", base)
    if opening_m:
        return opening_m.group(1)

    return base.strip()


def _load_hssd_lexicon(scene_dataset_dir: str) -> Dict[int, str]:
    lex_path = os.path.abspath(
        os.path.join(scene_dataset_dir, "semantics", "hssd-hab_semantic_lexicon.json")
    )
    if lex_path in _HSSD_LEXICON_CACHE:
        return _HSSD_LEXICON_CACHE[lex_path]

    out: Dict[int, str] = {}
    if os.path.exists(lex_path):
        try:
            with open(lex_path, "r", encoding="utf-8") as f:
                lex = json.load(f)
            classes = lex.get("classes", []) if isinstance(lex, dict) else []
            for row in classes:
                if not isinstance(row, dict):
                    continue
                sid = _safe_int(row.get("id"))
                name = _normalize_category_name(row.get("name", ""))
                if sid is None or not name:
                    continue
                out[sid] = name
        except Exception:
            out = {}

    _HSSD_LEXICON_CACHE[lex_path] = out
    return out


def _load_hssd_condensed(scene_dataset_dir: str) -> Dict[str, str]:
    path = os.path.abspath(
        os.path.join(scene_dataset_dir, "metadata", "hssd_obj_semantics_condensed.csv")
    )
    if path in _HSSD_CONDENSED_CACHE:
        return _HSSD_CONDENSED_CACHE[path]

    out: Dict[str, str] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    key = row[0].strip().lower() if len(row) > 0 else ""
                    condensed = row[3].strip() if len(row) > 3 else ""
                    primary = row[4].strip() if len(row) > 4 else ""
                    name = _normalize_category_name(condensed or primary)
                    if key and name:
                        out[key] = name
        except Exception:
            out = {}

    _HSSD_CONDENSED_CACHE[path] = out
    return out


def _load_hssd_fpmodels(scene_dataset_dir: str) -> Dict[str, str]:
    """
    metadata/objects.json 로드 (경로: /home/yuchaehee/long_term_memory_project/scene_data/scene_datasets/hssd-hab/metadata/objects.json)
    - 해시 ID로 name을 매칭해 object_name 얻을 때 사용
    """

    path = os.path.abspath(
        os.path.join(scene_dataset_dir, "metadata", "fpmodels-with-decomposed.csv")
    )
    if path in _HSSD_FPMODEL_CACHE:
        return _HSSD_FPMODEL_CACHE[path]

    out: Dict[str, str] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = str(row.get("id", "")).strip().lower()
                    name = _normalize_category_name(
                        row.get("main_category", "")
                        or row.get("super_category", "")
                        or row.get("name", "")
                    )
                    if key and name:
                        out[key] = name
        except Exception:
            out = {}

    _HSSD_FPMODEL_CACHE[path] = out
    return out


def _load_hssd_object_names(scene_dataset_dir: str) -> Dict[str, str]:
    path = os.path.abspath(os.path.join(scene_dataset_dir, "metadata", "objects.json"))
    if path in _HSSD_OBJECT_NAME_CACHE:
        return _HSSD_OBJECT_NAME_CACHE[path]

    out: Dict[str, str] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for object_id, meta in raw.items():
                    oid = str(object_id).strip().lower()
                    if not oid:
                        continue
                    name = ""
                    if isinstance(meta, dict):
                        name = _normalize_category_name(meta.get("name", ""))
                    elif isinstance(meta, str):
                        name = _normalize_category_name(meta)
                    if name:
                        out[oid] = name
        except Exception:
            out = {}

    _HSSD_OBJECT_NAME_CACHE[path] = out
    return out


def _candidate_hssd_config_paths(scene_dataset_dir: str, template_name: str):
    if not template_name:
        return []

    candidates = []
    t = template_name
    t_lower = t.lower()

    if "_part_" in t_lower:
        root = t_lower.split("_part_", 1)[0]
        candidates.append(
            os.path.join(scene_dataset_dir, "objects", "decomposed", root, f"{t_lower}.object_config.json")
        )

    if re.fullmatch(r"[0-9a-f]{40}", t_lower):
        candidates.append(
            os.path.join(scene_dataset_dir, "objects", t_lower[0], f"{t_lower}.object_config.json")
        )
        candidates.append(
            os.path.join(scene_dataset_dir, "urdf", t_lower, f"{t_lower}.ao_config.json")
        )

    if re.fullmatch(r"\d{2,5}-\d+", t_lower):
        candidates.append(
            os.path.join(scene_dataset_dir, "objects", "openings", f"{t_lower}.object_config.json")
        )

    # 일반 fallback
    if t_lower:
        candidates.append(
            os.path.join(scene_dataset_dir, "objects", t_lower[0], f"{t_lower}.object_config.json")
        )
        candidates.append(
            os.path.join(scene_dataset_dir, "objects", "x", f"{t_lower}.object_config.json")
        )
        candidates.append(
            os.path.join(scene_dataset_dir, "objects", "openings", f"{t_lower}.object_config.json")
        )
        candidates.append(
            os.path.join(scene_dataset_dir, "urdf", t_lower, f"{t_lower}.ao_config.json")
        )

    # 순서 유지하면서 중복 제거
    out = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _semantic_id_from_config(config_path: str) -> Optional[int]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return _safe_int(cfg.get("semantic_id"))
    except Exception:
        return None


def resolve_object_semantics_from_handle(
    handle: str,
    template_class: str = "",
    scene_dataset_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    handle 기반으로 object type / dataset semantic 정보를 복원한다.
    - Habitat-Lab에서 runtime semantic_id가 재할당되어도 사용 가능.
    """
    fallback_type = _legacy_object_type_from_handle(handle, template_class)
    dataset_dir = _default_scene_dataset_dir(scene_dataset_dir)
    template_name = _extract_template_name_from_handle(handle)

    result = {
        "template_name": template_name,
        "object_type": fallback_type,
        "dataset_semantic_id": None,
        "dataset_semantic_class": "",
        "dataset_object_id": "",
        "object_name": "",
        "source": "legacy_handle",
    }
    # ycb objects 처리 로직
    # - hsdd-hab 데이터셋이랑 달라서 처리하는 함수도 다른 거 썼어여
    ycb_type, ycb_name = _ycb_template_to_names(template_name)
    if ycb_type:
        result["object_type"] = ycb_type
        result["object_name"] = ycb_name
        result["dataset_semantic_class"] = ycb_type
        result["source"] = "ycb_template_name"
        return result

    if not template_name or not os.path.isdir(dataset_dir):
        return result

    # HSSD 템플릿 패턴이 아니면 기존 fallback만 사용한다.
    if not (
        re.fullmatch(r"[0-9a-f]{40}", template_name, flags=re.IGNORECASE)
        or re.fullmatch(r"[0-9a-f]{40}_part_\d+", template_name, flags=re.IGNORECASE)
        or re.fullmatch(r"\d{2,5}-\d+", template_name)
        or re.fullmatch(r"\d+", template_name)
    ):
        return result

    cache_key = (os.path.abspath(dataset_dir), template_name.lower())
    cached = _HSSD_RESOLUTION_CACHE.get(cache_key)
    
    # 해시 ID로 name을 매칭해 object_name 얻기
    if cached is not None:
        out = dict(cached)
        if not out.get("object_type"):
            out["object_type"] = fallback_type
        return out

    template_key = template_name.lower()
    base_hash = template_key.split("_part_", 1)[0]

    dataset_object_id = ""
    if re.fullmatch(r"[0-9a-f]{40}", base_hash, flags=re.IGNORECASE):
        dataset_object_id = base_hash
    object_name = ""
    if dataset_object_id:
        object_name = _normalize_category_name(
            _load_hssd_object_names(dataset_dir).get(dataset_object_id, "")
        )

    semantic_id = None
    semantic_class = ""
    source = ""

    for p in _candidate_hssd_config_paths(dataset_dir, template_key):
        if not os.path.exists(p):
            continue
        semantic_id = _semantic_id_from_config(p)
        if semantic_id is None:
            continue
        lex = _load_hssd_lexicon(dataset_dir)
        semantic_class = _normalize_category_name(lex.get(semantic_id, ""))
        source = "hssd_config_semantic_id"
        break

    if not semantic_class:
        condensed = _load_hssd_condensed(dataset_dir)
        semantic_class = _normalize_category_name(
            condensed.get(template_key, "") or condensed.get(base_hash, "")
        )
        if semantic_class:
            source = "hssd_condensed_csv"

    if not semantic_class:
        fpmodels = _load_hssd_fpmodels(dataset_dir)
        semantic_class = _normalize_category_name(
            fpmodels.get(template_key, "") or fpmodels.get(base_hash, "")
        )
        if not semantic_class and re.fullmatch(r"\d+", template_key):
            prefix = f"{template_key}-"
            for k, v in fpmodels.items():
                if k.startswith(prefix) and v:
                    semantic_class = _normalize_category_name(v)
                    break
        if semantic_class:
            source = "hssd_fpmodels_csv"

    if semantic_class:
        result["object_type"] = semantic_class
        result["dataset_semantic_class"] = semantic_class
        result["source"] = source or "hssd_semantics"
    if semantic_id is not None:
        result["dataset_semantic_id"] = semantic_id
    if dataset_object_id:
        result["dataset_object_id"] = dataset_object_id
    if object_name:
        result["object_name"] = object_name

    _HSSD_RESOLUTION_CACHE[cache_key] = dict(result)
    return result


def object_type_from_handle(handle: str, template_class: str = "") -> str:
    # 기존 인터페이스 유지: 내부적으로 HSSD semantic 복원을 우선 사용
    info = resolve_object_semantics_from_handle(
        handle=handle, template_class=template_class
    )
    return str(info.get("object_type") or template_class or "UnknownObject")
