from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

WAV_EXTENSIONS = {".wav"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_wav_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in WAV_EXTENSIONS:
            raise ValueError(f"Expected a .wav file, got: {path}")
        return [path]

    wavs = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in WAV_EXTENSIONS)
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under directory: {path}")
    return wavs


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "sample"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def read_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read config files. Install pyyaml or omit --config.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got: {type(data).__name__}")
    return data

