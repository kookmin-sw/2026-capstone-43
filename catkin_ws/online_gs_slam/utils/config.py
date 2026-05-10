from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(path).expanduser()) as f:
        return yaml.safe_load(f) or {}
