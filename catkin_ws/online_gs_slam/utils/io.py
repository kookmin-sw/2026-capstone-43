from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_trajectory(path: Union[str, Path], trajectory: List[Dict]) -> None:
    path = Path(path)
    with open(path, "w") as f:
        json.dump(trajectory, f, indent=2)


def save_matrix_txt(path: Union[str, Path], matrices: List[np.ndarray]) -> None:
    with open(path, "w") as f:
        for idx, mat in enumerate(matrices):
            values = " ".join(f"{x:.9f}" for x in mat.reshape(-1))
            f.write(f"{idx} {values}\n")
