from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from output_common import build_subset_overview


def generate(
    output_dir: Path,
    sample_dir: Path,
    columns: int = 4,
) -> list[Path]:
    optional_topdown = sample_dir / "optional" / "topdown_debug.png"
    extra_tail_paths: list[Path] = []
    if optional_topdown.exists():
        extra_tail_paths.append(optional_topdown)
    return build_subset_overview(output_dir, columns=columns, extra_tail_paths=extra_tail_paths)
