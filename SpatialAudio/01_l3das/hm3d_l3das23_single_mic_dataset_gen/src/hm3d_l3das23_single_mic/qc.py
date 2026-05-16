from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest_io import iter_dataset_rows
from .schemas import SampleMetadata


@dataclass
class QCAggregator:
    scenes_processed: int = 0
    candidate_mic_poses: int = 0
    candidate_sources: int = 0
    valid_samples: int = 0
    skipped_existing_samples: int = 0
    gcounts: Counter[str] = field(default_factory=Counter)
    fov_counts: Counter[str] = field(default_factory=Counter)
    joint_counts: Counter[str] = field(default_factory=Counter)
    failure_reasons: Counter[str] = field(default_factory=Counter)
    scene_failures: dict[str, str] = field(default_factory=dict)

    def record_scene_processed(self) -> None:
        self.scenes_processed += 1

    def record_scene_failure(self, scene_id: str, message: str) -> None:
        self.scene_failures[scene_id] = message

    def record_mic_candidates(self, count: int) -> None:
        self.candidate_mic_poses += int(count)

    def record_source_candidate(self) -> None:
        self.candidate_sources += 1

    def record_failure(self, reason: str) -> None:
        self.failure_reasons[str(reason)] += 1

    def record_sample(self, metadata: SampleMetadata | dict[str, Any], *, skipped_existing: bool = False) -> None:
        self.valid_samples += 1
        if skipped_existing:
            self.skipped_existing_samples += 1
        if isinstance(metadata, dict):
            gkey = str(metadata.get("geometry_los", "unknown"))
            in_fov = bool(metadata.get("is_in_fov", metadata.get("in_fov", False)))
        else:
            gkey = str(metadata.geometry_los)
            in_fov = bool(metadata.in_fov)
        fkey = "True" if in_fov else "False"
        self.gcounts[gkey] += 1
        self.fov_counts[fkey] += 1
        self.joint_counts[f"{'FOV' if in_fov else 'OOF'}+{gkey}"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes_processed": self.scenes_processed,
            "candidate_mic_poses": self.candidate_mic_poses,
            "candidate_sources": self.candidate_sources,
            "valid_samples": self.valid_samples,
            "skipped_existing_samples": self.skipped_existing_samples,
            "gLOS_gNLOS_counts": dict(self.gcounts),
            "in_fov_counts": dict(self.fov_counts),
            "joint_counts": dict(self.joint_counts),
            "failure_reasons": dict(self.failure_reasons),
            "scene_failures": dict(self.scene_failures),
        }


def write_qc_report(dataset_root: Path, report: dict[str, Any]) -> Path:
    manifests_dir = dataset_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    output_path = manifests_dir / "qc_report.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    return output_path


def build_qc_report_from_existing_metadata(dataset_root: Path) -> dict[str, Any]:
    qc = QCAggregator()
    for payload in iter_dataset_rows(dataset_root):
        geometry_los = str(payload.get("geometry_los", "unknown"))
        in_fov = bool(payload.get("is_in_fov", payload.get("in_fov", False)))
        qc.valid_samples += 1
        qc.gcounts[geometry_los] += 1
        qc.fov_counts["True" if in_fov else "False"] += 1
        qc.joint_counts[f"{'FOV' if in_fov else 'OOF'}+{geometry_los}"] += 1
    return qc.to_dict()
