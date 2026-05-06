from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from online_gs_slam.data.frame import Frame


@dataclass
class KeyframeManagerConfig:
    translation_threshold: float = 0.15
    rotation_threshold_deg: float = 12.0
    frame_interval: int = 10
    local_window_size: int = 8


@dataclass
class KeyframeManager:
    config: KeyframeManagerConfig
    keyframes: List[Frame] = field(default_factory=list)
    last_keyframe_pose: Optional[np.ndarray] = None
    last_keyframe_index: int = -1

    def should_add_keyframe(self, frame: Frame) -> bool:
        if self.last_keyframe_pose is None:
            return True
        if frame.index - self.last_keyframe_index >= self.config.frame_interval:
            return True
        delta_t = np.linalg.norm(frame.camera_to_world[:3, 3] - self.last_keyframe_pose[:3, 3])
        return bool(delta_t >= self.config.translation_threshold)

    def add_keyframe(self, frame: Frame) -> bool:
        if not self.should_add_keyframe(frame):
            return False
        self.keyframes.append(frame)
        self.last_keyframe_pose = frame.camera_to_world.copy()
        self.last_keyframe_index = frame.index
        return True

    def local_window(self) -> List[Frame]:
        return self.keyframes[-self.config.local_window_size :]
