from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch.nn.functional as F
import torch

from online_gs_slam.data.frame import Frame
from online_gs_slam.mapping.gaussian_insertion import GaussianInserter
from online_gs_slam.mapping.gaussian_map import GaussianMap
from online_gs_slam.mapping.keyframe_manager import KeyframeManager
from online_gs_slam.mapping.uncertainty import compute_uncertainty
from online_gs_slam.rendering.renderer import OnlineGaussianRenderer


@dataclass
class MapperConfig:
    local_opt_steps: int = 3
    learning_rate: float = 0.01
    opacity_increment: float = 0.02
    visible_radius: float = 5.0
    insert_every_n_frames: int = 1
    optimize_max_gaussians: int = 8000
    optimize_means: bool = True
    optimize_scales: bool = True
    optimize_colors: bool = True
    optimize_opacity: bool = True
    scale_reg: float = 0.001
    opacity_reg: float = 0.0001
    train_local_window: bool = True


class OnlineMapper:
    def __init__(
        self,
        config: MapperConfig,
        gaussian_map: GaussianMap,
        renderer: OnlineGaussianRenderer,
        inserter: GaussianInserter,
        keyframes: KeyframeManager,
    ):
        self.config = config
        self.gaussian_map = gaussian_map
        self.renderer = renderer
        self.inserter = inserter
        self.keyframes = keyframes

    def _target_rgb(self, frame: Frame, height: int, width: int) -> torch.Tensor:
        target = torch.from_numpy(frame.rgb).to(self.gaussian_map.device, dtype=torch.float32) / 255.0
        target = target.permute(2, 0, 1)[None, ...]
        target = F.interpolate(target, size=(height, width), mode="bilinear", align_corners=False)
        return target[0].permute(1, 2, 0).contiguous()

    def _collect_training_frames(self, current_frame: Frame) -> List[Frame]:
        if not self.config.train_local_window:
            return [current_frame]
        frames = self.keyframes.local_window()
        if not any(frame.index == current_frame.index for frame in frames):
            frames = frames + [current_frame]
        return frames

    def _visible_union(self, training_items: List[Tuple[Frame, torch.Tensor]]) -> torch.Tensor:
        chunks = []
        for _, pose in training_items:
            idx = self.gaussian_map.query_visible_gaussians(pose, radius=self.config.visible_radius)
            if idx.numel() > 0:
                chunks.append(idx)
        if not chunks:
            return torch.empty((0,), dtype=torch.long, device=self.gaussian_map.device)
        visible = torch.unique(torch.cat(chunks, dim=0))
        if visible.numel() > self.config.optimize_max_gaussians:
            visible = visible[-self.config.optimize_max_gaussians :]
        return visible

    def _optimize_visible(self, training_items: List[Tuple[Frame, torch.Tensor]], visible: torch.Tensor) -> float:
        if visible.numel() == 0 or self.config.local_opt_steps <= 0:
            return float("nan")

        means = self.gaussian_map.means[visible].detach().clone().requires_grad_(self.config.optimize_means)
        colors = self.gaussian_map.colors[visible].detach().clone().clamp(1e-4, 1.0 - 1e-4).logit().requires_grad_(self.config.optimize_colors)
        scales = self.gaussian_map.scales[visible].detach().clone().clamp(1e-4, 0.5).log().requires_grad_(self.config.optimize_scales)
        rotations = self.gaussian_map.rotations[visible].detach().clone()
        opacity = self.gaussian_map.opacity[visible].detach().clone().clamp(1e-4, 1.0 - 1e-4).logit().requires_grad_(self.config.optimize_opacity)

        params = [p for p in (means, colors, scales, opacity) if p.requires_grad]
        if not params:
            return float("nan")
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        last_loss = None
        target_cache = {}

        for _ in range(self.config.local_opt_steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for frame, camera_to_world in training_items:
                render = self.renderer.render_tensors(means, colors.sigmoid(), scales.exp().clamp(1e-4, 0.5), rotations, opacity.sigmoid(), camera_to_world, frame.intrinsics)
                cache_key = (frame.index, render.rgb.shape[0], render.rgb.shape[1])
                if cache_key not in target_cache:
                    target_cache[cache_key] = self._target_rgb(frame, render.rgb.shape[0], render.rgb.shape[1])
                losses.append(torch.mean(torch.abs(render.rgb - target_cache[cache_key])))
            rgb_loss = torch.stack(losses).mean()
            loss = rgb_loss
            if self.config.scale_reg > 0:
                loss = loss + self.config.scale_reg * torch.mean(scales.exp())
            if self.config.opacity_reg > 0:
                loss = loss + self.config.opacity_reg * torch.mean(opacity.sigmoid())
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

        with torch.no_grad():
            self.gaussian_map.means[visible] = means.detach()
            self.gaussian_map.colors[visible] = colors.detach().sigmoid().clamp(0.0, 1.0)
            self.gaussian_map.scales[visible] = scales.detach().exp().clamp(1e-4, 0.5)
            self.gaussian_map.opacity[visible] = opacity.detach().sigmoid().clamp(0.0, 1.0)
        return float(last_loss) if last_loss is not None else float("nan")

    def update(self, frame: Frame, camera_to_world: torch.Tensor) -> dict:
        stats = {"inserted": 0, "visible": 0, "loss": None, "keyframe": False, "train_frames": 1}
        visible = self.gaussian_map.query_visible_gaussians(camera_to_world, radius=self.config.visible_radius)
        stats["visible"] = int(visible.numel())

        if visible.numel() > 0:
            self.gaussian_map.observation_count[visible] += 1.0
            self.gaussian_map.opacity[visible] = (self.gaussian_map.opacity[visible] + self.config.opacity_increment).clamp(0.0, 1.0)

        if frame.index % self.config.insert_every_n_frames == 0:
            means, colors, scales = self.inserter.propose_from_frame(frame)
            self.gaussian_map.add_gaussians(means=means, colors=colors, scales=scales)
            stats["inserted"] = int(means.shape[0])

        stats["keyframe"] = self.keyframes.add_keyframe(frame)
        training_frames = self._collect_training_frames(frame)
        training_items = [(item, torch.from_numpy(item.camera_to_world).float().to(self.gaussian_map.device)) for item in training_frames]
        stats["train_frames"] = len(training_items)
        visible = self._visible_union(training_items)
        stats["visible"] = int(visible.numel())
        loss = self._optimize_visible(training_items, visible)
        if loss == loss:
            stats["loss"] = loss

        compute_uncertainty(self.gaussian_map)
        return stats

    def render_debug(self, frame: Frame, camera_to_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            render = self.renderer.render(self.gaussian_map, camera_to_world, frame.intrinsics)
            target = self._target_rgb(frame, render.rgb.shape[0], render.rgb.shape[1])
        return target, render.rgb
