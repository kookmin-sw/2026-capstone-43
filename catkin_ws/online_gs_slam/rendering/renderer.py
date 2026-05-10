from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch

from online_gs_slam.data.frame import CameraIntrinsics
from online_gs_slam.mapping.gaussian_map import GaussianMap


@dataclass
class RenderOutput:
    rgb: torch.Tensor
    depth: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None


@dataclass
class SimpleSplatRendererConfig:
    downscale: int = 8
    background: float = 0.0
    max_visible: int = 600
    min_sigma_px: float = 0.75
    max_sigma_px: float = 8.0
    patch_sigma_multiplier: float = 3.0


@dataclass
class GsplatRendererConfig:
    downscale: int = 4
    background: float = 0.0
    near_plane: float = 0.01
    far_plane: float = 100.0
    eps2d: float = 0.3
    packed: bool = True
    rasterize_mode: str = "classic"
    render_mode: str = "RGB"


class OnlineGaussianRenderer(Protocol):
    def render(self, gaussian_map: GaussianMap, camera_to_world: torch.Tensor, intrinsics: CameraIntrinsics) -> RenderOutput:
        ...


class NullGaussianRenderer:
    """Dependency-free placeholder.

    Replace this with a gsplat-backed renderer while keeping tracking/mapping
    code unchanged.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def render(self, gaussian_map: GaussianMap, camera_to_world: torch.Tensor, intrinsics: CameraIntrinsics) -> RenderOutput:
        rgb = torch.zeros((intrinsics.height, intrinsics.width, 3), dtype=torch.float32, device=self.device)
        alpha = torch.zeros((intrinsics.height, intrinsics.width, 1), dtype=torch.float32, device=self.device)
        return RenderOutput(rgb=rgb, alpha=alpha)


def opengl_c2w_to_opencv_w2c(camera_to_world_gl: torch.Tensor) -> torch.Tensor:
    gl_from_cv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32, device=camera_to_world_gl.device))
    camera_to_world_cv = camera_to_world_gl @ gl_from_cv
    return torch.linalg.inv(camera_to_world_cv)


class GsplatGaussianRenderer:
    def __init__(self, config: GsplatRendererConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        try:
            from gsplat import rasterization
        except Exception as exc:
            raise ImportError("gsplat is required for GsplatGaussianRenderer. Install with `python3 -m pip install --user gsplat`.") from exc
        self._rasterization = rasterization

    def _camera(self, camera_to_world: torch.Tensor, intrinsics: CameraIntrinsics) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        downscale = max(int(self.config.downscale), 1)
        width = max(int(intrinsics.width // downscale), 1)
        height = max(int(intrinsics.height // downscale), 1)
        viewmat = opengl_c2w_to_opencv_w2c(camera_to_world.to(self.device).float())[None, ...]
        K = torch.tensor(
            [
                [float(intrinsics.fx) / downscale, 0.0, float(intrinsics.cx) / downscale],
                [0.0, float(intrinsics.fy) / downscale, float(intrinsics.cy) / downscale],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )[None, ...]
        return viewmat, K, width, height

    def render(self, gaussian_map: GaussianMap, camera_to_world: torch.Tensor, intrinsics: CameraIntrinsics) -> RenderOutput:
        return self.render_tensors(
            gaussian_map.means,
            gaussian_map.colors,
            gaussian_map.scales,
            gaussian_map.rotations,
            gaussian_map.opacity,
            camera_to_world,
            intrinsics,
        )

    def render_tensors(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsics: CameraIntrinsics,
    ) -> RenderOutput:
        viewmats, Ks, width, height = self._camera(camera_to_world, intrinsics)
        if means.numel() == 0:
            rgb = torch.full((height, width, 3), self.config.background, dtype=torch.float32, device=self.device)
            alpha = torch.zeros((height, width, 1), dtype=torch.float32, device=self.device)
            return RenderOutput(rgb=rgb, alpha=alpha)
        renders, alphas, _ = self._rasterization(
            means=means.to(self.device).float(),
            quats=rotations.to(self.device).float(),
            scales=scales.to(self.device).float().clamp_min(1e-5),
            opacities=opacity.to(self.device).float().reshape(-1).clamp(0.0, 1.0),
            colors=colors.to(self.device).float().clamp(0.0, 1.0),
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            eps2d=self.config.eps2d,
            packed=self.config.packed,
            backgrounds=None,
            render_mode=self.config.render_mode,
            rasterize_mode=self.config.rasterize_mode,
        )
        rgb = renders[0, ..., :3]
        alpha = alphas[0]
        if alpha.ndim == 2:
            alpha = alpha[..., None]
        return RenderOutput(rgb=rgb.clamp(0.0, 1.0), alpha=alpha)


class SimpleTorchGaussianRenderer:
    """Small differentiable Gaussian splat renderer.

    This is deliberately simple and research-friendly. It optimizes local
    Gaussian colors/positions/scales without requiring gsplat yet. It is not a
    replacement for a production CUDA splatter.
    """

    def __init__(self, config: SimpleSplatRendererConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def render(self, gaussian_map: GaussianMap, camera_to_world: torch.Tensor, intrinsics: CameraIntrinsics) -> RenderOutput:
        indices = gaussian_map.query_visible_gaussians(camera_to_world)
        if indices.numel() > self.config.max_visible:
            indices = indices[-self.config.max_visible :]
        return self.render_tensors(
            gaussian_map.means[indices],
            gaussian_map.colors[indices],
            gaussian_map.scales[indices],
            gaussian_map.opacity[indices],
            camera_to_world,
            intrinsics,
        )

    def render_tensors(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsics: CameraIntrinsics,
    ) -> RenderOutput:
        downscale = max(int(self.config.downscale), 1)
        height = max(int(intrinsics.height // downscale), 1)
        width = max(int(intrinsics.width // downscale), 1)
        fx = float(intrinsics.fx) / downscale
        fy = float(intrinsics.fy) / downscale
        cx = float(intrinsics.cx) / downscale
        cy = float(intrinsics.cy) / downscale

        accum_rgb = torch.full((height, width, 3), self.config.background, dtype=torch.float32, device=self.device)
        accum_w = torch.zeros((height, width, 1), dtype=torch.float32, device=self.device)
        if means.numel() == 0:
            return RenderOutput(rgb=accum_rgb, alpha=accum_w)

        world_to_camera = torch.linalg.inv(camera_to_world.to(self.device))
        points_h = torch.cat([means, torch.ones((means.shape[0], 1), dtype=torch.float32, device=self.device)], dim=-1)
        points_c = (world_to_camera @ points_h.T).T[:, :3]

        # OpenGL camera convention: camera looks along -Z.
        z_forward = -points_c[:, 2].clamp(max=-1e-4)
        u = fx * (points_c[:, 0] / z_forward) + cx
        v = fy * (-points_c[:, 1] / z_forward) + cy
        sigma = (scales.mean(dim=-1).clamp_min(1e-4) * fx / z_forward).clamp(
            self.config.min_sigma_px,
            self.config.max_sigma_px,
        )

        valid = (points_c[:, 2] < -1e-4) & (u >= -self.config.max_sigma_px) & (u < width + self.config.max_sigma_px) & (v >= -self.config.max_sigma_px) & (v < height + self.config.max_sigma_px)
        valid_idx = valid.nonzero(as_tuple=False)[:, 0]
        if valid_idx.numel() > self.config.max_visible:
            valid_idx = valid_idx[-self.config.max_visible :]

        for idx in valid_idx.tolist():
            ui = u[idx]
            vi = v[idx]
            si = sigma[idx]
            radius = int(max(1, round(float(si.detach().cpu()) * self.config.patch_sigma_multiplier)))
            u0 = max(0, int(torch.floor(ui.detach()).item()) - radius)
            u1 = min(width, int(torch.floor(ui.detach()).item()) + radius + 1)
            v0 = max(0, int(torch.floor(vi.detach()).item()) - radius)
            v1 = min(height, int(torch.floor(vi.detach()).item()) + radius + 1)
            if u0 >= u1 or v0 >= v1:
                continue

            yy, xx = torch.meshgrid(
                torch.arange(v0, v1, dtype=torch.float32, device=self.device),
                torch.arange(u0, u1, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            dist2 = (xx - ui) ** 2 + (yy - vi) ** 2
            weight = opacity[idx].clamp(0.0, 1.0) * torch.exp(-0.5 * dist2 / (si ** 2 + 1e-6))
            weight = weight[..., None]
            accum_rgb[v0:v1, u0:u1] = accum_rgb[v0:v1, u0:u1] + weight * colors[idx].clamp(0.0, 1.0)
            accum_w[v0:v1, u0:u1] = accum_w[v0:v1, u0:u1] + weight

        rgb = accum_rgb / accum_w.clamp_min(1e-6)
        alpha = accum_w.clamp(0.0, 1.0)
        rgb = torch.where(accum_w > 1e-6, rgb, accum_rgb)
        return RenderOutput(rgb=rgb.clamp(0.0, 1.0), alpha=alpha)
