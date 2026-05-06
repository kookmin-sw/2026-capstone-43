#!/usr/bin/env python3
import os
import shutil

if os.path.isdir("/usr/local/cuda-12.1") and not os.environ.get("CUDA_HOME"):
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"

local_bin = os.path.expanduser("~/.local/bin")
cuda_bin = os.path.join(os.environ.get("CUDA_HOME", ""), "bin")
path_parts = [p for p in [cuda_bin, local_bin, os.environ.get("PATH", "")] if p]
os.environ["PATH"] = os.pathsep.join(path_parts)

import torch


def main():
    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '')}")
    print(f"nvcc: {shutil.which('nvcc')}")
    try:
        import gsplat

        print(f"gsplat: {getattr(gsplat, '__version__', 'unknown')}")
        from gsplat import rasterization

        print(f"gsplat rasterization: {rasterization}")
        means = torch.tensor([[0.0, 0.0, 1.5]], device="cuda")
        quats = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
        scales = torch.tensor([[0.05, 0.05, 0.05]], device="cuda")
        opacities = torch.tensor([0.8], device="cuda")
        colors = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
        viewmats = torch.eye(4, device="cuda")[None]
        Ks = torch.tensor([[[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]], device="cuda")
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=32,
            height=32,
        )
        print(f"tiny render OK: rgb={tuple(renders.shape)} alpha={tuple(alphas.shape)}")
    except Exception as exc:
        print(f"gsplat backend check failed: {exc!r}")
        raise


if __name__ == "__main__":
    main()
