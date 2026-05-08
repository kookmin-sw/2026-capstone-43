"""RGB -> Depth -> Point cloud -> Spherical projection MVP."""

from .pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
