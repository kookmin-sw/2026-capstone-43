"""Spatial branch package for Sci-Phi.

This package keeps SELD-based spatial code outside ``phi4mm_clean`` so the
vendor model code remains untouched.
"""

from .utils.input import FOASpatialFeatureExtractor, feature_maps_to_seld_tensor
from .models.spatial_encoder import SeldNetSpatialEncoder
from .projector import DualBranchAudioEncoder, SpatialFusionProjector
from .vision import FrozenDinoV2VisionBranch, build_dinov2_image_transform
from .model import (
    attach_spatial_branch,
    get_audio_embed_module,
    set_spatial_features,
    setup_spatial_lora,
)

__all__ = [
    "FOASpatialFeatureExtractor",
    "feature_maps_to_seld_tensor",
    "SeldNetSpatialEncoder",
    "DualBranchAudioEncoder",
    "SpatialFusionProjector",
    "FrozenDinoV2VisionBranch",
    "build_dinov2_image_transform",
    "attach_spatial_branch",
    "get_audio_embed_module",
    "set_spatial_features",
    "setup_spatial_lora",
]
