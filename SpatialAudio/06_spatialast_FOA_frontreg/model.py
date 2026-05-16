import torch.nn as nn

from backbone import build_backbone
from heads import build_heads


class FOASpatialASTModel(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone=None,
            heads=None,
            use_class_head=True,
            use_distance_head=True,
            use_azimuth_head=True,
            use_elevation_head=True,
            use_vector_head=True,
            azimuth_head_mode="full360_classification",
            azimuth_regression_range=45.0,
            **backbone_kwargs,
        ):
        super().__init__()
        self.backbone = backbone or build_backbone(**backbone_kwargs)
        self.heads = heads or build_heads(
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            use_class_head=use_class_head,
            use_distance_head=use_distance_head,
            use_azimuth_head=use_azimuth_head,
            use_elevation_head=use_elevation_head,
            use_vector_head=use_vector_head,
            azimuth_head_mode=azimuth_head_mode,
            azimuth_regression_range=azimuth_regression_range,
        )

    def forward(self, waveforms, reverbs=None, mask_t_prob=0.0, mask_f_prob=0.0, return_backbone_outputs=False):
        backbone_outputs = self.backbone(
            waveforms,
            reverbs=reverbs,
            mask_t_prob=mask_t_prob,
            mask_f_prob=mask_f_prob,
        )
        head_outputs = self.heads(backbone_outputs)
        if return_backbone_outputs:
            return {
                "outputs": head_outputs,
                "backbone": backbone_outputs,
            }
        return head_outputs

    def get_debug_shapes(self):
        return self.backbone.get_debug_shapes()


def build_model(num_classes, **kwargs):
    return FOASpatialASTModel(num_classes=num_classes, **kwargs)
