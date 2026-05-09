import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_layers import trunc_normal_


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        trunc_normal_(self.proj.weight, std=2e-5)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        return self.proj(x)


class UnitVectorHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 3)
        trunc_normal_(self.proj.weight, std=2e-5)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class AzimuthRegressionHead(nn.Module):
    def __init__(self, in_dim, azimuth_range=45.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)
        self.azimuth_range = float(azimuth_range)
        trunc_normal_(self.proj.weight, std=2e-5)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        raw = self.proj(x)
        pred_deg = self.azimuth_range * torch.tanh(raw)
        return pred_deg.squeeze(-1)


class FOASpatialHeads(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_classes,
            use_class_head=True,
            use_distance_head=True,
            use_azimuth_head=True,
            use_elevation_head=True,
            use_vector_head=True,
            azimuth_head_mode="full360_classification",
            azimuth_regression_range=45.0,
        ):
        super().__init__()
        self.use_class_head = use_class_head and num_classes > 0
        self.use_distance_head = use_distance_head
        self.use_azimuth_head = use_azimuth_head
        self.use_elevation_head = use_elevation_head
        self.use_vector_head = use_vector_head
        self.azimuth_head_mode = azimuth_head_mode

        self.class_head = LinearHead(embed_dim, num_classes) if self.use_class_head else None
        self.distance_head = LinearHead(embed_dim, 21) if self.use_distance_head else None
        self.azimuth_head = None
        self.azimuth_regression_head = None
        if self.use_azimuth_head:
            if azimuth_head_mode == "full360_classification":
                self.azimuth_head = LinearHead(embed_dim, 360)
            elif azimuth_head_mode == "front_regression":
                self.azimuth_regression_head = AzimuthRegressionHead(
                    embed_dim,
                    azimuth_range=azimuth_regression_range,
                )
            else:
                raise ValueError(f"Unsupported azimuth_head_mode: {azimuth_head_mode}")
        self.elevation_head = LinearHead(embed_dim, 180) if self.use_elevation_head else None
        self.vector_head = UnitVectorHead(embed_dim) if self.use_vector_head else None

    def forward(self, backbone_outputs):
        outputs = {}

        if self.class_head is not None:
            outputs["class_logits"] = self.class_head(backbone_outputs["class_token"])
        if self.distance_head is not None:
            outputs["distance_logits"] = self.distance_head(backbone_outputs["distance_token"])
        if self.azimuth_head is not None:
            outputs["azimuth_logits"] = self.azimuth_head(backbone_outputs["doa_token"])
        if self.azimuth_regression_head is not None:
            outputs["azimuth_regression"] = self.azimuth_regression_head(backbone_outputs["doa_token"])
        if self.elevation_head is not None:
            outputs["elevation_logits"] = self.elevation_head(backbone_outputs["doa_token"])
        if self.vector_head is not None:
            outputs["vector"] = self.vector_head(backbone_outputs["doa_token"])

        return outputs


def build_heads(embed_dim, num_classes, **kwargs):
    return FOASpatialHeads(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
