import torch
import torch.nn.functional as F


def labels_to_unit_vectors(azimuth, elevation):
    azimuth = torch.as_tensor(azimuth).float()
    elevation = torch.as_tensor(elevation).float()
    azimuth = torch.where(azimuth > 180, azimuth - 360, azimuth)
    azimuth = azimuth * torch.pi / 180.0
    elevation = (elevation - 90.0) * torch.pi / 180.0
    x = torch.cos(elevation) * torch.cos(azimuth)
    y = torch.sin(elevation)
    z = -torch.cos(elevation) * torch.sin(azimuth)
    return torch.stack([x, y, z], dim=-1)


def compute_losses(outputs, batch, loss_weights):
    reference_tensor = next(iter(outputs.values()))
    zero = reference_tensor.new_zeros(())

    class_loss = zero
    if "class_logits" in outputs and loss_weights["class"] != 0:
        class_loss = F.binary_cross_entropy_with_logits(outputs["class_logits"], batch["class_target"])

    distance_loss = zero
    if "distance_logits" in outputs and loss_weights["distance"] != 0:
        distance_loss = F.cross_entropy(outputs["distance_logits"], batch["distance_target"])

    azimuth_loss = zero
    if "azimuth_logits" in outputs and loss_weights["azimuth"] != 0:
        azimuth_loss = F.cross_entropy(outputs["azimuth_logits"], batch["azimuth_target"])

    elevation_loss = zero
    if "elevation_logits" in outputs and loss_weights["elevation"] != 0:
        elevation_loss = F.cross_entropy(outputs["elevation_logits"], batch["elevation_target"])

    vector_loss = zero
    if "vector" in outputs and loss_weights["vector"] != 0:
        target_vector = labels_to_unit_vectors(batch["azimuth_target"], batch["elevation_target"]).to(outputs["vector"].device)
        vector_loss = F.smooth_l1_loss(outputs["vector"], target_vector)

    total = (
        loss_weights["class"] * class_loss
        + loss_weights["distance"] * distance_loss
        + loss_weights["azimuth"] * azimuth_loss
        + loss_weights["elevation"] * elevation_loss
        + loss_weights["vector"] * vector_loss
    )

    return total, {
        "class": class_loss.detach(),
        "distance": distance_loss.detach(),
        "azimuth": azimuth_loss.detach(),
        "elevation": elevation_loss.detach(),
        "vector": vector_loss.detach(),
    }
