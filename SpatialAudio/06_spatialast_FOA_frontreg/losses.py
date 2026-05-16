import torch
import torch.nn.functional as F


FRONT_CONE_AZIMUTH_SUPPORT = {-40, -30, -20, -10, 0, 10, 20, 30, 40}


def azimuth_to_signed_deg(azimuth):
    azimuth = torch.as_tensor(azimuth).float()
    return torch.where(azimuth > 180.0, azimuth - 360.0, azimuth)


def azimuth_to_signed_front_deg(azimuth, strict_support=False):
    signed = azimuth_to_signed_deg(azimuth)
    if torch.any(signed < -45.0) or torch.any(signed > 45.0):
        raise ValueError(
            f"Front-cone azimuth target must stay within [-45, 45], got range "
            f"[{signed.min().item()}, {signed.max().item()}]"
        )
    if strict_support:
        rounded = signed.round().to(torch.int64)
        invalid = sorted(set(rounded.detach().cpu().tolist()) - FRONT_CONE_AZIMUTH_SUPPORT)
        if invalid:
            raise ValueError(
                f"Current dataset is expected to use front-cone support "
                f"{sorted(FRONT_CONE_AZIMUTH_SUPPORT)}, got {invalid}"
            )
    return signed


def labels_to_unit_vectors(azimuth, elevation):
    azimuth = azimuth_to_signed_deg(azimuth)
    elevation = torch.as_tensor(elevation).float()
    azimuth = azimuth * torch.pi / 180.0
    elevation = (elevation - 90.0) * torch.pi / 180.0
    x = torch.cos(elevation) * torch.cos(azimuth)
    y = torch.sin(elevation)
    z = -torch.cos(elevation) * torch.sin(azimuth)
    return torch.stack([x, y, z], dim=-1)


def compute_losses(
    outputs,
    batch,
    loss_weights,
    azimuth_head_mode="full360_classification",
    azimuth_regression_loss="smoothl1",
):
    reference_tensor = next(iter(outputs.values()))
    zero = reference_tensor.new_zeros(())

    class_loss = zero
    if "class_logits" in outputs and loss_weights["class"] != 0:
        class_loss = F.binary_cross_entropy_with_logits(outputs["class_logits"], batch["class_target"])

    distance_loss = zero
    if "distance_logits" in outputs and loss_weights["distance"] != 0:
        distance_loss = F.cross_entropy(outputs["distance_logits"], batch["distance_target"])

    azimuth_loss = zero
    if azimuth_head_mode == "full360_classification" and "azimuth_logits" in outputs and loss_weights["azimuth"] != 0:
        azimuth_loss = F.cross_entropy(outputs["azimuth_logits"], batch["azimuth_target"])
    elif azimuth_head_mode == "front_regression" and "azimuth_regression" in outputs and loss_weights["azimuth"] != 0:
        target = azimuth_to_signed_front_deg(batch["azimuth_target"], strict_support=True).to(
            outputs["azimuth_regression"].device
        )
        if azimuth_regression_loss == "smoothl1":
            azimuth_loss = F.smooth_l1_loss(outputs["azimuth_regression"], target)
        elif azimuth_regression_loss == "huber":
            azimuth_loss = F.huber_loss(outputs["azimuth_regression"], target)
        else:
            raise ValueError(f"Unsupported azimuth_regression_loss: {azimuth_regression_loss}")

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
