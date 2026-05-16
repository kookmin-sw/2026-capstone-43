import json
from pathlib import Path

import torch
import torch.nn.functional as F


def azimuth_to_signed_deg(azimuth):
    azimuth = torch.as_tensor(azimuth).float()
    return torch.where(azimuth > 180.0, azimuth - 360.0, azimuth)


def azimuth_raw_to_signed_front_deg(azimuth):
    signed = azimuth_to_signed_deg(azimuth)
    if torch.any(signed < -45.0) or torch.any(signed > 45.0):
        raise ValueError(
            f"Front-cone azimuth target must stay within [-45, 45], got range "
            f"[{signed.min().item()}, {signed.max().item()}]"
        )
    return signed


def circular_difference_deg(pred, target):
    pred = torch.as_tensor(pred).float()
    target = torch.as_tensor(target).float()
    return (pred - target + 180.0) % 360.0 - 180.0


def azimuth_deg_to_sincos(azimuth):
    azimuth = torch.as_tensor(azimuth).float() * torch.pi / 180.0
    return torch.stack([torch.sin(azimuth), torch.cos(azimuth)], dim=-1)


def azimuth_sincos_to_deg(pred):
    pred = torch.as_tensor(pred).float()
    if pred.shape[-1] != 2:
        raise ValueError(f"azimuth_sincos_to_deg expects a final dimension of 2, got shape {tuple(pred.shape)}")
    pred = F.normalize(pred, dim=-1)
    angle = torch.atan2(pred[..., 0], pred[..., 1])
    return torch.remainder(torch.rad2deg(angle), 360.0)


def _load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def build_front9_label_space_from_manifest(manifest_paths):
    if isinstance(manifest_paths, (str, Path)):
        manifest_paths = [manifest_paths]

    entries = []
    for path in manifest_paths:
        if not path:
            continue
        entries.extend(_load_manifest(Path(path)))

    if not entries:
        raise ValueError("build_front9_label_space_from_manifest requires at least one manifest with entries")

    raw_unique = sorted({int(item["azimuth"]) for item in entries})
    signed_unique = sorted({int(round(value)) for value in azimuth_raw_to_signed_front_deg(raw_unique).tolist()})
    raw_to_signed = {
        raw_value: int(round(signed_value))
        for raw_value, signed_value in zip(raw_unique, azimuth_raw_to_signed_front_deg(raw_unique).tolist())
    }
    signed_to_class = {signed_value: index for index, signed_value in enumerate(signed_unique)}
    raw_to_class = {raw_value: signed_to_class[signed_value] for raw_value, signed_value in raw_to_signed.items()}
    class_to_raw = [next(raw for raw, signed in raw_to_signed.items() if signed == signed_value) for signed_value in signed_unique]

    label_space = {
        "raw_unique_azimuth": raw_unique,
        "signed_unique_azimuth": signed_unique,
        "raw_to_signed": raw_to_signed,
        "signed_to_class": signed_to_class,
        "raw_to_class": raw_to_class,
        "class_to_signed": signed_unique,
        "class_to_raw": class_to_raw,
        "num_classes": len(signed_unique),
    }

    print("[FRONT9 LABEL SPACE]")
    print("raw unique azimuth:", raw_unique)
    print("signed unique azimuth:", signed_unique)
    print("class mapping (raw->class):", raw_to_class)
    print("class mapping (signed->class):", signed_to_class)
    return label_space


def azimuth_raw_to_front9_class(azimuth, label_space):
    signed = azimuth_raw_to_signed_front_deg(azimuth)
    signed_rounded = signed.round().to(torch.int64)
    signed_to_class = label_space["signed_to_class"]
    class_values = [signed_to_class[int(value)] for value in signed_rounded.detach().cpu().tolist()]
    return torch.tensor(class_values, device=signed.device, dtype=torch.long)


def front9_class_to_signed_deg(class_index, label_space):
    class_tensor = torch.as_tensor(class_index, dtype=torch.long)
    class_to_signed = torch.as_tensor(label_space["class_to_signed"], dtype=torch.float32, device=class_tensor.device)
    return class_to_signed[class_tensor]


def azimuth_to_signed_front_deg(azimuth, strict_support=False):
    return azimuth_raw_to_signed_front_deg(azimuth)


def labels_to_unit_vectors(azimuth, elevation):
    azimuth = azimuth_to_signed_deg(azimuth)
    elevation = torch.as_tensor(elevation).float()
    azimuth = azimuth * torch.pi / 180.0
    elevation = (elevation - 90.0) * torch.pi / 180.0
    x_front = torch.cos(elevation) * torch.cos(azimuth)
    y_left = torch.cos(elevation) * torch.sin(azimuth)
    z_up = torch.sin(elevation)
    return torch.stack([x_front, y_left, z_up], dim=-1)


def compute_losses(
    outputs,
    batch,
    loss_weights,
    azimuth_head_mode="front9_classification",
    azimuth_regression_loss="smoothl1",
    front9_label_space=None,
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
    elif azimuth_head_mode == "front9_classification" and "azimuth_front9_logits" in outputs and loss_weights["azimuth"] != 0:
        if front9_label_space is None:
            raise ValueError("front9_label_space is required for front9_classification")
        target = azimuth_raw_to_front9_class(batch["azimuth_target"], front9_label_space).to(
            outputs["azimuth_front9_logits"].device
        )
        azimuth_loss = F.cross_entropy(outputs["azimuth_front9_logits"], target)
    elif azimuth_head_mode == "front_regression" and "azimuth_regression" in outputs and loss_weights["azimuth"] != 0:
        target = azimuth_raw_to_signed_front_deg(batch["azimuth_target"]).to(
            outputs["azimuth_regression"].device
        )
        if azimuth_regression_loss == "smoothl1":
            azimuth_loss = F.smooth_l1_loss(outputs["azimuth_regression"], target)
        elif azimuth_regression_loss == "huber":
            azimuth_loss = F.huber_loss(outputs["azimuth_regression"], target)
        else:
            raise ValueError(f"Unsupported azimuth_regression_loss: {azimuth_regression_loss}")
    elif (
        azimuth_head_mode in {"full360_regression", "full360_sincos_regression"}
        and "azimuth_regression" in outputs
        and loss_weights["azimuth"] != 0
    ):
        azimuth_target = batch.get("azimuth_target_deg", batch["azimuth_target"])
        target = azimuth_deg_to_sincos(azimuth_target).to(outputs["azimuth_regression"].device)
        pred = F.normalize(outputs["azimuth_regression"], dim=-1)
        if azimuth_regression_loss == "smoothl1":
            azimuth_loss = F.smooth_l1_loss(pred, target)
        elif azimuth_regression_loss == "huber":
            azimuth_loss = F.huber_loss(pred, target)
        else:
            raise ValueError(f"Unsupported azimuth_regression_loss: {azimuth_regression_loss}")

    elevation_loss = zero
    if "elevation_logits" in outputs and loss_weights["elevation"] != 0:
        elevation_loss = F.cross_entropy(outputs["elevation_logits"], batch["elevation_target"])

    vector_loss = zero
    if "vector" in outputs and loss_weights["vector"] != 0:
        azimuth_target = batch.get("azimuth_target_deg", batch["azimuth_target"])
        target_vector = labels_to_unit_vectors(azimuth_target, batch["elevation_target"]).to(outputs["vector"].device)
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
