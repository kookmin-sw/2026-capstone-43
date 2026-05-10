from contextlib import nullcontext

import numpy as np
import open_clip
from PIL import Image
import torch


CLIP_DIM = {
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
}


def _normalize_clip_model_name(model_type: str) -> str:
    if model_type == "ViT-L/14@336px":
        return "ViT-L-14"
    return model_type


def load_clip_model(model_type: str, checkpoint: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    normalized_model = _normalize_clip_model_name(model_type)
    precision = "fp16" if device == "cuda" else "fp32"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        normalized_model,
        pretrained=str(checkpoint),
        precision=precision,
        device=device,
    )
    clip_model.eval()
    return clip_model, preprocess, CLIP_DIM[normalized_model]


def _get_model_device_and_dtype(clip_model):
    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "conv1"):
        param = clip_model.visual.conv1.weight
        return param.device, param.dtype
    param = next(clip_model.parameters())
    return param.device, param.dtype


def _get_autocast_context(device, dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def get_img_feats(img, preprocess, clip_model):
    img_pil = Image.fromarray(np.uint8(img))
    device, dtype = _get_model_device_and_dtype(clip_model)
    img_in = preprocess(img_pil)[None, ...].to(device=device, dtype=dtype)
    with torch.no_grad():
        with _get_autocast_context(device, dtype):
            img_feats = clip_model.encode_image(img_in).float()
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    return np.float32(img_feats.cpu())


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    device, dtype = _get_model_device_and_dtype(clip_model)
    text_tokens = open_clip.tokenize(in_text).to(device=device)
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):
        current_batch = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + current_batch]
        with torch.no_grad():
            with _get_autocast_context(device, dtype):
                batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + current_batch, :] = batch_feats
        text_id += current_batch
    return text_feats


def get_text_feats_multiple_templates(in_text, clip_model, clip_feat_dim, batch_size=64):
    templates = [
        "{}",
        "There is the {} in the scene.",
    ]
    templated = [template.format(label) for label in in_text for template in templates]
    text_feats = get_text_feats(templated, clip_model, clip_feat_dim, batch_size=batch_size)
    text_feats = text_feats.reshape((-1, len(templates), text_feats.shape[-1]))
    return np.mean(text_feats, axis=1)

