from __future__ import annotations

from typing import Any

import numpy as np

EIGHT_WAY_LABELS = [
    "front-left",
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
]
EIGHT_WAY_CENTER_DEG = np.asarray([-45.0, 0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0], dtype=np.float32)


def _wrap_degrees(angles_deg: np.ndarray) -> np.ndarray:
    wrapped = (angles_deg + 180.0) % 360.0 - 180.0
    wrapped = np.where(np.isclose(wrapped, -180.0), 180.0, wrapped)
    return np.asarray(wrapped, dtype=np.float32)


def azimuth_to_8way_index(azimuth_rad: np.ndarray | float, mapping_mode: str = "sector") -> np.ndarray:
    azimuth_deg = _wrap_degrees(np.degrees(np.asarray(azimuth_rad, dtype=np.float32)))
    if mapping_mode == "nearest":
        diffs = np.abs(_wrap_degrees(azimuth_deg[..., None] - EIGHT_WAY_CENTER_DEG[None, ...]))
        return np.argmin(diffs, axis=-1).astype(np.int32)
    if mapping_mode != "sector":
        raise ValueError(f"Unsupported mapping_mode: {mapping_mode}")
    sector_idx = np.floor(((azimuth_deg + 22.5) % 360.0) / 45.0).astype(np.int32)
    sector_to_label_index = np.asarray([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32)
    return sector_to_label_index[sector_idx]


def map_azimuth_bins_to_8way(azimuth_centers_rad: np.ndarray, mapping_mode: str = "sector") -> np.ndarray:
    return azimuth_to_8way_index(azimuth_centers_rad, mapping_mode=mapping_mode).astype(np.int32)


def _pool(values: np.ndarray, mode: str) -> np.ndarray:
    if values.size == 0:
        return np.zeros((values.shape[-1] if values.ndim else 1,), dtype=np.float32)
    if mode == "max":
        return np.max(values, axis=0).astype(np.float32)
    return np.mean(values, axis=0).astype(np.float32)


def pool_audio_azimuth_to_8way(
    azimuth_tensor: np.ndarray,
    azimuth_centers_rad: np.ndarray,
    channel_names: list[str],
    pooling_mode: str = "mean",
    mapping_mode: str = "sector",
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if azimuth_tensor.ndim != 2:
        raise ValueError(f"Expected azimuth tensor [A,C], got {azimuth_tensor.shape}.")
    if azimuth_tensor.shape[0] != azimuth_centers_rad.shape[0]:
        raise ValueError("Azimuth tensor/bin center size mismatch.")
    if pooling_mode not in {"mean", "max", "both"}:
        raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")

    bin_to_sector = map_azimuth_bins_to_8way(azimuth_centers_rad, mapping_mode=mapping_mode)

    def build(mode: str) -> np.ndarray:
        pooled = np.zeros((len(EIGHT_WAY_LABELS), azimuth_tensor.shape[1]), dtype=np.float32)
        for sector_idx in range(len(EIGHT_WAY_LABELS)):
            selected = azimuth_tensor[bin_to_sector == sector_idx]
            pooled[sector_idx] = _pool(selected, mode=mode)
        return pooled

    pooled_mean = build("mean")
    pooled_max = build("max") if pooling_mode in {"max", "both"} else None
    primary = pooled_max if pooling_mode == "max" else pooled_mean
    assert primary is not None

    score_channel = "dp_reliability" if "dp_reliability" in channel_names else channel_names[0]
    score_idx = channel_names.index(score_channel)
    top_idx = int(np.argmax(primary[:, score_idx]))
    meta = {
        "labels": list(EIGHT_WAY_LABELS),
        "label_order": list(EIGHT_WAY_LABELS),
        "sector_centers_deg": EIGHT_WAY_CENTER_DEG.tolist(),
        "sector_ranges": [
            {"label": "front-left", "range_deg": [-67.5, -22.5]},
            {"label": "front", "range_deg": [-22.5, 22.5]},
            {"label": "front-right", "range_deg": [22.5, 67.5]},
            {"label": "right", "range_deg": [67.5, 112.5]},
            {"label": "back-right", "range_deg": [112.5, 157.5]},
            {"label": "back", "range_deg": [157.5, 180.0], "wraps_to_deg": [-180.0, -157.5]},
            {"label": "back-left", "range_deg": [-157.5, -112.5]},
            {"label": "left", "range_deg": [-112.5, -67.5]},
        ],
        "mapping_mode": mapping_mode,
        "pooling_mode": pooling_mode,
        "azimuth_bin_centers_deg": np.degrees(azimuth_centers_rad).tolist(),
        "azimuth_bin_to_sector": bin_to_sector.tolist(),
        "channel_names": list(channel_names),
        "primary_output_shape": list(primary.shape),
        "mean_output_shape": list(pooled_mean.shape),
        "max_output_shape": None if pooled_max is None else list(pooled_max.shape),
        "score_channel_for_top_label": score_channel,
        "top_label": EIGHT_WAY_LABELS[top_idx],
        "top_label_score": float(primary[top_idx, score_idx]),
    }
    return primary.astype(np.float32), None if pooled_max is None else pooled_max.astype(np.float32), meta
