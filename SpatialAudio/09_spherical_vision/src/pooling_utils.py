from __future__ import annotations

import math
from typing import Any

import numpy as np

from .spherical_projection import DEPTH_LIKE_CHANNELS, FeatureBundle

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
    wrapped[np.isclose(wrapped, -180.0)] = 180.0
    return wrapped.astype(np.float32)


def azimuth_to_8way_index(
    azimuth_rad: np.ndarray | float,
    mapping_mode: str = "sector",
) -> np.ndarray:
    azimuth_deg = _wrap_degrees(np.degrees(np.asarray(azimuth_rad, dtype=np.float32)))

    if mapping_mode == "nearest":
        diffs = np.abs(_wrap_degrees(azimuth_deg[..., None] - EIGHT_WAY_CENTER_DEG[None, ...]))
        return np.argmin(diffs, axis=-1).astype(np.int32)

    sector_idx = np.floor(((azimuth_deg + 22.5) % 360.0) / 45.0).astype(np.int32)
    sector_to_label_index = np.asarray([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32)
    return sector_to_label_index[sector_idx]


def map_azimuth_bins_to_8way(
    azimuth_centers_rad: np.ndarray,
    mapping_mode: str = "sector",
) -> np.ndarray:
    return azimuth_to_8way_index(azimuth_centers_rad, mapping_mode=mapping_mode).astype(np.int32)


def _pool_channel_values(
    values: np.ndarray,
    selector: np.ndarray,
    mode: str,
) -> float:
    if values.size == 0 or not np.any(selector):
        return 0.0
    selected = values[selector]
    if selected.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(selected))
    return float(np.mean(selected))


def pool_azimuth_features_to_8way(
    azimuth_bundle: FeatureBundle,
    pooling_mode: str = "mean",
    mapping_mode: str = "sector",
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if azimuth_bundle.valid_count.ndim != 1:
        raise ValueError("8-way pooling expects the azimuth-aggregated 1D feature bundle.")

    if pooling_mode not in {"mean", "max", "both"}:
        raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")

    bin_to_sector = map_azimuth_bins_to_8way(azimuth_bundle.azimuth_centers, mapping_mode=mapping_mode)
    observed_mask = azimuth_bundle.channels["observed_mask"] > 0.5
    has_points_mask = azimuth_bundle.channels["has_points"] > 0.5

    def build_tensor(mode: str) -> np.ndarray:
        pooled = np.zeros((len(EIGHT_WAY_LABELS), len(azimuth_bundle.channel_names)), dtype=np.float32)
        for sector_idx in range(len(EIGHT_WAY_LABELS)):
            sector_selector = bin_to_sector == sector_idx
            for channel_idx, channel_name in enumerate(azimuth_bundle.channel_names):
                values = azimuth_bundle.channels[channel_name]
                if channel_name in DEPTH_LIKE_CHANNELS:
                    channel_selector = sector_selector & has_points_mask
                elif channel_name in {"valid_ratio", "occupancy", "density", "has_points"}:
                    channel_selector = sector_selector & observed_mask
                else:
                    channel_selector = sector_selector
                pooled[sector_idx, channel_idx] = _pool_channel_values(values, channel_selector, mode=mode)
        return pooled

    pooled_mean = build_tensor("mean")
    pooled_max = build_tensor("max") if pooling_mode in {"max", "both"} else None
    if pooling_mode == "max":
        primary = pooled_max
        assert primary is not None
    else:
        primary = pooled_mean

    sector_ranges = [
        {"label": "front-left", "range_deg": [-67.5, -22.5]},
        {"label": "front", "range_deg": [-22.5, 22.5]},
        {"label": "front-right", "range_deg": [22.5, 67.5]},
        {"label": "right", "range_deg": [67.5, 112.5]},
        {"label": "back-right", "range_deg": [112.5, 157.5]},
        {"label": "back", "range_deg": [157.5, 180.0], "wraps_to_deg": [-180.0, -157.5]},
        {"label": "back-left", "range_deg": [-157.5, -112.5]},
        {"label": "left", "range_deg": [-112.5, -67.5]},
    ]

    meta = {
        "labels": list(EIGHT_WAY_LABELS),
        "label_order": list(EIGHT_WAY_LABELS),
        "sector_centers_deg": EIGHT_WAY_CENTER_DEG.tolist(),
        "sector_ranges": sector_ranges,
        "mapping_mode": mapping_mode,
        "pooling_mode": pooling_mode,
        "azimuth_bin_centers_deg": np.degrees(azimuth_bundle.azimuth_centers).tolist(),
        "azimuth_bin_to_sector": bin_to_sector.tolist(),
        "primary_output_shape": list(primary.shape),
        "mean_output_shape": list(pooled_mean.shape),
        "max_output_shape": None if pooled_max is None else list(pooled_max.shape),
    }
    return primary.astype(np.float32), None if pooled_max is None else pooled_max.astype(np.float32), meta
