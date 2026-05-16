from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

AZIMUTH_RANGE_RAD = (-math.pi, math.pi)
ELEVATION_RANGE_RAD = (-math.pi / 2.0, math.pi / 2.0)
EPSILON = 1.0e-8

AUDIO_CHANNEL_NAMES = [
    "beam_power",
    "aiv_score",
    "diffuseness",
    "dp_reliability",
    "energy",
    "stability",
]
WINDOW_CHANNEL_NAMES = ["beam_power", "aiv_score", "diffuseness", "dp_reliability", "energy"]

CHANNEL_DESCRIPTIONS: dict[str, str] = {
    "beam_power": "First-order FOA cardioid beamformer power for each spherical direction bin.",
    "aiv_score": "Active-intensity-vector magnitude accumulated into the matching direction bin.",
    "diffuseness": "Uncertainty proxy; high values indicate diffuse or ambiguous directional evidence.",
    "dp_reliability": "Direct-path reliability proxy combining directional energy, local contrast, and low diffuseness.",
    "energy": "Normalized directional energy proxy from beam and active-intensity evidence.",
    "stability": "Temporal stability proxy from window-wise directional energy consistency.",
}


@dataclass(frozen=True)
class AngularGrid:
    num_az_bins: int
    num_el_bins: int
    azimuth_edges: np.ndarray
    azimuth_centers: np.ndarray
    elevation_edges: np.ndarray
    elevation_centers: np.ndarray

    def get_bin_centers(self) -> dict[str, np.ndarray]:
        azimuth_grid, elevation_grid = np.meshgrid(
            self.azimuth_centers,
            self.elevation_centers,
            indexing="xy",
        )
        return {
            "azimuth_rad": azimuth_grid.astype(np.float32),
            "azimuth_deg": np.degrees(azimuth_grid).astype(np.float32),
            "elevation_rad": elevation_grid.astype(np.float32),
            "elevation_deg": np.degrees(elevation_grid).astype(np.float32),
        }

    def direction_vectors(self) -> np.ndarray:
        centers = self.get_bin_centers()
        az = centers["azimuth_rad"]
        el = centers["elevation_rad"]
        cos_el = np.cos(el)
        x = cos_el * np.sin(az)
        y = np.sin(el)
        z = cos_el * np.cos(az)
        return np.stack([x, y, z], axis=-1).astype(np.float32)

    def to_meta_dict(self) -> dict[str, Any]:
        centers = self.get_bin_centers()
        return {
            "num_az_bins": int(self.num_az_bins),
            "num_el_bins": int(self.num_el_bins),
            "azimuth_edges_rad": self.azimuth_edges.tolist(),
            "azimuth_edges_deg": np.degrees(self.azimuth_edges).tolist(),
            "azimuth_centers_rad": self.azimuth_centers.tolist(),
            "azimuth_centers_deg": np.degrees(self.azimuth_centers).tolist(),
            "elevation_edges_rad": self.elevation_edges.tolist(),
            "elevation_edges_deg": np.degrees(self.elevation_edges).tolist(),
            "elevation_centers_rad": self.elevation_centers.tolist(),
            "elevation_centers_deg": np.degrees(self.elevation_centers).tolist(),
            "bin_centers_grid_deg": {
                "azimuth": centers["azimuth_deg"].tolist(),
                "elevation": centers["elevation_deg"].tolist(),
            },
        }


@dataclass(frozen=True)
class AudioSphere:
    tensor: np.ndarray
    azimuth_tensor: np.ndarray
    tensor_max: np.ndarray | None
    azimuth_tensor_max: np.ndarray | None
    channel_names: list[str]
    grid: AngularGrid
    aggregation_mode: str
    metadata: dict[str, Any]


def coordinate_convention_dict() -> dict[str, Any]:
    return {
        "coordinate_system": {
            "x": "right",
            "y": "up",
            "z": "forward",
            "reference_frame": "camera/listener centered",
        },
        "azimuth": {
            "definition": "atan2(x, z)",
            "zero_degrees": "forward",
            "positive_direction": "right",
            "range_degrees": [-180.0, 180.0],
            "range_radians": [-math.pi, math.pi],
        },
        "elevation": {
            "definition": "atan2(y, sqrt(x^2 + z^2))",
            "zero_degrees": "horizontal",
            "positive_direction": "up",
            "range_degrees": [-90.0, 90.0],
            "range_radians": [-math.pi / 2.0, math.pi / 2.0],
        },
        "alignment": "Matches 09_spherical_vision V_sphere angular convention.",
    }


def build_angular_grid(num_az_bins: int, num_el_bins: int) -> AngularGrid:
    if num_az_bins <= 0:
        raise ValueError(f"num_az_bins must be positive, got {num_az_bins}.")
    if num_el_bins <= 0:
        raise ValueError(f"num_el_bins must be positive, got {num_el_bins}.")
    azimuth_edges = np.linspace(AZIMUTH_RANGE_RAD[0], AZIMUTH_RANGE_RAD[1], num_az_bins + 1, dtype=np.float32)
    elevation_edges = np.linspace(ELEVATION_RANGE_RAD[0], ELEVATION_RANGE_RAD[1], num_el_bins + 1, dtype=np.float32)
    return AngularGrid(
        num_az_bins=int(num_az_bins),
        num_el_bins=int(num_el_bins),
        azimuth_edges=azimuth_edges,
        azimuth_centers=((azimuth_edges[:-1] + azimuth_edges[1:]) * 0.5).astype(np.float32),
        elevation_edges=elevation_edges,
        elevation_centers=((elevation_edges[:-1] + elevation_edges[1:]) * 0.5).astype(np.float32),
    )


def get_bin_centers(num_az_bins: int, num_el_bins: int) -> dict[str, np.ndarray]:
    return build_angular_grid(num_az_bins, num_el_bins).get_bin_centers()


def _wrap_azimuth_rad(azimuth_rad: np.ndarray) -> np.ndarray:
    return ((azimuth_rad + math.pi) % (2.0 * math.pi) - math.pi).astype(np.float32)


def angle_to_bin(
    azimuth_rad: np.ndarray | float,
    elevation_rad: np.ndarray | float,
    grid: AngularGrid,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth = _wrap_azimuth_rad(np.asarray(azimuth_rad, dtype=np.float32))
    elevation = np.asarray(elevation_rad, dtype=np.float32)
    elevation = np.clip(elevation, ELEVATION_RANGE_RAD[0], ELEVATION_RANGE_RAD[1])
    azimuth_idx = np.clip(np.digitize(azimuth, grid.azimuth_edges) - 1, 0, grid.num_az_bins - 1).astype(np.int32)
    elevation_idx = np.clip(np.digitize(elevation, grid.elevation_edges) - 1, 0, grid.num_el_bins - 1).astype(np.int32)
    return azimuth_idx, elevation_idx


def bin_to_angle(
    azimuth_idx: int | np.ndarray,
    elevation_idx: int | np.ndarray,
    grid: AngularGrid,
    output_in_degrees: bool = False,
) -> dict[str, np.ndarray]:
    az_idx = np.clip(np.asarray(azimuth_idx, dtype=np.int32), 0, grid.num_az_bins - 1)
    el_idx = np.clip(np.asarray(elevation_idx, dtype=np.int32), 0, grid.num_el_bins - 1)
    azimuth = grid.azimuth_centers[az_idx]
    elevation = grid.elevation_centers[el_idx]
    if output_in_degrees:
        azimuth = np.degrees(azimuth)
        elevation = np.degrees(elevation)
    return {"azimuth": np.asarray(azimuth), "elevation": np.asarray(elevation)}


def directions_to_angles(unit_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if unit_vectors.ndim != 2 or unit_vectors.shape[1] != 3:
        raise ValueError(f"Expected [N,3] unit vectors, got {unit_vectors.shape}.")
    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]
    azimuth = np.arctan2(x, z).astype(np.float32)
    horizontal_radius = np.sqrt(np.maximum(x * x + z * z, EPSILON)).astype(np.float32)
    elevation = np.arctan2(y, horizontal_radius).astype(np.float32)
    return azimuth, elevation


def azimuth_aggregate(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor [E,A,C], got {tensor.shape}.")
    return np.mean(tensor, axis=0).astype(np.float32)


def _compute_stability(window_feature_maps: np.ndarray) -> np.ndarray:
    if window_feature_maps.shape[0] <= 1:
        energy = window_feature_maps[0, :, :, WINDOW_CHANNEL_NAMES.index("energy")]
        return (energy > EPSILON).astype(np.float32)
    energy_stack = window_feature_maps[:, :, :, WINDOW_CHANNEL_NAMES.index("energy")]
    mean_energy = np.mean(energy_stack, axis=0)
    std_energy = np.std(energy_stack, axis=0)
    stability = mean_energy / (mean_energy + std_energy + EPSILON)
    stability[mean_energy <= EPSILON] = 0.0
    return np.clip(stability, 0.0, 1.0).astype(np.float32)


def _aggregate_primary(window_feature_maps: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return np.max(window_feature_maps, axis=0)
    if mode == "mean":
        return np.mean(window_feature_maps, axis=0)
    raise ValueError(f"Unsupported aggregation mode for primary tensor: {mode}")


def aggregate_window_feature_maps(
    window_feature_maps: np.ndarray,
    grid: AngularGrid,
    aggregation_mode: str = "mean",
    extra_metadata: dict[str, Any] | None = None,
) -> AudioSphere:
    if window_feature_maps.ndim != 4:
        raise ValueError(f"Expected window feature maps [W,E,A,Cw], got {window_feature_maps.shape}.")
    if window_feature_maps.shape[1] != grid.num_el_bins or window_feature_maps.shape[2] != grid.num_az_bins:
        raise ValueError(
            "Window feature map grid mismatch: "
            f"maps={window_feature_maps.shape[1:3]} grid={(grid.num_el_bins, grid.num_az_bins)}"
        )
    if window_feature_maps.shape[3] != len(WINDOW_CHANNEL_NAMES):
        raise ValueError(f"Expected {len(WINDOW_CHANNEL_NAMES)} window channels, got {window_feature_maps.shape[3]}.")
    if aggregation_mode not in {"mean", "max", "both"}:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")

    primary_mode = "mean" if aggregation_mode == "both" else aggregation_mode
    primary_without_stability = _aggregate_primary(window_feature_maps, primary_mode)
    stability = _compute_stability(window_feature_maps)
    tensor = np.concatenate([primary_without_stability, stability[:, :, None]], axis=-1).astype(np.float32)

    tensor_max = None
    if aggregation_mode == "both":
        max_without_stability = _aggregate_primary(window_feature_maps, "max")
        tensor_max = np.concatenate([max_without_stability, stability[:, :, None]], axis=-1).astype(np.float32)

    azimuth_tensor = azimuth_aggregate(tensor)
    azimuth_tensor_max = None if tensor_max is None else azimuth_aggregate(tensor_max)

    metadata = {
        "tensor_name": "A_sphere",
        "tensor_shape": list(tensor.shape),
        "azimuth_tensor_shape": list(azimuth_tensor.shape),
        "channel_names": list(AUDIO_CHANNEL_NAMES),
        "channel_descriptions": CHANNEL_DESCRIPTIONS,
        "aggregation_mode": aggregation_mode,
        "primary_tensor_aggregation": primary_mode,
        "has_max_tensor": tensor_max is not None,
        "coordinate_convention": coordinate_convention_dict(),
        "angular_grid": grid.to_meta_dict(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return AudioSphere(
        tensor=tensor,
        azimuth_tensor=azimuth_tensor,
        tensor_max=tensor_max,
        azimuth_tensor_max=azimuth_tensor_max,
        channel_names=list(AUDIO_CHANNEL_NAMES),
        grid=grid,
        aggregation_mode=aggregation_mode,
        metadata=metadata,
    )

