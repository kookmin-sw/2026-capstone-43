import csv
import json
import math
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import Dataset


def normalize_audio(audio_data, target_dbfs=-14.0):
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms == 0:
        return audio_data
    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20)
    return audio_data * gain_linear


def labels_to_unit_vectors(azimuth, elevation):
    azimuth = torch.as_tensor(azimuth).float()
    elevation = torch.as_tensor(elevation).float()
    azimuth = torch.where(azimuth > 180, azimuth - 360, azimuth)
    azimuth = azimuth * torch.pi / 180.0
    elevation = (elevation - 90.0) * torch.pi / 180.0
    x_front = torch.cos(elevation) * torch.cos(azimuth)
    y_left = torch.cos(elevation) * torch.sin(azimuth)
    z_up = torch.sin(elevation)
    return torch.stack([x_front, y_left, z_up], dim=-1)


def reorder_raw_foa_wyzx_to_wxyz(waveform):
    # Raw AmbiX/ACN order is WYZX; internally use WXYZ = W, X(front), Y(left), Z(up).
    if waveform.ndim != 2 or waveform.shape[1] != 4:
        raise AssertionError(f"Expected raw FOA waveform [T, 4], got {tuple(waveform.shape)}")
    return waveform[:, [0, 3, 1, 2]]


def validate_manifest_conventions(data):
    expected = {
        "audio_channel_order": "WYZX",
        "foa_raw_channel_order": "WYZX",
        "foa_canonical_axes": "AmbiX_ACN_SN3D_W,Y_left,Z_up,X_front",
        "local_coordinate_frame": "mic_local_right_front_up",
        "azimuth_reference": "mic_local_source_relative",
        "azimuth_convention": "ambix_acn_sn3d_positive_left",
    }
    for index, datum in enumerate(data):
        for key, value in expected.items():
            if key in datum and datum[key] != value:
                sample_id = datum.get("sample_id", index)
                raise ValueError(
                    f"Manifest sample {sample_id} has {key}={datum[key]!r}; expected {value!r} "
                    "for mic-local AmbiX ACN/SN3D WYZX training."
                )


def build_label_index(label_csv):
    if not label_csv:
        return {}
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        for index, row in enumerate(csv_reader):
            index_lookup[row["mid"]] = index
    return index_lookup


class FOAWaveDataset(Dataset):
    def __init__(
            self,
            json_path,
            audio_path_root="",
            num_classes=0,
            label_csv="",
            sample_rate=32000,
            clip_seconds=10,
            normalize=True,
            limit_samples=0,
        ):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        if limit_samples:
            self.data = self.data[:limit_samples]
        validate_manifest_conventions(self.data)

        self.audio_path_root = audio_path_root
        self.num_classes = num_classes
        self.label_index = build_label_index(label_csv)
        self.sample_rate = sample_rate
        self.target_samples = sample_rate * clip_seconds
        self.normalize = normalize
        self._debug_printed = False
        self._label_debug_count = 0
        if self.data:
            first = self.data[0]
            print("[FOA dataset] expected stored format: AmbiX ACN/SN3D WYZX")
            print("[FOA dataset] internal canonical order: WXYZ = W, X(front), Y(left), Z(up)")
            if "azimuth_reference" in first:
                print(f"[FOA dataset] azimuth_reference: {first['azimuth_reference']}")
            if "azimuth_convention" in first:
                print(f"[FOA dataset] azimuth_convention: {first['azimuth_convention']}")

    def __len__(self):
        return len(self.data)

    def _resolve_audio_path(self, datum):
        audio_path = datum.get("audio_path", datum.get("wav", datum.get("path", datum.get("fname", ""))))
        if not audio_path:
            raise KeyError("Each datum must provide audio_path/wav/path/fname.")
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(self.audio_path_root, audio_path)
        return audio_path

    def _class_target(self, datum):
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if self.num_classes <= 0:
            return target

        if "class_index" in datum:
            target[int(datum["class_index"])] = 1.0
            return target

        if "class_indices" in datum:
            for index in datum["class_indices"]:
                target[int(index)] = 1.0
            return target

        labels = datum.get("label", datum.get("labels", []))
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            if label in self.label_index:
                target[self.label_index[label]] = 1.0
        return target

    def _spatial_target(self, datum):
        if all(key in datum for key in ("distance", "azimuth", "elevation")):
            raw_azimuth = float(datum["azimuth"])
            raw_elevation = float(datum["elevation"])
            continuous_azimuth = float(datum.get("azimuth_deg", raw_azimuth)) % 360.0
            distance = int(round(float(datum["distance"])))
            azimuth = int(round(raw_azimuth)) % 360
            elevation = int(round(raw_elevation)) % 180
        elif "sensor_position" in datum and "source_position" in datum:
            sensor_position = np.array([float(x) for x in datum["sensor_position"].split(",")])
            source_position = np.array([float(x) for x in datum["source_position"].split(",")])
            distance = int(round(np.linalg.norm(sensor_position - source_position) * 2))
            dx = source_position[0] - sensor_position[0]
            dy = source_position[1] - sensor_position[1]
            dz = source_position[2] - sensor_position[2]
            raw_azimuth = math.degrees(math.atan2(-dz, dx))
            raw_elevation = math.degrees(math.atan(dy / math.sqrt(dx ** 2 + dz ** 2)))
            continuous_azimuth = raw_azimuth % 360.0
            azimuth = (round(raw_azimuth) + 360) % 360
            elevation = (round(raw_elevation) + 90) % 180
        else:
            raise KeyError("Each datum must provide distance/azimuth/elevation or sensor_position/source_position.")

        if self._label_debug_count < 10:
            print("\n[DATASET DEBUG]")
            print("azimuth (raw):", raw_azimuth)
            print("azimuth (continuous deg):", continuous_azimuth)
            print("elevation (raw):", raw_elevation)
            print("azimuth (target):", azimuth)
            print("elevation (target):", elevation)
            self._label_debug_count += 1

        return {
            "distance": torch.tensor(distance, dtype=torch.long),
            "azimuth": torch.tensor(azimuth, dtype=torch.long),
            "azimuth_deg": torch.tensor(continuous_azimuth, dtype=torch.float32),
            "elevation": torch.tensor(elevation, dtype=torch.long),
        }

    def _load_waveform(self, audio_path):
        waveform, sr = sf.read(audio_path, always_2d=True)
        if waveform.shape[1] != 4:
            raise AssertionError(f"FOA input must have shape [T, 4] in raw WYZX order, got {waveform.shape}")

        if not self._debug_printed:
            print(f"[FOA dataset] raw waveform shape: {waveform.shape}")
            print("[FOA dataset] raw channel order: WYZX")

        waveform = reorder_raw_foa_wyzx_to_wxyz(waveform)
        if sr != self.sample_rate:
            waveform = signal.resample_poly(waveform, self.sample_rate, sr, axis=0)

        waveform = waveform.T
        if self.normalize:
            waveform = normalize_audio(waveform, -14.0)
        waveform = torch.from_numpy(waveform).float()

        padding = self.target_samples - waveform.shape[1]
        if padding > 0:
            waveform = F.pad(waveform, (0, padding), "constant", 0)
        elif padding < 0:
            waveform = waveform[:, :self.target_samples]

        if not self._debug_printed:
            print(f"[FOA dataset] canonical waveform shape: {tuple(waveform.shape)}")
            self._debug_printed = True

        return waveform

    def __getitem__(self, index):
        datum = self.data[index]
        audio_path = self._resolve_audio_path(datum)
        waveform = self._load_waveform(audio_path)
        spatial_target = self._spatial_target(datum)
        class_target = self._class_target(datum)

        return {
            "waveform": waveform,
            "class_target": class_target,
            "distance_target": spatial_target["distance"],
            "azimuth_target": spatial_target["azimuth"],
            "azimuth_target_deg": spatial_target["azimuth_deg"],
            "elevation_target": spatial_target["elevation"],
            "audio_path": audio_path,
        }


class SyntheticFOADataset(Dataset):
    def __init__(self, num_samples, num_classes, sample_rate=32000, clip_seconds=1):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.target_samples = sample_rate * clip_seconds
        self._label_debug_count = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        time = torch.linspace(0.0, 1.0, self.target_samples)
        azimuth = (index * 37) % 360
        elevation = (index * 17) % 180
        distance = index % 21
        class_index = index % max(1, self.num_classes)

        w = 0.2 * torch.sin(2 * torch.pi * (220 + class_index * 5) * time)
        y = 0.15 * torch.sin(2 * torch.pi * (280 + azimuth) * time / 8.0)
        z = 0.1 * torch.sin(2 * torch.pi * (330 + elevation) * time / 8.0)
        x = 0.05 * torch.sin(2 * torch.pi * (400 + distance * 10) * time / 8.0)

        raw_wyzx = torch.stack([w, y, z, x], dim=0)
        waveform = raw_wyzx[[0, 3, 1, 2], :]

        class_target = torch.zeros(self.num_classes, dtype=torch.float32)
        if self.num_classes > 0:
            class_target[class_index] = 1.0

        if self._label_debug_count < 10:
            print("\n[DATASET DEBUG]")
            print("azimuth (raw):", azimuth)
            print("azimuth (continuous deg):", float(azimuth))
            print("elevation (raw):", elevation)
            print("azimuth (target):", azimuth)
            print("elevation (target):", elevation)
            self._label_debug_count += 1

        return {
            "waveform": waveform,
            "class_target": class_target,
            "distance_target": torch.tensor(distance, dtype=torch.long),
            "azimuth_target": torch.tensor(azimuth, dtype=torch.long),
            "azimuth_target_deg": torch.tensor(float(azimuth), dtype=torch.float32),
            "elevation_target": torch.tensor(elevation, dtype=torch.long),
            "audio_path": f"synthetic_{index}.wav",
        }


def foa_collate_fn(batch):
    waveforms = torch.stack([item["waveform"] for item in batch], dim=0)
    class_target = torch.stack([item["class_target"] for item in batch], dim=0)
    distance_target = torch.stack([item["distance_target"] for item in batch], dim=0)
    azimuth_target = torch.stack([item["azimuth_target"] for item in batch], dim=0)
    azimuth_target_deg = torch.stack([item["azimuth_target_deg"] for item in batch], dim=0)
    elevation_target = torch.stack([item["elevation_target"] for item in batch], dim=0)
    audio_paths = [item["audio_path"] for item in batch]

    return {
        "waveforms": waveforms,
        "class_target": class_target,
        "distance_target": distance_target,
        "azimuth_target": azimuth_target,
        "azimuth_target_deg": azimuth_target_deg,
        "elevation_target": elevation_target,
        "audio_paths": audio_paths,
    }
