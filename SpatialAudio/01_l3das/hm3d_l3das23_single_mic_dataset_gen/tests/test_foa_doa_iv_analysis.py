from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import wavfile

from hm3d_l3das23_single_mic.foa_doa_iv_analysis import (
    AnalysisItem,
    AnalysisOptions,
    ChannelMapping,
    _aggregate_frame_vectors,
    _beam_scan_frame_vectors,
    _prepare_item,
    _stft_matrix,
    all_channel_mappings,
    angular_error_deg,
    build_tf_mask,
    local_vector_to_angles,
    run_analysis,
    select_best_channel_mapping,
)
from hm3d_l3das23_single_mic.geometry import world_to_local
from hm3d_l3das23_single_mic.manifest_io import append_manifest_row


def _make_options(dataset_root: Path, config_path: Path, out_dir: Path, mode: str) -> AnalysisOptions:
    return AnalysisOptions(
        dataset_root=dataset_root,
        config_path=config_path,
        mode=mode,
        split="train",
        limit_los=1,
        limit_nlos=0,
        sample_ids=None,
        out_dir=out_dir,
        stft_win=256,
        hop=128,
        nfft=256,
        energy_db_below_peak=20.0,
        diffuseness_max=0.5,
        beam_az_step=15.0,
        beam_el_step=15.0,
        probe_signals=("white", "pink", "chirp"),
        save_rendered_probes=False,
    )


def _write_minimal_config(config_path: Path, dataset_root: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                'version: "0.1"',
                "paths:",
                f'  dataset_root: "{dataset_root}"',
                f'  hm3d_root: "{dataset_root}"',
                '  hm3d_scene_glob: "**/*.basis.glb"',
                f'  hm3d_scene_dataset_config: "{dataset_root / "dummy.scene_dataset_config.json"}"',
                "audio:",
                "  sample_rate: 48000",
                '  channel_layout: "ambisonics"',
                "  channel_count: 4",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _synthetic_foa_waveform(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    sample_rate: int = 48000,
    duration_s: float = 0.5,
) -> np.ndarray:
    num_samples = int(round(duration_s * float(sample_rate)))
    time_axis = np.linspace(0.0, duration_s, num_samples, endpoint=False, dtype=np.float64)
    dry = (
        np.sin(2.0 * np.pi * 440.0 * time_axis)
        + 0.5 * np.sin(2.0 * np.pi * 880.0 * time_axis)
        + 0.25 * np.sin(2.0 * np.pi * 1760.0 * time_axis)
    ).astype(np.float32)
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    elevation_rad = np.deg2rad(float(elevation_deg))
    cos_el = np.cos(elevation_rad)
    direction = np.array(
        [
            -cos_el * np.sin(azimuth_rad),
            cos_el * np.cos(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=np.float32,
    )
    waveform = np.stack(
        [
            dry,
            direction[0] * dry,
            direction[1] * dry,
            direction[2] * dry,
        ],
        axis=0,
    )
    peak = float(np.max(np.abs(waveform)))
    return (0.8 * waveform / peak).astype(np.float32)


def _synthetic_stored_wyzx_waveform(local_rfu: np.ndarray, *, sample_rate: int = 48000) -> np.ndarray:
    num_samples = sample_rate // 2
    time_axis = np.linspace(0.0, 0.5, num_samples, endpoint=False, dtype=np.float64)
    dry = (
        np.sin(2.0 * np.pi * 523.25 * time_axis)
        + 0.4 * np.sin(2.0 * np.pi * 1174.66 * time_axis)
        + 0.2 * np.sin(2.0 * np.pi * 2217.46 * time_axis)
    ).astype(np.float32)
    local = np.asarray(local_rfu, dtype=np.float32)
    local = local / max(float(np.linalg.norm(local)), 1.0e-8)
    y_left = -local[0] * dry
    z_up = local[2] * dry
    x_front = local[1] * dry
    waveform = np.stack([dry, y_left, z_up, x_front], axis=0)
    peak = float(np.max(np.abs(waveform)))
    return (0.8 * waveform / peak).astype(np.float32)


def _write_manifest_row(dataset_root: Path, audio_relpath: str, *, sample_id: str = "sample_a") -> None:
    append_manifest_row(
        dataset_root,
        {
            "sample_id": sample_id,
            "scene_id": "scene_a",
            "split": "train",
            "geometry_los": "gLOS",
            "rendering_status": "success",
            "audio_format": "foa_ambisonics_4ch",
            "audio_channel_layout": "ambisonics",
            "is_in_fov": True,
            "source_mic_relative_position": [0.0, 1.0, 0.0],
            "continuous_azimuth_deg": 0.0,
            "continuous_elevation_deg": 0.0,
            "foa_audio_path": audio_relpath,
            "mic_pose_world": {
                "position_xyz": [0.0, 1.6, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "yaw_rad": 0.0,
                "yaw_deg": 0.0,
            },
            "mic_world_position": [0.0, 1.6, 0.0],
            "mic_world_rotation": [1.0, 0.0, 0.0, 0.0],
            "source_world_position": [0.0, 1.6, -1.0],
        },
    )


class FoaDoaIvAnalysisTests(unittest.TestCase):
    def test_local_vector_to_angles_matches_dataset_convention(self) -> None:
        self.assertEqual(local_vector_to_angles([0.0, 1.0, 0.0]), (0.0, 0.0))
        self.assertEqual(local_vector_to_angles([-1.0, 0.0, 0.0]), (90.0, 0.0))
        self.assertEqual(local_vector_to_angles([1.0, 0.0, 0.0]), (-90.0, 0.0))
        _, elevation_deg = local_vector_to_angles([0.0, 0.0, 1.0])
        self.assertAlmostEqual(elevation_deg, 90.0, places=6)

    def test_origin_identity_final_sanity_axes_match_iv_and_beam(self) -> None:
        listener_world = [0.0, 0.0, 0.0]
        yaw_rad = 0.0
        distance = 1.0
        cases = {
            "front": [0.0, 0.0, -distance],
            "right": [distance, 0.0, 0.0],
            "left": [-distance, 0.0, 0.0],
            "back": [0.0, 0.0, distance],
            "up": [0.0, distance, 0.0],
        }
        expected_audio_xyz = {
            "front": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
            "left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "back": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        wyzx_to_local_rfu = ChannelMapping((0, 2, 1), (-1, 1, 1))

        for label, source_world in cases.items():
            local_rfu = world_to_local(listener_world, yaw_rad, source_world)
            local_unit = local_rfu / np.linalg.norm(local_rfu)
            stored_gains_yzx = np.array(
                [-local_unit[0], local_unit[2], local_unit[1]],
                dtype=np.float64,
            )
            audio_xyz = np.array(
                [stored_gains_yzx[2], stored_gains_yzx[0], stored_gains_yzx[1]],
                dtype=np.float64,
            )
            np.testing.assert_allclose(audio_xyz, expected_audio_xyz[label], atol=1.0e-6)

            waveform = _synthetic_stored_wyzx_waveform(local_unit)
            _, _, stft_matrix = _stft_matrix(waveform, 48000, win=512, hop=256, nfft=512)
            selection = build_tf_mask(
                stft_matrix,
                wyzx_to_local_rfu,
                energy_db_below_peak=80.0,
                diffuseness_max=1.0,
            )
            iv_vectors, iv_valid = _aggregate_frame_vectors(selection["raw_iv"], selection["tf_mask"])
            self.assertTrue(np.any(iv_valid), label)
            iv_mean = np.mean(iv_vectors[iv_valid], axis=0)
            self.assertLess(angular_error_deg(iv_mean, local_unit), 1.0e-6, label)

            beam_vectors, beam_valid = _beam_scan_frame_vectors(
                stft_matrix,
                wyzx_to_local_rfu,
                selection["tf_mask"],
                azimuth_step_deg=5.0,
                elevation_step_deg=5.0,
            )
            self.assertTrue(np.any(beam_valid), label)
            beam_mean = np.mean(beam_vectors[beam_valid], axis=0)
            self.assertLess(angular_error_deg(beam_mean, local_unit), 1.0e-6, label)

    def test_select_best_channel_mapping_recovers_permuted_directional_channels(self) -> None:
        waveform = _synthetic_foa_waveform(azimuth_deg=35.0, elevation_deg=15.0)
        mapping = next(
            candidate
            for candidate in all_channel_mappings()
            if candidate.permutation == (2, 0, 1) and candidate.signs == (-1, 1, 1)
        )
        canonical = waveform[1:4]
        raw_directional = np.empty_like(canonical)
        for canonical_index in range(3):
            raw_index = mapping.permutation[canonical_index]
            raw_directional[raw_index] = canonical[canonical_index] * float(mapping.signs[canonical_index])
        azimuth_rad = np.deg2rad(35.0)
        elevation_rad = np.deg2rad(15.0)
        gt_unit_vector = np.array(
            [
                -np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.sin(elevation_rad),
            ],
            dtype=np.float64,
        )
        raw_waveform = np.concatenate([waveform[0:1], raw_directional], axis=0)

        item = AnalysisItem(
            item_id="sanity__sample_a__white",
            sample_id="sample_a",
            scene_id="scene_a",
            geometry_los="gLOS",
            source_kind="sanity",
            signal_name="white",
            gt_unit_vector=gt_unit_vector,
            gt_azimuth_deg=35.0,
            gt_elevation_deg=15.0,
            waveform=raw_waveform,
            sample_rate=48000,
            direct_window_s=(0.0, 0.2),
            metadata={},
            notes=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = _make_options(Path(tmp_dir), Path(tmp_dir) / "cfg.yaml", Path(tmp_dir) / "out", "sanity")
            prepared = _prepare_item(item, options)
            best = select_best_channel_mapping([prepared], [], options)
            self.assertEqual(best["mapping"].permutation, mapping.permutation)
            self.assertEqual(best["mapping"].signs, mapping.signs)

    def test_build_tf_mask_stays_finite_for_random_waveform(self) -> None:
        rng = np.random.default_rng(1234)
        waveform = rng.normal(0.0, 0.1, size=(4, 4096)).astype(np.float32)
        item = AnalysisItem(
            item_id="existing__sample_a",
            sample_id="sample_a",
            scene_id="scene_a",
            geometry_los="gLOS",
            source_kind="existing",
            signal_name="speech",
            gt_unit_vector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            gt_azimuth_deg=0.0,
            gt_elevation_deg=0.0,
            waveform=waveform,
            sample_rate=48000,
            metadata={},
            notes=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            options = _make_options(Path(tmp_dir), Path(tmp_dir) / "cfg.yaml", Path(tmp_dir) / "out", "existing")
            prepared = _prepare_item(item, options)
            identity_mapping = next(
                candidate
                for candidate in all_channel_mappings()
                if candidate.permutation == (0, 1, 2) and candidate.signs == (1, 1, 1)
            )
            selection = build_tf_mask(
                prepared.stft_matrix,
                identity_mapping,
                energy_db_below_peak=options.energy_db_below_peak,
                diffuseness_max=options.diffuseness_max,
            )
            self.assertTrue(np.isfinite(selection["frame_energy_db"]).all())
            self.assertTrue(np.isfinite(selection["diffuseness"]).all())
            self.assertEqual(selection["tf_mask"].shape, selection["total_energy"].shape)

    def test_run_analysis_existing_mode_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_dir = dataset_root / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            waveform = _synthetic_foa_waveform(azimuth_deg=0.0, elevation_deg=0.0)
            wavfile.write(audio_dir / "sample.wav", 48000, waveform.T.astype(np.float32))
            _write_manifest_row(dataset_root, "audio/sample.wav")

            config_path = root / "config.yaml"
            _write_minimal_config(config_path, dataset_root)
            out_dir = root / "out"
            summary = run_analysis(_make_options(dataset_root, config_path, out_dir, "existing"))

            self.assertTrue((out_dir / "aggregate_summary.json").exists())
            self.assertTrue((out_dir / "aggregate_summary.md").exists())
            self.assertTrue((out_dir / "sample_metrics.csv").exists())
            self.assertEqual(summary["analysis_mode"], "existing")
            self.assertEqual(summary["num_items_analyzed"], 1)

    def test_run_analysis_sanity_mode_writes_outputs_with_mocked_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            dataset_root.mkdir(parents=True, exist_ok=True)
            _write_manifest_row(dataset_root, "audio/sample.wav")

            config_path = root / "config.yaml"
            _write_minimal_config(config_path, dataset_root)
            out_dir = root / "out"
            waveform = _synthetic_foa_waveform(azimuth_deg=0.0, elevation_deg=0.0)
            mocked_items = [
                AnalysisItem(
                    item_id="sanity__sample_a__white",
                    sample_id="sample_a",
                    scene_id="scene_a",
                    geometry_los="gLOS",
                    source_kind="sanity",
                    signal_name="white",
                    gt_unit_vector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    gt_azimuth_deg=0.0,
                    gt_elevation_deg=0.0,
                    waveform=waveform,
                    sample_rate=48000,
                    direct_window_s=(0.0, 0.2),
                    metadata={},
                    notes=[],
                )
            ]

            with mock.patch(
                "hm3d_l3das23_single_mic.foa_doa_iv_analysis.build_sanity_items",
                return_value=(mocked_items, []),
            ):
                summary = run_analysis(_make_options(dataset_root, config_path, out_dir, "sanity"))

            self.assertTrue((out_dir / "aggregate_summary.json").exists())
            self.assertTrue((out_dir / "aggregate_summary.md").exists())
            self.assertTrue((out_dir / "sample_metrics.csv").exists())
            aggregate = json.loads((out_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["analysis_mode"], "sanity")
            self.assertEqual(aggregate["mapping_selection_source"], "sanity")
            self.assertEqual(summary["num_items_analyzed"], 1)


if __name__ == "__main__":
    unittest.main()
