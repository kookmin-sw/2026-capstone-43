from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from hm3d_l3das23_single_mic.config import DatasetGenerationConfig, SourceSamplingConfig
from hm3d_l3das23_single_mic.export_audio_manifest import azimuth_to_eight_way_label, build_audio_jsonl
from hm3d_l3das23_single_mic.fov_labeler import CameraModel, compute_in_fov
from hm3d_l3das23_single_mic.geometry import local_to_world, local_xyz_from_cylindrical, spherical_from_local_xyz
from hm3d_l3das23_single_mic.los_labeler import RayHit, classify_geometry_los_from_hits
from hm3d_l3das23_single_mic.main_generate import _filter_scenes
from hm3d_l3das23_single_mic.manifest_io import append_manifest_row, dataset_manifest_path, iter_dataset_rows
from hm3d_l3das23_single_mic.qc import build_qc_report_from_existing_metadata
from hm3d_l3das23_single_mic.scene_loader import parse_hm3d_semantic_annotation_txt
from hm3d_l3das23_single_mic.schemas import MicPose, SceneInfo
from hm3d_l3das23_single_mic.source_sampler import generate_source_candidates
from hm3d_l3das23_single_mic.spatial_conventions import (
    ambix_unit_vector_xyz,
    direction_8way_from_azimuth,
    local_angles_from_relative_xyz,
)


class ManifestPipelineTests(unittest.TestCase):
    def test_axis_alignment_front_source_is_zero_azimuth_and_in_fov(self) -> None:
        local_xyz = local_xyz_from_cylindrical(rho=2.0, theta_rad=0.0, z=0.0)
        distance, azimuth_deg, elevation_deg = spherical_from_local_xyz(local_xyz)
        world_xyz = local_to_world([0.0, 0.0, 0.0], 0.0, local_xyz)
        camera_model = CameraModel.from_hfov(512, 512, 90.0)
        projection = compute_in_fov(camera_model, [0.0, 0.0, 0.0], 0.0, world_xyz.tolist())

        self.assertEqual(distance, 2.0)
        self.assertEqual(azimuth_deg, 0.0)
        self.assertEqual(elevation_deg, 0.0)
        self.assertTrue(projection.in_fov)
        self.assertIsNotNone(projection.pixel_xy)
        assert projection.pixel_xy is not None
        self.assertLess(abs(projection.pixel_xy[0] - camera_model.cx), 1.0)

    def test_ambix_azimuth_positive_is_listener_left(self) -> None:
        self.assertEqual(azimuth_to_eight_way_label(0.0), "front")
        self.assertEqual(azimuth_to_eight_way_label(45.0), "front-left")
        self.assertEqual(azimuth_to_eight_way_label(90.0), "left")
        self.assertEqual(azimuth_to_eight_way_label(-45.0), "front-right")
        self.assertEqual(azimuth_to_eight_way_label(-90.0), "right")

    def test_spatial_convention_maps_local_rfu_to_ambix_axes(self) -> None:
        azimuth, elevation, distance = local_angles_from_relative_xyz([0.0, 2.0, 0.0])
        self.assertEqual((azimuth, elevation, distance), (0.0, 0.0, 2.0))
        self.assertEqual(direction_8way_from_azimuth(azimuth), "front")
        self.assertEqual(ambix_unit_vector_xyz([0.0, 2.0, 0.0]), [1.0, -0.0, 0.0])

        azimuth, _, _ = local_angles_from_relative_xyz([-1.0, 0.0, 0.0])
        self.assertEqual(azimuth, 90.0)
        self.assertEqual(direction_8way_from_azimuth(azimuth), "left")
        self.assertEqual(ambix_unit_vector_xyz([-1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])

        azimuth, _, _ = local_angles_from_relative_xyz([1.0, 0.0, 0.0])
        self.assertEqual(azimuth, -90.0)
        self.assertEqual(direction_8way_from_azimuth(azimuth), "right")
        self.assertEqual(ambix_unit_vector_xyz([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0])

    def test_l3das_arc_length_sampling_adds_outer_ring_points(self) -> None:
        mic_pose = MicPose(
            mic_index=0,
            floor_point_world=[0.0, 0.0, 0.0],
            position_world=[0.0, 0.0, 0.0],
            quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            yaw_rad=0.0,
            yaw_deg=0.0,
        )
        source_config = SourceSamplingConfig(
            rho_min_m=1.0,
            rho_max_m=2.0,
            rho_step_m=1.0,
            z_min_m=0.0,
            z_max_m=0.0,
            z_step_m=1.0,
            theta_sampling_mode="arc_length",
            theta_arc_step_m=0.5,
            max_sources_per_mic=10_000,
            shuffle_candidates=False,
        )

        candidates = generate_source_candidates(mic_pose, source_config, seed=0)
        counts = Counter(round(candidate.cylindrical.rho, 6) for candidate in candidates)

        self.assertEqual(counts[1.0], 13)
        self.assertEqual(counts[2.0], 25)
        self.assertAlmostEqual((2.0 * 3.141592653589793 * 2.0) / counts[2.0], 0.5, delta=0.01)

    def test_classify_geometry_los_counts_occluders(self) -> None:
        result = classify_geometry_los_from_hits(
            [
                RayHit(ray_distance=0.50, object_id=10),
                RayHit(ray_distance=0.70, object_id=11),
                RayHit(ray_distance=1.20, object_id=12),
            ],
            source_distance=1.0,
            eps_end=0.05,
            max_dist_margin=0.02,
        )

        self.assertEqual(result.geometry_los, "gNLOS")
        self.assertEqual(result.occluder_count, 2)
        self.assertEqual(result.occluding_object_id, 10)

    def test_parse_hm3d_semantic_annotation_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotation_path = Path(tmp_dir) / "scene.semantic.txt"
            annotation_path.write_text(
                'HM3D Semantic Annotations\n1,ABCDEF,"chair",4\n2,123456,"table",7\n',
                encoding="utf-8",
            )

            categories, room_ids = parse_hm3d_semantic_annotation_txt(annotation_path)

            self.assertEqual(categories, {1: "chair", 2: "table"})
            self.assertEqual(room_ids, {1: 4, 2: 7})

    def test_global_manifest_is_source_of_truth_for_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            append_manifest_row(
                dataset_root,
                {
                    "sample_id": "sample_a",
                    "scene_id": "scene_a",
                    "split": "train",
                    "geometry_los": "gLOS",
                    "is_in_fov": True,
                },
            )
            append_manifest_row(
                dataset_root,
                {
                    "sample_id": "sample_b",
                    "scene_id": "scene_b",
                    "split": "val",
                    "geometry_los": "gNLOS",
                    "is_in_fov": False,
                },
            )

            rows = list(iter_dataset_rows(dataset_root))
            report = build_qc_report_from_existing_metadata(dataset_root)

            self.assertTrue(dataset_manifest_path(dataset_root).exists())
            self.assertEqual([row["sample_id"] for row in rows], ["sample_a", "sample_b"])
            self.assertEqual(report["valid_samples"], 2)
            self.assertEqual(report["gLOS_gNLOS_counts"], {"gLOS": 1, "gNLOS": 1})

    def test_build_audio_jsonl_uses_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            audio_path = dataset_root / "audio" / "sample.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"RIFF")
            append_manifest_row(
                dataset_root,
                {
                    "sample_id": "sample_a",
                    "scene_id": "scene_a",
                    "split": "train",
                    "geometry_los": "gLOS",
                    "is_in_fov": True,
                    "continuous_azimuth_deg": 0.0,
                    "foa_audio_path": "audio/sample.wav",
                },
            )

            output_path = dataset_root / "manifests" / "audio.jsonl"
            summary = build_audio_jsonl(
                dataset_root=dataset_root,
                output_path=output_path,
                split="train",
                require_geometry_los="gLOS",
                require_in_fov=True,
                audio_output_key="audio_mic_wav",
                path_mode="absolute",
                relative_base=None,
            )

            row = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(summary["num_written"], 1)
            self.assertEqual(row["label"], "front")
            self.assertEqual(row["audio_path"], str(audio_path.resolve()))

    def test_scene_sharding_selects_disjoint_scene_halves(self) -> None:
        scenes = [
            SceneInfo(scene_id=f"scene_{idx:02d}", scene_path=Path(f"/tmp/scene_{idx:02d}.basis.glb"))
            for idx in range(8)
        ]
        shard0 = DatasetGenerationConfig()
        shard0.generation.scene_shard_count = 2
        shard0.generation.scene_shard_index = 0
        shard1 = DatasetGenerationConfig()
        shard1.generation.scene_shard_count = 2
        shard1.generation.scene_shard_index = 1

        shard0_ids = [scene.scene_id for scene in _filter_scenes(shard0, scenes)]
        shard1_ids = [scene.scene_id for scene in _filter_scenes(shard1, scenes)]

        self.assertEqual(shard0_ids, ["scene_00", "scene_02", "scene_04", "scene_06"])
        self.assertEqual(shard1_ids, ["scene_01", "scene_03", "scene_05", "scene_07"])
        self.assertEqual(sorted(shard0_ids + shard1_ids), [scene.scene_id for scene in scenes])


if __name__ == "__main__":
    unittest.main()
