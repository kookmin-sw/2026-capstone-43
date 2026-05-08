# SpatialAudio

GitHub-ready collection of the selected spatial-audio experiment folders. Original project indices were compacted and reordered so reusable dataset/pipeline/model code appears before experiment-result bundles.

## Folder Index

| New | Original | Type | Description |
| --- | --- | --- | --- |
| `01_l3das` | `02_l3das` | code/data | HM3D L3DAS23 single-mic dataset generator |
| `02_pipeline` | `03_pipeline` | code/pipeline | Spatial audio visualization pipeline |
| `03_spatialast_FOA` | `11_spatialast_FOA` | model code | SpatialAST FOA baseline |
| `04_spatialast_FOA_conv` | `12_spatialast_FOA_conv` | model code | SpatialAST FOA conv-stem ablation |
| `05_spatialast_FOA_conv64x2` | `13_spatialast_FOA_conv64x2` | model code | SpatialAST FOA deeper conv-stem ablation |
| `06_spatialast_FOA_frontreg` | `14_spatialast_FOA_frontreg` | model code | SpatialAST FOA front-cone regression ablation |
| `07_spatialast_FOA_front9_and_reg` | `15_spatialast_FOA_front9_and_reg` | model code | SpatialAST FOA front9/regression and AmbiX experiments |
| `08_multi_accdoa_head` | `17_multi_accdoa_head` | model code | Multi-ACCDOA source-slot head experiment |
| `09_spherical_vision` | `08_spherical_vision` | model/representation code | Spherical vision representation pipeline |
| `10_spherical_audio` | `09_spherical_audio` | model/representation code | Spherical audio representation pipeline |
| `11_decode_overfit_test` | `04_decode_overfit_test` | experiment results | Decode overfit result analysis |
| `12_overfit_baseline` | `05_overfit_baseline` | experiment results | Overfit baseline result bundle |
| `13_validation` | `06_validation` | experiment results | Validation metrics only |
| `14_curriculum_baseline` | `07_curriculum_baseline` | experiment results + analysis code | Curriculum vs end-to-end baseline analysis |

## Ordering Rule

- `01`-`10` contain dataset, pipeline, model, or representation code.
- `11`-`14` contain experiment-result bundles; their raw result files were removed and summarized in README files.

## Cleanup Policy

- Removed generated result directories or emptied them when the folder itself was result-only.
- Removed result files such as `.json`, `.jsonl`, `.csv`, `.png`, `.zip`, `.log`, `.npy`, `.npz`, `.pt`, `.pth`, and checkpoint artifacts unless they were source/config files needed by code.
- Removed Python bytecode caches and cache directories.
- Preserved code files and summarized removed results in each folder README.
