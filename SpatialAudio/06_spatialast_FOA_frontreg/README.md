## 06_spatialast_FOA_frontreg

Front-cone azimuth regression ablation package cloned from `05_spatialast_FOA_conv64x2`.

This project keeps the Stage-13 FOA-native conv stem experiments intact while changing only the
azimuth supervision path so we can test whether the current front-cone dataset is better matched
by continuous regression than by 360-way classification.

What stays the same:
- FOA-native feature extractor
- conv stem variant support from Stage-13
- adapter, patch embedding, transformer, elevation head, vector head
- slow adaptation recipe and best-checkpoint workflow

What changes in Stage-14:
- azimuth head now supports two modes
- `full360_classification`: previous 360-way CE path
- `front_regression`: `45 * tanh(linear(doa_token))` for signed front-cone prediction in `[-45, 45]`

Supported FOA-native stem variants:
- `baseline`: `7 -> 16 -> 1`
- `conv32_out4`: `7 -> 32 -> 4`
- `conv32_out8`: `7 -> 32 -> 8`
- `conv64_out8`: `7 -> 64 -> 8`
- `conv64_out16`: `7 -> 64 -> 16`
- `conv32_32_out8`: `7 -> 32 -> 32 -> 8`
- `conv64_64_out8`: `7 -> 64 -> 64 -> 8`
- `conv64_64_out16`: `7 -> 64 -> 64 -> 16`

Front-cone azimuth support in the current dataset:
- raw labels: `320, 330, 340, 350, 0, 10, 20, 30, 40`
- signed front mapping: `-40, -30, -20, -10, 0, 10, 20, 30, 40`

Key files:
- `heads.py`: classification and front-regression azimuth heads
- `losses.py`: azimuth target mapping + classification/regression loss routing
- `train.py`: metrics, CLI, and Stage-14 logging
- `tools/build_stage14_subset.py`: balanced Stage-14 subset builder
- `tools/compare_stage14_runs.py`: Stage-14 comparison report generator

Quick checks:

```bash
conda run -n spatial-ast python tools/forward_test.py
conda run -n spatial-ast python tools/train_smoke_test.py
conda run -n spatial-ast python tools/check_no_timm_import.py
```

Stage-14 entrypoints:

```bash
cd /home/yu/Project_git/SpatialAudio/06_spatialast_FOA_frontreg
python tools/build_stage14_subset.py

bash scripts/train_stage14_subset_foa_baseline_cls_slow.sh
bash scripts/train_stage14_subset_foa_baseline_reg_slow.sh
python tools/compare_stage14_runs.py

bash scripts/train_stage14_subset_foa_bestconv_reg_slow.sh
python tools/compare_stage14_runs.py

bash scripts/train_stage14_full_foa_baseline_cls_slow.sh
bash scripts/train_stage14_full_foa_baseline_reg_slow.sh
bash scripts/train_stage14_full_foa_bestconv_reg_slow.sh
python tools/compare_stage14_runs.py
```

## GitHub Cleanup Notes

This GitHub-ready copy was renumbered from `14_spatialast_FOA_frontreg` to `06_spatialast_FOA_frontreg`.
Generated experiment artifacts, logs, Python caches, and model weights were removed from this copy. The useful result metadata is summarized below so the repository stays lightweight.

### Removed Artifacts
- `__pycache__/`: 12 files, 137.5 KB
- `tools/__pycache__/`: 18 files, 107.6 KB
- `utils/__pycache__/`: 6 files, 48.0 KB

### Result Summary
- No experiment result JSON or model weight files were present in the selected original folder; only generated Python caches were excluded.
