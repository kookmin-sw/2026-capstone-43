## 07_spatialast_FOA_front9_and_reg

Front-cone azimuth supervision and patch-interface ablation package cloned from `06_spatialast_FOA_frontreg`.

This project keeps the widened FOA stem, adapter, patch embedding, transformer, elevation head, and vector head
from Stage-14, while changing only the azimuth supervision and patch embedding training policy.

What stays the same:
- FOA-native feature extractor
- conv stem variant support from Stage-14
- adapter, patch embedding, transformer, elevation head, vector head
- slow adaptation recipe and best-checkpoint workflow

What changes in Stage-15:
- azimuth supervision supports `front9_classification`
- azimuth supervision supports `front_regression`
- front9 label mapping is built from the manifests actually used by the run
- patch embedding is pretrained-initialized, shape-adapted when needed, and trainable by default

Supported azimuth modes:
- `front9_classification`: predict the manifest-derived front-cone label space
- `front_regression`: predict signed front-cone degrees in `[-45, 45]` with `45 * tanh(...)`
- `full360_sincos_regression`: predict full-azimuth `[sin(theta), cos(theta)]` and decode to raw degrees

Supported FOA-native stem variants:
- `baseline`: `7 -> 16 -> 1`
- `conv32_out4`: `7 -> 32 -> 4`
- `conv32_out8`: `7 -> 32 -> 8`
- `conv64_out8`: `7 -> 64 -> 8`
- `conv64_out16`: `7 -> 64 -> 16`
- `conv32_32_out8`: `7 -> 32 -> 32 -> 8`
- `conv64_64_out8`: `7 -> 64 -> 64 -> 8`
- `conv64_64_out16`: `7 -> 64 -> 64 -> 16`

Key files:
- `heads.py`: front9 classification and front regression azimuth heads
- `losses.py`: manifest-based front9 mapping and loss routing
- `train.py`: metrics, patch embed optimizer grouping, and Stage-15 logging
- `tools/build_stage15_subset.py`: balanced Stage-15 subset builder
- `tools/compare_stage15_runs.py`: Stage-15 comparison report generator

Stage-15 entrypoints:

```bash
cd /home/yu/Project_git/SpatialAudio/07_spatialast_FOA_front9_and_reg
python tools/build_stage15_subset.py

bash scripts/train_stage15_subset_foa_baseline_front9_patchunfreeze.sh
bash scripts/train_stage15_subset_foa_baseline_reg_patchunfreeze.sh
python tools/compare_stage15_runs.py

bash scripts/train_stage15_subset_foa_bestconv_reg_patchunfreeze.sh
python tools/compare_stage15_runs.py

bash scripts/train_stage15_full_foa_baseline_front9_patchunfreeze.sh
bash scripts/train_stage15_full_foa_baseline_reg_patchunfreeze.sh
bash scripts/train_stage15_full_foa_bestconv_reg_patchunfreeze.sh
python tools/compare_stage15_runs.py
```

Full360 staged training:

```bash
cd /home/yu/Project_git/SpatialAudio/07_spatialast_FOA_front9_and_reg

# Stage A: azimuth-only pretraining.
# Stage B: azimuth-protected multitask warmup.
# Stage C: balanced multitask finetuning.
bash scripts/train_stage15_hm3d_losnlos_full360_staged_sincos.sh
```

Useful overrides:

```bash
RUN_PREFIX=my_full360_staged \
BATCH_SIZE=8 \
LIMIT_TRAIN_SAMPLES=0 \
LIMIT_VAL_SAMPLES=0 \
STAGE_A_EPOCHS=30 \
STAGE_B_EPOCHS=15 \
STAGE_C_EPOCHS=15 \
bash scripts/train_stage15_hm3d_losnlos_full360_staged_sincos.sh
```

Quick smoke test:

```bash
RUN_PREFIX=smoke_staged \
LIMIT_TRAIN_SAMPLES=8 \
LIMIT_VAL_SAMPLES=8 \
MAX_TRAIN_STEPS=1 \
MAX_VAL_STEPS=1 \
STAGE_A_EPOCHS=1 \
STAGE_B_EPOCHS=1 \
STAGE_C_EPOCHS=1 \
bash scripts/train_stage15_hm3d_losnlos_full360_staged_sincos.sh
```

## GitHub Cleanup Notes

This GitHub-ready copy was renumbered from `15_spatialast_FOA_front9_and_reg` to `07_spatialast_FOA_front9_and_reg`.
Generated experiment artifacts, logs, Python caches, and model weights were removed from this copy. The useful result metadata is summarized below so the repository stays lightweight.

### Removed Artifacts
- `__pycache__/`: 13 files, 179.8 KB
- `logs_stage15/`: 31 files, 394.8 MB
- `manifests_stage17/`: 6 files, 146.4 MB
- `outputs_stage16/`: 2 files, 381.5 MB
- `outputs_stage17/`: 3 files, 412.6 MB
- `tools/__pycache__/`: 24 files, 179.7 KB
- `utils/__pycache__/`: 6 files, 48.0 KB
- Weight/checkpoint files removed:
  - `finetuned_transformer_only.pth`: 326.8 MB

### Manifest Summaries
- `hm3d_losnlos_100k_balanced_front9`: train=14047 samples/124 scenes; val=3902 samples/10 scenes; test=3878 samples/11 scenes; front_sample_count=21827; front_scene_count=145
- `hm3d_losnlos_100k_balanced_full360`: summary metadata captured before file removal
- `hm3d_losnlos_100k_balanced_full360_glos`: train=55415; val=16594
- `hm3d_losnlos_100k_balanced_local_full360`: train=55415 samples/122 scenes; val=16594 samples/11 scenes; test=16256 samples/12 scenes
- `hm3d_losnlos_100k_balanced_local_full360_glos`: train=38210 samples/121 scenes; val=14098 samples/11 scenes; test=13402 samples/12 scenes
- `hm3d_losnlos_100k_balanced_world_full360`: train=55415; val=16594; test=16256
- `hm3d_losnlos_100k_balanced_world_full360_glos`: train=38210; val=14098; test=13402
- `hm3d_audio_only_ambix_full360_70k`: train=57639 samples/88 scenes; val=12361 samples/8 scenes
- `hm3d_audio_only_ambix_full360_70k_glos`: train=39867 samples/82 scenes; val=10133 samples/8 scenes

### Training Result Summary
- Stage-16 AmbiX local full360 gLOS azimuth-only: 2 logged epochs; best epoch=1, val_loss=0.853, val_azimuth_mae=88.132 deg, train_steps=9553, val_steps=3525.
- Stage-17 audio-only AmbiX 70k azimuth-only: 12 logged epochs; best epoch=10, val_loss=0.062, val_azimuth_mae=11.483 deg, train_steps=14410, val_steps=3091.
  Last epoch=11, val_loss=0.064, val_azimuth_mae=11.702 deg.
