## 04_spatialast_FOA_conv

Stem-width ablation package cloned from `03_spatialast_FOA`.

This project keeps the same FOA dataset pipeline, SpatialAST transformer, heads, and loss setup,
but makes the FOA-native stem compression ratio configurable to test whether the default
`7 -> 16 -> 1` stem is over-compressing spatial cues before patch embedding.

Supported FOA-native stem variants:
- `baseline`: `7 -> 16 -> 1`
- `conv32_out4`: `7 -> 32 -> 4`
- `conv32_out8`: `7 -> 32 -> 8`
- `conv64_out8`: `7 -> 64 -> 8`
- `conv64_out16`: `7 -> 64 -> 16`

Default FOA input stem is now DCASE2024 SELD-baseline-inspired:
- `WXYZ` 4-channel log-mel
- normalized FOA intensity vectors `IV_x, IV_y, IV_z`
- default stacked input channels: `7`
- optional `diffuseness` / `beam_proxy` remain available behind flags

Included files:
- `backbone.py`: FOA-native stem + SpatialAST transformer backbone
- `heads.py`: class / distance / azimuth / elevation / vector heads
- `model.py`: composition layer joining backbone and heads
- `spatial_ast.py`: backward-compatible wrapper with tuple-style output
- `dataset.py`: minimal direct FOA dataset loader and synthetic dataset
- `losses.py`: multitask loss helpers
- `train.py`: small single-process training entrypoint
- `tools/build_stage12_subset.py`: balanced Stage-12 subset builder
- `tools/compare_stage12_runs.py`: Stage-12 comparison report generator
- `requirements.txt`: minimal pip dependencies without `timm`
- `environment.yml`: minimal conda environment without `timm`
- `utils/stft.py`
- `utils/vision_transformer.py`
- `utils/torch_layers.py`: local replacements for `to_2tuple`, `DropPath`, `trunc_normal_`
- `utils/foa_features.py`
- `tools/forward_test.py`
- `tools/train_smoke_test.py`
- `tools/check_no_timm_import.py`

FOA channel convention:
- raw disk order: `WYZX`
- internal canonical order: `WXYZ`
- reorder rule: `x = x[:, [0, 3, 1, 2], :]` for batched tensors

Quick checks:

```bash
conda run -n spatial-ast python tools/forward_test.py
conda run -n spatial-ast python tools/train_smoke_test.py
conda run -n spatial-ast python tools/check_no_timm_import.py
```

Real-data training example:

```bash
conda run -n spatial-ast python train.py \
  --train_json /path/to/train.json \
  --val_json /path/to/val.json \
  --audio_path_root /path/to/audio_root \
  --num_classes 355 \
  --batch_size 4 \
  --epochs 10 \
  --audio_normalize \
  --class_loss_weight 0.0 \
  --distance_loss_weight 0.0 \
  --azimuth_loss_weight 2.0 \
  --elevation_loss_weight 2.0 \
  --vector_loss_weight 0.5
```

Stage-12 ablation entrypoints:

```bash
cd /home/yu/Project_git/SpatialAudio/04_spatialast_FOA_conv
python tools/build_stage12_subset.py

bash scripts/train_stage12_subset_foa_baseline_slow.sh
bash scripts/train_stage12_subset_foa_conv32_out8_slow.sh
bash scripts/train_stage12_subset_foa_conv64_out8_slow.sh
bash scripts/train_stage12_subset_foa_conv64_out16_slow.sh

python tools/compare_stage12_runs.py
```

## GitHub Cleanup Notes

This GitHub-ready copy was renumbered from `12_spatialast_FOA_conv` to `04_spatialast_FOA_conv`.
Generated experiment artifacts, logs, Python caches, and model weights were removed from this copy. The useful result metadata is summarized below so the repository stays lightweight.

### Removed Artifacts
- `__pycache__/`: 12 files, 131.4 KB
- `tools/__pycache__/`: 14 files, 80.9 KB
- `utils/__pycache__/`: 6 files, 48.0 KB
- Weight/checkpoint files removed:
  - `finetuned_transformer_only.pth`: 326.8 MB

### Model Artifact Summary
- `finetuned_transformer_only.pth` was present in the original folder and removed from this GitHub copy. Recreate or download weights separately before running finetuning scripts that require it.
