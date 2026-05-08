# 01_l3das

## GitHub Cleanup Notes

This GitHub-ready copy was renumbered from `02_l3das` to `01_l3das`.
Generated experiment artifacts, logs, Python caches, and model weights were removed from this copy. The useful result metadata is summarized below so the repository stays lightweight.

### Removed Artifacts
- `hm3d_l3das23_single_mic_dataset_gen/_tmp_topdown_check.png`: 105.1 KB
- `hm3d_l3das23_single_mic_dataset_gen/logs/`: 2 files, 1.2 MB
- `hm3d_l3das23_single_mic_dataset_gen/outputs/`: 123 files, 20.5 MB
- `hm3d_l3das23_single_mic_dataset_gen/scripts/__pycache__/`: 5 files, 33.1 KB
- `hm3d_l3das23_single_mic_dataset_gen/src/hm3d_l3das23_single_mic/__pycache__/`: 94 files, 936.8 KB
- `hm3d_l3das23_single_mic_dataset_gen/tests/__pycache__/`: 5 files, 41.3 KB
- `hm3d_l3das23_single_mic_dataset_gen/tmp/`: 91 files, 9.3 MB

### Result Summary
- FOA/DOA existing-data check: split=train, analyzed=24, selected_mapping=`perm=1,2,3;signs=--+`, source=existing_fallback, LOS median=55.386 deg, NLOS cluster-ratio median=14.715%.
- Controlled WYZX sanity check: expected=`perm=1,3,2;signs=-++`, selected=`perm=3,2,1;signs=-++`, matches_expected=False, rendered_items=17.
- RIR/spatial rendering check: scene=00006-HkseAnWCgqk, rendering=success, rir_generation=success, sample_rate=48000, samples=298985, peak=0.891, rms=0.041.
