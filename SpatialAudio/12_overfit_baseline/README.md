# 12_overfit_baseline

Overfit baseline result bundle copied from `05_overfit_baseline`. Raw result files were removed; empty experiment folder structure and the visualization script are retained.

## Code Analysis
- Kept `analysis_outputs/generate_pairwise_visuals.py`, which rebuilds pairwise dashboards from CSV summaries.
- `analysis_outputs/generate_pairwise_visuals.py`: functions `_load_csv`, `_as_percent`, `_style_axes`, `make_pair_dashboard`, `make_overview_threshold_plot`, `make_overview_decode_plot`

## Result Summary
| Run | Modality | Best Decode Acc | Best Decode Epoch | Best Val Loss Epoch | Best Val Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| `overfit_01_audio_3way_fov_glos_gnlos_300` | audio | 99.67% | 18 | 18 | 0.0022 |
| `overfit_02_av_3way_fov_glos_gnlos_300` | av | 100.00% | 18 | 18 | 0.0037 |
| `overfit_03_audio_8way_glos_800` | audio | 100.00% | 20 | 20 | 0.0139 |
| `overfit_04_av_8way_glos_800` | av | 26.75% | 19 | 19 | 1.4689 |
| `overfit_05_audio_8way_gnlos_800` | audio | 98.62% | 15 | 16 | 0.0194 |
| `overfit_06_av_8way_gnlos_800` | av | 100.00% | 20 | 20 | 0.0021 |
| `overfit_07_audio_8way_mixed_glos_gnlos_800` | audio | 97.00% | 18 | 18 | 0.0092 |
| `overfit_08_av_8way_mixed_glos_gnlos_800` | av | 100.00% | 20 | 15 | 0.0134 |

### Notes
- Most reruns reached near-perfect overfit decode accuracy; the notable failure case is `overfit_04_av_8way_glos_800` with 26.75% best decode accuracy.
- Audio-vs-AV gaps from the original analysis: AV improved small 3-way/gNLOS/mixed tasks, but dropped sharply on the 8-way gLOS run.

## Cleanup Notes
- Original source contained 341 files across 66 subdirectories (14.6 MB). Generated result files were removed or reduced to empty folder structure in this GitHub copy.
- Overfit run directories and nested decode/validation folders were kept as empty structure; raw metrics, decode JSONL, CSV, Markdown reports, and dashboard images were removed.
