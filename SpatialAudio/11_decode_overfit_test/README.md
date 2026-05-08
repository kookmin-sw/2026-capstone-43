# 11_decode_overfit_test

Decode overfit experiment results copied from `04_decode_overfit_test`. Raw result files were removed; folder structure and analysis code are retained.

## Code Analysis
- Kept `analysis_20260416_182603/analyze_decode_results.py` as the reproducible analysis script.
- `analysis_20260416_182603/analyze_decode_results.py`: functions `safe_float`, `circular_diff_deg`, `format_pct`, `extract_epoch`, `extract_base_sample_id`, `normalize_sample_uid`

## Result Summary
| Experiment | Best Epoch | Best Accuracy | Valid Epochs | Weakest Classes |
| --- | ---: | ---: | --- | --- |
| `01_decoded_3way` | 14 | 93.33% | 11..19 (9) | front-left 86.00%, front-right 96.00%, front 98.00% |
| `02_decode_3way_av` | 35 | 100.00% | 6..35 (30) | front-left 100.00%, front 100.00%, front-right 100.00% |
| `03_decode_8way` | 19 | 99.75% | 15..20 (6) | front 99.00%, back 99.00%, front-left 100.00% |
| `04_decode_8way_av` | 24 | 92.50% | 14..25 (12) | left 82.00%, back-right 86.00%, right 91.00% |
| `05_decode_gnlos_3way` | 12 | 99.67% | 1..12 (12) | front-right 99.00%, front-left 100.00%, front 100.00% |
| `06_decode_gnlos_5way` | 8 | 82.67% | 1..12 (11) | back-right 66.67%, back 78.33%, back-left 85.00% |
| `07_decode_gnlos_8way` | 12 | 100.00% | 1..14 (14) | front-left 100.00%, front 100.00%, front-right 100.00% |
| `08_decode_gnlos_8way_av` | 18 | 85.83% | 10..19 (10) | left 73.33%, back 76.67%, front-left 86.67% |
| `09_decode_glos_gnlos_8way_av` | 35 | 100.00% | 20..35 (16) | front-left 100.00%, front 100.00%, front-right 100.00% |

### Pairwise Findings
- `01_decoded_3way vs 02_decode_3way_av`: audio=93.33%, AV=100.00%, delta=+6.67pp, shared samples=150.
- `03_decode_8way vs 04_decode_8way_av`: audio=99.75%, AV=92.50%, delta=-7.25pp, shared samples=800.
- `07_decode_gnlos_8way vs 08_decode_gnlos_8way_av`: audio=100.00%, AV=85.83%, delta=-14.17pp, shared samples=480.

### gNLOS Granularity
- `05_decode_gnlos_3way`: classes=3, best_epoch=12, accuracy=99.67%, samples=300.
- `06_decode_gnlos_5way`: classes=5, best_epoch=8, accuracy=82.67%, samples=300.
- `07_decode_gnlos_8way`: classes=8, best_epoch=12, accuracy=100.00%, samples=480.

### Mixed Condition 09
- Best epoch=35, total samples=960, missing subset labels=0.
- FOV+LOS: N=104, accuracy=100.00%.
- OOF+LOS: N=376, accuracy=100.00%.
- FOV+NLOS: N=180, accuracy=100.00%.
- OOF+NLOS: N=300, accuracy=100.00%.

## Cleanup Notes
- Original source contained 178 files across 12 subdirectories (32.2 MB). Generated result files were removed or reduced to empty folder structure in this GitHub copy.
- Result-only epoch folders were left as empty directories; `.json`, `.jsonl`, `.csv`, `.zip`, and generated report files were removed.
