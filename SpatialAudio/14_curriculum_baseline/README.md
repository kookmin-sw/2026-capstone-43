# 14_curriculum_baseline

Curriculum-vs-end-to-end baseline analysis copied from `07_curriculum_baseline`. Raw decode/metric outputs were removed; analysis scripts and an empty result folder tree remain.

## Code Analysis
- Kept `analysis_utils.py`, `analyze_curriculum_runs.py`, and `build_curriculum_analysis.py` for reproducing the analysis from raw decode outputs.
- `analysis_utils.py`: classes `ParsedRecord`; functions `_build_label_aliases`, `load_json`, `load_jsonl`, `extract_epoch_from_name`, `iter_named_values`, `extract_candidate_value`
- `analyze_curriculum_runs.py`: functions `parse_args`, `parse_run_metadata`, `discover_runs`, `inspect_run_schema`, `sanitize_filename`, `analyze_run_epoch`
- `build_curriculum_analysis.py`: functions `load_json`, `iter_jsonl`, `markdown_table`, `pct`, `infer_subset`, `scan_metrics`

## Result Summary
| Run | Strategy | Data Size | Best Epoch | Best Accuracy | Final Epoch | Final Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `01_curriculum_1000` | curriculum | 1000 | 15 | 17.50% | 20 | 16.25% |
| `02_endtoend_1000` | endtoend | 1000 | 19 | 19.25% | 20 | 19.25% |
| `03_curriculum_2400` | curriculum | 2400 | 20 | 22.50% | 20 | 22.50% |
| `04_endtoend_2400` | endtoend | 2400 | 20 | 26.00% | 20 | 26.00% |

### Main Analysis Takeaway
- Original analysis concluded that curriculum was not consistently better than end-to-end. At the 2400-sample scale, end-to-end reached the strongest final result: 26.00% accuracy / 25.75% macro F1 at epoch 20.
- The 1000-sample curriculum run showed front/boundary bias but weak side/back recall, so its gains were not stable across classes.

## Cleanup Notes
- Original source contained 134 files across 10 subdirectories (15.1 MB). Generated result files were removed or reduced to empty folder structure in this GitHub copy.
- Run folders and `analysis_outputs/` were kept as empty result structure; raw decode JSONL, metrics JSON, CSV, images, Markdown reports, zip archives, and caches were removed.
