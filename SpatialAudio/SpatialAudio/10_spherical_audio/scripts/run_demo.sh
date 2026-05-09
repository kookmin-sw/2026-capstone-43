#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python run_audio_mvp.py \
  --input ../99_archive/Sci-Phi/spatial_branch/test_data/convolved_foa.wav \
  --output_dir outputs/demo \
  --channel_order WXYZ \
  --num_az_bins 24 \
  --num_el_bins 8 \
  --window_sec 2.0 \
  --hop_sec 1.0 \
  --aggregation both \
  --pooling_mode both

