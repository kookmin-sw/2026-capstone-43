#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# You can pass a custom input image as the first argument.
INPUT_IMAGE="${1:-${REPO_ROOT}/01_dataset/99_archive/hm3d_fov_glos_pool_2500_diverse/scenes/00683-KCvzhHEhdwB/samples/00683-KCvzhHEhdwB__mic0001__src0380__dry_598ffa13/image/rgb_front.png}"

python "${PROJECT_ROOT}/run_mvp.py" \
  --input "${INPUT_IMAGE}" \
  --output_dir "${PROJECT_ROOT}/outputs/demo" \
  --hfov_deg 69 \
  --num_az_bins 24 \
  --num_el_bins 8 \
  --max_points 120000 \
  --include_extra_depth_channels \
  --pooling_mode mean
