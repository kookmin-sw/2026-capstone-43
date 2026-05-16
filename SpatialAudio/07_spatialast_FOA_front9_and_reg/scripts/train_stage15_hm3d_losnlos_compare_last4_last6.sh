#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEM_VARIANT="${1:-conv64_64_out8}"
if [ "$#" -gt 0 ]; then
  shift
fi

RUN_BASE="${RUN_BASE:-hm3d_losnlos_front9_${STEM_VARIANT}_reg}"
LAST4_RUN_NAME="${LAST4_RUN_NAME:-${RUN_BASE}_last4_patchunfreeze}"
LAST6_RUN_NAME="${LAST6_RUN_NAME:-${RUN_BASE}_last6_patchunfreeze}"

echo "[stage15-hm3d] starting last4 run: ${LAST4_RUN_NAME}"
RUN_NAME="${LAST4_RUN_NAME}" \
UNFREEZE_LAST_N_BLOCKS=4 \
BATCH_SIZE="${BATCH_SIZE_LAST4:-16}" \
"${SCRIPT_DIR}/train_stage15_hm3d_losnlos_bestconv_reg_patchunfreeze.sh" \
  "${STEM_VARIANT}" \
  "$@"

echo "[stage15-hm3d] starting last6 run: ${LAST6_RUN_NAME}"
RUN_NAME="${LAST6_RUN_NAME}" \
UNFREEZE_LAST_N_BLOCKS=6 \
BATCH_SIZE="${BATCH_SIZE_LAST6:-8}" \
"${SCRIPT_DIR}/train_stage15_hm3d_losnlos_bestconv_reg_patchunfreeze.sh" \
  "${STEM_VARIANT}" \
  "$@"
