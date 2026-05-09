#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPARE_SCRIPT="${PROJECT_ROOT}/tools/compare_stage4_runs.py"
HISTORY_SCRIPT="${PROJECT_ROOT}/tools/report_stage4_history.py"

run_or_skip() {
  local run_name="$1"
  local script_path="$2"
  local metrics_path="${PROJECT_ROOT}/outputs_stage4/${run_name}/metrics_summary.json"

  if [ -f "${metrics_path}" ]; then
    echo "[skip] ${run_name} already completed"
  else
    echo "[run] ${run_name}"
    bash "${script_path}"
  fi

  python "${COMPARE_SCRIPT}"
  python "${HISTORY_SCRIPT}"
}

run_or_skip "foa_last4_longer" "${SCRIPT_DIR}/train_stage4_2400_foa_last4_longer.sh"
run_or_skip "foa_last4_longer_seed2024" "${SCRIPT_DIR}/train_stage4_2400_foa_last4_longer_seed2024.sh"
run_or_skip "foa_last4_longer_seed3407" "${SCRIPT_DIR}/train_stage4_2400_foa_last4_longer_seed3407.sh"
run_or_skip "foa_last4_longer_cosine" "${SCRIPT_DIR}/train_stage4_2400_foa_last4_longer_cosine.sh"
run_or_skip "foa_last4_longer_lower_head_lr" "${SCRIPT_DIR}/train_stage4_2400_foa_last4_longer_lower_head_lr.sh"
