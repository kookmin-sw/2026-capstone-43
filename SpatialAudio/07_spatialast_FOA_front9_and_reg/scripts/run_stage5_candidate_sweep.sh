#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_SUBSET_SCRIPT="${PROJECT_ROOT}/tools/build_stage5_subset.py"
SELECT_SCRIPT="${PROJECT_ROOT}/tools/select_stage5_candidates.py"
COMPARE_SCRIPT="${PROJECT_ROOT}/tools/compare_stage5_runs.py"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs_stage5"
SWEEP_LOG="${OUTPUT_ROOT}/stage5_candidate_sweep.log"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

python "${BUILD_SUBSET_SCRIPT}"

if [ "$#" -gt 0 ]; then
  RUN_NAMES=("$@")
else
  mapfile -t RUN_NAMES < <(python "${SELECT_SCRIPT}" --run-names)
fi

TOTAL_RUNS="${#RUN_NAMES[@]}"
if [ "${TOTAL_RUNS}" -eq 0 ]; then
  echo "[stage5] no candidate runs selected"
  exit 1
fi

echo "[stage5] selected candidate shortlist"
python "${SELECT_SCRIPT}"
echo

for idx in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$idx]}"
  script_path="${SCRIPT_DIR}/train_stage5_2400_${run_name}.sh"
  metrics_path="${OUTPUT_ROOT}/${run_name}/metrics_summary.json"
  step=$((idx + 1))

  echo "[stage5] (${step}/${TOTAL_RUNS}) ${run_name}"
  if [ ! -f "${script_path}" ]; then
    echo "[stage5][error] missing script: ${script_path}"
    exit 1
  fi

  if [ -f "${metrics_path}" ]; then
    echo "[stage5][skip] ${run_name} already completed"
  else
    bash "${script_path}"
  fi

  python "${COMPARE_SCRIPT}"
  echo
done

echo "[stage5] candidate sweep finished"
