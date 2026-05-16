#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yu/Project_git/02_l3das/hm3d_l3das23_single_mic_dataset_gen"
SRC_ROOT="$PROJECT_ROOT/src"
LOG_DIR="$PROJECT_ROOT/logs"
MERGED_ROOT="/home/yu/Project_git/01_dataset/hm3d/hm3d-train-semantic-habitat-v0.2"
CONFIG_SHARD0="$PROJECT_ROOT/configs/hm3d_losnlos_100k_balanced_shard0.yaml"
CONFIG_SHARD1="$PROJECT_ROOT/configs/hm3d_losnlos_100k_balanced_shard1.yaml"
CONDA_BIN="/home/yu/miniconda3/bin/conda"
LOG_SHARD0="$LOG_DIR/hm3d_100k_shard0.log"
LOG_SHARD1="$LOG_DIR/hm3d_100k_shard1.log"

mkdir -p "$LOG_DIR"

if [ ! -f "$MERGED_ROOT/hm3d_annotated_train_basis.scene_dataset_config.json" ]; then
  "$CONDA_BIN" run --no-capture-output -n habitat python "$PROJECT_ROOT/scripts/prepare_hm3d_semantic_habitat_root.py" \
    --habitat-root /home/yu/Project_git/01_dataset/hm3d/hm3d-train-habitat-v0.2 \
    --semantic-annots-root /home/yu/Project_git/01_dataset/hm3d/hm3d-train-semantic-annots-v0.2 \
    --output-root "$MERGED_ROOT" \
    --source-dataset-config /home/yu/Project_git/01_dataset/hm3d/hm3d-train-semantic-configs-v0.2/hm3d_annotated_train_basis.scene_dataset_config.json
fi

run_shard() {
  local label="$1"
  local config_path="$2"
  local log_path="$3"

  : > "$log_path"

  (
    set -euo pipefail
    echo "$(date '+%F %T') starting"
    echo "config: $config_path"
    echo "log   : $log_path"
    "$CONDA_BIN" run --no-capture-output -n habitat bash -lc \
      "cd '$PROJECT_ROOT' && export PYTHONUNBUFFERED=1 PYTHONPATH='$SRC_ROOT' && python -m hm3d_l3das23_single_mic.main_generate generate --config '$config_path' --mode full"
  ) 2>&1 | stdbuf -oL tee "$log_path" | stdbuf -oL sed -u "s/^/[$label] /"
}

cleanup() {
  trap - INT TERM
  if [ "${pid0:-}" != "" ]; then
    kill "$pid0" 2>/dev/null || true
  fi
  if [ "${pid1:-}" != "" ]; then
    kill "$pid1" 2>/dev/null || true
  fi
  wait || true
}

trap 'cleanup; exit 130' INT TERM

echo "project root: $PROJECT_ROOT"
echo "log dir     : $LOG_DIR"
echo "shard0 cfg  : $CONFIG_SHARD0"
echo "shard1 cfg  : $CONFIG_SHARD1"
echo "watch logs in another terminal:"
echo "tail -f $LOG_SHARD0 $LOG_SHARD1"
echo "watch progress in another terminal:"
echo "$CONDA_BIN run --no-capture-output -n habitat python $PROJECT_ROOT/scripts/count_100k_progress.py --watch 10 --check-files --show-processes"

run_shard "shard0" "$CONFIG_SHARD0" "$LOG_SHARD0" &
pid0=$!
run_shard "shard1" "$CONFIG_SHARD1" "$LOG_SHARD1" &
pid1=$!

echo "runner pid : $$"
echo "shard0 pid : $pid0"
echo "shard1 pid : $pid1"

set +e
wait "$pid0"
status0=$?
wait "$pid1"
status1=$?
set -e

echo "shard0 exit code: $status0"
echo "shard1 exit code: $status1"

if [ "$status0" -ne 0 ] || [ "$status1" -ne 0 ]; then
  exit 1
fi
