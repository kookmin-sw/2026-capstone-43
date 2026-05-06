#!/usr/bin/env bash
set -eo pipefail

CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/miniconda3/envs/ns310}"
if [ ! -x "$CONDA_ENV_DIR/bin/ns-train" ]; then
  echo "[nerfstudio-conda] ns-train not found at $CONDA_ENV_DIR/bin/ns-train"
  exit 127
fi

export NERFSTUDIO_BIN_DIR="$CONDA_ENV_DIR/bin"
export PATH="$NERFSTUDIO_BIN_DIR:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

exec "$(dirname "$0")/train_nerfstudio_splatfacto.sh"
