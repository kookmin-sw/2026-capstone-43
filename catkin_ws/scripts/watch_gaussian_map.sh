#!/usr/bin/env bash
set -eo pipefail

PLY_PATH="${PLY_PATH:-$PWD/outputs/scene01/gaussians_latest.ply}"

echo "[visualize] watching $PLY_PATH"
echo "[visualize] Open this file in MeshLab, CloudCompare, or Open3D."
echo "[visualize] Refresh/reload the file while online GS is running."

while true; do
  if [ -f "$PLY_PATH" ]; then
    ls -lh "$PLY_PATH"
  else
    echo "[visualize] waiting for $PLY_PATH"
  fi
  sleep 2
done
