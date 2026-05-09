#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yu/Project_git/SpatialAudio/01_l3das/hm3d_l3das23_single_mic_dataset_gen"
SRC_ROOT="$PROJECT_ROOT/src"
FULL_CONFIG="$PROJECT_ROOT/configs/hm3d_losnlos_100k_balanced.yaml"
DATASET_ROOT="/home/yu/Project_git/01_dataset/hm3d_losnlos_100k_balanced"

conda run -n habitat bash -lc "cd '$PROJECT_ROOT' && PYTHONPATH='$SRC_ROOT' python -m hm3d_l3das23_single_mic.main_generate build-splits --config '$FULL_CONFIG' --mode full"
conda run -n habitat bash -lc "cd '$PROJECT_ROOT' && PYTHONPATH='$SRC_ROOT' python -m hm3d_l3das23_single_mic.main_generate qc --dataset-root '$DATASET_ROOT'"
