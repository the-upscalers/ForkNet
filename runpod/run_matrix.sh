#!/usr/bin/env bash
# Run a list of training jobs, one per GPU at a time; jobs are resumable (--resume auto).
#   bash runpod/run_matrix.sh runpod/matrix_short.txt 0,1,2,3
# Each non-empty, non-# line of the matrix file: <config> [KEY=VALUE ...] (overrides for scripts/train.py --set).
set -uo pipefail
MATRIX=${1:?matrix file}; GPUS=${2:-0}
IFS=',' read -ra GPU_LIST <<< "$GPUS"
mkdir -p logs
mapfile -t JOBS < <(grep -v '^\s*#' "$MATRIX" | grep -v '^\s*$')
worker() {  # worker <gpu> <slot>
  local gpu=$1 slot=$2 i
  for ((i = slot; i < ${#JOBS[@]}; i += ${#GPU_LIST[@]})); do
    read -ra parts <<< "${JOBS[$i]}"
    cfg=${parts[0]}; overrides=("${parts[@]:1}")
    tag=$(basename "$cfg" .yaml)$(printf '_%s' "${overrides[@]}" | tr '=/.' '---' | cut -c1-60)
    echo "[gpu $gpu] $(date +%T) start $cfg ${overrides[*]}"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py --config "$cfg" --resume auto --set "${overrides[@]}" > "logs/$tag.log" 2>&1
    echo "[gpu $gpu] $(date +%T) done  $cfg (exit $?)"
  done
}
for s in "${!GPU_LIST[@]}"; do worker "${GPU_LIST[$s]}" "$s" & done
wait
