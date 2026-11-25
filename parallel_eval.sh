#!/bin/bash

TASK_NAME=${1}
TASK_CONFIG=${2}
POLICY_CONFIG=${3}
GPU=${4:-0}
NUM_PROCESSES=2
TOTAL_NUM=100
shift "$(( $# < 4 ? $# : 4 ))"

if [[ $# -gt 0 && "$1" != -* ]]; then
    NUM_PROCESSES=${1:-2}
    shift
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
    TOTAL_NUM=${1:-100}
    shift
fi

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/parallel_eval_policy.py "$TASK_NAME" "$TASK_CONFIG" "$POLICY_CONFIG" \
    --total_num "$TOTAL_NUM" \
    --workers "$NUM_PROCESSES" "$@"
