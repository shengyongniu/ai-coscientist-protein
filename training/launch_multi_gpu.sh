#!/usr/bin/env bash
# Launch multi-GPU ESM2 fine-tuning with torchrun.
#
# Usage:
#   ./training/launch_multi_gpu.sh [NUM_GPUS] [STRATEGY] [BASE_MODEL]
# Examples:
#   ./training/launch_multi_gpu.sh 4 ddp  facebook/esm2_t12_35M_UR50D
#   ./training/launch_multi_gpu.sh 4 fsdp facebook/esm2_t33_650M_UR50D
set -euo pipefail

NUM_GPUS="${1:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
STRATEGY="${2:-ddp}"
BASE_MODEL="${3:-facebook/esm2_t12_35M_UR50D}"
DATASET="${DATASET:-synthetic}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUT="${OUT:-checkpoints/esm2_fitness}"

echo "Launching: gpus=${NUM_GPUS} strategy=${STRATEGY} base=${BASE_MODEL} dataset=${DATASET}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m training.train \
  --dataset "${DATASET}" \
  --base-model "${BASE_MODEL}" \
  --strategy "${STRATEGY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --out "${OUT}"
