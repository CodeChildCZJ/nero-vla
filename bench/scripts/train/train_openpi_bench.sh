#!/bin/bash
# NERO b2 backbone bench: openpi training launcher.
#   usage: train_openpi_bench.sh <config_name> <gpu_id> <exp_name> [wandb_project]
#   e.g.   train_openpi_bench.sh pi05_nero_b2_train 5 pi05_b2train nero_backbone_bench
#
# Pinned to a single GPU on purpose -- without CUDA_VISIBLE_DEVICES JAX builds a mesh over ALL
# visible devices (including other people's cards) and then dies on batch_size % n_devices.
#
# exp_name doubles as the wandb run name AND the checkpoint subdir:
#   $NERO_CKPT/<config_name>/<exp_name>/<step>/
set -euo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
CKPT_BASE="${NERO_CKPT:-$NERO_ROOT/checkpoints}"
CFG="${1:?config name}"
GPU="${2:?gpu id}"
EXP="${3:?exp name}"
PROJ="${4:-nero_backbone_bench}"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HUB_OFFLINE=1     # datasets/weights are all local; must NOT gate wandb (different lib)
export WANDB_MODE=online
cd "$NERO_ROOT/third_party/openpi-agilex"
exec .venv/bin/python scripts/train.py "$CFG" \
  --exp-name "$EXP" \
  --project-name "$PROJ" \
  --checkpoint-base-dir "$CKPT_BASE" \
  --overwrite
