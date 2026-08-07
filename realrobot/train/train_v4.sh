#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# Pi0.5 NERO v4: CLEAN dataset (50 ep), action_horizon=16, 5000 step, save every 1000
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
unset OPENPI_TRACE
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/train.py \
  pi05_nero_pick_pink_sponge_v4 \
  --exp-name v4 \
  --checkpoint-base-dir ${NERO_CKPT} \
  --no-wandb-enabled \
  --overwrite
