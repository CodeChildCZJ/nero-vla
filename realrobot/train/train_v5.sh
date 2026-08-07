#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# Pi0.5 NERO v5: 51ep (50 clean + ep050), action_horizon=16, 20000 step, save every 2000
# czj 方案: 测长训到收敛 (v4 才 5k 没收敛), 不锐化不补位置 (留下轮)
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # GPU2 空了94GB free
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
unset OPENPI_TRACE
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/train.py \
  pi05_nero_pick_pink_sponge_v5 \
  --exp-name v5 \
  --checkpoint-base-dir ${NERO_CKPT} \
  --no-wandb-enabled \
  --overwrite
