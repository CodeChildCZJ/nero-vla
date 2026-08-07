#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# v7/ckpt_14999 (=15k末步) server — czj 真机终极候选, 量腕j7漂移 vs 6k(51/80) vs 9k
# 端口 9099 (9098被占), GPU2 (与9k共卡, 训练已释放~80GB空). 独立trace.
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_v7_15k.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA --port 9099 \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge_v7 \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v7/v7/14999
