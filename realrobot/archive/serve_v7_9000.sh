#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# v7/ckpt_9000 server — czj 真机验腕漂移随step改善 (9k vs 6k卡83)
# 端口 9097 (3k:9095 / 6k:9096 都留着), GPU3 三server并存. MEM0.25省着点.
# 新 trace 文件
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_v7_9000.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA --port 9097 \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge_v7 \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v7/v7/9000
