#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# v7/ckpt_6000 server — czj 真机试 6k 手感 (默认同步broker, 异步预取已弃用)
# 新端口 9096 (保留 3k 在 9095 不动), GPU3 与 3k server 并存 (各 MEM0.3, 总~65G<98G)
# 新 trace 文件, 不污染 3k 的
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_v7_6000.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA --port 9096 \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge_v7 \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v7/v7/6000
