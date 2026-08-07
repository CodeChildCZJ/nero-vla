#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# v6_gmask/ckpt_7999 server — 真机测mask版 (病A修复后验病B)
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_v6_7999.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA --port 9095 \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge_v6_gmask \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v6_gmask/v6_gmask/7999
