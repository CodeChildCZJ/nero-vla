#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA \
  --port 9095 \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge/v1/2999
