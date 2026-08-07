#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# v7/ckpt_3000 server — 早期ckpt真机试手感 (loss~0.006还在降, 部署~20Hz, 主要看夹爪会不会commit)
# config = pi05_nero_pick_pink_sponge_v7 (mask off, 官方recipe, 数据已修copycat)
# GPU3 (65G空, 不碰GPU2训练). MEM0.3 推理够用且不挤GPU3邻居.
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_v7_3000.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA --port 9095 \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge_v7 \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v7/v7/3000
