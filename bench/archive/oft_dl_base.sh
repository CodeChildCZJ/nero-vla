#!/bin/bash
# Download openvla/openvla-7b base weights via hf-mirror (the corp proxy is ~400 B/s to hf.co).
: "${NERO_ROOT:?please 'source env.sh' first}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
exec "$NERO_ROOT/third_party/lerobot/.venv/bin/hf" download openvla/openvla-7b --max-workers 8
