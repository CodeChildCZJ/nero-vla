#!/usr/bin/env bash
# Build a Blackwell(sm_120)-capable env for GR00T N1.5 (n1d5 branch).
# Upstream pins torch==2.5.1 which has no sm_120 kernels -> override with cu128 build.
set -euo pipefail
# Run from the repo root (this script lives there).
cd "$(dirname "$0")"
export UV_HTTP_TIMEOUT=600

uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"

# 1. torch first (cu128 for Blackwell sm_120)
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 2. gr00t core deps (extras [base] are skipped: they pin torch==2.5.1 / tensorflow)
uv pip install -e .

# 3. runtime bits from [base] that we actually need (minus torch/tensorflow/pytorch3d)
uv pip install decord==0.6.0 diffusers==0.30.2 opencv_python==4.8.0.74 pyzmq

# 4. flash-attn: a LOCALLY BUILT 2.8.3 wheel containing sm_120 cubins.
# No public wheel ships sm_120 at this version -- you have to build it yourself
# (`MAX_JOBS=8 pip wheel flash-attn==2.8.3 --no-build-isolation`, takes ~1h) and then
# point FLASH_ATTN_WHEEL at the result.
uv pip install --no-deps "${FLASH_ATTN_WHEEL:?set FLASH_ATTN_WHEEL to your locally built flash_attn sm_120 wheel}"

echo "=== install done ==="
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
