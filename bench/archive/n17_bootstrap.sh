#!/usr/bin/env bash
# One-shot CPU-only validation of the GR00T N1.7 stack, before any GPU is allocated.
#
#   1. generate meta/stats.json + meta/relative_stats.json for the N1.7 train/val dirs
#      (LeRobotEpisodeLoader hard-asserts stats.json; relative_stats.json is what the
#      RELATIVE single_arm action config normalizes against)
#   2. run check_n17_stack.py, which verifies the modality config is joint-space and
#      that the val loader lines up with val_anchors frame-for-frame
#
# Stats are computed separately per directory, so the val split never informs the
# train normalization -- same no-leakage property as the N1.5 dirs.
set -uo pipefail

: "${NERO_ROOT:?please 'source env.sh' first}"
REPO="$NERO_ROOT/third_party/Isaac-GR00T"
BENCH="$NERO_ROOT/bench"
PY=$REPO/.venv/bin/python
CFG=$BENCH/configs/nero_n17_config.py

cd "$REPO"
export PYTHONPATH="$BENCH/configs:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy='*' NO_PROXY='*'
# CPU-only unless the caller allocated a card: never squat on someone else's GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

for split in train val; do
  echo "=== stats: b2_n17_$split ==="
  $PY gr00t/data/stats.py \
    --dataset-path "$BENCH/data/b2_n17_$split" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CFG" || exit 1
  ls -la "$BENCH/data/b2_n17_$split/meta/"
done

echo
echo "=== check_n17_stack ==="
exec $PY "$BENCH/archive/check_n17_stack.py"
