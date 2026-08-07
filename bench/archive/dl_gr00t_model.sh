#!/usr/bin/env bash
# The `hf` CLI wedges partway through every LFS shard here (both via the HTTP
# proxy and direct to hf-mirror), always at a power-of-two boundary. Plain curl
# with --continue-at sustains ~5 MB/s, so fetch the repo file-by-file instead.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first}"
REPO=${1:-nvidia/GR00T-N1.5-3B}
DEST=${2:-$NERO_ROOT/models/GR00T-N1.5-3B}
BASE=https://hf-mirror.com/$REPO/resolve/main

FILES=(
  config.json
  model.safetensors.index.json
  experiment_cfg/metadata.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
)

mkdir -p "$DEST/experiment_cfg"
for f in "${FILES[@]}"; do
  out="$DEST/$f"
  for attempt in $(seq 1 60); do
    curl -sS --noproxy '*' -L -C - -o "$out" "$BASE/$f" && break
    echo "  retry $attempt for $f (have $(stat -c%s "$out" 2>/dev/null || echo 0) bytes)"
    sleep 3
  done
  echo "OK $f -> $(stat -c%s "$out" 2>/dev/null || echo MISSING) bytes"
done
echo "=== DOWNLOAD COMPLETE ==="
du -sh "$DEST"
