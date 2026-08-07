#!/usr/bin/env bash
# Short training smoke: does the real train command actually launch, how much VRAM
# does it peak at, and how fast is a step? Run this before committing a card to a
# multi-hour run -- it answers "will batch 64 fit next to whatever else is on this
# GPU" with a number instead of a guess.
#
#   ./lerobot_train_smoke.sh act     4 20
#   ./lerobot_train_smoke.sh smolvla 4 20
set -euo pipefail

: "${NERO_ROOT:?please 'source env.sh' first}"
BB=${1:?usage: $0 <act|smolvla> <gpu> [steps]}
GPU=${2:?usage: $0 <act|smolvla> <gpu> [steps]}
STEPS=${3:-20}
BENCH="$NERO_ROOT/bench"
VENV="$NERO_ROOT/third_party/lerobot/.venv"
ROOT="${NERO_DATA_V30:-$HOME/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench}"
REPO_ID=local/pick_pink_sponge_b2_bench

EPS=$(BENCH="$BENCH" "$VENV/bin/python" - <<'PY'
import json, os
s = json.load(open(os.environ["BENCH"] + "/data/split.json"))
print("[" + ",".join(map(str, s["train_episodes"])) + "]")
PY
)

case "$BB" in
  act)     POLICY_ARGS=(--policy.type=act); BATCH=8 ;;
  smolvla) POLICY_ARGS=(--policy.path=lerobot/smolvla_base
                        --policy.freeze_vision_encoder=false
                        --policy.train_expert_only=false
                        --policy.scheduler_decay_steps=20000); BATCH=64 ;;
  *) echo "unknown backbone: $BB" >&2; exit 1 ;;
esac

# Scratch paths live under a directory THIS user owns, not /tmp. On this shared box
# /tmp already holds other users' files with exactly these generic names (e.g.
# /tmp/smoke_bl.json and /tmp/smoke_log.txt, owned by other users): a foreign owner makes
# the write fail with EACCES while the READ below still succeeds against their stale
# file -- syntactically perfect, plausibly sized, completely wrong data. openvla-oft
# lost 9 hours to that exact shape (/tmp/ps_snap.txt, another user's, 48 days stale).
SCRATCH=$BENCH/logs/smoke_scratch
mkdir -p "$SCRATCH"
OUT=$SCRATCH/run_${BB}
VRAM=$SCRATCH/vram_${BB}.txt
rm -rf "$OUT"
LOG=$BENCH/logs/smoke_${BB}.log

# sample peak VRAM of our own process while it runs (nvidia-smi totals include
# whatever else is sharing the card, which would overstate our footprint).
# Truncate-and-check BEFORE backgrounding: a redirect failure inside `( ... ) &` exits
# only the subshell, so the sampler would be silently absent and the read at the end
# would report whatever was already at that path.
: > "$VRAM" || { echo "FATAL: cannot write $VRAM" >&2; exit 1; }
( while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
    sleep 2
  done ) >> "$VRAM" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=$GPU HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
"$VENV/bin/lerobot-train" \
  "${POLICY_ARGS[@]}" \
  --dataset.repo_id="$REPO_ID" --dataset.root="$ROOT" --dataset.episodes="$EPS" \
  --batch_size=$BATCH --steps=$STEPS --save_freq=$STEPS \
  --output_dir="$OUT" --job_name=smoke_${BB} \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false 2>&1 | tee "$LOG"

kill $SAMPLER 2>/dev/null || true
echo "=== peak VRAM seen on this box during run (all procs, MiB) ==="
sort -t, -k2 -n "$VRAM" | tail -3
echo "=== step timings ==="
grep -oE "step:[0-9K ]+ .*" "$LOG" | tail -5 || tail -5 "$LOG"
rm -rf "$OUT"
