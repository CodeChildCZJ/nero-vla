#!/usr/bin/env bash
# Cross-host watcher for the pi0 training running on 147 GPU4.
# Emits: hourly progress, any crash signature, and the terminal exit event.
# Both boxes mounted the repo at the same path, which is why one LOG works on either side.
: "${NERO_ROOT:?please 'source env.sh' first}"
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
LOG="$NERO_ROOT/bench/logs/train_pi0_b2train.log"
SSH="ssh $NERO_REMOTE_SSH_OPTS -o ConnectTimeout=20 -o ServerAliveInterval=30 $NERO_REMOTE_HOST"
tick=0
while true; do
  out=$($SSH "pgrep -f 'train.py pi0_nero_b2_train' >/dev/null && echo ALIVE || echo DEAD;
              grep -oE 'Progress on: [0-9.k]+it/30.0kit rate:[^ ]+ remaining:[^ ]+' $LOG | tail -1;
              grep -hoE 'Traceback|RESOURCE_EXHAUSTED|OutOfMemory|CUDA_ERROR|Killed|OOM' $LOG | tail -1;
              pgrep -f 'post_train_pipeline.sh pi0_nero_b2_train' >/dev/null && echo PIPE_ARMED || echo PIPE_GONE" 2>&1) || out="SSH_FAIL"
  state=$(printf '%s' "$out" | head -1)
  prog=$(printf '%s' "$out" | grep -m1 'Progress on' || echo 'no-progress-line')
  bad=$(printf '%s' "$out" | grep -m1 -E 'Traceback|RESOURCE_EXHAUSTED|OutOfMemory|CUDA_ERROR|Killed|OOM' || true)
  pipe=$(printf '%s' "$out" | grep -m1 -E 'PIPE_ARMED|PIPE_GONE' || echo PIPE_UNKNOWN)
  if [ "$out" = "SSH_FAIL" ]; then
    echo "[147 pi0] SSH unreachable at $(date +%H:%M) — cannot confirm training state"
  elif [ "$state" = "DEAD" ]; then
    echo "[147 pi0] TRAIN PROCESS GONE at $(date +%H:%M) | last: $prog | err: ${bad:-none} | $pipe"
    exit 0
  elif [ -n "$bad" ]; then
    echo "[147 pi0] CRASH SIGNATURE '$bad' at $(date +%H:%M) | $prog"
  elif [ $((tick % 12)) -eq 0 ]; then
    echo "[147 pi0] alive $(date +%H:%M) | $prog | $pipe"
  fi
  tick=$((tick+1))
  sleep 300
done
