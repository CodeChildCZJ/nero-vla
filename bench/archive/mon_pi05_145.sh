#!/bin/bash
# Health monitor for the pi0.5 training on 145 GPU5. Emits one line per EVENT, not per log line.
#
# Coverage is the point: a filter that only matches progress stays silent through a crash, and
# silence is indistinguishable from "still running". This emits on the hourly heartbeat AND on
# death, traceback, OOM, and each new checkpoint step dir (the step-20000 dir is the norm_stats
# md5 evidence window, which max_to_keep=1 closes when 29999 lands).
#
# Liveness is checked BY PID, never by `pgrep -f <pattern>` or by grepping a ps snapshot for the
# config name: this script's own argv contains that string, so those match themselves. `ps -p PID
# -o args=` prints only that one process's argv and cannot see this monitor.
set -uo pipefail
# LOG/CKPT are overridable ONLY so the shipped file itself can be exercised against temp dirs --
# the alternative (editing a copy) tests text that will never run. Never point them at the real
# checkpoint dir for a test: the janitor, the load-tester and the post-train pipeline all key off
# what lives there, so a fake step dir would be acted on.
: "${NERO_ROOT:?please 'source env.sh' first}"
PID=${1:?train pid}
LOG=${2:-$NERO_ROOT/bench/logs/train_pi05_b2train.log}
CKPT=${3:-${NERO_CKPT:-$NERO_ROOT/checkpoints}/pi05_nero_b2_train/pi05_b2train}
off=$(wc -c < "$LOG" 2>/dev/null || echo 0)
seen_steps=" $(ls "$CKPT" 2>/dev/null | tr '\n' ' ')"
last_hb=0

prog() { grep -o 'Progress on: [^ ]* [^ ]*' "$LOG" 2>/dev/null | tail -1; }

while true; do
  if ! ps -p "$PID" -o args= 2>/dev/null | grep -q nero_b2_train; then
    echo "[145 pi05] TRAIN PID $PID GONE at $(date '+%F %T') | last=$(prog) | steps=$(ls "$CKPT" 2>/dev/null | tr '\n' ' ')"
    exit 0
  fi

  # new bytes only, so a stale traceback is not re-reported every poll
  now=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  if [ "$now" -gt "$off" ]; then
    tail -c +$((off + 1)) "$LOG" 2>/dev/null \
      | grep -aE 'Traceback|RESOURCE_EXHAUSTED|OutOfMemory|out of memory|Killed|CUDA_ERROR|XlaRuntimeError|^E[0-9]|FAILED' \
      | head -5 | while IFS= read -r l; do echo "[145 pi05] !! $l"; done
    off=$now
  fi

  # checkpoint step dirs: appearance is the evidence window opening
  cur=" $(ls "$CKPT" 2>/dev/null | tr '\n' ' ')"
  for s in $cur; do
    case "$seen_steps" in *" $s "*) ;; *) echo "[145 pi05] CKPT DIR APPEARED: $s at $(date '+%F %T') (orbax writes in place for ~7 min; not committed yet)";; esac
  done
  seen_steps="$cur"

  h=$(date +%H)
  if [ "$h" != "$last_hb" ]; then
    last_hb=$h
    echo "[145 pi05] alive $(date '+%H:%M') | $(prog) | free=$(df -BG --output=avail /home2 | tail -1 | tr -d ' ')"
  fi
  sleep 120
done
