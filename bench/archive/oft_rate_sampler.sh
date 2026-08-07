#!/bin/bash
# Session-local sampler (no cron, no systemd, nothing persistent in the shared env): every 120s record
# step / loadavg / who else holds GPU4, so the final walltime can be ATTRIBUTED rather than guessed.
# Exits by itself when the trainer exits.
PID=$1; LOG=$2; OUT=$3
while kill -0 "$PID" 2>/dev/null; do
  read -r one five _ < /proc/loadavg
  step=$(tr '\r' '\n' < "$LOG" | grep -oE '\| [0-9]+/10500 \[[0-9:]+<' | tail -1)
  others=$(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader | grep -vc "^$PID$")
  printf '{"t":"%s","load1":%s,"load5":%s,"gpu4_other_procs":%s,"tqdm":"%s"}\n' \
    "$(date +%H:%M:%S)" "$one" "$five" "$others" "$step" >> "$OUT"
  sleep 120
done
printf '{"t":"%s","event":"trainer_pid_%s_gone"}\n' "$(date +%H:%M:%S)" "$PID" >> "$OUT"
