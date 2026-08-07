#!/usr/bin/env bash
# Push the step-20000 checkpoint load-test verdicts (145 pi0.5 + 147 pi0) into the conversation.
# Exits once both have reported -- two events, then done.
#
# Coverage note: it must emit on FAIL and on "waiter vanished without a verdict" too, not just
# on PASS. A load-test that dies silently is indistinguishable from one still waiting, and that
# is the failure mode this whole check exists to remove.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first}"
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
B="$NERO_ROOT/bench"
L145=$B/logs/ckpt_loadtest_pi05.log
L147=$B/logs/ckpt_loadtest_pi0.log
SSH="timeout 45 ssh $NERO_REMOTE_SSH_OPTS -o BatchMode=yes -o ConnectTimeout=20 $NERO_REMOTE_HOST"
d145=0; d147=0
while [ $d145 -eq 0 ] || [ $d147 -eq 0 ]; do
  if [ $d145 -eq 0 ]; then
    v=$(grep -h CKPT_LOADTEST "$L145" 2>/dev/null | tr '\n' ' ')
    if [ -n "$v" ]; then
      echo "[loadtest pi0.5@20000 /145] $v"; d145=1
    elif ! ps -eo args 2>/dev/null | grep -q '[c]kpt_loadtest.sh .*pi05_b2train'; then
      echo "[loadtest pi0.5@20000 /145] WAITER GONE with no verdict | last: $(tail -1 "$L145" 2>/dev/null)"; d145=1
    fi
  fi
  if [ $d147 -eq 0 ]; then
    out=$($SSH "grep -h CKPT_LOADTEST $L147 2>/dev/null | tr '\n' ' '; echo '|'; ps -eo args | grep -c '[c]kpt_loadtest.sh .*pi0_b2train'" 2>/dev/null)
    if [ -n "$out" ]; then
      v=${out%%|*}; alive=$(echo "$out" | sed -n 's/.*|//p' | tr -dc '0-9')
      if [ -n "${v// /}" ]; then
        echo "[loadtest pi0@20000 /147] $v"; d147=1
      elif [ "${alive:-1}" = "0" ]; then
        echo "[loadtest pi0@20000 /147] WAITER GONE with no verdict"; d147=1
      fi
    fi
  fi
  [ $d145 -eq 1 ] && [ $d147 -eq 1 ] && break
  sleep 300
done
echo "[loadtest] both step-20000 checkpoints reported; watcher done"
