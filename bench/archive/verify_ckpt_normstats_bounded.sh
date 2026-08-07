#!/usr/bin/env bash
# Independent, DEADLINE-BOUNDED verifier for the ckpt-baked norm_stats md5 gate.
# Runs ALONGSIDE watch_ckpt_normstats.sh (which is armed and untouched) -- this is not a
# replacement, it closes two silence-is-not-success holes in it:
#
#  1. its remote probe is `ssh ... 2>/dev/null`, so an unreachable remote (network, auth, reboot)
#     produces an empty result that is indistinguishable from "pi0 hasn't reached 20k yet" ->
#     it would wait forever and never say why. Here reachability is probed SEPARATELY from
#     file presence, and an unreachable host is its own reported state.
#  2. it has no deadline, but the window it depends on CLOSES: max_to_keep=1 means orbax
#     deletes step 20000 when 29999 lands. If the verdict is missed, the gate is lost silently
#     and the board keeps reading "ckpt pending" forever. Here, 29999 appearing while 20000 is
#     unverified is an explicit reported outcome, as is a wall-clock deadline.
#
# Everything is APPEND-ONLY to a log file as well as stdout: a harness notification can be
# missed, a file cannot. Emits on every terminal state, never only on success.
set -u
: "${NERO_ROOT:?please 'source env.sh' first}"
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
BENCH="$NERO_ROOT/bench"
LOG=$BENCH/logs/verify_ckpt_normstats_bounded.log
EXPECT=9467e876b49b33168d5c2654482f0d09
REL=assets/local/pick_pink_sponge_b2_train/norm_stats.json
P05_DIR="${NERO_CKPT:?set NERO_CKPT}/pi05_nero_b2_train/pi05_b2train"
P0_DIR="$NERO_CKPT/pi0_nero_b2_train/pi0_b2train"
SSH="timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=20 $NERO_REMOTE_SSH_OPTS $NERO_REMOTE_HOST"
DEADLINE=$(( $(date +%s) + 12*3600 ))

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- power control: prove the comparison can report FAIL, not just PASS ---------------------
# A gate that has only ever produced PASS on synthetic input has not been shown to discriminate.
probe_md5_of() { md5sum "$1" 2>/dev/null | awk '{print $1}'; }
t=$(mktemp); printf 'not the norm stats\n' > "$t"
bad=$(probe_md5_of "$t"); rm -f "$t"
if [ -z "$bad" ]; then say "FATAL power control: md5sum produced nothing on a readable file"; exit 90; fi
if [ "$bad" = "$EXPECT" ]; then say "FATAL power control: junk file md5 == EXPECT ($EXPECT) -- comparison is degenerate"; exit 90; fi
say "power control OK: junk md5 $bad != EXPECT $EXPECT (a mismatch IS reportable)"
say "bounded verifier armed; EXPECT=$EXPECT deadline=$(date -d "@$DEADLINE" '+%F %T')"

d05=0; d0=0; unreach_streak=0; unreach_reported=0
while [ $d05 -eq 0 ] || [ $d0 -eq 0 ]; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    [ $d05 -eq 0 ] && say "CKPT_NORMSTATS pi05 TIMEOUT: no verdict within deadline -- gate NOT satisfied, do not mark the board verified"
    [ $d0  -eq 0 ] && say "CKPT_NORMSTATS pi0  TIMEOUT: no verdict within deadline -- gate NOT satisfied, do not mark the board verified"
    break
  fi

  # ---- pi0.5 (local box) ----
  if [ $d05 -eq 0 ]; then
    if [ -s "$P05_DIR/20000/$REL" ]; then
      sleep 20                                    # let orbax finish the assets callback
      m=$(probe_md5_of "$P05_DIR/20000/$REL")
      if [ -z "$m" ];        then say "CKPT_NORMSTATS pi05 FAIL: file present but md5 unreadable"
      elif [ "$m" = "$EXPECT" ]; then say "CKPT_NORMSTATS pi05 PASS: ckpt-baked md5=$m == proven-clean source"
      else say "CKPT_NORMSTATS pi05 FAIL: ckpt-baked md5=$m != source $EXPECT -- leak proof does NOT transfer"; fi
      d05=1
    elif [ -d "$P05_DIR/29999" ]; then
      say "CKPT_NORMSTATS pi05 WINDOW CLOSED: step 29999 exists but step-20000 norm_stats was never verified (max_to_keep=1 deleted it). Verify the 29999 baked copy instead."
      d05=1
    fi
  fi

  # ---- pi0 (second machine): reachability probed SEPARATELY from file presence ----
  if [ $d0 -eq 0 ]; then
    out=$($SSH "echo REACHABLE; [ -s '$P0_DIR/20000/$REL' ] && md5sum '$P0_DIR/20000/$REL' | awk '{print \"MD5 \"\$1}'; [ -d '$P0_DIR/29999' ] && echo FINALEXISTS" 2>&1)
    if ! printf '%s' "$out" | grep -q REACHABLE; then
      unreach_streak=$((unreach_streak+1))
      if [ $unreach_streak -ge 3 ] && [ $unreach_reported -eq 0 ]; then
        say "REMOTE UNREACHABLE x$unreach_streak (this is NOT 'pi0 hasn't reached 20k') | last: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
        unreach_reported=1
      fi
    else
      [ $unreach_streak -ge 3 ] && say "remote reachable again after $unreach_streak failures"
      unreach_streak=0; unreach_reported=0
      m=$(printf '%s' "$out" | sed -n 's/^MD5 //p')
      if [ -n "$m" ]; then
        if [ "$m" = "$EXPECT" ]; then say "CKPT_NORMSTATS pi0 PASS: ckpt-baked md5=$m == proven-clean source"
        else say "CKPT_NORMSTATS pi0 FAIL: ckpt-baked md5=$m != source $EXPECT -- leak proof does NOT transfer"; fi
        d0=1
      elif printf '%s' "$out" | grep -q FINALEXISTS; then
        say "CKPT_NORMSTATS pi0 WINDOW CLOSED: step 29999 exists but step-20000 norm_stats was never verified. Verify the 29999 baked copy instead."
        d0=1
      fi
    fi
  fi

  [ $d05 -eq 1 ] && [ $d0 -eq 1 ] && break
  sleep 120
done
say "bounded verifier done (pi05_reported=$d05 pi0_reported=$d0)"
