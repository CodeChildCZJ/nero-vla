#!/usr/bin/env bash
# Launch the real N1.7 b2 training and arm the post-training chain, in one step,
# behind a gate that refuses unless the card is genuinely free.
#
#   usage:  n17_go.sh --check-only                 # run the gate, launch nothing
#           n17_go.sh --i-have-main-go [gpu] [steps]
#
# Why a script instead of typing it at handoff time: the two things most likely to
# go wrong are done here once and reviewed while there is no time pressure --
#   * the trainer must be started with nohup AND its PID captured correctly. That
#     works only because train_gr00t_n17.sh ends in `exec python ...`: the shell is
#     REPLACED by the trainer, so $! stays valid for the whole run. Without exec, $!
#     would be a wrapper that exits early and n17_post_train.sh would start its
#     chain against a half-written checkpoint.
#   * n17_post_train.sh must be armed with THAT pid, in the same breath. Arming it
#     later, by hand, off a `ps` reading, is how you end up waiting on the wrong pid.
#
# The gate is deliberately paranoid about the card. A sentinel firing, a waiter
# notification, or another agent's message is NOT authorization -- only main's
# explicit go is, and that is what --i-have-main-go asserts. The gate cannot check
# that, so it checks everything it can and refuses on anything unexpected.
set -uo pipefail

: "${NERO_ROOT:?please 'source env.sh' first}"
MODE="${1:?use --check-only or --i-have-main-go}"
GPU="${2:-4}"
STEPS="${3:-10000}"
BENCH="$NERO_ROOT/bench"
REPO="$NERO_ROOT/third_party/Isaac-GR00T"
RUN="${NERO_CKPT:?set NERO_CKPT}/gr00t_n17_nero_b2/gr00t_n17"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$BENCH/logs/n17_train_$STAMP.log
MIN_FREE_G=30

say() { echo "[$(date '+%F %T')] $*"; }
fail=0
bad() { echo "  REFUSE: $*"; fail=1; }

say "gate for GPU $GPU, $STEPS steps"

# --- 0. the HOST. "GPU 4" names a different physical card on each machine -----
# This leg's card was GPU 4 on the SECOND training machine. GPU 4 on the primary
# box is a different card and belonged to another leg. Every other check in this
# gate reads the LOCAL nvidia-smi, so running this script on the wrong host does
# not merely check the wrong card -- it would pass its gate and then launch a 14h
# job onto a teammate's GPU. Caught by running --check-only on the primary box and
# watching it gate THAT machine's GPU 4 (60 GB, another leg's) while reporting
# "GPU 4" exactly as it would for mine. Nothing in the flags distinguishes the two.
N17_EXPECTED_HOST="${NERO_EXPECTED_HOST:?set NERO_EXPECTED_HOST to the hostname where this GPU lives}"
if [ "$(hostname)" != "$N17_EXPECTED_HOST" ]; then
  bad "running on $(hostname), but this leg's card is GPU $GPU on $N17_EXPECTED_HOST."
  bad "  '$GPU' names a DIFFERENT physical card here. Refusing before any launch."
fi

# --- 1. the card, by process not by free memory ------------------------------
# A memory threshold alone can be fooled both ways: a just-exited job can still show
# hundreds of MiB, and a small tenant looks like noise. Ask which PROCESSES hold the
# card, matched by uuid so the index can never drift between the two queries.
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" 2>/dev/null)
[ -z "$UUID" ] && bad "cannot read uuid for GPU $GPU"
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
       | grep -F "$UUID")
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
if [ -n "$APPS" ]; then
  bad "GPU $GPU still has compute processes:"; echo "$APPS" | sed 's/^/          /'
else
  say "  ok: no compute processes on GPU $GPU (uuid $UUID)"
fi
if [ -n "$USED" ] && [ "$USED" -gt 2000 ] 2>/dev/null; then
  if [ -n "$APPS" ]; then
    bad "GPU $GPU reports ${USED} MiB used (consistent with the processes above)"
  else
    # Worth its own branch: memory held with NOTHING listed is a stale context or a
    # process in another container/namespace, and it is exactly the case where
    # "looks idle" is most tempting and most wrong.
    bad "GPU $GPU reports ${USED} MiB used but NO compute process is visible -- stale context or another namespace, not free"
  fi
else
  say "  ok: GPU $GPU memory.used = ${USED} MiB"
fi

# --- 2. disk ------------------------------------------------------------------
FREE_G=$(df -BG --output=avail /home2 | tail -1 | tr -dc '0-9')
if [ -z "$FREE_G" ] || [ "$FREE_G" -lt "$MIN_FREE_G" ]; then
  bad "/home2 free ${FREE_G}G < ${MIN_FREE_G}G (12G per checkpoint, 24G rotation peak)"
else
  say "  ok: /home2 free ${FREE_G}G"
fi

# --- 3. nothing of mine already running --------------------------------------
# pgrep would match this script's own command line (and, under the agent harness,
# the wrapper's argv too), so snapshot ps and match the snapshot instead.
ps -eo pid,args > /tmp/n17_go_ps.$$
if grep -q "n17_launch_finetune" /tmp/n17_go_ps.$$; then
  bad "an n17_launch_finetune process already exists:"
  grep "n17_launch_finetune" /tmp/n17_go_ps.$$ | sed 's/^/          /' | head -3
else
  say "  ok: no n17_launch_finetune process running"
fi
rm -f /tmp/n17_go_ps.$$

# --- 4. the output dir is not already a finished run --------------------------
if [ -d "$RUN" ] && [ -n "$(ls "$RUN" 2>/dev/null)" ]; then
  bad "$RUN already exists and is not empty -- a relaunch would mix two runs: $(ls "$RUN" | tr '\n' ' ')"
else
  say "  ok: $RUN is clear"
fi

# --- 5. inputs the post chain needs, checked BEFORE 3 hours of GPU ------------
for p in "$BENCH/data/val_anchors.npz" "$BENCH/logs/variance_subset200.npy" \
         "$BENCH/data/b2_n17_train" "$BENCH/data/b2_n17_val" \
         "${NERO_DATA:?set NERO_DATA}" \
         "$BENCH/scripts/predict/n17_post_train.sh" "$REPO/.venv/bin/python"; do
  [ -e "$p" ] || bad "missing input: $p"
done
say "  ok: post-chain inputs present (or listed above)"

# The shared anchor subset is the one input where PRESENT is not the same as CORRECT:
# a file with the right name and shape but different anchors makes the sampling footnote
# silently incomparable (the local fallback overlaps the shared set by 24/200). Existence
# was checked above; identity is checked here, at launch, rather than 3 h later when
# variance_gr00t_n17.py would refuse. Same md5 the variance script gates on.
SUBSET_MD5=9007df1fcec5eca2a1f01591c6266b43
if [ -e "$BENCH/logs/variance_subset200.npy" ]; then
  got=$(md5sum "$BENCH/logs/variance_subset200.npy" | cut -d' ' -f1)
  if [ "$got" = "$SUBSET_MD5" ]; then
    say "  ok: shared anchor subset md5 $got"
  else
    bad "shared anchor subset md5 $got != $SUBSET_MD5 -- footnote would be incomparable"
  fi
fi

if [ "$fail" -ne 0 ]; then
  say "GATE FAILED -- nothing launched"; exit 1
fi
say "GATE PASSED"

if [ "$MODE" = "--check-only" ]; then
  say "--check-only: stopping here, nothing launched"; exit 0
fi
if [ "$MODE" != "--i-have-main-go" ]; then
  say "refusing: '$MODE' is not --i-have-main-go. A sentinel/waiter/teammate message is"
  say "not authorization; only main's explicit go is."; exit 2
fi

cd "$REPO" || exit 1

# --- pin the shell scripts OUTSIDE the synced folder -------------------------
# $BENCH is inside the syncthing folder shared 145<->147 (.stignore excludes
# refs/, data/raw/, data/processed/ and **/log/ -- note `log/`, NOT `logs/` --
# so vla_backbone_bench/scripts IS synced). bash reads a script INCREMENTALLY by
# byte offset, so a replacement landing mid-run matters. Measured both ways:
#   in-place rewrite (same inode): the running shell resumes at the old offset in
#       the NEW bytes -> `unexpected EOF while looking for matching quote`, rc=2.
#   temp+rename (new inode): the running shell keeps its fd on the old inode and
#       finishes cleanly, rc=0.
# Syncthing and rsync both land files the second way, so propagation is expected
# to be safe -- but "expected to be safe" is a claim about someone else's writer,
# and these two scripts run for ~14h. Copying them out costs nothing and removes
# the dependency entirely. Bonus: RUNDIR pins the exact bytes this run used, so a
# later edit cannot rewrite the history of what executed.
RUNDIR=/tmp/n17_run_$STAMP
mkdir -p "$RUNDIR" || exit 1
cp "$BENCH/scripts/train/train_gr00t_n17.sh" "$BENCH/scripts/predict/n17_post_train.sh" "$RUNDIR/" || exit 1
say "pinned run scripts -> $RUNDIR ($(md5sum "$RUNDIR"/*.sh | tr '\n' ' '))"

say "launching training -> $LOG"
nohup env CUDA_VISIBLE_DEVICES="$GPU" bash "$RUNDIR/train_gr00t_n17.sh" > "$LOG" 2>&1 &
TRAIN_PID=$!
sleep 5
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  say "!!! trainer died within 5s -- see $LOG"; tail -20 "$LOG"; exit 1
fi
say "train pid $TRAIN_PID alive"

PLOG=$BENCH/logs/n17_post_train_$STAMP.log
nohup bash "$RUNDIR/n17_post_train.sh" "$TRAIN_PID" "$GPU" "$STEPS" > "$PLOG" 2>&1 &
say "post-train chain armed on pid $TRAIN_PID (pid $!), log $PLOG"
say "watch: tail -f $LOG"
