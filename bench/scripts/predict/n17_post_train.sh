#!/bin/bash
# Wait for the N1.7 bench training to exit, then run the whole post-training chain
# unattended:  predict -> score -> sampling-variance -> floor round-trip -> audits
# -> drop the duplicate top-level save.
#
#   usage: n17_post_train.sh <train_pid> <gpu_id> [steps]
#   e.g.   n17_post_train.sh 123456 4 10000
#
# PUBLIC RELEASE NOTE: this leg was originally run on a SECOND training machine, and the
# script pushed its artifacts (preds npz, variance footnote, audit logs) back to the host
# that renders the leaderboard. Those cross-machine transfer blocks have been REMOVED from
# this version -- everything now stays local. If you run this on a machine other than your
# scoring host, copy the npz it writes into $BENCH/preds/ yourself. The engineering lessons
# those blocks encoded are kept as comments at the points where they applied.
#
# Waits on a PID, not on `pgrep -f <pattern>`. Pattern waits are unreliable here:
# a pgrep run over ssh matches the remote shell's own command line, and inside
# Claude Code the harness wrapper's argv also embeds the pattern, so the loop can
# see a process that does not exist and wait forever. `kill -0 <pid>` cannot lie.
#
# bash reads a running script by byte offset, so editing this file while an armed
# instance is executing corrupts its control flow -- kill, edit, re-arm.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
TRAIN_PID="${1:?train pid}"; GPU="${2:?gpu id}"; STEPS="${3:-10000}"
BENCH="$NERO_ROOT/bench"
REPO="$NERO_ROOT/third_party/Isaac-GR00T"
PY=$REPO/.venv/bin/python
RUN="${NERO_CKPT:-$NERO_ROOT/checkpoints}/gr00t_n17_nero_b2/gr00t_n17"
CKPT=$RUN/checkpoint-$STEPS
PREDS=$BENCH/preds/preds_gr00t_n17.npz

log() { echo "[$(date '+%F %T')] $*"; }
# Count every loud failure so the final line can REPORT status instead of asserting
# success. An unconditional "DONE" is a claim the script is in no position to make.
WARN=0
# NOT `bad "$*"` -- that is what this line said until 2026-08-05, because the edit that
# introduced bad() rewrote every `log "!!! ..."` call site and this definition IS one.
# Self-recursive, syntactically valid, invisible to `bash -n`, and it only executes on
# the first warning: measured, bash segfaults (exit 139) with the message never printed,
# so the whole unattended chain dies at the exact moment something else had gone wrong.
bad() { WARN=$((WARN+1)); log "!!! $*"; }

log "waiting for train pid $TRAIN_PID to exit..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
log "training process gone"

# HF writes the shards then the index; wait for the tree to stop changing so a
# partially flushed checkpoint is never loaded.
if [ ! -d "$CKPT" ]; then
  log "checkpoint-$STEPS missing; present: $(ls "$RUN" 2>/dev/null | tr '\n' ' ')"
  # the launcher also leaves a final save at the run root
  if [ -f "$RUN/model.safetensors.index.json" ]; then
    CKPT=$RUN; log "falling back to the run-root final save"
  else
    log "ABORT: no usable checkpoint"; exit 1
  fi
fi
while [ -n "$(find "$CKPT" -newermt '-90 seconds' -print -quit 2>/dev/null)" ]; do
  log "checkpoint still settling..."; sleep 30
done
log "using $CKPT ($(du -sh "$CKPT" | cut -f1))"

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

log "predict -> $PREDS (noise pinned per global anchor index, seed 0)"
$PY "$BENCH/scripts/predict/predict_gr00t_n17.py" --model-path "$CKPT" --out "$PREDS"
rc=$?; log "predict rc=$rc"
if [ $rc -ne 0 ]; then
  log "predict FAILED -- stopping here, keeping everything for debugging"
  exit 1
fi

log "score"
$PY "$BENCH/scripts/score/score.py"
SRC=$?; log "score rc=$SRC"   # not fatal: the .npz is the deliverable and score.py is
# re-runnable any time -- but "not fatal" is not "not worth reporting". A nonzero here
# means the board table above was never printed, so it must reach the warning count
# rather than sit in a log line nobody greps (openvla-oft/gr00t-n15 flagged the
# neighbouring shape: a closing line that RESTATES an rc it never captured).
[ $SRC -eq 0 ] || bad "score.py FAILED (rc=$SRC) -- board table not rendered; preds unaffected"

# --- deliver the product before running any diagnostics ---------------------
# The cross-machine transfer that used to sit here is removed in the public version
# (see the header). Two lessons it encoded generalize to any delivery step, so they
# stay:
#   * VERIFY THE TRANSPORT, NOT THE EXIT CODE. The first version shipped over the
#     wrong port and was refused every single time; because the failure was swallowed
#     by `|| log`, this side reported a clean finish while the leaderboard silently
#     lost the row. Read the artifact back through the READER that will consume it.
#   * ORDERING. Every transfer sat immediately after the step that WROTE its payload,
#     and the deliverable went first. Hanging delivery off the END of the chain lets a
#     crash or a hang in a *verification* step -- none of which the leaderboard needs
#     -- withhold the row.
# Local equivalent of the readback: prove the npz is on disk and parses, so a truncated
# or half-written file cannot masquerade as success.
if [ -s "$PREDS" ]; then
  log "preds readback: $($PY -c \
    'import numpy,sys; a=numpy.load(sys.argv[1])["pred"]; print(a.shape, a.dtype)' "$PREDS")"
else
  bad "$PREDS absent or empty -- the leaderboard is missing the n17 row"
fi

VARJSON="$BENCH/logs/variance/variance_gr00t_n17.json"
rm -f "$VARJSON"   # so a stale file from an earlier attempt cannot satisfy the gate below

# Re-measure seed non-aliasing HERE rather than trusting the copy made earlier: the
# check reads the SHIPPED pinner and ast-parses the SHIPPED caller, so its verdict is
# only about the code as it stood when it ran. Scripts change between the pre-flight
# and the run that ships the row. Cheap (CPU, ~15 s, no model, no GPU) and it feeds
# the hard gate inside variance_gr00t_n17.py, so a stale PASS here is a stale PASS on
# the board's sampling footnote. rm first for the same reason as VARJSON above.
SEEDCHK="$BENCH/logs/variance/seedcheck_gr00t_n17.json"
rm -f "$SEEDCHK"
log "seed-collision measurement (shipped pinner + power control)"
$PY "$BENCH/archive/n17_seed_collision_check.py" --out "$SEEDCHK" 2>&1 | tee -a "$LOG"
SCRC=${PIPESTATUS[0]}; log "seed-collision check rc=$SCRC"
[ $SCRC -eq 0 ] || bad "seed-collision check FAILED (rc=$SCRC) -- variance will refuse to run"

log "sampling-variance footnote (shared 200-anchor subset x 5 seeds)"
$PY "$BENCH/archive/variance_gr00t_n17.py" --model-path "$CKPT"
VRC=$?; log "variance rc=$VRC"
[ $VRC -eq 0 ] || bad "variance FAILED (rc=$VRC) -- continuing, preds are already delivered"

# PER-ARTIFACT GATING (the removed transfer block's other lesson, kept because it is
# about gates, not about transports): `logs/variance/` is a SHARED directory that
# already holds other legs' footnotes, so any operation keyed on the DIRECTORY succeeds
# and logs "ok" even when MY json was never written. Gate on the specific artifact --
# a directory existing is not evidence that the file I owe the board is in it. Same
# family as the `|| log` swallow: a green line for work that never happened.
# Writer-side self-check: can the board's OWN reader read back what I just wrote?
# Not hypothetical -- until 2026-08-05 this script wrote a flat {mae_std:..} schema
# that paired_signif.py's strict reader did not know, so the footnote would have
# SystemExit'd the shared --all the moment my row landed. A file existing is not the
# same as a file being ingestible, and the two sides live in different scripts, so
# nothing but an actual round trip catches a divergence between them.
if [ -f "$VARJSON" ]; then
  $PY - "$BENCH/scripts/score/paired_signif.py" <<'PYEOF'
import sys, importlib.util, pathlib
spec = importlib.util.spec_from_file_location("ps", sys.argv[1])
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
for ch in ps.CHANNELS:
    v = ps.sampling_std("preds_gr00t_n17", ch)
    if v is None:
        sys.exit(f"reader sees NO footnote for {ch}")
    print(f"  readback {ch:<15} = {v:.6f}")
PYEOF
  [ $? -eq 0 ] && log "footnote readback OK (paired_signif can ingest it)" \
    || bad "footnote written but the board reader CANNOT parse it -- --all will die on my row"
else
  bad "$VARJSON absent -- the board has no n17 sampling footnote"
fi

# The floor depends on the checkpoint's own statistics.json, so the number quoted
# on the leaderboard has to come from the FINAL checkpoint, not the smoke one.
# --n-anchors 0 = ALL 1397, stated explicitly rather than relying on the default:
# this call used to inherit a 200-anchor subsample and would have written 0.02347
# into the final log against the board's 0.01757 -- a 33% overstatement, silent,
# and indistinguishable from "the floor changed with the checkpoint" (it cannot;
# the floor depends only on the baked stats and the ground truth).
log "inverse-clip floor round-trip on the final checkpoint (all anchors)"
$PY "$BENCH/archive/n17_unit_roundtrip.py" --model-path "$CKPT" --n-anchors 0 \
  | tee "$BENCH/logs/n17_floor_final.log"

# The 10k run bakes its OWN statistics.json; proving the smoke checkpoint was
# clean proves nothing about this one. Re-identify the table the postprocessor
# actually holds against train/val meta -- a val-derived table would normalize
# predictions with held-out statistics and still yield a plausible MAE.
log "statistics provenance on the final checkpoint"
$PY "$BENCH/archive/n17_stats_provenance.py" --model-path "$CKPT" \
  2>&1 | tee "$BENCH/logs/n17_stats_provenance_final.log"
if grep -q "PASS: live table is the TRAIN one" "$BENCH/logs/n17_stats_provenance_final.log"; then
  log "stats provenance PASS"
else
  bad "stats provenance DID NOT PASS -- read logs/n17_stats_provenance_final.log before quoting any number"
fi

# Provenance above proves live == {ckpt, experiment_cfg, train-dir meta} -- i.e. they
# are all the SAME table, which is consistency, not SCOPE. If that one table had been
# generated over all 131 episodes every one of those comparisons would still pass.
# The direct recompute re-derives it from raw source parquet over exactly the 111
# train episodes, and shows a 131-episode table would have been detected.
log "direct stats recompute (positive leak proof) on the final checkpoint"
$PY "$BENCH/archive/n17_leak_direct_recompute.py" --ckpt "$CKPT" \
  --json-out "$BENCH/logs/n17_leak_direct_recompute_final.json" \
  2>&1 | tee "$BENCH/logs/n17_leak_direct_recompute_final.log"
if grep -q "^PASS: baked stats reproduce EXACTLY" "$BENCH/logs/n17_leak_direct_recompute_final.log"; then
  log "direct recompute PASS"
else
  bad "direct recompute DID NOT PASS -- the leaderboard row is not trustworthy"
fi

# Channel 3: predict opens a loader over the VAL directory, which ships its own
# meta/stats.json + relative_stats.json. Measured (not argued) by corrupting every
# stat leaf to NaN and requiring the observation bytes to be identical, with a power
# control proving the corruption would have been visible. The modality config comes
# from THIS checkpoint, so re-running also re-checks the action convention it bakes.
log "val-directory stats reachability on the final checkpoint"
$PY "$BENCH/archive/n17_valstats_channel.py" --ckpt "$CKPT" \
  --json-out "$BENCH/logs/n17_valstats_channel_final.json" \
  2>&1 | tee "$BENCH/logs/n17_valstats_channel_final.log"
if grep -q "^PASS: val-directory normalization metadata cannot reach" \
     "$BENCH/logs/n17_valstats_channel_final.log"; then
  log "val-stats channel PASS"
else
  bad "val-stats channel DID NOT PASS -- held-out statistics may be reaching the model"
fi

# The board quotes the floor / provenance / recompute numbers, so they have to be
# reachable wherever the board is rendered. The transfer that did this is removed in
# the public version; the artifacts stay in $BENCH/logs/ next to the row:
#   n17_floor_final.log, n17_stats_provenance_final.log,
#   n17_leak_direct_recompute_final.{log,json}, n17_valstats_channel_final.{log,json}

# Only after the preds exist: the run root holds a second full copy of the same
# weights (12G). Contract says keep params only, and disk was at 100% at the time.
if [ "$CKPT" != "$RUN" ] && [ -f "$RUN/model.safetensors.index.json" ]; then
  rm -f "$RUN"/model-*-of-*.safetensors "$RUN"/model.safetensors.index.json \
    && log "dropped the duplicate run-root save (~12G)"
fi
log "df: $(df -h /home2 | tail -1)"

# State only what is observable AT THIS INSTANT. The two things a bare "DONE" implies
# are both things this line cannot know: that the audits passed, and that the card is
# free. The second is a prediction about the future, not an observation -- this very
# process still holds the CUDA context on GPU$GPU and will until it exits, seconds
# from now. A downstream leg that reads a "GPU is free" string and takes the card is
# racing a live context. Report the warning count; let the reader decide.
if [ "$WARN" -eq 0 ]; then
  log "compute stages finished, 0 warnings. This pid ($$) STILL HOLDS GPU$GPU until it exits."
else
  log "compute stages finished with $WARN WARNING(S) -- grep '!!!' above before quoting any number."
  log "This pid ($$) STILL HOLDS GPU$GPU until it exits."
fi
