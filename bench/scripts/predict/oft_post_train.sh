#!/bin/bash
# OFT post-training: predict -> determinism/provenance -> leak recheck -> score -> significance.
#
# Discipline (learned from n15's post_train.sh, which printed "GPU is FREE" unconditionally from a
# process that still held the CUDA context):
#   * every stage's rc is READ and aborts the pipeline;
#   * no log line names a number it did not read;
#   * no line asserts anything about the FUTURE (a process cannot print proof of its own exit).
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"

BENCH="$NERO_ROOT/bench"
PY=${NERO_ROOT}/third_party/openvla-oft/.venv/bin/python
CKPT=${CKPT:-$BENCH/runs/openvla_oft}   # overridable so each stage-0 guard can be exercised alone
GPU=${1:-4}
TMP="$NERO_ROOT/tmp/oft_post"
PREDS=$BENCH/preds/preds_openvla_oft.npz
TS=$(date +%Y%m%d_%H%M%S)
LOG=$BENCH/logs/oft/post_train_${TS}.log
# rc checked explicitly: `set -e` is deliberately OFF (every stage reads PIPESTATUS instead, which
# -e cannot do through a `| tee` pipeline), so an unchecked mkdir would be invisible -- and on a 99%
# -full /home2 it is a live failure, not a hypothetical. Cannot route through die(): if the logs dir
# is what failed, log()'s `tee` has nowhere to write, so the abort message would be lost with it.
mkdir -p "$TMP" "$BENCH/logs/oft" "$BENCH/preds" || {
  echo "ABORT: mkdir failed for one of $TMP / $BENCH/logs/oft / $BENCH/preds (disk full? perms?)" >&2
  exit 1; }

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

export CUDA_VISIBLE_DEVICES=$GPU
export ROBOT_PLATFORM=NERO
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log "=== OFT post-train | GPU $GPU | ckpt $CKPT ==="

# Flag sanity BEFORE any environment check. Placed here because the first version sat after the
# GPU-busy guard, and a test proved the consequence: with the card busy I got "GPU busy" and never
# learned my flags were nonsense -- I would have fixed the GPU, re-run, and only then found out.
# An error in MY invocation should never be reported second, behind an error about the world.
RUN_STAGE5=${RUN_STAGE5:-1}
ONLY_STAGE5=${ONLY_STAGE5:-0}
[ "$ONLY_STAGE5" = "1" ] && [ "$RUN_STAGE5" = "0" ] && die \
  "ONLY_STAGE5=1 with RUN_STAGE5=0 asks to run nothing at all; refusing rather than exiting 0 and "\
"looking like a successful run."

# ------------------------------------------------------- stage -1: WHICH MACHINE, WHICH CARD
# `--gpu 4` names a different physical card on every host, and this leg shared two machines with
# three other legs that coordinated in chat using the bare phrase "GPU 4".
# Another leg's launcher passed its own pre-flight while sitting on the WRONG host and would have
# thrown a 14 h training run onto THIS card; the only thing that stopped it was that this card
# happened to be busy at that moment.
# So: refuse on host identity BEFORE any nvidia-smi, because nvidia-smi on the wrong host answers
# confidently about the wrong hardware. Then pin the UUID, which unlike the index is global.
#
# Deliberately NO defaults: these must name YOUR machine and YOUR card. A default here would be
# fail-OPEN on every other host -- the guard would disarm itself exactly when it is needed. Get
# the two values with `hostname` and
#   nvidia-smi -i <gpu> --query-gpu=uuid --format=csv,noheader
EXPECT_HOST=${EXPECT_HOST:?set EXPECT_HOST to the hostname this run is registered to run on}
EXPECT_GPU_UUID=${EXPECT_GPU_UUID:?set EXPECT_GPU_UUID to the uuid of the card you registered}
THIS_HOST=$(hostname)
[ "$THIS_HOST" = "$EXPECT_HOST" ] || die \
  "host is $THIS_HOST, this leg runs only on $EXPECT_HOST. GPU $GPU here is NOT the card this "\
"run was pre-registered on -- on another host it is someone else's."
GOT_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
[ "$GOT_UUID" = "$EXPECT_GPU_UUID" ] || die \
  "GPU $GPU on $THIS_HOST has uuid '$GOT_UUID', expected '$EXPECT_GPU_UUID'. Index reassigned or "\
"wrong card: an index is a local name, the uuid is the card."
log "stage -1 OK: host $THIS_HOST, GPU $GPU uuid $GOT_UUID (both pinned, not inferred)"

# ---------------------------------------------------------------- stage 0: guards
# The delivered ckpt must be the one the pre-registered run produced. A ckpt dir that exists is
# NOT evidence it is the right one -- `save_latest_checkpoint_only=True` puts NO step number in
# any filename, so a run that died at 7000 leaves a dir that is complete, loadable, and wrong.
# Checking the files exist cannot see that; the step number lives only in the training log.
EXPECT_STEP=${EXPECT_STEP:-10500}
TRAIN_LOG=${TRAIN_LOG:-$(ls -t "$BENCH"/logs/oft/train_*.log 2>/dev/null | head -1)}
[ -d "$CKPT" ] || die "no ckpt dir $CKPT"
for f in action_head--latest_checkpoint.pt proprio_projector--latest_checkpoint.pt \
         dataset_statistics.json config.json model.safetensors.index.json; do
  [ -s "$CKPT/$f" ] || die "missing/empty $CKPT/$f"
done
[ -s "$TRAIN_LOG" ] || die "no training log found (looked at $TRAIN_LOG)"
LAST_SAVE=$(tr '\r' '\n' < "$TRAIN_LOG" | grep -oE 'Saved merged model for Step [0-9]+' \
            | tail -1 | grep -oE '[0-9]+$')
[ -n "$LAST_SAVE" ] || die "no completed save in $TRAIN_LOG"
[ "$LAST_SAVE" = "$EXPECT_STEP" ] || die \
  "last completed save is step $LAST_SAVE, pre-registered delivery step is $EXPECT_STEP "\
"($TRAIN_LOG). Refusing: delivering an earlier ckpt would break the no-val-selection commitment."
if nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  die "GPU $GPU still has a compute process; refusing to run predict beside the trainer"
fi
log "stage 0 OK: ckpt complete, last save = step $LAST_SAVE (= pre-registered $EXPECT_STEP), GPU $GPU idle"

# Stage 5 is ~69% of this script's GPU time, is pure diagnostic (a generalization footnote), and
# cannot change the delivery row. Two knobs so the card can be handed to another leg in between:
#   RUN_STAGE5=0   -> stages -1..4 only (delivery), then stop and release the card
#   ONLY_STAGE5=1  -> stages -1,0 then jump straight to 5 (guards ALWAYS run; they are what makes
#                     the second invocation trustworthy, and they cost nothing)
# Deliberately NOT a way to skip the delivery predict when producing the row: stage 4 reads $PREDS.
if [ "$ONLY_STAGE5" = "1" ]; then
  log "stages 1-4 SKIPPED (ONLY_STAGE5=1): the delivery row was produced by an earlier invocation"
  [ -s "$PREDS" ] || die "ONLY_STAGE5=1 but $PREDS does not exist -- the delivery row was never "\
"produced, so there is nothing for this diagnostic to be a footnote to."
fi

if [ "$ONLY_STAGE5" != "1" ]; then
# ---------------------------------------------------------------- stage 1: leak recheck on FINAL ckpt
# The probe-run diff proved nothing about this checkpoint: it baked its own dataset_statistics.json.
log "stage 1: leak falsification + soft floor against the FINAL ckpt's own stats"
$PY "$BENCH/archive/oft_soft_floor.py" --ckpt-stats "$CKPT/dataset_statistics.json" \
    --out "$BENCH/logs/oft/oft_soft_floor_final.json" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "oft_soft_floor.py rc=$rc"
$PY - "$BENCH/logs/oft/oft_soft_floor_final.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
diff = d["ckpt_vs_recomputed"]
bad = [k for k, v in diff.items() if not v["bit_identical"]]
print(f"  ckpt stats vs recompute over split.json train-111: "
      f"{'ALL %d FIELDS BIT-IDENTICAL' % len(diff) if not bad else 'DIFFERS on ' + str(bad)}")
# Both normalization channels the inference path reads, each gated. Action drives the OUTPUT
# (unnormalize); proprio drives an INPUT (ProprioProjector). Gating on action alone would pass
# while half the fields went untested.
verdicts = {}
for chan, key in (("action", "leakage_falsification"), ("proprio", "leakage_falsification_proprio")):
    v = d[key]; verdicts[chan] = v["verdict"].split()[0]
    print(f"  leak falsification [{chan}]: {v['verdict'].split(' --')[0]} "
          f"({v['n_informative_dims']}/8 informative dims)")
print(f"  image channel: no dataset statistic exists (pretrained-preprocessor constants)")
# The leg's #1 trap: preds must come back in ABSOLUTE joint units. The floor above is measured
# through `nero_dataset.unnormalize`, but delivery runs the model's own `_unnormalize_actions`,
# which has a `mask` branch returning the head's RAW [-1,1] on any masked-off dim. GATE on the
# agreement, do not merely print it -- a check nothing acts on is a comment.
u = d["delivery_inverse_identity"]
unit_ok = (u["two_inverse_impls_agree"] and u["mask_all_true"]
           and abs(u["roundtrip_via_MODEL_inverse"]["mean"] - d["floor_at_K_score"]["total"]) < 1e-5)
print(f"  delivery inverse: impls agree {u['two_inverse_impls_agree']} "
      f"(dev {u['two_inverse_impls_max_abs_dev']:.2e}), mask_all_true {u['mask_all_true']}, "
      f"roundtrip {u['roundtrip_via_MODEL_inverse']['mean']:.6f} vs floor "
      f"{d['floor_at_K_score']['total']:.6f} -> {'UNITS OK' if unit_ok else 'UNIT MISMATCH'}")
sys.exit(0 if (not bad and unit_ok and set(verdicts.values()) == {"CLEAN"}) else 1)
PY
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "final-ckpt leak check did not pass (rc=$rc)"
log "stage 1 PASS"

# ---------------------------------------------------------------- stage 2: delivery predict
log "stage 2: full predict -> $PREDS"
$PY "$BENCH/scripts/predict/oft_predict.py" --ckpt "$CKPT" --out "$PREDS" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "predict (delivery) rc=$rc"

$PY - "$PREDS" "$BENCH/data/val_anchors.npz" <<'PY' 2>&1 | tee -a "$LOG"
import numpy as np, sys
P = np.load(sys.argv[1])["pred"]; a = np.load(sys.argv[2])
K, n = int(a["K"]), len(a["episodes"])
assert P.shape[0] == n and P.shape[1] >= K and P.shape[2] == 8, P.shape
assert np.isfinite(P).all(), "non-finite values in preds"
assert not (P == 0).all(axis=(1, 2)).any(), "some anchor row is all-zero (partial run?)"
# Non-degeneracy (n15): rho vs the pass line is a free collapse readout, but a model collapsed to a
# CONSTANT scores rho~0 too -- rho only measures correlation WITH hold-state, so it cannot see a
# collapse that isn't toward hold-state. Per-dim spread across anchors can. Hard-fail on a spread
# RATIO below 1e-3 (full collapse or one dead dim); milder shrinkage is REPORTED, not fatal, since
# partial shrinkage is a finding about the model rather than a pipeline fault.
sd_p = P[:, :K, :].std(axis=(0, 1)); sd_g = a["gt"].astype(np.float32)[:, :K, :].std(axis=(0, 1))
# Threshold the RATIO, never `sd_p == 0`. Measured: a pred that is bit-identically constant
# (np.unique -> 1 value) still reports std 5.9e-05, because float32 `std` sums 11176 identical
# values pairwise and the resulting mean is not bit-identical to the value itself, so (x - mean)
# never cancels. `== 0` is therefore a guard with ZERO power -- it cannot fire on any real input.
# 1e-3 sits ~3 orders below any plausible healthy ratio (every one of the 8 dims genuinely moves).
# gr00t-n15 2026-08-05 sharpened this twice, and both corrections are load-bearing:
#  (a) that zero power is INPUT-DEPENDENT -- std(f32) of 11176 copies of 3.7 is 2.4e-07 but of
#      1.0 or 1e-3 it is exactly 0.0. So a must-fail control built from `np.full(..., 1.0)` would
#      have gone GREEN against the broken `== 0` gate. The control that exposed the bug only
#      worked because the constant happened to be 3.7; a "cleaner" constant would have hidden it.
#  (b) sd_g == 0 on some dim (a locked joint / pinned gripper) makes ratio NaN, and `NaN < 1e-3`
#      is False, so the gate would wave the collapse through. Not live on b2 (min sd_g 13.32 over
#      1397x8, measured by n15) but the gate gets copied, so close it where it is one line.
ratio = sd_p / sd_g
assert np.isfinite(ratio).all(), (
    f"spread ratio is non-finite on dim(s) {np.flatnonzero(~np.isfinite(ratio)).tolist()} "
    f"(sd_gt {sd_g.tolist()}) -- a constant GT dim makes NaN, and NaN passes every `<` test")
dead = [int(d) for d in np.flatnonzero(ratio < 1e-3)]
assert not dead, f"pred spread is ~0 vs GT on dim(s) {dead} (ratio {ratio[dead]}) -- collapsed"
print(f"  shape {P.shape} finite, no all-zero rows, dtype {P.dtype}")
print(f"  per-dim spread pred/gt: {np.round(sd_p / sd_g, 3).tolist()}  (<<1 on every dim = collapse)")
PY
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "delivery npz failed its shape/finite/zero-row gate (rc=$rc)"
log "stage 2 PASS"

# ---------------------------------------------------------------- stage 3: determinism + provenance
# 5 draws total. Draw 2 is a FULL independent re-run (strongest determinism evidence); draws 3-5 are
# 200-anchor runs whose value is PROVENANCE: a fresh process re-loads the checkpoint from disk, so a
# byte-match proves the delivery npz came from THIS ckpt, not a stale one that happens to be finite.
log "stage 3: determinism (draw 2 = full re-run) + provenance (draws 3-5 = 200-anchor re-loads)"
$PY "$BENCH/scripts/predict/oft_predict.py" --ckpt "$CKPT" --out "$TMP/draw2_full.npz" 2>&1 | tail -3 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "predict draw2 rc=$rc"
for d in 3 4 5; do
  # board-shared subset (md5 9007df1f..., spans all 20 val episodes) so this footnote is comparable
  # with n15/n17/ACT; --limit 200 would be a contiguous prefix covering ~3 episodes.
  $PY "$BENCH/scripts/predict/oft_predict.py" --ckpt "$CKPT" --subset-idx "$BENCH/logs/variance_subset200.npy" \
    --out "$TMP/draw${d}_200.npz" 2>&1 | tail -2 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "predict draw$d rc=$rc"
done
$PY - "$PREDS" "$TMP" "$BENCH/logs/variance_subset200.npy" \
     "$BENCH/logs/oft/determinism_oft.json" "$BENCH/data/val_anchors.npz" \
     "$BENCH/logs/variance/variance_openvla_oft.json" <<'PY' 2>&1 | tee -a "$LOG"
import hashlib, json, numpy as np, pathlib, sys
ref = np.load(sys.argv[1])["pred"]; tmp = pathlib.Path(sys.argv[2])
idx = np.load(sys.argv[3]); ok = True
full = np.load(tmp / "draw2_full.npz")["pred"]
eq2 = bool(np.array_equal(full, ref)); ok &= eq2
maxdiffs = [float(np.abs(full - ref).max())]
print(f"  draw2 full re-run (1397): bit_equal={eq2} maxdiff={maxdiffs[0]:g}")
subs = {}
for d in (3, 4, 5):
    p = np.load(tmp / f"draw{d}_200.npz")["pred"]
    eq = bool(np.array_equal(p, ref[idx])); ok &= eq
    subs[f"draw{d}"] = {"bit_equal_vs_full_idx": eq,
                        "maxdiff": float(np.abs(p - ref[idx]).max())}
    maxdiffs.append(subs[f"draw{d}"]["maxdiff"])
    print(f"  draw{d} shared-200 subset: bit_equal={eq} maxdiff={subs[f'draw{d}']['maxdiff']:g}")
# std is MEASURED, never asserted. A hardcoded 0.0 would restate the conclusion the draws are
# supposed to establish, and would keep printing 0.0 in exactly the case where the draws DISAGREED.
gt = np.load(sys.argv[5])["gt"].astype(np.float32)
Kc = gt.shape[1]
def chans(P, g):
    e = np.abs(P[:, :Kc, :].astype(np.float32) - g)
    return {"mae": float(e.mean()), "arm": float(e[..., :7].mean()), "grip": float(e[..., 7].mean())}
full_draws = [chans(ref, gt), chans(full, gt)]
sub_draws = [chans(ref[idx], gt[idx])] + \
            [chans(np.load(tmp / f"draw{d}_200.npz")["pred"], gt[idx]) for d in (3, 4, 5)]
std = {k: float(np.std([d[k] for d in full_draws], ddof=1)) for k in ("mae", "arm", "grip")}
std_sub = {k: float(np.std([d[k] for d in sub_draws], ddof=1)) for k in ("mae", "arm", "grip")}

# Power control, EXECUTED IN THIS RUN rather than quoted. This used to be a static string that
# recited a maxdiff measured on some earlier day -- which prints identically whether or not the
# comparator still has any power, i.e. exactly the defect ("std_mae: 0.0", "maxdiff: 0.0 if ok")
# that the rest of this stage was written to avoid. Perturb ONE cell by 1 ULP and require the same
# np.array_equal that produced `ok` to reject it.
probe = ref.copy()
probe[0, 0, 0] = np.nextafter(probe[0, 0, 0], np.float32(np.inf))
pc_equal = bool(np.array_equal(probe, ref))
pc_maxdiff = float(np.abs(probe.astype(np.float64) - ref.astype(np.float64)).max())
if pc_equal or not (pc_maxdiff > 0):
    print(f"  [FATAL] power control failed: a 1-ULP perturbation was NOT rejected "
          f"(equal={pc_equal}, maxdiff={pc_maxdiff:g}). The bit-equality above proves nothing.")
    sys.exit(1)
pc = {
    "claim": "a 0.0 std means the decoder is deterministic, NOT that the comparator is vacuous",
    "method": "flip preds[0,0,0] by one ULP, re-run the SAME np.array_equal used for the draws",
    "measured_rejects_1ulp": not pc_equal,
    "measured_1ulp_maxdiff": pc_maxdiff,
    "why_this_substitutes_for_seeds": "distinct seeds prove the draws COULD differ; for a "
                                      "seedless stack the equivalent is proving the COMPARATOR "
                                      "would notice if they did",
}
print(f"  power control: 1-ULP perturbation rejected={not pc_equal} (maxdiff {pc_maxdiff:g})")
det = {
    # maxdiff is the MEASURED max over all 4 comparisons, not `0.0 if ok`. Writing a constant
    # under a passing condition restates the conclusion instead of reporting the observation --
    # and it prints the same 0.0 whether the code measured anything or not (same defect as a
    # hardcoded std_mae, and as a log line quoting an rc it never read).
    "draws": 5, "bit_exact_across_draws": ok, "maxdiff": max(maxdiffs),
    "std_mae": std["mae"], "std_full_1397": std, "std_shared_subset_200": std_sub,
    "per_draw_mae_full": [d["mae"] for d in full_draws],
    "why_zero_is_expected": "L1RegressionActionHead + do_sample=False has NO noise source; unlike "
                            "flow-matching legs a NON-zero std here would be a bug, not variance",
    "per_draw_seeds": None,
    "seedless_stack": "no seed is consumed anywhere on this predict path, so the board's usual "
                      "'distinct per-draw seeds' discriminator for a 0.0 std is UNAVAILABLE here "
                      "-- it cannot be supplied, not merely omitted",
    "harness_power_control": pc,
    "batch_size": 1,
    "subset_md5": hashlib.md5(open(sys.argv[3], "rb").read()).hexdigest(),
    "subset_spans_episodes": 20,
    "subset_run_is_independent_process": True,
    "full_rerun_is_independent_process": True,
    "subsets": subs,
}
json.dump(det, open(sys.argv[4], "w"), indent=1)
# Board footnote in the flat schema paired_signif.py parses. Posting it explicitly matters: a
# MISSING file is read as "no footnote yet" and the sampling term is silently dropped, which is
# indistinguishable from a stack measured at 0 -- the tool's own docstring flags that ambiguity.
pathlib.Path(sys.argv[6]).parent.mkdir(parents=True, exist_ok=True)
json.dump({f"{k}_std": v for k, v in std.items()} |
          {"draws": 5, "population": 1397, "per_draw_seeds": None,
           "note": det["seedless_stack"], "harness_power_control": det["harness_power_control"]},
          open(sys.argv[6], "w"), indent=1)
print(f"  measured std over full draws: mae {std['mae']:g} arm {std['arm']:g} grip {std['grip']:g}")
print("  DETERMINISTIC + PROVENANCE OK" if ok else "  MISMATCH")
sys.exit(0 if ok else 1)
PY
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "determinism/provenance mismatch (rc=$rc)"
log "stage 3 PASS"

# ---------------------------------------------------------------- stage 4: score + significance
log "stage 4: score.py (whole board) then paired significance"
$PY "$BENCH/scripts/score/score.py" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "score.py rc=$rc"
$PY "$BENCH/scripts/score/paired_signif.py" --all 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "paired_signif.py rc=$rc"
log "stage 4 PASS (score.py and paired_signif.py both exited 0)"
fi   # end of stages 1-4 (skipped when ONLY_STAGE5=1)

if [ "$RUN_STAGE5" != "1" ]; then
  log "stage 5 DEFERRED (RUN_STAGE5=$RUN_STAGE5). The DELIVERY ROW IS COMPLETE -- stage 5 is a "\
"diagnostic footnote only. Releasing GPU $GPU. Resume later with: ONLY_STAGE5=1 bash $0 $GPU"
  log "=== OFT post-train: stages -1..4 done, stage 5 deferred ==="
  exit 0
fi

# ---------------------------------------------------------------- stage 5: generalization footnote
# Runs AFTER the delivery row is on disk, so a failure here cannot cost the board row.
# FULL populations on both sides (7475 train / 1397 val), no subsampling: a 400-anchor draw carries
# sd 0.0633 on b2, i.e. +-0.13 on a ratio -- the same order as the effect being measured.
log "stage 5: generalization ratio (full train split, 7475 anchors, same eval mode)"
$PY "$BENCH/archive/oft_build_split_anchors.py" --split train 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "train anchor build / val self-check rc=$rc"
$PY "$BENCH/scripts/predict/oft_predict.py" --ckpt "$CKPT" --anchors "$BENCH/logs/oft/train_anchors.npz" \
    --out "$TMP/pred_train_split.npz" 2>&1 | tail -3 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "train-split predict rc=$rc"
$PY - "$TMP/pred_train_split.npz" "$BENCH/logs/oft/train_anchors.npz" "$PREDS" \
      "$BENCH/data/val_anchors.npz" "$BENCH/logs/oft/trainsplit_eval_oft.json" "$BENCH" \
      <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, numpy as np, pathlib
BENCH = pathlib.Path(sys.argv[6]).resolve()
assert (BENCH / "data" / "val_anchors.npz").exists(), f"argv[6] is not the bench root: {BENCH}"
def stats(pred_f, anc_f, split):
    P = np.load(pred_f)["pred"].astype(np.float32); a = np.load(anc_f)
    K = int(a["K"]); gt = a["gt"].astype(np.float32); st = a["state"].astype(np.float32)
    P = P[:, :K, :]; e = np.abs(P - gt); h = np.abs(np.broadcast_to(st[:, None, :], gt.shape) - gt)
    # Per-anchor dump in lerobot's convention, so a SECOND implementation can recompute this
    # ratio (and the frame-weighted/episode-equal calibers) without re-running the 7B model.
    # Same keys ACT ships; anchor_idx is the row index into the anchor file, which IS the
    # delivery row order for val.
    np.savez_compressed(BENCH / f"logs/trainsplit_preds_openvla_oft_{split}.npz",
                        pred=P, gt=gt, state=st, episodes=a["episodes"].astype(np.int32),
                        frames=a["frames"].astype(np.int32),
                        anchor_idx=np.arange(len(gt), dtype=np.int32))
    return dict(n=int(len(gt)), mae=float(e.mean()), arm=float(e[..., :7].mean()),
                grip=float(e[..., 7].mean()), hold_mae=float(h.mean()),
                hold_arm=float(h[..., :7].mean()), hold_grip=float(h[..., 7].mean()))
tr, va = stats(sys.argv[1], sys.argv[2], "train"), stats(sys.argv[3], sys.argv[4], "val")
out = {"split_train": tr, "split_val": va,
       "generalization_ratio": {k: va[k] / tr[k] for k in ("mae", "arm", "grip")},
       "subsampling": "NONE -- full 7475 / 1397 populations on both sides",
       # State the weighting as a FIELD, not a convention (lerobot 2026-08-05): n15 measured
       # corr(episode length, hold MAE) = -0.62, so a per-frame vs per-episode reweighting moves
       # ratios by ~7% with zero randomness. Both splits here are unweighted means over anchors,
       # i.e. a long episode contributes proportionally more -- the same caliber lerobot reports
       # ACT in, which is what makes the two comparable at all.
       "weighting": "frame-length-weighted (unweighted mean over anchors), BOTH splits; "
                    "matches lerobot's ACT caliber. NOT recomputed episode-equal.",
       # Two ACT denominators, deliberately. 2.141x was the n=400 number visible when this leg's
       # prereg ("OFT r32 LoRA ratio < ACT") was written; 2.301x is lerobot's later full-population
       # value. The revision moved the bar in the direction that FAVOURS the prereg (gap over 1:
       # 1.141 -> 1.301, +15.5%), so beating only 2.301 is not a clean hit -- the prereg settles
       # against the number that was on the table when the bet was placed.
       "act_denominator_prereg": 2.141, "act_denominator_fullpop": 2.301,
       "caveat": "ACT's 2.141x is an n=400 subsample (sd 0.0633/side => ~+-0.13 on the ratio); "
                 "lerobot's full-population re-run gives 2.301x (subsampling biased it DOWN 7.5%). "
                 "Report against BOTH: 2.141 is the prereg-time bar, 2.301 is ACT's truth. AND ACT "
                 "was trained 21.0 epochs vs this run's ~9.0, so a ratio difference cannot be "
                 "attributed to LoRA/backbone alone -- epoch count is an uncontrolled confound."}
json.dump(out, open(sys.argv[5], "w"), indent=1)
print(f"  train {tr['mae']:.4f} (hold {tr['hold_mae']:.4f}) | val {va['mae']:.4f} "
      f"(hold {va['hold_mae']:.4f}) | ratio {out['generalization_ratio']['mae']:.3f}x "
      f"(arm {out['generalization_ratio']['arm']:.3f}x grip {out['generalization_ratio']['grip']:.3f}x)")
PY
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || die "generalization ratio computation rc=$rc"
log "stage 5 PASS"

log "=== all 6 stages exited 0; artifacts: $PREDS , $LOG ==="
log "=== this pid still holds GPU $GPU until it exits ==="
