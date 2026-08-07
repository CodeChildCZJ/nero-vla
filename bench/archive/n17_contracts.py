#!/usr/bin/env python3
"""Gates that every N1.7 script must pass before it produces a board number.

Two independent contracts live here, both of them team-lead board pins, and both
of them written ONCE so no second copy of a constant can drift (this stack has
already been bitten twice by two-source splits: config-object vs checkpoint-config,
and the action_horizon 50-vs-40 registry clash).

  1. `load_val_anchors` -- gate the shared anchor file on IDENTITY, not existence.
  2. `normalization_contract` -- read the flags the floor number's *label* depends
     on, instead of asserting the assumption in a docstring.

Both raise SystemExit on violation. Neither has an override flag: a mismatch here
means the board rows are not comparable, which is a stop-the-line condition.
"""
import hashlib
import pathlib

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]

# Frozen 2026-08-05 00:03:53, 414998 bytes. Pinned by team-lead across all legs:
# every leg's preds must be row-aligned to the SAME anchors or the leaderboard is
# comparing different questions. Same discipline as variance_subset200.npy's md5
# gate in variance_gr00t_n17.py -- gate identity, because "the file exists" is
# satisfied by a regenerated file with a different anchor order.
EXPECTED_ANCHORS_MD5 = "cc34dda2a99ade1011c192e2b3e4e343"

# MEASURED off this leg's own checkpoints, not chosen. The N1.7 recipe sets
# `use_percentiles=True`, so normalization bounds are q01/q99 rather than true
# min/max -- and the published gripper floor already reflects that (0.1326 = 0.0144
# irreducible OOD + 0.1182 self-inflicted q99 truncation; j7 true_max 94.318 vs q99
# 85.995). So this is a CHANGE DETECTOR, not a claim that True is the right value:
# the invariant being protected is that the floor and the preds come from the SAME
# normalization configuration. The first version of this gate refused True outright
# and was caught by the pre-flight predict-gate refusing a real checkpoint -- a
# must-PASS control failing, which is the direction that a fail-closed gate hides
# until it blocks production.
EXPECTED_USE_PERCENTILES = True


def load_val_anchors(path=None):
    """np.load(val_anchors.npz) after checking its md5 against the board pin.

    Returns the loaded npz. Raises SystemExit if the file is missing or is a
    different file than the one the rest of the board scored against.
    """
    p = pathlib.Path(path) if path else BENCH / "data" / "val_anchors.npz"
    if not p.exists():
        raise SystemExit(f"[FATAL] val_anchors not found at {p}")
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    if md5 != EXPECTED_ANCHORS_MD5:
        raise SystemExit(
            f"[FATAL] {p} md5 {md5} != board pin {EXPECTED_ANCHORS_MD5}. The anchor "
            f"set was regenerated or edited; every preds_*.npz on the board is "
            f"row-indexed to the pinned file, so this run would produce rows that "
            f"silently do not line up with the others. Restore the pinned file or "
            f"escalate -- a mismatch is a whole-board rerun, not a local problem.")
    print(f"val_anchors: {p} (md5 {md5}, board pin OK)")
    return np.load(p)


def normalization_contract(modality, sap):
    """Measure -- not assume -- the three flags the floor number's KIND rests on.

    The published K=8 floor (0.0176) is a round-trip: absolute action ->
    apply_action -> unapply_action -> compare. It was labelled HARD on the strength
    of a docstring claiming "unnormalize_values_minmax clips to [-1,1]
    unconditionally". Reading the shipped source, that sentence is wrong three ways:

      * WHERE   the clip is in `apply_action` (state_action_processor.py:414) and
                `apply_state` (:263) -- the FORWARD direction. `unapply_action`
                (:421-480) contains no clip at all.
      * WHETHER it is `if self.clip_outliers:`, a constructor flag serialized into
                the checkpoint. Class default is True, so nothing ever complained.
      * WHICH   `unapply_action` picks per key between the meanstd and minmax
                inverses on `mean_std_embedding_keys` (:463-469); the floor math
                hardcodes minmax.

    The measured floor is still whatever the checkpoint's own flags produce -- the
    number is empirical. What is unverified is its LABEL: with clip_outliers=False
    the round trip is float32-lossless and there is no hard floor to report, and
    with use_percentiles=True the bounds become q01/q99, which clips ~2.7% of
    gripper frames and moves the floor's magnitude (the same q99-as-bounds unit
    error openvla-oft measured landing only 0.63 above the pass line -- silently
    rankable, invisible to a do-nothing flag).

    Same family as the trap oft found in `_unnormalize_actions`' `mask` argument:
    one shipped inverse whose behaviour is selected by a config field nobody reads.

    Attributes are read WITHOUT a getattr default on purpose. A default would make
    an upstream rename fall through to the value that happens to pass, which is the
    exact silent-pass this function exists to prevent.
    """
    ac = modality["action"]
    msk = getattr(ac, "mean_std_embedding_keys", None) or ()
    branches = {k: ("meanstd" if k in msk else "minmax") for k in ac.modality_keys}
    clip = bool(sap.clip_outliers)        # AttributeError if upstream renames it
    pct = bool(sap.use_percentiles)       # ditto -- loud beats silently-passing
    contract = {
        "branches": branches,
        "clip_outliers": clip,
        "use_percentiles": pct,
        "bounds_source": "q01/q99" if pct else "min/max",
        "floor_kind": "HARD" if clip else "SOFT",
    }

    off = sorted(k for k, v in branches.items() if v != "minmax")
    if off:
        raise SystemExit(
            f"action keys {off} take the meanstd inverse branch, but this leg's floor "
            f"math (n17_unit_roundtrip.py) assumes minmax. The published floor would "
            f"describe a code path the delivery did not run.")
    if not clip:
        raise SystemExit(
            "checkpoint has clip_outliers=False: apply_action does NOT clip to "
            "[-1,1], so the round trip is float32-lossless and the floor published "
            "as HARD would not exist. Re-derive the floor's kind before shipping.")
    if pct != EXPECTED_USE_PERCENTILES:
        raise SystemExit(
            f"checkpoint has use_percentiles={pct}, this leg's floor was measured "
            f"under {EXPECTED_USE_PERCENTILES}. A preds file and a floor computed "
            f"under different normalization bounds are not comparable, and the "
            f"gripper floor is ~89% q99-truncation so the change would be large.")
    return contract
