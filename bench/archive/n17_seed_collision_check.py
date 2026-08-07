#!/usr/bin/env python3
"""Measure -- not declare -- that the 5 variance draws draw non-aliasing noise.

Why this file exists (gr00t-n15 2026-08-05, adopted board-wide by team-lead):
`variance_gr00t_n17.py` reports `per_draw_seeds`, `per_draw_seed_stride` and
`seed_stride_exceeds_anchor_count`. All three are DERIVED FROM CONSTANTS --
the first two from BASE_SEED/STRIDE arithmetic, the third from
`PinnedNoise.STRIDE > 1397`. Reading STRIDE off the class instead of retyping it
removes a two-source split, but it does NOT make the field evidence:
`PinnedNoise.STRIDE` was already 1000003 on 2026-08-04, when the caller passed
the draw index in as the BASE SEED and the effective stride was 1. So the field
would have printed `stride=1000003 / exceeds=True` on the exact code that reused
254 of 1000 (draw, anchor) noise tensors. **The bug was never in the constant.
It was in the caller.** A field computed from the constant cannot see it.

So this check has two layers, and neither reads a constant:

  A. MEASURED. Import the SHIPPED `PinnedNoise` (not a copy -- a copy proves
     nothing about what ships), drive it over the REAL 200-anchor subset for
     r=0..4, md5 all 1000 produced noise tensors, count collisions. Power
     control, always on: the same shipped class driven through the 2026-08-04
     call shape must come back NONZERO. A check that cannot fail reports 0
     collisions for free.

  B. STRUCTURAL. Layer A still encodes the call convention in THIS file, so if
     the caller regresses, layer A keeps testing the old (correct) convention
     and stays green. So parse the SHIPPED `variance_gr00t_n17.py` and assert
     the invariant that actually distinguishes the two versions: the loop
     variable `repeat` reaches the seed through `build(repeat=...)` and does NOT
     appear in the `PinnedNoise(...)` constructor argument. Power control: the
     same audit run against the 2026-08-04 call shape must FAIL.

Shape note: collisions are counted on tensor bytes, not on seeds, so the answer
could in principle depend on the prior's shape. The real prior is (1, 40, 132)
but that is learned at runtime by probing the model, and this check holds no
GPU and loads no model -- so instead of hardcoding it, run TWO shapes and
require the collision count to agree. Shape-invariance measured, not assumed.

CPU only, no model, no GPU, ~15 s. Output is a JSON that
`variance_gr00t_n17.py` refuses to run without.
"""
import argparse
import ast
import hashlib
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = BENCH / "scripts" / "predict"
EXPECTED_SUBSET_MD5 = "9007df1fcec5eca2a1f01591c6266b43"
N_ANCHORS_FULL = 1397

sys.path[:0] = [str(SCRIPTS), str(BENCH / "archive")]   # was one flat scripts/ dir
from predict_gr00t_n17 import PinnedNoise  # noqa: E402  the SHIPPED class


def md5_file(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Layer A -- measured collisions
# --------------------------------------------------------------------------
def draw_all(anchors, draws, shape, call_shape):
    """Produce every (draw, anchor) noise tensor through the SHIPPED PinnedNoise.

    call_shape="current": one pinner seeded BASE_SEED=0, draw index enters via
                          build(repeat=r)   -- what variance_gr00t_n17.py does now.
    call_shape="v20260804": a fresh pinner seeded with the draw index, build()
                          left at its repeat=0 default -- the shipped bug shape.
    """
    digests = []
    if call_shape == "current":
        pin = PinnedNoise(0)
        pin.want = shape
        for r in draws:
            for g in anchors:
                pin.build(int(g), repeat=r)
                digests.append(hashlib.md5(pin._pending.numpy().tobytes()).hexdigest())
                pin._pending = None
    elif call_shape == "v20260804":
        for r in draws:
            pin = PinnedNoise(r)          # <-- draw index as the BASE seed
            pin.want = shape
            for g in anchors:
                pin.build(int(g))         # <-- repeat left at its default 0
                digests.append(hashlib.md5(pin._pending.numpy().tobytes()).hexdigest())
                pin._pending = None
    elif call_shape == "draw_dropped":
        # my single likeliest slip: the pinner is built ONCE outside the draw
        # loop and `repeat=` is a one-token kwarg on the build call. Drop it and
        # every draw replays the same 200 tensors.
        pin = PinnedNoise(0)
        pin.want = shape
        for r in draws:
            for g in anchors:
                pin.build(int(g))         # <-- repeat kwarg omitted
                digests.append(hashlib.md5(pin._pending.numpy().tobytes()).hexdigest())
                pin._pending = None
    elif call_shape == "anchor_dropped":
        pin = PinnedNoise(0)
        pin.want = shape
        for r in draws:
            for _ in anchors:
                pin.build(0, repeat=r)    # <-- anchor index never folded in
                digests.append(hashlib.md5(pin._pending.numpy().tobytes()).hexdigest())
                pin._pending = None
    else:
        raise ValueError(call_shape)
    return digests


def delivery_draw(n_all, seed, shape, call_shape, local_idx=None):
    """Layer C: the DELIVERY call shape, i.e. what predict_gr00t_n17.py does.

    predict makes ONE pass (no draw dimension) and therefore legitimately calls
    `build(i)` with `repeat` at its default -- a shape that layer B's variance
    contract would reject. The property that matters for the shipped npz is not
    "the draw index is folded in" (there is no draw index) but "the 1397 noise
    tensors are pairwise DISTINCT", which is a direct measurement.
    """
    pin = PinnedNoise(seed)
    pin.want = shape
    idx = range(n_all) if call_shape == "current" else local_idx
    digests = []
    for i in idx:
        pin.build(int(i))
        digests.append(hashlib.md5(pin._pending.numpy().tobytes()).hexdigest())
        pin._pending = None
    return digests


def collisions(digests):
    c = Counter(digests)
    reused = sum(n - 1 for n in c.values() if n > 1)
    return {
        "n_noise_tensors": len(digests),
        "n_distinct": len(c),
        "collisions": reused,
        "max_multiplicity": max(c.values()) if c else 0,
    }


# --------------------------------------------------------------------------
# Layer B -- structural audit of the SHIPPED caller
# --------------------------------------------------------------------------
def audit_caller(src):
    """Does the draw index reach the seed through build(repeat=) or the ctor?

    Returns (ok, detail). Operates on source text so it can be pointed at the
    shipped file AND at a synthetic bug-shaped string (the power control).
    """
    tree = ast.parse(src)
    ctors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "PinnedNoise"]
    builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "build"]

    def names_in(node):
        return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}

    ctor_args = [sorted(names_in(a)) for c in ctors for a in c.args]
    ctor_mentions_repeat = any("repeat" in a for a in ctor_args)
    builds_pass_repeat = [
        any(kw.arg == "repeat" and "repeat" in names_in(kw.value) for kw in b.keywords)
        for b in builds
    ]
    ok = (
        len(ctors) == 1
        and len(builds) >= 1
        and not ctor_mentions_repeat
        and all(builds_pass_repeat)
    )
    return ok, {
        "n_constructors": len(ctors),
        "constructor_arg_names": ctor_args,
        "constructor_arg_mentions_repeat": ctor_mentions_repeat,
        "n_build_calls": len(builds),
        "build_calls_pass_repeat_kwarg": builds_pass_repeat,
    }


BUGGY_CALLER_SRC = (
    "for repeat in args.seeds:\n"
    "    pinner = PinnedNoise(repeat)\n"
    "    for i in sel:\n"
    "        pinner.build(int(i))\n"
)
GOOD_CALLER_SRC = (
    "pinner = PinnedNoise(BASE_SEED)\n"
    "for repeat in args.seeds:\n"
    "    for i in sel:\n"
    "        pinner.build(int(i), repeat=repeat)\n"
)


def main():
    ap = argparse.ArgumentParser()
    # openpi's dry-run rule (board, 2026-08-05): a script that writes a result
    # artifact takes --out so a pre-flight cannot overwrite the production file
    # with stand-in numbers, and records every input path + md5 it consumed.
    ap.add_argument("--out", default=str(BENCH / "logs/variance/seedcheck_gr00t_n17.json"))
    ap.add_argument("--subset", default=str(BENCH / "logs/variance_subset200.npy"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    # ---- inputs (identity-gated, not existence-gated) ----
    sub = pathlib.Path(args.subset)
    cands = [sub, BENCH / "logs/variance" / sub.name, BENCH / "logs" / sub.name]
    sub_path = sub_md5 = None
    for c in cands:
        if c.is_file() and md5_file(c) == EXPECTED_SUBSET_MD5:
            sub_path, sub_md5 = c, EXPECTED_SUBSET_MD5
            break
    if sub_path is None:
        sys.exit("[FATAL] board-shared 200-anchor subset not found with md5 "
                 f"{EXPECTED_SUBSET_MD5}; tried {[str(c) for c in cands]}. "
                 "Counting collisions on the wrong anchor set answers a question "
                 "nobody asked.")
    anchors = [int(x) for x in np.load(sub_path)]
    print(f"subset: {sub_path} (md5 {sub_md5}, {len(anchors)} anchors)")

    caller_path = BENCH / "archive" / "variance_gr00t_n17.py"
    pred_path = SCRIPTS / "predict_gr00t_n17.py"

    # ---- layer A: measured collisions, two shapes ----
    shapes = [(1, 40, 132), (1, 8, 16)]
    per_shape, ctrl_per_shape = {}, {}
    for shp in shapes:
        per_shape[str(shp)] = collisions(draw_all(anchors, args.seeds, shp, "current"))
        ctrl_per_shape[str(shp)] = collisions(draw_all(anchors, args.seeds, shp, "v20260804"))
        print(f"  shape {shp}: current {per_shape[str(shp)]['collisions']} collisions | "
              f"power control (v20260804 call shape) {ctrl_per_shape[str(shp)]['collisions']}")

    # ---- layer A2: anchor-set-IMMUNE power controls (lerobot-setup, 2026-08-05) ----
    # The stride-1 control above collides 254 times only because THIS anchor set
    # has minimum inter-anchor gap 1; on a sparse anchor set the same defect would
    # collide zero times. My verdict is fail-closed against that (power_ok demands
    # >0, so an innocent 0 aborts rather than passes), but ">0" is a loose gate: it
    # is satisfied by 1 collision as well as by 800. These two defects instead
    # collide a number fixed by CONSTRUCTION on any anchor set whatsoever, so the
    # expected count is a derived constant and the gate can be an equality.
    D, A_ = len(args.seeds), len(anchors)
    imm_spec = {"draw_dropped": (D - 1) * A_, "anchor_dropped": D * (A_ - 1)}
    immune = {}
    for name, expect in imm_spec.items():
        got = collisions(draw_all(anchors, args.seeds, shapes[0], name))["collisions"]
        immune[name] = {"collisions": got, "expected_by_construction": expect,
                        "ok": got == expect}
        print(f"  immune control {name}: {got} collisions "
              f"(construction requires {expect}) {'OK' if got == expect else 'BROKEN'}")
    immune_ok = all(v["ok"] for v in immune.values())

    shape_invariant = len({v["collisions"] for v in per_shape.values()}) == 1 and \
                      len({v["collisions"] for v in ctrl_per_shape.values()}) == 1
    measured = per_shape[str(shapes[0])]
    control = ctrl_per_shape[str(shapes[0])]
    power_ok = control["collisions"] > 0

    # ---- layer C: the DELIVERY path (team-lead's addendum, 2026-08-05) ----
    # Layers A/B cover the PinnedNoise class and the VARIANCE caller. The shipped
    # npz is written by a different caller with a different, also-correct contract
    # (one pass, no repeat kwarg), so answering "does the check drive the
    # production entry point?" with "the variance run loads a real policy" would
    # be answering about the wrong caller. Measure the delivery path's own
    # property directly: are the 1397 noise tensors pairwise distinct?
    anc = np.load(BENCH / "data" / "val_anchors.npz")
    ep_all = anc["episodes"]
    n_all = len(ep_all)
    seen = Counter()
    local_idx = []                     # within-episode ordinal per anchor
    for e in ep_all:
        local_idx.append(seen[int(e)])
        seen[int(e)] += 1
    dlv = collisions(delivery_draw(n_all, 0, shapes[0], "current"))
    dlv_ctrl = collisions(delivery_draw(n_all, 0, shapes[0], "local", local_idx))
    dlv_ok = dlv["collisions"] == 0 and dlv_ctrl["collisions"] > 0
    print(f"  delivery path ({n_all} anchors, predict call shape): "
          f"{dlv['collisions']} collisions | power control (within-episode index "
          f"instead of global) {dlv_ctrl['collisions']}")

    # ---- layer B: structural audit of the shipped caller ----
    caller_ok, caller_detail = audit_caller(caller_path.read_text())
    ctrl_bad_ok, _ = audit_caller(BUGGY_CALLER_SRC)     # must be False
    ctrl_good_ok, _ = audit_caller(GOOD_CALLER_SRC)     # must be True
    audit_power_ok = (ctrl_bad_ok is False) and (ctrl_good_ok is True)
    print(f"  caller audit: shipped {'PASS' if caller_ok else 'FAIL'} | "
          f"power control bug-shape={'FAIL (correct)' if not ctrl_bad_ok else 'PASS (BROKEN)'} "
          f"good-shape={'PASS' if ctrl_good_ok else 'FAIL (BROKEN)'}")

    verdict = (measured["collisions"] == 0 and power_ok and shape_invariant
               and caller_ok and audit_power_ok and immune_ok and dlv_ok)
    out = {
        "leg": "gr00t_n17",
        "verdict": "PASS (draws are non-aliasing)" if verdict else "FAIL",
        "measured_not_declared": True,
        # layer A
        "per_anchor_noise_collisions_measured": measured["collisions"],
        "n_noise_tensors": measured["n_noise_tensors"],
        "n_distinct_noise_tensors": measured["n_distinct"],
        # Board rule (team-lead 2026-08-05): emit the UNION of every leg's power
        # control key name. A reader that knows only its own name takes the bare
        # measured 0 and never learns whether it was powered -- which is the one
        # thing the measurement was added to establish. I made that mistake on
        # n15's file, so the fix belongs on the write side too, not only mine.
        **{k: control["collisions"] for k in (
            "power_control_collisions",              # generic
            "seed_collision_power_control",          # gr00t_n17 (mine)
            "seed_check_power_control_collisions",   # gr00t_n15
            "collision_power_control",               # lerobot-setup
        )},
        "power_control_key_aliases": [
            "power_control_collisions", "seed_collision_power_control",
            "seed_check_power_control_collisions", "collision_power_control",
        ],
        "power_control_ok": power_ok,
        "power_control_call_shape": "v20260804: PinnedNoise(draw_index) + build(idx) "
                                    "with repeat left at its default -> effective stride 1",
        "power_control_anchor_set_dependent": True,
        "power_control_anchor_set_note": (
            "254 is a property of THIS anchor set (min inter-anchor gap 1), not of "
            "the defect: on a sparse anchor set effective-stride-1 collides zero "
            "times. That reads as FAIL here, not as a false PASS (power_ok demands "
            ">0), but it is a loose gate -- hence the two construction-fixed "
            "controls below, whose expected counts hold on ANY anchor set "
            "(lerobot-setup, 2026-08-05)."
        ),
        "anchor_set_immune_power_controls": immune,
        "anchor_set_immune_power_controls_ok": immune_ok,
        "collision_count_shape_invariant": shape_invariant,
        "per_shape": {"current": per_shape, "power_control": ctrl_per_shape},
        # layer C -- delivery path
        "delivery_path": {
            "caller": str(pred_path),
            "n_anchors": n_all,
            "collisions_measured": dlv["collisions"],
            "n_distinct": dlv["n_distinct"],
            "power_control_within_episode_index": dlv_ctrl["collisions"],
            "ok": dlv_ok,
            "why_a_separate_layer": (
                "predict makes ONE pass and correctly calls build(i) with repeat at "
                "its default, so layer B's variance contract (every build passes "
                "repeat=) would REJECT the shipped delivery caller. The delivery "
                "property is pairwise distinctness over 1397, measured here."
            ),
            "structural_pin_on_the_global_index": (
                "predict_gr00t_n17.py asserts np.allclose(got, ref_state[i]) with the "
                "SAME i it passes to build(), and ref_state is the full 1397-row "
                "anchor array -- so a within-episode index would fire the state "
                "assert long before it could alias noise."
            ),
        },
        # layer B
        "caller_audit_ok": caller_ok,
        "caller_audit_detail": caller_detail,
        "caller_audit_power_control_ok": audit_power_ok,
        # provenance (openpi dry-run rule)
        "inputs": {
            "subset": {"path": str(sub_path), "md5": sub_md5, "n": len(anchors)},
            "caller": {"path": str(caller_path), "md5": md5_file(caller_path)},
            "pinner": {"path": str(pred_path), "md5": md5_file(pred_path),
                       "class": "PinnedNoise (imported, not copied)"},
        },
        "seeds": list(args.seeds),
        "n_anchors_full": N_ANCHORS_FULL,
        "note_not_cross_validation": (
            "If this power control returns the same number as another leg's, that is "
            "NOT independent corroboration: under effective stride 1 the collision "
            "count is a deterministic function of the shared anchor index set "
            "(pairs with r+g == r'+g'), independent of model, stack and author "
            "(gr00t-n15). It shows both legs reproduced the same bug SHAPE."
        ),
    }
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n{out['verdict']}  ->  {outp}")
    if not verdict:
        sys.exit(1)


if __name__ == "__main__":
    main()
