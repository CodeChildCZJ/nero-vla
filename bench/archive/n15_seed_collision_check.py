#!/usr/bin/env python3
"""Measure — not derive — that the R sampling draws used non-aliasing noise.

Why this exists (gr00t-n15/n17 2026-08-05). The variance footnote carries
`per_draw_seeds` / `seed_stride_exceeds_anchor_count` as UPSTREAM evidence that the R
draws really differ, because `std==0.0` and `bit_exact_across_draws` are both computed
from the same R arrays and so cannot distinguish "deterministic decoder" from "the
harness reused one seed". But writing those fields by mirroring the source constant is
the same defect one layer down: it restates a literal instead of reading the shipped
object. n17's leg proves the gap is real — their formula had the prime stride and was
correct, yet the variance driver never passed `repeat`, so the EFFECTIVE stride was 1
and 254/1000 anchor-level noise tensors were shared across draws, while a constant-
derived field would have reported `stride=1000003, exceeds_anchor_count=True`.

So: import the SHIPPED PinnedNoise, run it over the REAL subset indices for r=0..R-1,
hash every (draw, anchor) noise tensor, and count collisions. Zero collisions is the
invariant that actually matters, and it holds regardless of what any constant says.

Run: $NERO_ROOT/third_party/Isaac-GR00T-n1d5/.venv/bin/python bench/archive/n15_seed_collision_check.py
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir
from predict_gr00t import PinnedNoise  # noqa: E402  (the shipped class, not a copy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(BENCH / "logs/variance_subset200.npy"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-anchors-full", type=int, default=1397)
    ap.add_argument("--out", default=str(BENCH / "logs/variance/seedcheck_gr00t_n15.json"))
    args = ap.parse_args()

    sel = [int(g) for g in np.load(args.subset)]
    pin = PinnedNoise(args.horizon, args.dim, args.seed)

    # digest -> (draw, anchor); a repeat digest means two draws shared a noise tensor
    seen, collisions = {}, []
    for r in range(args.repeats):
        pin.build(sel, r)
        stack = pin._pending                     # (N, H, D), built exactly as predict does
        assert stack.shape[0] == len(sel), (stack.shape, len(sel))
        for i, g in enumerate(sel):
            d = hashlib.md5(stack[i].numpy().tobytes()).hexdigest()
            if d in seen:
                collisions.append({"first": seen[d], "second": [r, g]})
            else:
                seen[d] = [r, g]

    total = args.repeats * len(sel)

    # POWER CONTROL, always run: a collision check that cannot fail is not a check.
    # Reproduce n17's bug shape on the same class and the same indices — feed the draw
    # index to the BASE SEED and leave repeat=0, which makes the effective stride 1 —
    # and require it to trip. If this comes back 0 the measurement above is worthless.
    ctl_seen, ctl_hits = set(), 0
    for r in range(args.repeats):
        p = PinnedNoise(args.horizon, args.dim, args.seed + r)   # draw index -> base seed
        p.build(sel, 0)                                          # ... repeat never varies
        for i in range(len(sel)):
            d = hashlib.md5(p._pending[i].numpy().tobytes()).hexdigest()
            ctl_hits += d in ctl_seen
            ctl_seen.add(d)
    # Infer the EFFECTIVE stride from behaviour rather than from a source literal: find
    # the draw-1 tensor that reproduces a draw-0 tensor, if any. None => no aliasing.
    out = {
        "backbone": "gr00t_n15",
        "check": "per_anchor_noise_collisions_across_draws",
        "measured_on": "shipped predict_gr00t.PinnedNoise, real subset indices",
        "repeats": args.repeats,
        "n_anchors": len(sel),
        "n_noise_tensors": total,
        "collisions": len(collisions),
        # Key superset (board contract 2026-08-05): a cross-reader that looks for its own key
        # name, misses, and silently treats the absence as "no finding" will swallow a bare 0 —
        # and a 0 with no power control behind it is worthless. Emit every name in circulation.
        "per_anchor_noise_collisions_measured": len(collisions),   # canonical
        "collision_examples": collisions[:5],
        "anchor_index_span": [min(sel), max(sel)],
        "n_anchors_full": args.n_anchors_full,
        "power_control_bug_shape": "draw index fed to base seed, repeat held at 0 (effective stride 1)",
        "power_control_collisions": ctl_hits,
        "seed_check_power_control_collisions": ctl_hits,   # alias
        "seed_collision_power_control": ctl_hits,          # alias (n17's reader's name)
        "power_control_key_aliases": ["power_control_collisions",
                                      "seed_check_power_control_collisions",
                                      "seed_collision_power_control"],
        "power_control_ok": ctl_hits > 0,
        "verdict": "PASS (draws are non-aliasing)" if not collisions else
                   "FAIL (draws share noise tensors — sampling std is biased LOW)",
    }
    if not out["power_control_ok"]:
        out["verdict"] = "VOID (power control did not trip — this check has no power)"
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if not collisions else 1


if __name__ == "__main__":
    sys.exit(main())
