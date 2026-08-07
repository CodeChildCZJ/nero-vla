#!/usr/bin/env python3
"""Build anchors for either split from data/b2_oft/traj.npz, mirroring build_val_anchors.py.

Purpose: the generalization ratio (train MAE / val MAE, same eval mode, same anchor construction)
is the only honest read on overfitting -- a train-mode running loss confounds the regime.

Two things this file exists to get right:

1. **The rebuild is validated against the frozen artifact before it is trusted.** Running with
   --split val must reproduce val_anchors.npz bit-for-bit on all four arrays, or it exits nonzero.
   A train-split number produced by unvalidated mirror logic is uninterpretable.

2. **No subsampling.** `lerobot_train_split_eval.py` defaults to --n 400, and on b2 a single
   400-anchor draw moves ACT's val MAE by +-0.124 at 95% (sd 0.0633 over 4000 draws) = 18% of the
   real N1.5-vs-ACT gap; propagated to a val/train ratio that is +-0.13 on 2.141. The full train
   split is 7475 anchors -- ~32 min of predict, which is 4.7% of an 11.4 h training run and buys
   an exactly-zero-noise denominator. Cheaper to pay it than to argue about it afterwards.
   (The hold-state line does NOT rescue a subsample: corr(hold_sub, model_sub) = +0.43 over 4000
   draws, so it explains only ~18% of the variance -- weak evidence of representativeness, not
   none, but nowhere near enough to correct a ratio with.)

Note the pass line is split-dependent: hold-state over the full train anchors is 3.0799
(arm 2.6384 / grip 6.1704) vs val's 3.0720 (arm 2.5352 / grip 6.8295) -- so judge a train MAE
against 3.0799, never against 3.072. That the two agree to 0.26% is itself the finding that the
val split is not intrinsically harder; the arm and gripper channels disagree in opposite
directions (train arm harder, train gripper easier) and cancel in the 7+1 weighted mean.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parent.parent
K = 8
STRIDE = 5


def build(traj, eps):
    """Byte-for-byte the logic of scripts/prepare/build_val_anchors.py, over the given episodes."""
    E, F, G, S = [], [], [], []
    for ep in sorted(eps):
        act = traj[f"ep{ep:03d}_action"]
        st = traj[f"ep{ep:03d}_state"]
        for t in range(0, len(act) - K, STRIDE):
            E.append(ep)
            F.append(t)
            G.append(act[t : t + K])
            S.append(st[t])
    return (np.array(E), np.array(F),
            np.stack(G).astype(np.float32), np.stack(S).astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--data-root", default=str(BENCH / "data" / "b2_oft"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    split = json.loads((BENCH / "data" / "split.json").read_text())
    tf = np.load(pathlib.Path(args.data_root) / "traj.npz")
    traj = {k: tf[k] for k in tf.files}

    # ---- tool validation FIRST, always, regardless of which split was asked for ----
    A = np.load(BENCH / "data" / "val_anchors.npz")
    ve, vf, vg, vs = build(traj, split["val_episodes"])
    checks = {"episodes": np.array_equal(ve, A["episodes"]),
              "frames": np.array_equal(vf, A["frames"]),
              "gt": np.array_equal(vg, A["gt"]),
              "state": np.array_equal(vs, A["state"])}
    print(f"[validate] rebuild vs frozen val_anchors.npz: {checks}")
    if not all(checks.values()):
        sys.exit("FAIL: mirror logic does not reproduce the frozen anchors; any split number "
                 "built with it is uninterpretable")
    print("[validate] PASS -- reproduces the frozen harness bit-for-bit")

    eps = split[f"{args.split}_episodes"]
    e, f, g, s = build(traj, eps)
    hold = np.abs(np.broadcast_to(s[:, None, :], g.shape) - g)
    print(f"[{args.split}] {len(e)} anchors over {len(eps)} episodes, K={K} stride={STRIDE}")
    print(f"[{args.split}] hold-state line: mae {hold.mean():.4f}  "
          f"arm {hold[..., :7].mean():.4f}  grip {hold[..., 7].mean():.4f}")

    out = args.out or str(BENCH / "logs" / "oft" / f"{args.split}_anchors.npz")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, episodes=e, frames=f, gt=g, state=s, K=K, stride=STRIDE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
