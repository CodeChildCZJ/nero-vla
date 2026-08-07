#!/usr/bin/env python3
"""Two-sided train/val out-of-bounds audit for the b2 action space.

Written because openvla-oft caught that the out-of-range joint set the rest of us
were circulating ({j2, j5, j7}) came from a ONE-SIDED test (`val_max > train_max`).
A joint that only breaks the LOWER bound is invisible to that test, and one such
joint (j6) turned out to carry the second-largest absolute-space clip penalty.

Reports, per dimension and per side:
  - train [min,max] over the 111 train episodes (all frames)
  - val   [min,max] over the two populations that matter
  - counts + fraction of val values outside train range, LOW and HIGH separately
  - the absolute-space clip floor each side would cost
and, for every dimension with lower-side violations, HOW THOSE VIOLATIONS ARE
DISTRIBUTED OVER VAL EPISODES -- the question oft actually asked: a few odd
trajectories reads very differently from a broad distribution shift.

Two populations, because they answer different questions:
  ANCHORS  the (1397, K, 8) gt tensor score.py actually grades  -> leaderboard-relevant
  ALLVAL   every frame of the 20 val episodes                   -> distribution-shift claim

CPU-only, no model, no GPU.
"""
import os
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")))


def ep_parquet(ep: int) -> Path:
    return SOURCE / f"data/chunk-000/episode_{ep:06d}.parquet"


def load_actions(eps):
    """(N, 8) float32 raw absolute action rows, concatenated over episodes."""
    out = []
    for ep in eps:
        df = pd.read_parquet(ep_parquet(ep), columns=["action"])
        out.append(np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]]))
    return np.concatenate(out, axis=0)


def load_actions_by_ep(eps):
    return {ep: load_actions([ep]) for ep in eps}


def main():
    split = json.loads((BENCH / "data" / "split.json").read_text())
    train_eps = split["train_episodes"]
    val_eps = split["val_episodes"]
    assert not set(train_eps) & set(val_eps)
    print(f"train {len(train_eps)} eps | val {len(val_eps)} eps")

    tr = load_actions(train_eps)
    lo, hi = tr.min(axis=0), tr.max(axis=0)
    D = tr.shape[1]
    print(f"train action rows {tr.shape}")

    A = np.load(BENCH / "data" / "val_anchors.npz")
    anchors = A["gt"].reshape(-1, D)  # (1397*K, 8) -- exactly what score.py grades
    per_ep = load_actions_by_ep(val_eps)
    allval = np.concatenate([per_ep[e] for e in val_eps], axis=0)

    for name, V in (("ANCHORS(gt score.py grades)", anchors), ("ALLVAL(every val frame)", allval)):
        print(f"\n===== {name}: {V.shape[0]} rows/dim =====")
        print(f"{'dim':>4} {'train[min,max]':>26} {'val[min,max]':>26} "
              f"{'#low':>6} {'#high':>6} {'oob%':>6} {'floorLO':>9} {'floorHI':>9} {'floor':>9}")
        tot = 0.0
        for j in range(D):
            v = V[:, j]
            nlo = int((v < lo[j]).sum())
            nhi = int((v > hi[j]).sum())
            # clip floor = MAE contribution of clamping val into the train box
            f_lo = float(np.clip(lo[j] - v, 0, None).mean())
            f_hi = float(np.clip(v - hi[j], 0, None).mean())
            tot += (f_lo + f_hi) / D
            flag = "  <-- LOWER-side only" if nlo and not nhi else ""
            print(f"{j:>4} [{lo[j]:>10.3f},{hi[j]:>10.3f}] [{v.min():>10.3f},{v.max():>10.3f}] "
                  f"{nlo:>6} {nhi:>6} {100*(nlo+nhi)/len(v):>5.2f}% "
                  f"{f_lo:>9.4f} {f_hi:>9.4f} {f_lo+f_hi:>9.4f}{flag}")
        print(f"{'':>4} total absolute-space clip floor (mean over dims): {tot:.4f}")

    # --- the question oft asked: are the violations concentrated? ---------------
    # Run BOTH sides: if one episode drove only the lower side we would call it an
    # odd trajectory, but if the same episode also drives the upper side then the
    # whole absolute-space OOB penalty is one episode, not a distribution shift.
    print("\n===== violation concentration over val episodes (ALLVAL) =====")
    for side in ("LOWER", "UPPER"):
      for j in range(D):
        if side == "LOWER":
            viol = {e: int((per_ep[e][:, j] < lo[j]).sum()) for e in val_eps}
        else:
            viol = {e: int((per_ep[e][:, j] > hi[j]).sum()) for e in val_eps}
        n = sum(viol.values())
        if n == 0:
            continue
        print(f"\n[{side}]", end=" ")
        hit = {e: c for e, c in viol.items() if c}
        share = sorted(hit.values(), reverse=True)
        top = share[0] / n
        top3 = sum(share[:3]) / n
        print(f"dim {j}: {n} {side}-side violations across {len(hit)}/{len(val_eps)} val episodes")
        print(f"  top-1 episode holds {100*top:.1f}%, top-3 hold {100*top3:.1f}%")
        for e, c in sorted(hit.items(), key=lambda kv: -kv[1]):
            frac_of_ep = c / len(per_ep[e])
            worst = (float(lo[j] - per_ep[e][:, j].min()) if side == "LOWER"
                     else float(per_ep[e][:, j].max() - hi[j]))
            print(f"    ep{e:>4}: {c:>5} frames ({100*frac_of_ep:>5.1f}% of episode), "
                  f"max overshoot {worst:>7.3f}")
        verdict = ("CONCENTRATED (few odd trajectories)" if top3 > 0.8
                   else "BROAD (distribution shift)")
        print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
