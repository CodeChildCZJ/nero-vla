#!/usr/bin/env python3
"""How concentrated is the leaderboard metric itself?

Motivation: the two-sided OOB audit showed the ENTIRE absolute-space out-of-range
penalty comes from 4 of the 20 val episodes. That raises a separate question the
board needs answered before anyone reads a small MAE gap as a backbone difference:
**how many episodes actually decide the number?**

This computes the hold-state pass line (the delta=0 predictor, which is what
score.py compares against) per val episode, then reports:
  - each episode's share of the total MAE
  - the effective sample size  n_eff = (sum w)^2 / sum(w^2)  over per-episode means
  - leave-one-episode-out spread of the pass line
  - the same restricted to the 4 OOB-driving episodes

No model, no GPU, and it does NOT touch score.py (frozen harness) -- it recomputes
the pass line from val_anchors.npz with the same definition score.py uses.
"""
import os
import json
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
OOB_EPS = [2, 37, 51, 90]  # the episodes that own every train-box violation


def main():
    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, gt, state = A["episodes"], A["gt"], A["state"]
    K = int(A["K"])
    gt = gt[:, :K, :]

    # hold-state predictor: repeat the current state for the whole chunk
    hold = np.repeat(state[:, None, :], K, axis=1)
    err = np.abs(hold - gt)                     # (N, K, 8)
    per_anchor = err.mean(axis=(1, 2))          # (N,)
    total = per_anchor.mean()
    arm = err[:, :, :7].mean()
    grip = err[:, :, 7].mean()
    print(f"anchors {len(eps)} | K={K}")
    print(f"hold-state pass line: MAE {total:.4f}  arm {arm:.4f}  grip {grip:.4f}")

    uniq = sorted(set(int(e) for e in eps))
    counts = np.array([(eps == e).sum() for e in uniq])
    means = np.array([per_anchor[eps == e].mean() for e in uniq])
    # contribution to the anchor-weighted overall mean
    contrib = counts * means
    share = contrib / contrib.sum()

    print(f"\n{'ep':>5} {'anchors':>8} {'mean MAE':>10} {'share of total':>15}")
    for e, c, m, s in sorted(zip(uniq, counts, means, share), key=lambda t: -t[3]):
        mark = "  <-- OOB driver" if e in OOB_EPS else ""
        print(f"{e:>5} {c:>8} {m:>10.4f} {100*s:>14.2f}%{mark}")

    # effective sample size over episodes (anchor-weighted)
    w = contrib / contrib.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    print(f"\neffective #episodes deciding the metric: n_eff = {n_eff:.2f} of {len(uniq)}")
    print(f"top-1 episode share {100*share.max():.2f}% | "
          f"top-4 {100*np.sort(share)[-4:].sum():.2f}%")

    # leave-one-episode-out
    loo = []
    for e in uniq:
        keep = eps != e
        loo.append(per_anchor[keep].mean())
    loo = np.array(loo)
    print(f"\nleave-one-episode-out pass line: min {loo.min():.4f} max {loo.max():.4f} "
          f"spread {loo.max()-loo.min():.4f} ({100*(loo.max()-loo.min())/total:.2f}% of the line)")
    worst = uniq[int(np.argmax(np.abs(loo - total)))]
    print(f"  most influential single episode: ep{worst} "
          f"(removing it moves the line to {loo[uniq.index(worst)]:.4f})")

    # Delete-one-episode jackknife SE: the uncertainty of the metric with respect to
    # WHICH 20 episodes were held out. This is the resolution limit of the whole
    # leaderboard -- a gap smaller than a couple of these is episode-composition
    # wobble, not a backbone difference. It is INDEPENDENT of the per-stack sampling
    # variance footnote (that one holds the episodes fixed and varies the seed).
    n = len(uniq)
    se_jack = np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))
    print(f"\ndelete-one-episode jackknife SE of the pass line: {se_jack:.4f} "
          f"({100*se_jack/total:.2f}% of the line)")
    print("  NOTE: this is the SE of an ABSOLUTE metric. It is the WRONG ruler for")
    print("  comparing two stacks -- they are scored on the SAME episodes, so most of")
    print("  this wobble is COMMON to both and cancels in the difference. Using it as")
    print("  a significance threshold would wrongly dismiss real gaps. The right")
    print("  quantity is the jackknife SE of the PAIRED DIFFERENCE, measured below.")

    # Paired demonstration with the two predictors available without a trained model:
    # hold-state (delta=0) vs const-relative (train-fitted mean delta per step/joint).
    # Their MAEs are close, so this isolates the pairing effect rather than the gap.
    try:
        cr = fit_const_relative()
    except Exception as exc:                      # never let the diagnostic break the run
        print(f"  (paired demo skipped: {exc})")
        return total, se_jack

    pred2 = state[:, None, :] + cr[None, :K, :]
    per_anchor2 = np.abs(pred2 - gt).mean(axis=(1, 2))
    d = per_anchor - per_anchor2                  # per-anchor paired difference
    loo_d = np.array([d[eps != e].mean() for e in uniq])
    se_paired = np.sqrt((n - 1) / n * np.sum((loo_d - loo_d.mean()) ** 2))
    print(f"\npaired difference (hold-state - const-relative): {d.mean():+.4f}")
    print(f"  unpaired SEs: {se_jack:.4f} and (const-rel) comparable "
          f"-> naive threshold would be ~{2*se_jack:.3f}")
    print(f"  PAIRED jackknife SE: {se_paired:.5f}  "
          f"=> {se_jack/max(se_paired,1e-12):.0f}x tighter than the absolute SE")
    print(f"  so the resolvable gap on this val set is ~{2*se_paired:.4f} MAE, "
          f"not ~{2*se_jack:.3f}")
    return total, se_jack


def fit_const_relative():
    """Per-(step, joint) mean delta over the 111 TRAIN episodes -- train-only, no val."""
    import pandas as pd
    split = json.loads((BENCH / "data" / "split.json").read_text())
    src = Path(os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")))
    A = np.load(BENCH / "data" / "val_anchors.npz")
    K, stride = int(A["K"]), int(A["stride"])
    acc, cnt = None, 0
    for ep in split["train_episodes"]:
        df = pd.read_parquet(src / f"data/chunk-000/episode_{ep:06d}.parquet",
                             columns=["observation.state", "action"])
        st = np.vstack([np.asarray(x, np.float32) for x in df["observation.state"]])
        ac = np.vstack([np.asarray(x, np.float32) for x in df["action"]])
        need = (K - 1) * stride + 1
        for t in range(len(st) - need + 1):
            chunk = ac[t:t + need:stride][:K]
            d = chunk - st[t]
            acc = d if acc is None else acc + d
            cnt += 1
    return acc / cnt

    keep = ~np.isin(eps, OOB_EPS)
    print(f"\ndropping the 4 OOB-driving episodes {OOB_EPS}: "
          f"{keep.sum()}/{len(eps)} anchors remain, pass line "
          f"{per_anchor[keep].mean():.4f} (vs {total:.4f})")

    out = BENCH / "logs/n17_score_concentration.json"
    out.write_text(json.dumps({
        "K": K, "pass_line": float(total), "arm": float(arm), "grip": float(grip),
        "episodes": uniq, "anchors": counts.tolist(),
        "per_episode_mae": means.tolist(), "share": share.tolist(),
        "n_eff_episodes": float(n_eff),
        "loo_min": float(loo.min()), "loo_max": float(loo.max()),
        "pass_line_without_oob_eps": float(per_anchor[keep].mean()),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
