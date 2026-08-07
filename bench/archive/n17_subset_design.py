#!/usr/bin/env python3
"""Subset composition + weighting audit (board rule (d), 2026-08-05).

Three things a subset footnote has to state, and one thing it must not do:

  (d.1) how many anchors, stratified by episode or not, how many per episode;
  (d.2) the WEIGHTING -- per-anchor (== frame-length-weighted: long episodes
        contribute more) vs per-episode-equal. gr00t-n15 measured that this
        choice alone moves numbers in OPPOSITE directions for a model and for
        the do-nothing line (corr(ep length, ep preds MAE) = +0.338 vs
        corr(ep length, ep hold MAE) = -0.624), so it is a real term, not
        bookkeeping;
  (d.3) the pass line RECOMPUTED on the same subset. Comparing a subset MAE to
        the full-population pass line is the one combination that is
        systematically flattering.

Also settles, for THIS leg, gr00t-n15's follow-up on my "--n-anchors 200 inflates
the gripper floor by +33%" note: she proposed a second, additive mechanism
(reweighting) alongside my rare-event attribution. Reweighting can only be a
mechanism if the subset actually reweights, and that is decidable here from the
anchor list alone -- no checkpoint, no model, no GPU. (The floor number itself
does need the trained checkpoint's baked stats, so the floor half is deferred to
n17_unit_roundtrip.py on the final ckpt; this settles whether there is anything
for it to find.)

Two subsets are in play and they are NOT the same set:
  * linspace-200  -- what n17_unit_roundtrip.py used to default to; the subset
                     that produced the +33% gripper floor;
  * shared-200    -- logs/variance_subset200.npy, the board's variance anchors.
CPU, seconds.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_SUBSET_MD5 = "9007df1fcec5eca2a1f01591c6266b43"


def weighted_pair(err, eps):
    """(per-anchor mean, per-episode-equal mean) of a (n,K,8)-shaped error block."""
    per_anchor = err.reshape(len(err), -1).mean(axis=1)
    frame_w = float(per_anchor.mean())
    ep_ids = sorted(set(eps.tolist()))
    ep_w = float(np.mean([per_anchor[eps == e].mean() for e in ep_ids]))
    return frame_w, ep_w, len(ep_ids)


def channels(P, gt, eps):
    """frame- and episode-weighted MAE for total / arm / grip."""
    out = {}
    for name, sl in (("mae", slice(0, 8)), ("arm", slice(0, 7)), ("grip", slice(7, 8))):
        f, e, n_ep = weighted_pair(np.abs(P[..., sl] - gt[..., sl]), eps)
        out[name] = {"frame_weighted": f, "episode_equal": e, "delta": e - f,
                     "pct": 100.0 * (e - f) / f if f else float("nan")}
    out["n_episodes"] = n_ep
    return out


def composition(eps_sub, eps_full):
    """Per-episode counts and the share drift a subset introduces."""
    ids = sorted(set(eps_full.tolist()))
    full_c = np.array([(eps_full == e).sum() for e in ids], float)
    sub_c = np.array([(eps_sub == e).sum() for e in ids], float)
    full_share, sub_share = full_c / full_c.sum(), sub_c / max(sub_c.sum(), 1)
    return {
        "n_anchors": int(sub_c.sum()),
        "n_episodes_covered": int((sub_c > 0).sum()),
        "n_episodes_total": len(ids),
        "per_episode_counts": {str(e): int(c) for e, c in zip(ids, sub_c)},
        "min_per_episode": int(sub_c.min()),
        "max_per_episode": int(sub_c.max()),
        # L1 distance between the subset's episode mix and the full population's.
        # 0 == perfectly proportional (== no reweighting term at all);
        # 1 == disjoint. This is the number that decides whether n15's mechanism
        # can contribute here.
        "share_l1_vs_full": float(np.abs(sub_share - full_share).sum()),
        "max_abs_share_drift": float(np.abs(sub_share - full_share).max()),
        # DETECTED, not declared. I first wrote this as a hardcoded False and the
        # data immediately contradicted it (the board subset is exactly 10/episode).
        # Same failure mode as the seed fields: a constant that describes the
        # artifact instead of being read off it. Equal counts per episode is not a
        # cosmetic property -- it makes the two weightings in (d.2) the SAME
        # estimator, so the reweighting term is identically zero by construction.
        "stratified_by_episode": bool(sub_c.min() == sub_c.max() and sub_c.min() > 0),
        "weighting_is_degenerate": bool(sub_c.min() == sub_c.max() and sub_c.min() > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH / "logs/n17_subset_design.json"))
    ap.add_argument("--shared-subset", default=str(BENCH / "logs/variance_subset200.npy"))
    args = ap.parse_args()

    A = np.load(BENCH / "data" / "val_anchors.npz")
    gt = A["gt"].astype(np.float32)
    state = A["state"].astype(np.float32)
    eps = A["episodes"].astype(int)
    N, K, D = gt.shape
    hold = np.broadcast_to(state[:, None, :], (N, K, D))

    # subsets ------------------------------------------------------------------
    lin = np.linspace(0, N - 1, 200).astype(int)
    shared_p = pathlib.Path(args.shared_subset)
    for c in (shared_p, BENCH / "logs/variance" / shared_p.name):
        if c.is_file() and hashlib.md5(c.read_bytes()).hexdigest() == EXPECTED_SUBSET_MD5:
            shared_p = c
            break
    else:
        raise SystemExit(f"[FATAL] shared subset with md5 {EXPECTED_SUBSET_MD5} not found")
    shared = np.load(shared_p).astype(int)

    pops = {"full_1397": np.arange(N), "linspace_200": lin, "shared_200": shared}
    report = {"K": int(K), "n_anchors_full": int(N),
              "shared_subset": {"path": str(shared_p), "md5": EXPECTED_SUBSET_MD5},
              "populations": {}}

    print(f"anchors {N}, K={K}, episodes {len(set(eps.tolist()))}\n")
    for pname, idx in pops.items():
        blk = {"composition": composition(eps[idx], eps),
               "hold_state_pass_line": channels(hold[idx], gt[idx], eps[idx])}
        # every preds row currently on the board, same two weightings
        blk["preds"] = {}
        for f in sorted((BENCH / "preds").glob("preds_*.npz")):
            P = np.load(f)["pred"].astype(np.float32)
            if P.shape[0] != N:
                continue
            blk["preds"][f.stem.replace("preds_", "")] = channels(P[idx][:, :K, :], gt[idx], eps[idx])
        report["populations"][pname] = blk

        c, h = blk["composition"], blk["hold_state_pass_line"]
        print(f"[{pname}] n={c['n_anchors']} over {c['n_episodes_covered']}/"
              f"{c['n_episodes_total']} eps, per-ep {c['min_per_episode']}-{c['max_per_episode']}, "
              f"share L1 vs full {c['share_l1_vs_full']:.4f}")
        print(f"    hold  MAE frame {h['mae']['frame_weighted']:.4f} -> episode-equal "
              f"{h['mae']['episode_equal']:.4f} ({h['mae']['pct']:+.2f}%)   "
              f"grip {h['grip']['frame_weighted']:.4f} -> {h['grip']['episode_equal']:.4f} "
              f"({h['grip']['pct']:+.2f}%)")
        for bb, ch in blk["preds"].items():
            print(f"    {bb:14s} MAE frame {ch['mae']['frame_weighted']:.4f} -> "
                  f"{ch['mae']['episode_equal']:.4f} ({ch['mae']['pct']:+.2f}%)")

    # the comparison rule (d.3) makes illegal, quantified on this board ---------
    full_pl = report["populations"]["full_1397"]["hold_state_pass_line"]["mae"]["frame_weighted"]
    report["rule_d3_illegal_comparison"] = {
        "full_pass_line": full_pl,
        "subset_pass_lines": {p: report["populations"][p]["hold_state_pass_line"]["mae"]
                              ["frame_weighted"] for p in ("linspace_200", "shared_200")},
        "note": ("a subset MAE compared against the FULL pass line borrows this much "
                 "slack; always recompute the line on the same anchors"),
    }
    print(f"\nrule (d.3): full pass line {full_pl:.4f} vs subset lines "
          + ", ".join(f"{p} {v:.4f}" for p, v in
                      report["rule_d3_illegal_comparison"]["subset_pass_lines"].items()))

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
