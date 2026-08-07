#!/usr/bin/env python3
"""Is a DIFFERENCE OF GENERALIZATION RATIOS between two legs resolvable?

lerobot's resolution limit ("two ratios need to differ by roughly 0.35 to separate at 2 sigma")
combines the two legs' SEs in quadrature, i.e. it models them as INDEPENDENT samples. They are
not: every leg on this board is evaluated on the same anchor file, and I verified ACT's
train/val per-anchor arrays are bit-identical to mine (episodes, frames, gt, state all
array_equal on both splits). Two models scored on the same episodes are a PAIRED comparison --
delete an episode and BOTH ratios move, usually together, and the common motion cancels in the
difference.

Direction is NOT automatic and must be measured, not assumed: reference_leaderboard_significance
records a case where the paired SE AMPLIFIED (rho = -0.16 against a hold-state line). So this
script reports paired and quadrature side by side and prints the observed factor.

Units of resampling are EPISODES (anchors within an episode are ~5-frame-strided views of one
trajectory, nowhere near independent). Train and val hold disjoint episode sets (verified: 111 /
20 / overlap 0), so deleting one episode perturbs exactly one side of each ratio.

A "leg" is either
  * a name  -> logs/trainsplit_preds_<name>_{train,val}.npz  (keys pred/gt/state/episodes), or
  * "hold"  -> the do-nothing predictor (state broadcast over the horizon), built from the anchor
               files themselves. It is the board's pass line and needs no checkpoint.
"""
import argparse, json, pathlib, sys
import numpy as np

BENCH = pathlib.Path(__file__).resolve().parent.parent
CH = {"total": slice(0, 8), "arm": slice(0, 7), "grip": slice(7, 8)}


def load_leg(name, split):
    """-> (per_anchor_abs_error [N,K,8], episodes [N])"""
    anc = BENCH / ("data/val_anchors.npz" if split == "val" else "logs/oft/train_anchors.npz")
    a = np.load(anc)
    gt, st, ep = a["gt"].astype(np.float64), a["state"].astype(np.float64), a["episodes"]
    if name == "hold":
        pred = np.broadcast_to(st[:, None, :], gt.shape)
    else:
        f = BENCH / f"logs/trainsplit_preds_{name}_{split}.npz"
        if not f.exists():
            sys.exit(f"[FATAL] {f} does not exist -- leg '{name}' has no {split}-split per-anchor npz")
        d = np.load(f)
        # Pairing is the whole premise; refuse to proceed if the anchor grids differ at all.
        for k in ("episodes", "frames", "gt", "state"):
            if not np.array_equal(d[k].astype(a[k].dtype), a[k]):
                sys.exit(f"[FATAL] leg '{name}' {split}: '{k}' differs from {anc.name}; "
                         "the two legs are NOT on a common anchor grid, pairing is invalid")
        pred = d["pred"].astype(np.float64)[:, : gt.shape[1], :]
    return np.abs(pred - gt), ep.astype(np.int64)


def ratio(errA_tr, errA_va, keep_tr, keep_va, ch):
    """val/train MAE ratio on a channel, restricted to the kept anchors."""
    s = CH[ch]
    return errA_va[keep_va][..., s].mean() / errA_tr[keep_tr][..., s].mean()


def jackknife(vals):
    """Delete-one jackknife SE from the n leave-one-out replicates."""
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    return float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))


def analyse(legA, legB, permute_seed=None):
    A_tr, ep_tr = load_leg(legA, "train")
    A_va, ep_va = load_leg(legA, "val")
    B_tr, _ = load_leg(legB, "train")
    B_va, _ = load_leg(legB, "val")
    if permute_seed is not None:
        # Power control: destroy the pairing by permuting leg B's anchors WITHIN each split.
        # If the paired SE is small because of genuine common motion, it must rise toward
        # quadrature here. If it stays small regardless, the shrink is a bug.
        rng = np.random.default_rng(permute_seed)
        B_tr = B_tr[rng.permutation(len(B_tr))]
        B_va = B_va[rng.permutation(len(B_va))]

    eps_tr, eps_va = np.unique(ep_tr), np.unique(ep_va)
    out = {}
    for ch in CH:
        allm_tr = np.ones(len(ep_tr), bool)
        allm_va = np.ones(len(ep_va), bool)
        rA = ratio(A_tr, A_va, allm_tr, allm_va, ch)
        rB = ratio(B_tr, B_va, allm_tr, allm_va, ch)
        dA, dB, dD = [], [], []
        for e in np.concatenate([eps_tr, eps_va]):
            k_tr = ep_tr != e
            k_va = ep_va != e
            a = ratio(A_tr, A_va, k_tr, k_va, ch)
            b = ratio(B_tr, B_va, k_tr, k_va, ch)
            dA.append(a); dB.append(b); dD.append(a - b)
        seA, seB, sePaired = jackknife(dA), jackknife(dB), jackknife(dD)
        seQuad = float(np.hypot(seA, seB))
        diff = rA - rB
        # rho implied by the two SEs: Var(A-B) = VarA + VarB - 2 rho sdA sdB
        rho = (seA ** 2 + seB ** 2 - sePaired ** 2) / (2 * seA * seB) if seA and seB else float("nan")
        out[ch] = {
            "ratio_A": float(rA), "ratio_B": float(rB), "diff": float(diff),
            "se_A": seA, "se_B": seB,
            "se_diff_paired": sePaired, "se_diff_quadrature": seQuad,
            "paired_over_quadrature": sePaired / seQuad if seQuad else float("nan"),
            "implied_rho": float(rho),
            "sigma_paired": abs(diff) / sePaired if sePaired else float("inf"),
            "sigma_quadrature": abs(diff) / seQuad if seQuad else float("inf"),
            "min_resolvable_diff_2sigma_paired": 2 * sePaired,
            "min_resolvable_diff_2sigma_quadrature": 2 * seQuad,
            # What the 2-sigma bar WOULD be at other correlations, from the two marginal SEs
            # alone: 2*sqrt(sA^2 + sB^2 - 2 rho sA sB). The quadrature answer is the rho=0 row.
            # This is the honest way to quote a resolution limit before the second leg exists --
            # rho is a property of the specific PAIR, so a limit measured against one leg cannot
            # be carried over to another.
            "min_resolvable_2sigma_by_rho": {
                f"{rho_i:+.1f}": float(2 * np.sqrt(max(seA ** 2 + seB ** 2
                                                       - 2 * rho_i * seA * seB, 0.0)))
                for rho_i in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8)},
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="leg name, or 'hold'")
    p.add_argument("--b", required=True, help="leg name, or 'hold'")
    p.add_argument("--permute-seed", type=int, default=None, help="power control: break pairing")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    r = analyse(args.a, args.b, args.permute_seed)
    if not args.quiet:
        tag = "" if args.permute_seed is None else f"  [PAIRING BROKEN, seed={args.permute_seed}]"
        print(f"=== ratio({args.a}) - ratio({args.b}){tag}")
        for ch, d in r.items():
            print(f"  {ch:5s} {d['ratio_A']:.3f} - {d['ratio_B']:.3f} = {d['diff']:+.3f}  "
                  f"SE paired {d['se_diff_paired']:.4f} vs quad {d['se_diff_quadrature']:.4f} "
                  f"({d['paired_over_quadrature']:.2f}x, rho {d['implied_rho']:+.3f})  "
                  f"=> {d['sigma_paired']:.2f}sigma paired / {d['sigma_quadrature']:.2f}sigma quad")
    if args.out:
        payload = {"leg_A": args.a, "leg_B": args.b, "permute_seed": args.permute_seed,
                   "resampling_unit": "episode (131 = 111 train + 20 val, disjoint)",
                   "channels": r}
        json.dump(payload, open(args.out, "w"), indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
