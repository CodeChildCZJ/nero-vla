#!/usr/bin/env python3
"""Report every model-vs-hold quantity in BOTH weighting calibers, because one of them lies.

Board rule (d), from gr00t-n15's measurement. An MAE averaged over anchors carries a hidden
choice: FRAME-LENGTH-WEIGHTED (plain mean over anchors, so a long episode contributes more)
or EPISODE-EQUAL (mean of per-episode means). Both are legitimate. Neither is correct.

The reason it is not cosmetic is that the two calibers move a model and its baseline in
OPPOSITE directions:

    corr(episode length, MODEL per-episode MAE) = +0.34   long episodes are harder for a model
    corr(episode length, HOLD  per-episode MAE) = -0.62   long episodes are easier for do-nothing
                                                          (long = slower; hold IS the delta=0 predictor)

so anything built from both moves further than either side. Measured consequence on ACT:
the val arm margin vs hold is -0.1404 frame-weighted and +0.0029 episode-equal -- it CHANGES
SIGN, so "ACT's arm is worse than do-nothing" was never a finding, it was a caliber. The same
switch moves the 2.301x overfit ratio to 2.260, inside its own +-0.122 SE.

  ⇒ reweighting threatens DIFFERENCES, not RATIOS.

Subtracting two numbers of the same size makes the opposite-sign responses first-order;
dividing numbers that differ by >100% makes them second-order. So this script exists to stop
anyone (me) from reporting a small margin without saying which caliber produced it.

CPU only, no model, no GPU: reads the per-anchor npz written by lerobot_train_split_eval.py.
"""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
CH = {"TOTAL": slice(0, 8), "ARM": slice(0, 7), "GRIP": slice(7, 8)}


def weighted(err, eps, mode, sl):
    """err (A,K,D) -> scalar MAE over channel slice `sl` under weighting `mode`."""
    e = err[..., sl]
    if mode == "frame":
        return float(e.mean())
    return float(np.array([e[eps == u].mean() for u in np.unique(eps)]).mean())


def margin_signif(err, hold, eps, mode, sl):
    """Paired delete-one-episode jackknife of (hold - model) under one caliber.

    PAIRED, i.e. delete an episode and recompute BOTH sides, because model and baseline are
    scored on the same anchors of the same episodes and their per-episode errors are
    correlated (negatively, for a hold-state line -- rho -0.16 -- which AMPLIFIES the paired
    SE rather than shrinking it, so quadrature of the two marginal SEs is not conservative).
    """
    def m(keep):
        a, b, e = err[keep][..., sl], hold[keep][..., sl], eps[keep]
        if mode == "frame":
            return float(b.mean() - a.mean())
        u = np.unique(e)
        return float(np.array([b[e == q].mean() for q in u]).mean()
                     - np.array([a[e == q].mean() for q in u]).mean())
    uniq = np.unique(eps)
    full = m(np.ones(len(eps), bool))
    loo = np.array([m(eps != q) for q in uniq])
    n = len(uniq)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return full, se, (abs(full) / se if se else float("inf"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", default="act")
    ap.add_argument("--out", default=None,
                    help="output json (default logs/weighting_calibers_<backbone>.json); "
                         "point at /tmp for a pre-flight")
    args = ap.parse_args()

    data, src_md5 = {}, {}
    for s in ("train", "val"):
        f = BENCH / "logs" / f"trainsplit_preds_{args.backbone}_{s}.npz"
        if not f.exists():
            sys.exit(f"missing {f}\nRun: lerobot_train_split_eval.py --backbone {args.backbone} "
                     f"--eps {s} --n 0")
        z = np.load(f)
        src_md5[str(f)] = hashlib.md5(f.read_bytes()).hexdigest()
        err = np.abs(z["pred"][:, : z["gt"].shape[1], :] - z["gt"])
        hold = np.abs(np.broadcast_to(z["state"][:, None, :], z["gt"].shape) - z["gt"])
        data[s] = (err, hold, z["episodes"])

    out = {"backbone": args.backbone, "input_md5": src_md5,
           "why": "board rule (d): a margin reported without its caliber is not reportable",
           "calibers": {}}

    print(f"{'chan':7}{'weighting':10}{'train':>9}{'trHold':>9}{'trMargin':>10}"
          f"{'val':>9}{'vaHold':>9}{'vaMargin':>10}{'ratio':>8}")
    flips = []
    for chan, sl in CH.items():
        margins = {}
        for mode in ("frame", "episode"):
            row = {}
            for s in ("train", "val"):
                err, hold, eps = data[s]
                row[s] = weighted(err, eps, mode, sl)
                row[f"{s}_hold"] = weighted(hold, eps, mode, sl)
                row[f"{s}_margin"] = row[f"{s}_hold"] - row[s]
            row["ratio_val_over_train"] = row["val"] / row["train"]
            out["calibers"][f"{chan}_{mode}"] = row
            margins[mode] = row["val_margin"]
            print(f"{chan:7}{mode:10}{row['train']:9.4f}{row['train_hold']:9.4f}"
                  f"{row['train_margin']:+10.4f}{row['val']:9.4f}{row['val_hold']:9.4f}"
                  f"{row['val_margin']:+10.4f}{row['ratio_val_over_train']:8.3f}")
        if margins["frame"] * margins["episode"] < 0:
            flips.append(chan)

    # The whole point of the script is this line. A sign flip is not a small discrepancy to
    # note in passing -- it means the two legitimate calibers disagree about the DIRECTION of
    # the claim, so the claim is unresolved and must not be reported either way.
    # ---- does the CALIBER change the SIGNIFICANCE VERDICT on the val split? -------------
    # Stronger than the sign-flip check above and it fires on quantities the sign check
    # calls clean: a margin can keep its sign and still cross the 2-sigma decision boundary,
    # because the caliber moves the SE as well as the point estimate (episode-equal removes
    # the episode-length heterogeneity that inflates the frame-weighted spread). Whenever the
    # two calibers land on opposite sides of the threshold, "does this model beat do-nothing"
    # has no caliber-free answer and must be reported as both numbers, never one.
    err_v, hold_v, eps_v = data["val"]
    sig = {}
    print(f"\n{'chan':7}{'caliber':10}{'margin':>10}{'jackSE':>9}{'sigma':>8}   resolved(>2s)?")
    for chan, sl in CH.items():
        verdicts = {}
        for mode in ("frame", "episode"):
            m0, se, s = margin_signif(err_v, hold_v, eps_v, mode, sl)
            sig[f"{chan}_{mode}"] = {"margin_vs_hold": m0, "jackknife_se": se, "sigma": s,
                                     "resolved_at_2sigma": bool(s > 2)}
            verdicts[mode] = s > 2
            print(f"{chan:7}{mode:10}{m0:+10.4f}{se:9.4f}{s:8.2f}   "
                  f"{'YES' if s > 2 else 'no'}")
        if verdicts["frame"] != verdicts["episode"]:
            sig[f"{chan}_VERDICT_IS_CALIBER_DEPENDENT"] = True
    out["val_margin_significance"] = sig
    out["val_margin_sign_flips"] = flips
    swung = [c for c in CH if sig.get(f"{c}_VERDICT_IS_CALIBER_DEPENDENT")]
    parts = []
    parts.append(
        f"SIGN FLIP between calibers on: {', '.join(flips)} -- do not report a direction for "
        f"those margins." if flips else
        "no val margin changes sign between calibers.")
    parts.append(
        f"SIGNIFICANCE VERDICT is caliber-dependent on: {', '.join(swung)} -- the two legitimate "
        f"calibers land on OPPOSITE sides of 2 sigma, so 'beats do-nothing' has no caliber-free "
        f"answer for these channels and BOTH numbers must be reported. This is strictly stronger "
        f"than the sign check: the margin can keep its sign and still cross the threshold, because "
        f"the caliber moves the SE too." if swung else
        "no significance verdict changes across calibers.")
    parts.append(
        "Rescue for anything unresolved: ask a question that survives the switch -- e.g. train-vs-val "
        "for the SAME model, which compares one model across two splits instead of two models on one "
        "split, and whose ratio moved only 2.301 -> 2.260 (inside its own +-0.122 SE).")
    out["verdict"] = " ".join(parts)
    print(f"\n{out['verdict']}")

    p = pathlib.Path(args.out) if args.out else BENCH / "logs" / f"weighting_calibers_{args.backbone}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[write] {p}")


if __name__ == "__main__":
    main()
