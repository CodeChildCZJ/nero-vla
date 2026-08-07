#!/usr/bin/env python3
"""Put a scale on the train/val overfit RATIO, which is currently a bare point estimate.

Why: gr00t-n15 -- "那个具体的比值目前是无标度的" -- a ratio of two means computed on
differently-sized CLUSTERED samples has no stated uncertainty, and openvla-oft measured
how much that matters here: a 400-anchor draw carries ~±0.13 on ACT's 2.141, the same
order as the real gaps between backbones. Two legs are about to compare their ratios
(ACT full-finetune vs OFT LoRA r32), and without an SE that comparison is unresolvable
in either direction.

n15's instruction, followed literally: **jackknife the RATIO, not the two MAEs
separately.** Doing it separately and eyeballing the two error bars ignores that the
excess-over-difficulty is a PAIRED quantity -- the model and the constant baseline are
evaluated on the same anchors of the same episodes, so their episode-level errors move
together and the paired SE is not the quadrature of the marginals.

Three numbers, three different independence structures, so three different estimators:

1. R = val_MAE / train_MAE. The two splits hold DISJOINT episodes, so the two means are
   independent: delta method over each split's OWN cluster jackknife SE. Reported as the
   primary, and cross-checked against a pooled delete-one-episode jackknife over all 131
   episodes; if those two disagree materially the ratio is not smooth enough to trust.

2. R_diff = the same ratio for the image-blind const-abs-OWN predictor (pure difficulty).

3. EXCESS = R / R_diff, the actual claim ("the gap is not explained by split difficulty").
   Here model and constant SHARE episodes, so this one is PAIRED: delete one episode and
   recompute BOTH ratios, which keeps the correlation instead of assuming it away.

Reads only the per-anchor npz written by lerobot_train_split_eval.py -- no GPU, no model.
"""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
CH = {"mae": slice(0, 8), "arm": slice(0, 7), "grip": slice(7, 8)}


def per_anchor_err(pred, gt, ch):
    return np.abs(pred - gt)[..., CH[ch]].mean(axis=(1, 2))


def jack_se(vals, eps):
    """delete-one-episode jackknife SE of mean(vals); eps = cluster label per anchor."""
    uniq = np.unique(eps)
    loo = np.array([vals[eps != e].mean() for e in uniq])
    n = len(uniq)
    return float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="act")
    # Board rule (openpi, after they pre-flighted a tool on someone else's preds and wrote
    # ACT's and N1.5's MAEs into their own result json -- right filename, right schema,
    # right magnitude, undetectable downstream). Any script that produces a result
    # artifact takes --out so a pre-flight can be sent to /tmp: a dry run that writes to
    # the production path is not a dry run. The input md5s below are the other half --
    # they make a mis-fed run visible in the artifact itself rather than only in the
    # operator's memory of what they fed it.
    ap.add_argument("--out", default=None,
                    help="output json (default logs/ratio_signif_<backbone>.json); "
                         "point at /tmp for a pre-flight")
    args = ap.parse_args()

    d, src_md5 = {}, {}
    for s in ("train", "val"):
        f = BENCH / "logs" / f"trainsplit_preds_{args.backbone}_{s}.npz"
        if not f.exists():
            sys.exit(f"missing {f}\nRun: lerobot_train_split_eval.py --backbone {args.backbone} "
                     f"--eps {s} --n 0   (the npz is only written by versions after 2026-08-05)")
        d[s] = np.load(f)
        src_md5[str(f)] = hashlib.md5(f.read_bytes()).hexdigest()
        j = json.loads((BENCH / "logs" / f"trainsplit_eval_{args.backbone}_{s}.json").read_text())
        if j.get("subsampled"):
            print(f"[WARNING] the {s} split ran SUBSAMPLED (n={j['n_anchors']} of "
                  f"{j.get('n_anchors_full_population')}). The SE below is the SE of THIS DRAW's "
                  f"ratio over episode composition; it does NOT include the extra spread from "
                  f"having drawn 400 of the anchors. Re-run with --n 0.")

    out = {"backbone": args.backbone,
           "n_anchors": {s: int(len(d[s]["gt"])) for s in d},
           "n_episodes": {s: int(len(np.unique(d[s]["episodes"]))) for s in d},
           "input_md5": src_md5,
           # Board rule (d): a ratio of two means is silently caliber-dependent, and the two
           # calibers move numerator and denominator in OPPOSITE directions, so the field is
           # load-bearing rather than documentation. See reference-weighting-threatens-differences.
           "weighting": "frame-length-weighted (unweighted mean over anchors) on BOTH splits"}

    for ch in ("mae", "arm", "grip"):
        e = {s: per_anchor_err(d[s]["pred"], d[s]["gt"], ch) for s in d}
        # the image-blind difficulty null, recomputed here on the SAME anchors so the
        # paired excess below is genuinely paired (never read from another script's json)
        c = {s: per_anchor_err(np.broadcast_to(d[s]["gt"].mean(axis=0), d[s]["gt"].shape),
                               d[s]["gt"], ch) for s in d}
        ep = {s: d[s]["episodes"] for s in d}

        mv, mt = e["val"].mean(), e["train"].mean()
        R = mv / mt
        sev, set_ = jack_se(e["val"], ep["val"]), jack_se(e["train"], ep["train"])
        # delta method for a ratio of INDEPENDENT means (disjoint episode sets)
        se_R = R * np.sqrt((sev / mv) ** 2 + (set_ / mt) ** 2)

        # cross-check: pooled delete-one-episode over the union of both splits
        loo = []
        for s in ("train", "val"):
            for q in np.unique(ep[s]):
                a = e["val"][ep["val"] != q].mean() if s == "val" else mv
                b = e["train"][ep["train"] != q].mean() if s == "train" else mt
                loo.append(a / b)
        loo = np.array(loo); n = len(loo)
        se_pooled = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))

        Rd = c["val"].mean() / c["train"].mean()
        # PAIRED: one episode leaves, both ratios move together
        ex_loo = []
        for s in ("train", "val"):
            for q in np.unique(ep[s]):
                kv, kt = ep["val"] != q, ep["train"] != q
                if s == "val":
                    r = (e["val"][kv].mean() / mt) / (c["val"][kv].mean() / c["train"].mean())
                else:
                    r = (mv / e["train"][kt].mean()) / (c["val"].mean() / c["train"][kt].mean())
                ex_loo.append(r)
        ex_loo = np.array(ex_loo); n = len(ex_loo)
        excess = R / Rd
        se_ex = float(np.sqrt((n - 1) / n * np.sum((ex_loo - ex_loo.mean()) ** 2)))

        out[ch] = {"val_mae": float(mv), "train_mae": float(mt),
                   "ratio": float(R), "se_ratio_delta_method": float(se_R),
                   "se_ratio_pooled_jackknife": se_pooled,
                   "difficulty_ratio_const_abs_own": float(Rd),
                   "excess_over_difficulty": float(excess),
                   "se_excess_paired": se_ex,
                   "sigma_excess_vs_1": float(abs(excess - 1.0) / se_ex) if se_ex else float("inf")}
        print(f"\n=== {ch} ===")
        print(f"  val {mv:.4f} / train {mt:.4f}  ->  ratio {R:.4f} "
              f"+- {se_R:.4f} (delta) | +- {se_pooled:.4f} (pooled jackknife)")
        print(f"  difficulty null ratio {Rd:.4f}   excess {excess:.4f} +- {se_ex:.4f} "
              f"= {out[ch]['sigma_excess_vs_1']:.1f} sigma from 1.0 (no overfit)")

    p = pathlib.Path(args.out) if args.out else BENCH / "logs" / f"ratio_signif_{args.backbone}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[write] {p}")
    print("(SE answers 'would this ratio survive a different set of episodes'. It does NOT "
          "cover a different TRAINING run -- one seed, one ckpt, per no-val-selection.)")


if __name__ == "__main__":
    main()
