#!/usr/bin/env python3
"""Is ACT's train/val MAE gap MEMORIZATION, or just a harder val split?

openvla-oft's rider on my train-split eval: a large train-vs-val gap does NOT by
itself prove overfit, because the two splits may differ in INTRINSIC DIFFICULTY
(20 val episodes could simply move more). They suggested subtracting a control's
own train-vs-val difference. This does that with NO model and NO GPU, using
constant (image-blind) predictors on the exact same anchors:

    hold-state       pred = current state          (delta=0; the board's pass line)
    const-abs-TRAIN  pred[k,j] = mean action over TRAIN anchors
    const-abs-OWN    pred[k,j] = mean action over THAT SPLIT's own anchors

Each isolates a different null:

  * const-abs-OWN is the DIFFICULTY proxy matched to ACT's parameterization
    (ACT is absolute-space): it is the residual spread of a split around its own
    best constant, i.e. how much of the action a constant can never explain.
    Its train-vs-val ratio is difficulty ALONE -- no memorization is possible,
    the constant is fitted on the split it is scored on.

  * const-abs-TRAIN is the MARGINAL-MEMORIZATION null: it memorizes the training
    action distribution and nothing else -- no image, no per-episode structure.
    Its train-vs-val ratio is the gap a model gets for free from marginal
    memorization + difficulty, with zero image-conditioned overfit.

So ACT's ratio in EXCESS of const-abs-TRAIN's is the part that needs an
image-conditioned explanation, and ACT's ratio in excess of const-abs-OWN's is
the part not attributable to split difficulty.

On OFT's other caveat (anchor construction must be identical across splits):
build_anchors is uniform-in-time (stride 5, drop the last K), NOT "N anchors per
episode", so anchor count is proportional to episode length in BOTH splits and
the density is identical by construction. The val rebuild is asserted bit-exact
against the frozen val_anchors.npz in lerobot_train_split_eval.py.

Reads only; writes one JSON to logs/.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
K = 8
N_SUB = 400  # must match lerobot_train_split_eval.py --n
SEED = 0  # must match its --seed


def subsample(n: int, seed: int = SEED, n_sub: int = N_SUB) -> np.ndarray:
    """Byte-for-byte the subsample of lerobot_train_split_eval.py."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n_sub and n_sub < n:
        idx = np.sort(rng.choice(idx, size=n_sub, replace=False))
    return idx


def mae3(pred: np.ndarray, gt: np.ndarray) -> dict:
    e = np.abs(pred - gt)
    return {"mae": float(e.mean()), "arm": float(e[..., :7].mean()), "grip": float(e[..., 7].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="act",
                    help="which leg's trainsplit_eval_<bb>_{train,val}.json to compare against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                    str(BENCH / "archive")]   # was one flat scripts/ dir
    from lerobot_train_split_eval import build_anchors

    split = json.loads((BENCH / "data" / "split.json").read_text())
    data = {}
    for s in ("train", "val"):
        eps = sorted(split[f"{s}_episodes"])
        e, f, gt, st = build_anchors(eps)
        data[s] = {"eps": eps, "gt": gt.astype(np.float64), "state": st.astype(np.float64)}
        print(f"[anchors] {s}: {len(e)} anchors over {len(eps)} episodes")

    # cross-check the val rebuild against the frozen harness (read-only)
    A = np.load(BENCH / "data" / "val_anchors.npz")
    assert np.array_equal(data["val"]["gt"].astype(np.float32), A["gt"]), "val rebuild != frozen"
    print("[validate] val rebuild bit-matches val_anchors.npz")

    const_train_full = data["train"]["gt"].mean(axis=0)  # (K,8) from ALL train anchors

    out = {"K": K, "n_sub": N_SUB, "seed": SEED, "full": {}, "sub": {}}
    for scope in ("full", "sub"):
        for s in ("train", "val"):
            gt, st = data[s]["gt"], data[s]["state"]
            if scope == "sub":
                i = subsample(len(gt))
                gt, st = gt[i], st[i]
            hold = np.broadcast_to(st[:, None, :], gt.shape)
            const_own = gt.mean(axis=0)
            out[scope][s] = {
                "n": int(len(gt)),
                "hold_state": mae3(hold, gt),
                "const_abs_train": mae3(np.broadcast_to(const_train_full, gt.shape), gt),
                "const_abs_own": mae3(np.broadcast_to(const_own, gt.shape), gt),
            }

    # ---- the model's measured numbers (same anchors, same K, eval mode) ----------
    # The constants above are free at both scopes, but the MODEL is not: it was run at
    # whatever `--n` its eval used. Read that back from its own json and compare against
    # the matching scope -- pairing a full-population constant with a 400-anchor model
    # number reintroduces exactly the sampling error this control exists to remove.
    act, model_scope = {}, None
    for s in ("train", "val"):
        p = BENCH / "logs" / f"trainsplit_eval_{args.backbone}_{s}.json"
        if p.exists():
            r = json.loads(p.read_text())
            act[s] = {"mae": r["mae"], "arm": r["arm"], "grip": r["grip"]}
            sc = "sub" if r.get("subsampled", r.get("n_anchors") == N_SUB) else "full"
            if model_scope and sc != model_scope:
                sys.exit(f"{args.backbone}: train and val evals ran at DIFFERENT scopes "
                         f"({model_scope} vs {sc}) -- their ratio mixes two populations")
            model_scope = sc
    out["backbone"] = args.backbone
    out["model_scope"] = model_scope
    out["act" if args.backbone == "act" else args.backbone] = act

    def ratio(d: dict, key: str, ch: str = "mae") -> float:
        return d["val"][key][ch] / d["train"][key][ch]

    if len(act) == 2:
        sub = out[model_scope]
        r_act = act["val"]["mae"] / act["train"]["mae"]
        r_hold = ratio(sub, "hold_state")
        r_ctr = ratio(sub, "const_abs_train")
        r_own = ratio(sub, "const_abs_own")
        out["ratios_val_over_train"] = {
            "act": r_act,
            "hold_state": r_hold,
            "const_abs_train": r_ctr,
            "const_abs_own_DIFFICULTY": r_own,
            "act_excess_over_difficulty": r_act / r_own,
            "act_excess_over_marginal_memorization": r_act / r_ctr,
        }
        scope_label = ("FULL population (7475 train / 1397 val)" if model_scope == "full"
                       else f"{N_SUB}-anchor subsets -- carries ~+-0.13 on the ratio, see WARNING")
        print(f"\n=== val/train MAE ratio (>1 = val is worse). {scope_label}, K={K} ===")
        print(f"  const-abs-OWN   {r_own:6.3f}   <- DIFFICULTY only (constant fitted on the split it scores)")
        print(f"  const-abs-TRAIN {r_ctr:6.3f}   <- difficulty + marginal memorization, image-blind")
        print(f"  hold-state      {r_hold:6.3f}   <- delta-space difficulty proxy")
        print(f"  {args.backbone:<15s}{r_act:6.3f}   <- measured")
        if model_scope == "sub":
            print(f"  [WARNING] the model eval ran at n={N_SUB}; even a MODEL-FREE constant predictor "
                  f"moves 2.4% between that draw and the population. Re-run with --n 0.")
        print(f"\n  ACT / difficulty          = {r_act / r_own:.3f}x  unexplained by split difficulty")
        print(f"  ACT / marginal-memoriz.   = {r_act / r_ctr:.3f}x  needs an image-conditioned explanation")
        for s in ("train", "val"):
            d = sub[s]
            print(f"\n  [{s}] n={d['n']}  ACT {act[s]['mae']:.4f} | hold {d['hold_state']['mae']:.4f} "
                  f"| const-abs-TRAIN {d['const_abs_train']['mae']:.4f} | const-abs-OWN {d['const_abs_own']['mae']:.4f}")

    # ACT keeps the unsuffixed name it has been cited under since it landed; any other leg
    # gets its own file rather than silently overwriting a published artifact.
    p = pathlib.Path(args.out) if args.out else BENCH / "logs" / (
        "split_difficulty_control.json" if args.backbone == "act"
        else f"split_difficulty_control_{args.backbone}.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[write] {p}")


if __name__ == "__main__":
    main()
