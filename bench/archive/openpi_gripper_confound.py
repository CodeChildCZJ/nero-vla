#!/usr/bin/env python3
"""The falsifiable gripper-normalization-confound check (team-lead, 2026-08-05).

Setup: pi0 and pi0.5 read the SAME byte-identical norm_stats (md5 9467e876...), but pi0 uses
z-score and pi0.5 uses quantile normalization (`use_quantile_norm = model_type != PI0`). Because
normalized loss weight per raw unit is 1/std vs 2/(q99-q01), and the gripper is bimodal while the
arm deltas are peaked-with-tails, pi0 down-weights the gripper relative to the arm by 2.05x in
amplitude / 4.21x in squared loss vs pi0.5.

PREDICTION (registered before the numbers existed): if that normalization difference -- not a
backbone capability difference -- drives the gripper column, then

    (grip_MAE / arm_MAE)_pi0  >  (grip_MAE / arm_MAE)_pi05

and, the stronger form, the gap should SHRINK when each model's gripper error is expressed in its
OWN normalized units (pi0: err/std_j7; pi0.5: err/((q99-q01)_j7/2)) -- i.e. measured in the units
each model actually optimized, the two should look much more alike than they do in raw degrees/mm.

A raw-space gap that SURVIVES the normalized-space rescaling is a real capability difference and
the footnote must say so. Reporting either outcome is the point; this is not a check we can only
"pass".

QUANTITATIVE NULL (added 2026-08-05, still before any MAE exists -- derived from norm_stats, not
hardcoded, so it cannot drift from the file). "Directionally greater" is a weak prediction. The
normalization argument actually pins a NUMBER. If a model's per-dim error in its OWN normalized
space is e on every dim, then raw grip MAE = e*scale_grip and raw arm MAE = e*mean(scale_arm), so

    NULL (grip/arm)_raw = scale_grip / mean(scale_arm)
      pi0   (z-score)  : 26.0853 / 5.5823  = 4.673
      pi0.5 (quantile) : 42.9901 / 18.0431 = 2.383
    => NULL ratio-of-ratios = 1.961    and    NULL ratio-of-ratios in own-normalized units = 1.000

So the decision rule is a comparison against 1.961, not against 1:
    measured ratio-of-ratios ~= 1.961  -> the ENTIRE gripper gap is the normalizer. Pure artifact.
    measured  > 1.961                  -> pi0 is genuinely worse on the gripper beyond the artifact.
    measured  < 1.961                  -> pi0 is genuinely BETTER on the gripper, partly MASKED by
                                          the artifact (a raw-space read would have gotten the sign
                                          of the capability difference backwards).
The last branch is the one worth having built this for: it is invisible to the directional test,
which would report "CONFIRMED" and let us write the wrong conclusion.

K-DEPENDENCE (gr00t-n15 measured it): a grip/arm ratio drifts with score.py's K (hold-state -16%,
a real learner +23% over K=1..8), so a cross-model ratio gap under ~20% sits inside K-drift and
must not be read as K-independent. Two notes: (1) every number here is K=8, annotate it as such;
(2) the NULL above is K-INDEPENDENT -- it is a property of norm_stats alone -- and the null gap is
96%, far outside the ~20% drift band, so comparing the measured ratio against the null is safe at
fixed K even though the raw ratio itself is not a K-invariant quantity.

SECOND MECHANISM (openpi_roundtrip_floor.py, same population): z-score vs quantile also changes
where the TARGET sits relative to the flow prior N(0,1). Fraction of GT cells outside [-1,1]:
pi0 14.28% (gripper 16.64%, max |z| 11.95) vs pi0.5 1.27% (gripper 2.73%, max |z| 2.65). The
gripper is the most out-of-range dim in BOTH recipes. This is independent of the loss-weighting
argument -- a target at |z|=12 is a harder regression than one at |z|=1.2 regardless of weight --
and it pushes the same direction, so a confirmed result does not discriminate between the two.

Unit note (checked, not assumed): the arm rows of norm_stats are DELTA-space while the scored MAE
is in absolute units -- but the *error* is identical in both spaces, since
(pred_abs - gt_abs) = (pred_delta + state) - (gt_delta + state). So dividing an absolute-space MAE
by a delta-space std is the right rescaling, not a unit mismatch. The gripper is absolute in both.

Usage:  python openpi_gripper_confound.py [--preds-dir .../preds]
        python openpi_gripper_confound.py --selftest   # no preds needed; proves the verdict can flip
"""
import argparse, hashlib, json, pathlib

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
NS = (pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi-agilex" / "assets" /
      "pi05_nero_b2_train" / "local" / "pick_pink_sponge_b2_train" / "norm_stats.json")


def mae_block(pred, gt):
    K = gt.shape[1]
    err = np.abs(pred[:, :K, :] - gt)
    return {
        "overall": float(err.mean()),
        "arm": float(err[..., :7].mean()),
        "grip": float(err[..., 7].mean()),
        "per_joint": [float(v) for v in err.mean(axis=(0, 1))],
    }


def analyze(blocks, scale):
    """blocks: {tag: mae_block}; scale: {tag: per-dim raw size of 1 normalized unit}."""
    out = {}
    for tag, m in blocks.items():
        m = dict(m)
        s = scale[tag]
        m["grip_in_own_norm_units"] = m["grip"] / float(s[7])
        m["arm_in_own_norm_units"] = float(np.mean([m["per_joint"][j] / s[j] for j in range(7)]))
        m["ratio_raw"] = m["grip"] / m["arm"]
        m["ratio_norm"] = m["grip_in_own_norm_units"] / m["arm_in_own_norm_units"]
        m["norm_scheme"] = "z-score (std)" if tag == "pi0" else "quantile ((q99-q01)/2)"
        m["grip_scale_raw_units"] = float(s[7])
        out[tag] = m
    r0, r5 = out["pi0"]["ratio_raw"], out["pi05"]["ratio_raw"]
    n0, n5 = out["pi0"]["ratio_norm"], out["pi05"]["ratio_norm"]

    # The quantitative null, derived from the scale table so it tracks norm_stats automatically.
    null0 = float(scale["pi0"][7] / np.mean(scale["pi0"][:7]))
    null5 = float(scale["pi05"][7] / np.mean(scale["pi05"][:7]))
    null_ror = null0 / null5
    excess = float((r0 / r5) / null_ror)

    out["_verdict"] = {
        "prediction_confirmed": bool(r0 > r5),
        "ratio_of_ratios_raw": float(r0 / r5),
        "ratio_of_ratios_norm": float(n0 / n5),
        "gap_shrinks_in_norm_units": bool(
            abs(np.log(r0 / r5)) > abs(np.log(max(n0 / n5, 1e-9)))),
        "null_ratio_raw_pi0": null0,
        "null_ratio_raw_pi05": null5,
        "null_ratio_of_ratios": null_ror,
        "excess_over_null": excess,
        # >1 pi0 genuinely worse on gripper; <1 pi0 genuinely better but MASKED by the artifact.
        "capability_direction": (
            "pi0_worse_beyond_artifact" if excess > 1.10 else
            "pi0_better_masked_by_artifact" if excess < 0.909 else
            "fully_explained_by_normalizer"),
        "K": 8,
    }
    return out


def report(out):
    print(f"{'':6s} {'overall':>9s} {'arm':>9s} {'grip':>9s} {'grip/arm':>9s} "
          f"{'grip/arm (own norm units)':>28s}")
    for tag in ("pi0", "pi05"):
        m = out[tag]
        print(f"{tag:6s} {m['overall']:9.4f} {m['arm']:9.4f} {m['grip']:9.4f} "
              f"{m['ratio_raw']:9.4f} {m['ratio_norm']:28.4f}   [{m['norm_scheme']}]")
    print()
    for tag in ("pi0", "pi05"):
        print(f"{tag:6s} per-joint MAE: " + " ".join(f"{v:.3f}" for v in out[tag]["per_joint"]))
    print()
    v = out["_verdict"]
    print(f"PREDICTION  (grip/arm)_pi0 > (grip/arm)_pi05 : "
          f"{out['pi0']['ratio_raw']:.4f} vs {out['pi05']['ratio_raw']:.4f} -> "
          f"{'CONFIRMED' if v['prediction_confirmed'] else 'FALSIFIED'}")
    print(f"  raw-space ratio-of-ratios            = {v['ratio_of_ratios_raw']:.3f}x  "
          f"(normalization argument predicts > 1)")
    print(f"  own-normalized-units ratio-of-ratios = {v['ratio_of_ratios_norm']:.3f}x  "
          f"(if the raw gap is normalization, this should sit much closer to 1)")
    print(f"  gap shrinks in normalized units: "
          f"{'YES' if v['gap_shrinks_in_norm_units'] else 'NO'}")
    print()
    print(f"  QUANTITATIVE NULL (K-independent, from norm_stats): pi0 {v['null_ratio_raw_pi0']:.3f} "
          f"/ pi05 {v['null_ratio_raw_pi05']:.3f} -> ratio-of-ratios null = "
          f"{v['null_ratio_of_ratios']:.3f}x")
    print(f"  EXCESS OVER NULL = {v['excess_over_null']:.3f}x  -> {v['capability_direction']}")
    print(f"  (all measured numbers are K={v['K']}; the measured ratio drifts with K, the null "
          f"does not)")
    print()
    print("INTERPRETATION RULE: confirmed + shrinks => the pi0/pi0.5 gripper column difference is "
          "at least partly a normalization artifact, and the leaderboard footnote must say so. "
          "Survives the rescaling => real capability difference; say that instead.")


def load_scale():
    ns = json.loads(NS.read_text())["norm_stats"]["actions"]
    std = np.asarray(ns["std"], dtype=np.float64)
    q01 = np.asarray(ns["q01"], dtype=np.float64)
    q99 = np.asarray(ns["q99"], dtype=np.float64)
    return {"pi0": std, "pi05": (q99 - q01) / 2.0}


def selftest():
    """Fabricate two synthetic rows with a KNOWN injected gripper ratio and check that the
    verdict tracks it in BOTH directions. A check that can only print CONFIRMED is not a check."""
    scale = load_scale()
    rng = np.random.default_rng(0)
    gt = np.asarray(np.load(BENCH / "data" / "val_anchors.npz")["gt"], dtype=np.float64)
    A, K, D = gt.shape
    ok = True
    for name, (g0, g5) in (("pi0 worse on gripper", (3.0, 1.0)),
                           ("pi05 worse on gripper", (1.0, 3.0))):
        blocks = {}
        for tag, gmul in (("pi0", g0), ("pi05", g5)):
            err = rng.normal(0.0, 1.0, size=(A, K, D))
            err[..., 7] *= gmul                      # inject the gripper-vs-arm imbalance
            blocks[tag] = mae_block(gt + err, gt)
        out = analyze(blocks, scale)
        want = (g0 > g5)
        got = out["_verdict"]["prediction_confirmed"]
        status = "ok" if got == want else "MISMATCH"
        if got != want:
            ok = False
        print(f"  [{status}] {name:24s} injected grip x{g0}/{g5} -> "
              f"ratio_raw {out['pi0']['ratio_raw']:.3f} vs {out['pi05']['ratio_raw']:.3f}, "
              f"verdict={'CONFIRMED' if got else 'FALSIFIED'} (expected "
              f"{'CONFIRMED' if want else 'FALSIFIED'})")
    # --- positive control ON THE NULL ITSELF ------------------------------------------------
    # Fabricate the exact world the null describes: both models make the SAME error in their OWN
    # normalized space (same e_norm draw, each scaled by that model's own scale table). The raw
    # grip/arm ratios must then differ by exactly the null, and excess_over_null must be ~1 with
    # the verdict "fully_explained_by_normalizer". This checks the null is the RIGHT number, not
    # merely a number I computed -- without it, a wrong null would silently mislabel every result.
    e_norm = rng.normal(0.0, 1.0, size=(A, K, D))
    blocks = {tag: mae_block(gt + e_norm * scale[tag][None, None, :], gt)
              for tag in ("pi0", "pi05")}
    out = analyze(blocks, scale)
    v = out["_verdict"]
    excess_ok = abs(v["excess_over_null"] - 1.0) < 0.02
    verdict_ok = v["capability_direction"] == "fully_explained_by_normalizer"
    if not (excess_ok and verdict_ok):
        ok = False
    print(f"  [{'ok' if excess_ok and verdict_ok else 'MISMATCH'}] "
          f"{'null positive control':24s} equal error in own norm units -> "
          f"ror_raw {v['ratio_of_ratios_raw']:.3f} vs null {v['null_ratio_of_ratios']:.3f}, "
          f"excess {v['excess_over_null']:.4f} (want ~1.000), verdict={v['capability_direction']}")

    # --- the branch the directional test is BLIND to ----------------------------------------
    # pi0 genuinely BETTER on the gripper, by less than the artifact: raw space still shows
    # (grip/arm)_pi0 > (grip/arm)_pi05 so the directional test prints CONFIRMED, but the truth is
    # the opposite sign. excess_over_null must catch it.
    blocks = {}
    for tag, gmul in (("pi0", 0.6), ("pi05", 1.0)):
        blocks[tag] = mae_block(gt + e_norm * scale[tag][None, None, :] * np.where(
            np.arange(D) == 7, gmul, 1.0)[None, None, :], gt)
    out = analyze(blocks, scale)
    v = out["_verdict"]
    masked_ok = (v["prediction_confirmed"] and
                 v["capability_direction"] == "pi0_better_masked_by_artifact")
    if not masked_ok:
        ok = False
    print(f"  [{'ok' if masked_ok else 'MISMATCH'}] {'masked-inversion case':24s} "
          f"pi0 truly BETTER on grip (x0.6) -> directional says "
          f"{'CONFIRMED' if v['prediction_confirmed'] else 'FALSIFIED'} (misleading), "
          f"excess {v['excess_over_null']:.3f} -> {v['capability_direction']}")

    # the scale table itself must be non-degenerate, or ratio_norm is meaningless
    for tag, s in scale.items():
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            print(f"  [MISMATCH] scale[{tag}] has non-positive / non-finite entries: {s}")
            ok = False
    print(f"  scale j7: pi0 std={scale['pi0'][7]:.4f}  pi05 (q99-q01)/2={scale['pi05'][7]:.4f}  "
          f"ratio={scale['pi05'][7] / scale['pi0'][7]:.4f}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default=str(BENCH / "preds"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(BENCH / "logs" / "openpi_gripper_confound.json"),
                    help="point a PRE-FLIGHT run at /tmp so stand-in numbers cannot "
                         "impersonate the real pi0/pi0.5 result")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    A = np.load(BENCH / "data" / "val_anchors.npz")
    gt = A["gt"]
    scale = load_scale()

    blocks = {}
    for tag in ("pi0", "pi05"):
        p = pathlib.Path(args.preds_dir) / f"preds_{tag}.npz"
        if not p.exists():
            print(f"missing {p} -- rerun when both rows have landed")
            return
        z = np.load(p)
        pred = z["pred"].astype(np.float32)
        # anchor_idx is optional: predict_openpi.py always writes it, but the board convention is
        # POSITIONAL alignment to val_anchors (score.py relies on that and reads no index at all),
        # and other legs' npz omit the key. Fall back to identity rather than raising, so this tool
        # can also be pointed at another leg's file -- but only when the length matches exactly, so
        # a genuinely misaligned file still fails loudly instead of being silently reindexed.
        if "anchor_idx" in z:
            idx = z["anchor_idx"]
        elif len(pred) == len(gt):
            idx = np.arange(len(gt))
            print(f"[{tag}] no anchor_idx in npz -- assuming positional val_anchors order "
                  f"(board convention, {len(gt)} anchors)")
        else:
            raise SystemExit(f"[{tag}] {p.name} has no anchor_idx and length {len(pred)} != "
                             f"{len(gt)} val anchors -- cannot align, refusing to guess.")
        blocks[tag] = mae_block(pred, gt[idx])

    out = analyze(blocks, scale)
    # Provenance: record WHICH npz produced these numbers. A pre-flight run against stand-in files
    # (another leg's preds, to exercise the plumbing before pi0/pi0.5 land) otherwise writes
    # perfectly plausible pi0/pi05 numbers to the real results path, and nothing downstream can
    # tell them from the genuine article. Caught doing exactly that on 2026-08-05.
    out["source"] = {
        tag: {
            "path": str(pathlib.Path(args.preds_dir) / f"preds_{tag}.npz"),
            "md5": hashlib.md5((pathlib.Path(args.preds_dir) / f"preds_{tag}.npz").read_bytes()).hexdigest(),
        }
        for tag in ("pi0", "pi05")
    }
    report(out)
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
