#!/usr/bin/env python3
"""The two calibers are two POINTS on a one-parameter family. Measure the whole family.

Board rule (d) currently names two weightings -- frame-length-weighted and episode-equal --
and three findings have been built on the gap between them:

  * gr00t-n15 (me): the two move model and baseline in opposite directions, so a small
    margin can change SIGN (ACT val arm -0.1404 -> +0.0029).
  * gr00t-n17: the DISPLACEMENT |gap_EQ - gap_frame| across 9 board rows is 0.028-0.143,
    proposing the rule "a gap smaller than its channel's displacement (~0.14) is not
    determined by this val set".
  * lerobot-setup: the caliber also moves the jackknife SE, so a VERDICT can flip at 2 sigma
    while the sign is stable (ACT TOTAL 1.89 sigma -> 2.97 sigma).

All three are two-point statements. Two points cannot distinguish "0.14 is the scale of this
effect" from "0.14 is what you get at these particular two points, and the effect is larger
elsewhere in the space of defensible weightings". They also cannot say WHERE between the two
calibers a sign or a verdict turns over -- i.e. whether the flip needs the full swing to the
endpoint or happens immediately.

So embed both calibers in the obvious family and sweep it:

    w_e proportional to L_e ** alpha        L_e = anchors in episode e

    alpha = 1  ->  every ANCHOR equal   == frame-length-weighted (plain mean over anchors)
    alpha = 0  ->  every EPISODE equal  == episode-equal
    alpha < 0  ->  short episodes upweighted (not defensible; included only to test whether
                   the frame<->episode segment is the extremum or just a chord)

MAE(alpha) = sum_e w_e m_e / sum_e w_e  with m_e the per-episode mean error, which is exact:
at alpha=1 it collapses to the grand mean over anchors, at alpha=0 to the mean of means.

Everything is a functional of the 20 per-episode means, so the whole sweep including the
paired delete-one-episode jackknife costs a few milliseconds. CPU only. No GPU. Read-only on
val_anchors.npz + preds/*.npz.

Power controls for the caliber machinery itself (both must hold, or the sweep is not
measuring what it claims):
  A. m_e constant across episodes -> displacement must be EXACTLY 0 at every alpha
     (no weighting can move a constant), and
  B. m_e proportional to L_e      -> displacement must be nonzero,
so a bug that silently ignores `alpha` fails B while passing A, and a bug that scrambles the
weights fails A.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
CH = {"TOTAL": slice(0, 8), "ARM": slice(0, 7), "GRIP": slice(7, 8)}


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def wmean(m, L, alpha, keep=None):
    """Weighted mean of per-episode means under w_e = L_e**alpha."""
    if keep is not None:
        m, L = m[keep], L[keep]
    w = L.astype(np.float64) ** alpha
    return float((w * m).sum() / w.sum())


def margin_sigma(m_hold, m_model, L, alpha):
    """Paired delete-one-episode jackknife of (hold - model) at one alpha.

    PAIRED: drop the episode from both sides and recompute, because the two per-episode error
    vectors are correlated (negatively for a hold-state line), which AMPLIFIES rather than
    cancels -- quadrature of two marginal SEs is not conservative here.
    """
    full = wmean(m_hold, L, alpha) - wmean(m_model, L, alpha)
    n = len(L)
    loo = np.array([
        wmean(m_hold, L, alpha, keep=np.arange(n) != i)
        - wmean(m_model, L, alpha, keep=np.arange(n) != i)
        for i in range(n)
    ])
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return full, se, (abs(full) / se if se else float("inf"))


def crossing(alphas, y, level):
    """First alpha in the swept order where y crosses `level`; None if it never does."""
    d = np.asarray(y) - level
    s = np.sign(d)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if not len(idx):
        return None
    i = idx[0]
    # linear interpolation between the bracketing grid points
    a0, a1, d0, d1 = alphas[i], alphas[i + 1], d[i], d[i + 1]
    return float(a0 + (a1 - a0) * (-d0) / (d1 - d0))


def per_episode_means(err, eps, uniq, sl):
    # accumulate in float64: the preds are float32, and a float32 mean over ~500 values leaves
    # ~1e-8 of slop -- harmless for the 4-decimal numbers, but it is 10x too coarse to check the
    # TOTAL == (7*ARM + GRIP)/8 identity, which is the only test that the slicing is right.
    return np.array([err[eps == u][..., sl].mean(dtype=np.float64) for u in uniq], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(BENCH / "logs" / "caliber_family_val.json"))
    args = ap.parse_args()

    a = np.load(BENCH / "data" / "val_anchors.npz")
    gt, state, eps = a["gt"], a["state"], a["episodes"]
    K = int(a["K"])
    uniq, L = np.unique(eps, return_counts=True)
    L = L.astype(np.float64)

    src = {"val_anchors.npz": md5(BENCH / "data" / "val_anchors.npz")}
    models = {}
    for name, rel in (("act", "preds/preds_act.npz"), ("gr00t_n15", "preds/preds_gr00t_n15.npz")):
        z = np.load(BENCH / rel)
        src[rel] = md5(BENCH / rel)
        if "anchor_idx" in z:
            assert (z["anchor_idx"] == np.arange(len(eps))).all(), f"{name}: anchor order"
        models[name] = np.abs(z["pred"][:, :K, :] - gt)
    hold = np.abs(np.broadcast_to(state[:, None, :], gt.shape) - gt)

    # ---- power controls on the caliber machinery (see docstring) ----------------------
    # NB: control A is `== 0` in theory and 2.2e-16 in float64 (the two weightings sum the same
    # constant in different orders). Written as exact equality it fails on arithmetic that is
    # working correctly -- the same exact-comparison-on-a-float family as the round(x,3)==const
    # pass-line cliff, just landing on the false-FAIL side. Tolerance, and require the two
    # controls to be orders of magnitude apart so the pair still has power in both directions.
    CTRL_TOL = 1e-12
    const = np.full(len(L), 1.234)
    ctrl_const = abs(wmean(const, L, 0.0) - wmean(const, L, 1.0))
    ctrl_prop = abs(wmean(L / L.mean(), L, 0.0) - wmean(L / L.mean(), L, 1.0))
    assert ctrl_const < CTRL_TOL, f"control A failed: constant moved by {ctrl_const}"
    assert ctrl_prop > 1e-6, f"control B failed: alpha has no effect ({ctrl_prop})"
    assert ctrl_prop / max(ctrl_const, 1e-300) > 1e6, "controls not separated"

    alphas = np.unique(np.concatenate([np.linspace(-1.0, 2.0, 301), [0.0, 1.0]]))
    out = {
        "what": "MAE and paired (hold - model) significance under w_e = L_e**alpha",
        "input_md5": src,
        "n_val_episodes": int(len(uniq)),
        "episode_lengths": L.astype(int).tolist(),
        "K_scored": K,
        "power_control_constant_displacement": ctrl_const,
        "power_control_proportional_displacement": ctrl_prop,
        "alpha_grid": {"lo": float(alphas.min()), "hi": float(alphas.max()), "n": int(len(alphas))},
        "n_eff": {},
        "channels": {},
    }

    # effective sample size of the episode weights: (sum w)^2 / sum w^2. At alpha=0 this is
    # exactly the episode count; at alpha=1 it is the classic unequal-cluster design effect.
    for al in (0.0, 0.5, 1.0):
        w = L ** al
        out["n_eff"][f"alpha={al}"] = float(w.sum() ** 2 / (w ** 2).sum())

    print(f"val: {len(uniq)} episodes, lengths {int(L.min())}-{int(L.max())}, "
          f"n_eff(alpha=1)={out['n_eff']['alpha=1.0']:.2f} of {len(uniq)}")
    print()

    for chan, sl in CH.items():
        mh = per_episode_means(hold, eps, uniq, sl)
        out["channels"][chan] = {}
        for mname, err in models.items():
            mm = per_episode_means(err, eps, uniq, sl)
            rows = [margin_sigma(mh, mm, L, al) for al in alphas]
            marg = np.array([r[0] for r in rows])
            se = np.array([r[1] for r in rows])
            sig = np.array([r[2] for r in rows])
            i1 = int(np.argmin(np.abs(alphas - 1.0)))
            i0 = int(np.argmin(np.abs(alphas - 0.0)))
            seg = (alphas >= 0.0) & (alphas <= 1.0)
            disp = marg - marg[i1]

            d = {
                "mae_frame_alpha1": wmean(mm, L, 1.0),
                "mae_episode_alpha0": wmean(mm, L, 0.0),
                "hold_frame_alpha1": wmean(mh, L, 1.0),
                "hold_episode_alpha0": wmean(mh, L, 0.0),
                "margin_alpha1_frame": float(marg[i1]),
                "margin_alpha0_episode": float(marg[i0]),
                "se_alpha1_frame": float(se[i1]),
                "se_alpha0_episode": float(se[i0]),
                "sigma_alpha1_frame": float(sig[i1]),
                "sigma_alpha0_episode": float(sig[i0]),
                "two_point_displacement": float(abs(marg[i0] - marg[i1])),
                "max_abs_displacement_on_segment": float(np.abs(disp[seg]).max()),
                "argmax_abs_displacement_alpha_on_segment": float(alphas[seg][np.abs(disp[seg]).argmax()]),
                "monotone_on_segment": bool(
                    np.all(np.diff(marg[seg]) >= -1e-12) or np.all(np.diff(marg[seg]) <= 1e-12)
                ),
                "max_abs_displacement_full_grid": float(np.abs(disp).max()),
                "argmax_abs_displacement_alpha_full_grid": float(alphas[np.abs(disp).argmax()]),
                "sign_flip_alpha": crossing(alphas[seg][::-1], marg[seg][::-1], 0.0),
                "sigma2_crossing_alpha": crossing(alphas[seg][::-1], sig[seg][::-1], 2.0),
                "sigma_monotone_on_segment": bool(
                    np.all(np.diff(sig[seg]) >= -1e-12) or np.all(np.diff(sig[seg]) <= 1e-12)
                ),
                "sigma_min_on_segment": float(sig[seg].min()),
                "sigma_max_on_segment": float(sig[seg].max()),
                "se_min_on_segment": float(se[seg].min()),
                "se_max_on_segment": float(se[seg].max()),
                # Why is the frame-weighted SE the larger one? Two candidate mechanisms, and
                # they are separable. (i) DESIGN EFFECT: unequal episode weights shrink the
                # effective number of clusters, predicting se(a=1)/se(a=0) = sqrt(n/n_eff)
                # under equal-variance per-episode terms. (ii) HETEROGENEITY: the paired
                # difference genuinely disperses more across episodes when long ones are
                # upweighted. Report the measured ratio against the design-effect prediction
                # so the residual is attributed rather than assumed.
                "se_ratio_frame_over_episode": float(se[i1] / se[i0]),
                "se_ratio_predicted_by_design_effect": float(
                    np.sqrt(len(L) / (L.sum() ** 2 / (L ** 2).sum()))
                ),
                "se_inflation_share_from_design_effect": float(
                    np.log(np.sqrt(len(L) / (L.sum() ** 2 / (L ** 2).sum()))) / np.log(se[i1] / se[i0])
                ) if se[i1] > se[i0] else None,
                "curve": {
                    "alpha": [float(x) for x in alphas[::10]],
                    "margin": [float(x) for x in marg[::10]],
                    "se": [float(x) for x in se[::10]],
                    "sigma": [float(x) for x in sig[::10]],
                },
            }
            out["channels"][chan][mname] = d
            print(f"{chan:6} {mname:10} margin {d['margin_alpha1_frame']:+.4f} (a=1) -> "
                  f"{d['margin_alpha0_episode']:+.4f} (a=0)   sigma {d['sigma_alpha1_frame']:5.2f} -> "
                  f"{d['sigma_alpha0_episode']:5.2f}   |disp| 2pt {d['two_point_displacement']:.4f} "
                  f"max-on-seg {d['max_abs_displacement_on_segment']:.4f} "
                  f"mono {str(d['monotone_on_segment']):5}  "
                  f"flip@a={d['sign_flip_alpha']}  2sig@a={d['sigma2_crossing_alpha']}")

    # ---- where the displacement actually lives -----------------------------------------
    # A 9-row table of pairwise displacements (3 channels x 3 pairs) looks like 9 samples of
    # "how big is reweighting", but a pairwise displacement is a DIFFERENCE of the two rows'
    # own displacements, so per channel there are only 2 free numbers, and TOTAL is fixed by
    # ARM and GRIP. Nine renderings of four numbers. Report the four.
    per_model = {}
    for chan, sl in CH.items():
        mh = per_episode_means(hold, eps, uniq, sl)
        per_model[chan] = {"hold": wmean(mh, L, 0.0) - wmean(mh, L, 1.0)}
        for mname, err in models.items():
            mm = per_episode_means(err, eps, uniq, sl)
            per_model[chan][mname] = wmean(mm, L, 0.0) - wmean(mm, L, 1.0)
    out["per_model_displacement_alpha0_minus_alpha1"] = per_model
    out["degrees_of_freedom_note"] = (
        "pairwise displacement = difference of two per-model displacements, so a 3-model "
        "3-channel table has 2 free values per channel and TOTAL is determined by ARM+GRIP: "
        "4 independent numbers, not 9"
    )
    out["displacement_is_carried_by_the_pass_line"] = {
        "hold_displacement_by_channel": {c: per_model[c]["hold"] for c in CH},
        "model_displacement_range": [
            min(per_model[c][m] for c in CH for m in models),
            max(per_model[c][m] for c in CH for m in models),
        ],
        "reading": "hold moves +0.064..+0.067 on ALL THREE channels and every model moves the "
                   "other way, so any gap measured against the pass line inherits a ~0.065 "
                   "floor; model-vs-model pairs do not",
    }

    # TOTAL = (7*ARM + 1*GRIP)/8 holds for ANY linear weighting, so it is an arithmetic
    # identity, not evidence about the mechanism. Verify it numerically anyway: if it ever
    # failed, the channel slicing would be wrong.
    ident = {}
    for mname in models:
        for al, tag in ((1.0, "alpha1"), (0.0, "alpha0")):
            t = out["channels"]["TOTAL"][mname][f"margin_{tag}_frame" if al == 1.0 else f"margin_{tag}_episode"]
            ar = out["channels"]["ARM"][mname][f"margin_{tag}_frame" if al == 1.0 else f"margin_{tag}_episode"]
            gr = out["channels"]["GRIP"][mname][f"margin_{tag}_frame" if al == 1.0 else f"margin_{tag}_episode"]
            ident[f"{mname}_{tag}"] = abs(t - (7 * ar + gr) / 8)
    out["total_equals_7arm_plus_grip_over_8_maxdev"] = max(ident.values())
    out["identity_note"] = ("exact for any linear weighting; holding to 1e-12 tests the channel "
                            "slicing, it is not evidence that all three channels move")
    assert out["total_equals_7arm_plus_grip_over_8_maxdev"] < 1e-9, ident

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nTOTAL == (7*ARM + GRIP)/8 max dev {out['total_equals_7arm_plus_grip_over_8_maxdev']:.2e} "
          f"(identity, not a coincidence)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
