#!/usr/bin/env python3
"""Board K-slope table: grip/arm error ratio as a function of horizon step k.

Why the ratio and not the raw columns: the gripper and the arm live in different units
(mm vs deg) and different parameterizations across legs, so their LEVELS are not
comparable. The way the ratio MOVES with k is a within-leg quantity, and its sign is
robust to both.

Two normalizations are reported and they answer different questions:
  raw ratio_k  = grip_k / arm_k                 -- group-internal, safe to compare across
                                                   k within one leg, NOT across legs
  ratio_k/null_k, null = hold-state's own ratio -- divides out the horizon growth that is
                                                   a property of the DATA (both channels
                                                   get harder with k) rather than of the
                                                   model. Cross-leg-comparable up to the
                                                   caveat below.

⚠ CAVEAT that must travel with the cross-leg column: openpi's ARM target is DELTA while
its gripper is ABSOLUTE, and ACT/n15 are absolute on both. A delta arm has a structurally
easier k=0 (the delta-0 predictor is a good answer there and stops being one as k grows),
so a delta leg's arm_k rises faster from a lower base -- which pushes its grip/arm ratio
DOWN with k for reasons that have nothing to do with the gripper. The null divides out the
data's horizon growth but NOT the parameterization's. So: slope SIGN within a leg is a
finding; slope MAGNITUDE across the delta/absolute boundary is not a like-for-like number.
See docs/CAVEATS.md section 5 (delta parameterization makes the pass line cheap).

Sources for the parameterization column (each MEASURED, none asserted):
  act    -- lerobot meta `action_parameterization`, residual-vs-hypothesis 7.3e-06 (abs)
            vs 126.0 (delta), separation 1.7e7
  n15    -- ckpt experiment_cfg/metadata.json `absolute: True` on all 4 modality keys
            (archive/n15_normalization_assert.py)
  pi0/05 -- ckpt norm_stats.json: action arm means ~0 (-0.046..-0.861) against state means
            of tens of degrees => DELTA arm; action gripper mean 14.022/std 26.085 vs state
            gripper 14.941/25.174 => ABSOLUTE gripper. Measured here, see `param_evidence`.
"""
import os
import json, pathlib, sys
import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
VAL_ANCHORS_MD5 = "cc34dda2a99ade1011c192e2b3e4e343"
NORM_STATS = (pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "pi05_nero_b2_train" / "pi05_b2train" /
              "29999" / "assets" / "local" / "pick_pink_sponge_b2_train" /
              "norm_stats.json")

PARAM = {  # (arm, gripper, source)
    "act":        ("absolute", "absolute", "lerobot meta action_parameterization (measured, sep 1.7e7)"),
    "gr00t_n15":  ("absolute", "absolute", "ckpt metadata.json absolute:True x4 (n15_normalization_assert)"),
    "pi0":        ("delta",    "absolute", "ckpt norm_stats action-vs-state means (measured below)"),
    "pi05":       ("delta",    "absolute", "ckpt norm_stats action-vs-state means (measured below)"),
}


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def measure_openpi_param():
    """Delta vs absolute from the shipped norm_stats: a delta target has ~zero mean on the
    joints while an absolute one carries the joint's working point. Reported with the
    numbers so a reader can disagree with the threshold rather than with a label."""
    d = json.load(open(NORM_STATS))
    n = d.get("norm_stats", d)
    a_mean = np.asarray(n["actions"]["mean"], float)
    s_mean = np.asarray(n["state"]["mean"], float)
    a_std = np.asarray(n["actions"]["std"], float)
    s_std = np.asarray(n["state"]["std"], float)
    return dict(
        norm_stats_md5=md5(NORM_STATS),
        action_mean=[round(float(x), 4) for x in a_mean],
        state_mean=[round(float(x), 4) for x in s_mean],
        arm_action_mean_absmax=float(np.abs(a_mean[:7]).max()),
        arm_state_mean_absmax=float(np.abs(s_mean[:7]).max()),
        arm_verdict="delta" if np.abs(a_mean[:7]).max() < 0.1 * np.abs(s_mean[:7]).max() else "absolute",
        grip_action_mean=float(a_mean[7]), grip_state_mean=float(s_mean[7]),
        grip_action_std=float(a_std[7]), grip_state_std=float(s_std[7]),
        grip_verdict="absolute" if abs(a_mean[7] - s_mean[7]) < 0.2 * abs(s_mean[7]) else "delta",
        note="thresholds are reported alongside the raw numbers on purpose; the separation "
             "here is ~2 orders (0.861 vs 103.1), not a borderline call",
    )


def main():
    got = md5(BENCH / "data" / "val_anchors.npz")
    if got != VAL_ANCHORS_MD5:
        sys.exit(f"[FATAL] val_anchors.npz md5 {got} != board pin {VAL_ANCHORS_MD5}")
    A = np.load(BENCH / "data" / "val_anchors.npz")
    gt = A["gt"].astype(np.float64)
    state = A["state"].astype(np.float64)
    N, K, D = gt.shape

    out = {"val_anchors_md5": got, "N": int(N), "K_scored": int(K),
           "openpi_param_evidence": measure_openpi_param(), "legs": {}}

    # hold-state null: the delta=0 predictor, computed on the same anchors
    hold = np.broadcast_to(state[:, None, :], (N, K, D))
    herr = np.abs(hold - gt)
    null_arm = herr[..., :7].mean(axis=(0, 2))
    null_grip = herr[..., 7].mean(axis=0)
    null_ratio = null_grip / null_arm

    def slope(y, x=None):
        x = np.arange(len(y), dtype=float) if x is None else x
        return float(np.polyfit(x, y, 1)[0])

    # --- OUT-OF-WINDOW gt, recovered without regenerating val_anchors.
    # openpi ships a 10-step chunk and its meta invites using the unscored tail; taken
    # literally that needs gt at K=10, i.e. a new val_anchors.npz, which would break the
    # board md5 pin. It isn't needed: anchors are stride-5 within an episode, so anchor i's
    # step 8/9 is anchor (i+1)'s step 3/4. The construction VALIDATES ITSELF on the 3-step
    # overlap it does not use (steps 5,6,7 of i == steps 0,1,2 of i+1) — measured bit-exact
    # on all 1377 pairs, maxdiff 0.0. 20 anchors (the last of each episode) have no
    # successor, so the extended curve runs on 1377/1397 and every k is recomputed on that
    # same subset — mixing a 1397-based k<=7 with a 1377-based k>=8 would put a population
    # change inside the trend being measured.
    KEXT, S = 16, int(A["stride"])
    KCOMMON = min(int(np.load(f)["pred"].shape[1]) for f in (BENCH / "preds").glob("preds_*.npz"))
    KCOMMON = min(KCOMMON, KEXT)
    ep, fr = A["episodes"], A["frames"]
    pos = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, fr))}
    MMAX = -(-(KEXT - 1 - (K - 1)) // S)                      # successors needed for k<KEXT
    succ = np.stack([np.array([pos.get((int(e), int(f) + S * m), -1) for e, f in zip(ep, fr)])
                     for m in range(MMAX + 1)])                # (MMAX+1, N)
    has = (succ >= 0).all(axis=0)
    gt_ext = np.empty((int(has.sum()), KEXT, D))
    # REDUNDANT-COVERAGE self-validation: frame f+k is reachable through every m with
    # 0 <= k-S*m <= K-1. Where two different m reach the same frame the two reads must be
    # bit-identical — that is the check, and it is on frames the extension actually USES,
    # not merely on the untouched overlap.
    dup_maxdiff, dup_n = 0.0, 0
    for k in range(KEXT):
        ms = [m for m in range(MMAX + 1) if 0 <= k - S * m <= K - 1]
        assert ms, f"k={k} unreachable with MMAX={MMAX}"
        reads = [gt[succ[m][has], k - S * m, :] for m in ms]
        gt_ext[:, k, :] = reads[0]
        for r in reads[1:]:
            dup_n += 1
            dup_maxdiff = max(dup_maxdiff, float(np.abs(reads[0] - r).max()))
    if dup_maxdiff != 0.0:
        sys.exit(f"[FATAL] redundant reads of the same frame disagree (maxdiff {dup_maxdiff}); "
                 f"the out-of-window gt reconstruction is invalid.")
    assert np.abs(gt_ext[:, :K, :] - gt[has]).max() == 0.0, "in-window rows must be untouched"
    ext = dict(mask=has, gt=gt_ext, n=int(has.sum()))
    out["out_of_window_reconstruction"] = dict(
        method=f"frame f+k read as anchor(i+m) step k-{S}*m, minimal m; stride {S}, same episode",
        K_extended=KEXT, successors_required=int(MMAX),
        n_anchors=int(has.sum()), n_dropped=int((~has).sum()),
        self_validation=f"{dup_n} redundantly-covered (k,m) reads must agree bit-exactly",
        redundant_read_maxdiff=dup_maxdiff,
        in_window_rows_unchanged=True, val_anchors_untouched=True,
        why=("openpi's meta invites using its unscored 10-step tail; taken literally that needs "
             "gt at K=10, i.e. a regenerated val_anchors.npz, which would break the board md5 "
             "pin. It is not needed — the tail is already IN val_anchors, spread across later "
             "anchors. Applies to every leg: act ships 100 steps, n15 16, openpi 10."),
    )

    def extend(P, ext, Kf):
        Kf = min(Kf, ext["gt"].shape[1])
        g = ext["gt"][:, :Kf, :]
        p = P[ext["mask"]][:, :Kf, :]
        e = np.abs(p - g)
        arm = [float(e[:, k, :7].mean()) for k in range(Kf)]
        grip = [float(e[:, k, 7].mean()) for k in range(Kf)]
        r = [g_ / a for g_, a in zip(grip, arm)]
        return dict(n_anchors=ext["n"], K=int(Kf), arm=arm, grip=grip, ratio=r,
                    slope_in_window=slope(np.array(r[:K])),
                    # COMMON range: legs ship different horizons (openpi 10, n15 16, act 100),
                    # so slope_full is over a different k-range per leg and is NOT a cross-leg
                    # number. KCOMMON = the shortest shipped horizon; same trap as mixing anchor
                    # populations, one axis over.
                    slope_common=slope(np.array(r[:KCOMMON])), K_common=KCOMMON,
                    slope_full_OWN_RANGE_ONLY=slope(np.array(r)),
                    monotone=bool(all(b >= a for a, b in zip(r, r[1:]))),
                    argmin_k=int(np.argmin(r)),
                    ratio_out_of_window=r[K:],
                    note=f"k>={K} is OUT OF THE SCORED WINDOW — an out-of-sample test of the "
                         f"trend the scored table reports. Every k on the same "
                         f"{ext['n']} anchors, so the population is not part of the trend.")

    out["legs"]["[hold-state]"] = dict(
        arm_param="n/a (delta=0 predictor)", grip_param="n/a", param_source="score.py pass line",
        arm=[float(v) for v in null_arm], grip=[float(v) for v in null_grip],
        ratio=[float(v) for v in null_ratio], ratio_over_null=[1.0] * K,
        slope_ratio=slope(null_ratio), slope_ratio_over_null=0.0,
        anchor_idx_checked=False, anchor_idx_note="derived from val_anchors itself",
    )

    for f in sorted((BENCH / "preds").glob("preds_*.npz")):
        name = f.stem.replace("preds_", "")
        z = np.load(f)
        P = z["pred"].astype(np.float64)
        if P.shape[0] != N:
            print(f"[skip {name}] anchor count {P.shape[0]} != {N}")
            continue
        # PREREGISTERED gate: a preds file that ships anchor_idx must be in val_anchors order.
        # A permuted file scores plausibly (same rows, wrong pairing) and would silently
        # produce a K-slope curve that is pure noise.
        aidx_ok = None
        if "anchor_idx" in z.files:
            aidx_ok = bool((z["anchor_idx"] == np.arange(N)).all())
            if not aidx_ok:
                sys.exit(f"[FATAL] {name}: anchor_idx != arange({N}) — preds are not in "
                         f"val_anchors order; every paired number below would be wrong.")
        Kf = P.shape[1]
        err = np.abs(P - gt[:, :min(Kf, K), :]) if Kf <= K else None
        arm, grip = [], []
        for k in range(Kf):
            if k < K:
                e = np.abs(P[:, k, :] - gt[:, k, :])
            else:
                e = None  # out-of-window: no gt beyond K in val_anchors
            arm.append(float(e[:, :7].mean()) if e is not None else None)
            grip.append(float(e[:, 7].mean()) if e is not None else None)
        ratio = [g / a if (g is not None and a) else None for g, a in zip(grip, arm)]
        ron = [r / n if r is not None else None for r, n in zip(ratio, null_ratio)]
        ap, gp, src = PARAM.get(name, ("?", "?", "NOT DECLARED — ask the leg owner"))
        out["legs"][name] = dict(
            arm_param=ap, grip_param=gp, param_source=src,
            horizon_shipped=int(Kf), horizon_scored=int(K),
            arm=arm[:K], grip=grip[:K], ratio=ratio[:K], ratio_over_null=ron[:K],
            slope_ratio=slope(np.array(ratio[:K])),
            slope_ratio_over_null=slope(np.array(ron[:K])),
            ratio_k0=ratio[0], ratio_k7=ratio[K - 1],
            ratio_pct_change_k0_to_k7=float((ratio[K - 1] / ratio[0] - 1) * 100),
            anchor_idx_checked=aidx_ok,
            out_of_window_steps_shipped=int(max(0, Kf - K)),
        )
        if Kf > K and ext is not None:
            out["legs"][name]["extended"] = extend(P, ext, Kf)

    with open(BENCH / "logs" / "kslope_table.json", "w") as fh:
        json.dump(out, fh, indent=1)

    ev = out["openpi_param_evidence"]
    print(f"openpi param (measured): arm={ev['arm_verdict']} "
          f"(|action mean|max {ev['arm_action_mean_absmax']:.3f} vs state "
          f"{ev['arm_state_mean_absmax']:.3f}), grip={ev['grip_verdict']} "
          f"(action {ev['grip_action_mean']:.3f}/{ev['grip_action_std']:.3f} vs state "
          f"{ev['grip_state_mean']:.3f}/{ev['grip_state_std']:.3f})\n")
    print(f"{'leg':<14}{'param':<12}{'k0':>7}{'k7':>7}{'slope':>8}{'%chg':>8}   ratio/null k0..k7")
    for name, r in out["legs"].items():
        p = f"{r['arm_param'][:3]}/{r['grip_param'][:3]}"
        rn = " ".join(f"{v:5.3f}" for v in r["ratio_over_null"])
        pc = r.get("ratio_pct_change_k0_to_k7")
        print(f"{name:<14}{p:<12}{r['ratio'][0]:7.3f}{r['ratio'][7]:7.3f}"
              f"{r['slope_ratio']:8.4f}{(pc if pc is not None else 0):+7.1f}%   {rn}")
    oo = out["out_of_window_reconstruction"]
    print(f"\nout-of-window gt (K={oo['K_extended']}): {oo['method']} | n={oo['n_anchors']} "
          f"(dropped {oo['n_dropped']}) | {oo['self_validation']}: maxdiff "
          f"{oo['redundant_read_maxdiff']}")
    for name, r in out["legs"].items():
        if "extended" in r:
            e = r["extended"]
            print(f"  {name:<10} ratio k0.. " + " ".join(f"{v:5.3f}" for v in e["ratio"])
                  + f"\n  {'':<10} slope: in-window(k<8) {e['slope_in_window']:+.4f} | "
                    f"COMMON(k<{e['K_common']}) {e['slope_common']:+.4f} | own-range(k<{e['K']}) "
                    f"{e['slope_full_OWN_RANGE_ONLY']:+.4f} | monotone {e['monotone']} "
                    f"(min at k={e['argmin_k']})")
    print("\nper-k RAW (arm | grip):")
    for name, r in out["legs"].items():
        print(f"  {name:<14} arm  " + " ".join(f"{v:5.3f}" for v in r["arm"]))
        print(f"  {'':<14} grip " + " ".join(f"{v:5.3f}" for v in r["grip"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
