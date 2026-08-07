#!/usr/bin/env python3
"""Sampling-variance footnote for the flow-matching openpi backbones (CONTRACT.md, mandatory).

pi0/pi0.5 draw a fresh Gaussian noise chunk per infer, so a single-sample MAE is not directly
comparable to a deterministic backbone (act / openvla_oft). This measures how much of a
backbone's MAE is sampling luck: same checkpoint, same 200 anchors, 5 different noise seeds.

Loads the policy ONCE and decodes each val episode ONCE, evaluating all seeds per anchor, so the
whole footnote costs ~1 extra predict pass rather than 5.

Writes to logs/variance/ (NEVER preds/ -- must not enter the leaderboard).

Cross-check: seed 0 here must be bit-identical to the corresponding rows of preds/preds_<bb>.npz,
since both use noise = normal(fold_in(key(seed), global_anchor_index)).

Everything that can fail is checked BEFORE the policy is loaded, so a fatal costs seconds rather
than a 40-minute GPU pass: subset identity (md5), pass-line recomputation, and the noise-collision
measurement all run up front.
"""
import argparse, hashlib, json, pathlib, sys, time

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
SRC_REPO = "local/pick_pink_sponge_b2"
PROMPT = "pick the pink sponge and place it in the blue bucket"

# The board's shared 200-anchor subset. IDENTITY, not existence (gr00t-n17 2026-08-05): a wrong
# subset and a missing subset are both possible, and the wrong one is worse -- it publishes a
# footnote on different anchors while the json still reads "n_anchors": 200. The old code here
# warned and fell back to linspace, which overlaps the shared set by 24/200.
SUBSET_MD5 = "9007df1fcec5eca2a1f01591c6266b43"
SUBSET_NAME = "variance_subset200.npy"

# Full-1397 hold-state pass line (score.py). Recomputed from val_anchors.npz at runtime and
# required to agree to 3 dp: these constants are the tripwire for "val_anchors.npz changed under
# us", which would silently make every subset/full comparison in this file incomparable to the
# board. Mismatch => SystemExit BEFORE anything is written.
PASS_FULL = {"mae": 3.072, "arm": 2.535, "grip": 6.829}


def md5_of(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def resolve_subset(primary, allow_nonstandard=False, subset_n=200, total=1397):
    """Locate the shared anchor subset and gate it on CONTENT, not existence.

    Search order: the path we were given -> logs/variance/ -> logs/ (openpi is the leg that
    consolidated artifacts into logs/variance/, so the file can legitimately live in either).
    Returns (path_or_None, sel, md5_or_None, is_standard, description).
    """
    cands, seen = [], set()
    for p in [pathlib.Path(primary), BENCH / "logs" / "variance" / SUBSET_NAME,
              BENCH / "logs" / SUBSET_NAME]:
        if p not in seen:
            seen.add(p)
            cands.append(p)
    found = []
    for p in cands:
        if not p.is_file():
            continue
        m = md5_of(p)
        found.append((p, m))
        if m == SUBSET_MD5:
            return p, np.load(p).astype(int), m, True, f"shared file {p} (md5 {m})"
    if not allow_nonstandard:
        lines = ["[FATAL] shared variance subset not found with the expected content.",
                 f"  expected md5 {SUBSET_MD5} ({SUBSET_NAME})"]
        lines += [f"  searched {p}" + ("" if p.is_file() else "  (absent)") for p in cands]
        lines += [f"  present but WRONG CONTENT: {p} md5 {m}" for p, m in found]
        lines.append("  A footnote on a different anchor set is not comparable to the other "
                     "backbones and is worse than no footnote (the json would still say 200 "
                     "anchors). Re-run with --allow-nonstandard-subset only if you intend that.")
        raise SystemExit("\n".join(lines))
    if found:
        p, m = found[0]
        return p, np.load(p).astype(int), m, False, f"NON-STANDARD file {p} (md5 {m})"
    sel = np.unique(np.linspace(0, total - 1, subset_n).round().astype(int))
    return None, sel, None, False, "NON-STANDARD linspace fallback (overlaps shared set 24/200)"


def pinned_noise(seed: int, g: int, h: int, adim: int) -> np.ndarray:
    """The shipped noise generator: noise = normal(fold_in(key(seed), global_anchor_index)).

    Keyed by the GLOBAL anchor index, so a row's prediction is a pure function of
    (ckpt, anchor, seed) and cannot depend on iteration order or on which subset is being run.
    predict_openpi.py builds the same expression independently; --ref-preds proves they agree
    byte-for-byte, which is stronger evidence than sharing a symbol.
    """
    import jax
    import jax.numpy as jnp
    return np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(int(seed)), int(g)),
                                        (h, adim), dtype=jnp.float32))


def noise_fingerprint(digests) -> str:
    """Order-independent fingerprint of a MULTISET of noise-tensor md5s.

    Sorted, so the gate (which walks seed-major over `sel`) and the driver (which walks
    episode-grouped) can be compared without either being rewritten to match the other's order --
    order-independence is the property we want, not an inconvenience to work around.
    """
    return hashlib.md5("|".join(sorted(digests)).encode()).hexdigest()


def verify_driver_noise(noise_md5, gate_fingerprint: str) -> dict:
    """Close the gap between what the GATE measured and what the DRIVER actually fed to infer.

    measure_noise_collisions proves `pinned_noise(s, g)` does not alias over the subset. It does
    NOT prove the production loop calls it with those arguments -- and that gap is precisely where
    gr00t-n17's real defect lived: a correct generator, a correct-looking declared field, and a
    caller that passed the wrong index, so the check and the run were measuring different things.
    A local row index `r` substituted for the global `g` would leave the gate at 0 collisions and
    still alias every draw.

    So the driver hashes each noise chunk at the point it hands it to policy.infer, and here we
    require (a) the driver's own multiset is collision-free, and (b) it is the SAME multiset the
    gate certified. Any (s, g) divergence changes the fingerprint.
    """
    flat = np.asarray(noise_md5).reshape(-1).tolist()
    if any(not d for d in flat):
        raise SystemExit(f"[FATAL] driver noise ledger has {sum(1 for d in flat if not d)} "
                         f"unwritten cells of {len(flat)} -- the loop skipped (draw, anchor) "
                         f"pairs, so preds contains rows no noise was recorded for.")
    collisions = len(flat) - len(set(flat))
    fp = noise_fingerprint(flat)
    if collisions:
        raise SystemExit(f"[FATAL] the DRIVER fed {collisions} duplicate noise chunks to infer "
                         f"({len(flat)} draws, {len(set(flat))} distinct). The pre-GPU gate passed, "
                         f"so the generator is fine and the CALL SITE is wrong. Sampling std from "
                         f"this run is understated -- not publishable.")
    if fp != gate_fingerprint:
        raise SystemExit(f"[FATAL] driver/gate noise multiset MISMATCH: driver {fp} vs gate "
                         f"{gate_fingerprint}. Same count, same collision-freedom, DIFFERENT "
                         f"tensors -- the loop is not calling pinned_noise with the (seed, global "
                         f"anchor index) pairs the gate certified.")
    return {"driver_noise_chunks": len(flat),
            "driver_noise_collisions_measured": collisions,
            "driver_noise_multiset_fingerprint": fp,
            "driver_matches_gate_multiset": True,
            "tapped_at": "the value handed to policy.infer(noise=), not re-derived from the formula"}


def measure_noise_collisions(sel, seeds, h, adim) -> dict:
    """MEASURE that the draws use non-aliasing noise -- do not derive it from the formula.

    `std == 0` and `bit_exact_across_draws` are both computed from the same arrays, so neither can
    tell "the model is deterministic" from "the harness reused a seed". Declared seed fields are
    no better: gr00t-n17's formula had a correct prime stride yet its driver never varied the
    repeat, so the EFFECTIVE stride was 1 and 254/1000 tensors were shared across draws while a
    literal-derived field reported "exceeds_anchor_count: True".

    So: run the SHIPPED pinned_noise over the REAL subset indices for every seed, hash each
    tensor, count repeats. Two power controls reproduce the two ways this leg could alias; a check
    that cannot fail reports 0 for free, so if either control comes back 0 the measurement is VOID.
    """
    import jax
    import jax.numpy as jnp

    def _dupes(fn):
        seen, hits, ex, digests = {}, 0, [], []
        for s in seeds:
            for g in sel:
                d = hashlib.md5(fn(int(s), int(g)).tobytes()).hexdigest()
                digests.append(d)
                if d in seen:
                    hits += 1
                    if len(ex) < 5:
                        ex.append({"first": seen[d], "second": [int(s), int(g)]})
                else:
                    seen[d] = [int(s), int(g)]
        return hits, ex, digests

    hits, ex, digests = _dupes(lambda s, g: pinned_noise(s, g, h, adim))

    # POWER CONTROL A -- n17's bug shape: seed = base + global index, i.e. an ADDITIVE stride of 1
    # between draws, so (draw 0, anchor g+1) reproduces (draw 1, anchor g).
    ctl_a, _, _ = _dupes(lambda s, g: np.asarray(
        jax.random.normal(jax.random.key(s + g), (h, adim), dtype=jnp.float32)))
    # POWER CONTROL B -- this leg's own most plausible slip: forget to fold in the anchor index,
    # so all N anchors within a draw share one tensor.
    ctl_b, _, _ = _dupes(lambda s, g: np.asarray(
        jax.random.normal(jax.random.key(s), (h, adim), dtype=jnp.float32)))

    ok = ctl_a > 0 and ctl_b > 0
    if not ok:
        verdict = "VOID (a power control did not trip -- this check has no power)"
    elif hits:
        verdict = "FAIL (draws share noise tensors -- sampling std is biased LOW)"
    else:
        verdict = "PASS (draws are non-aliasing)"
    return {
        "check": "per_anchor_noise_collisions_across_draws",
        "measured_on": "shipped variance_openpi.pinned_noise, real subset indices",
        "n_noise_tensors": len(seeds) * len(sel),
        "noise_shape": [int(h), int(adim)],
        "collisions": hits,
        "collision_examples": ex,
        # Fingerprint of the exact tensors this gate certified, so the driver can prove it fed
        # THOSE and not merely "some collision-free set" (see verify_driver_noise).
        "noise_multiset_fingerprint": noise_fingerprint(digests),
        "power_control_A_additive_stride1": ctl_a,
        "power_control_B_anchor_not_folded_in": ctl_b,
        "power_controls_ok": bool(ok),
        "verdict": verdict,
        # Board key contract: a cross-leg reader that looks for its OWN key name, misses, and
        # accepts the bare 0 turns a power-controlled check into an uncontrolled one (n17 hit
        # exactly this reading n15's json). So publish control A under every name any leg is
        # known to look for, and list the aliases so the next reader can see they are one number.
        "power_control_collisions": ctl_a,
        "seed_check_power_control_collisions": ctl_a,
        "seed_collision_power_control": ctl_a,
        "power_control_key_aliases": ["power_control_A_additive_stride1",
                                      "power_control_collisions",
                                      "seed_check_power_control_collisions",
                                      "seed_collision_power_control"],
        # Which of the two controls actually carries the weight, stated so the board does not
        # read the shared 254 as corroboration. A's count is a property of THIS anchor set (its
        # min inter-anchor gap is 1); on a sparse subset an effective stride of 1 collides zero
        # times, which reads as VOID here (fails safe) but makes it a loose gate. B's count is
        # fixed by construction -- (200-1)*5 = 995 duplicates whatever the indices are.
        "power_control_A_anchor_set_dependent": True,
        "power_control_B_expected_by_construction": (len(sel) - 1) * len(seeds),
        "power_control_B_matches_construction": ctl_b == (len(sel) - 1) * len(seeds),
        "power_control_note": (
            "A=254 is the same number n15/n17/lerobot measured, but that is cross-implementation "
            "agreement on a deterministic function of (anchor indices x stride 1), NOT independent "
            "corroboration -- it is one quantity computed four times. The independent coverage is "
            "that A and B are DIFFERENT bug shapes (254 vs 995) and both trip."),
    }


def _weighted(err, ep_ids):
    """(frame-length-weighted mean, episode-equal-weighted mean) of a per-anchor error array."""
    flat = err.reshape(len(ep_ids), -1)
    per_anchor = flat.mean(axis=1)
    per_ep = [per_anchor[ep_ids == e].mean() for e in np.unique(ep_ids)]
    return float(per_anchor.mean()), float(np.mean(per_ep))


def _hold_err(state, gt, K):
    hold = np.broadcast_to(state[:, None, :], (len(gt), K, gt.shape[2]))
    return np.abs(hold - gt)


def subset_design(sel, eps_full, state, gt, K, ref_pred_full=None) -> dict:
    """Board rule (d): the footnote must state HOW the 200 anchors were drawn, HOW they are
    weighted, and carry a pass line recomputed ON THE SUBSET.

    Comparing a subset MAE to the full-1397 pass line is not a small error and it is not
    unbiased -- it systematically flatters the model, because the subset is episode-BALANCED
    while the full population is frame-length-weighted, and the two move the model and the pass
    line in OPPOSITE directions (so the ratio moves more than either term).
    """
    ep_sub = eps_full[sel]
    _, counts = np.unique(ep_sub, return_counts=True)
    balanced = bool(counts.min() == counts.max())

    herr_full, herr_sub = _hold_err(state, gt, K), _hold_err(state[sel], gt[sel], K)
    full_fw, full_ew = _weighted(herr_full, eps_full)
    sub_fw, sub_ew = _weighted(herr_sub, ep_sub)
    pass_full = {"mae": full_fw,
                 "arm": float(herr_full[..., :7].mean()), "grip": float(herr_full[..., 7].mean())}
    pass_sub = {"mae": sub_fw,
                "arm": float(herr_sub[..., :7].mean()), "grip": float(herr_sub[..., 7].mean())}

    # Tripwire: the constants above are what the board's score.py prints. If val_anchors.npz has
    # been regenerated, every subset-vs-full statement in this file is void -- die before writing.
    # TOLERANCE, not round-3 equality. `pass_full` here is the RAW float: grip = 6.829487323760986,
    # which sits 1.27e-05 BELOW the 6.8295 rounding boundary. So `round(x,3) == 6.829` does hold
    # today -- the hazard is latent, not live: any summation-order shift of that size flips it to
    # 6.830 and kills the footnote for no reason. |diff| < 1e-3 states "agrees to 3 dp" without the
    # cliff, and is still ~60x tighter than the smallest val_anchors change worth catching.
    #
    # Two things NOT to conclude from the number 6.8295, both of which I got wrong once:
    #   * it is not a second implementation's answer -- 145 and 147 and score.py all produce the
    #     bit-identical float32 0x40da8b29, so the "two implementations straddle the cliff" story
    #     is false; 6.8295 is just this value displayed at 4 dp (line ~301 below);
    #   * that display rounding is itself the trap for READERS: round(6.8295, 3) == 6.83, so anyone
    #     validating the STORED pass_line_full1397 against the 3-dp board constant fails
    #     DETERMINISTICALLY. Double rounding converts "1.27e-05 from the cliff" into "exactly on
    #     it". Validate against the raw value, never against a value you rounded for display.
    bad = {k: (round(pass_full[k], 5), v) for k, v in PASS_FULL.items()
           if abs(pass_full[k] - v) >= 1e-3}
    if bad:
        raise SystemExit(f"[FATAL] recomputed full-1397 pass line disagrees with the board "
                         f"constants (recomputed, expected): {bad}. val_anchors.npz changed; "
                         f"this footnote's subset/full comparison would be meaningless.")

    out = {
        "sampling_scheme": "stratified by episode" if balanced else "unbalanced across episodes",
        "episodes_covered": int(len(counts)),
        "episodes_total": int(len(np.unique(eps_full))),
        "anchors_per_episode": {"min": int(counts.min()), "median": int(np.median(counts)),
                                "max": int(counts.max())},
        "episode_balanced": balanced,
        "weighting_within_subset": "frame-length-weighted (plain mean over anchors)",
        "weighting_measured": {
            "subset200_hold_frame_weighted": sub_fw,
            "subset200_hold_episode_equal": sub_ew,
            "full1397_hold_frame_weighted": full_fw,
            "full1397_hold_episode_equal": full_ew,
        },
        "weighting_note": (
            "the subset is episode-BALANCED (10/episode), so its plain mean IS the episode-equal "
            "mean (measured identical above), while the full-1397 board row is frame-length-"
            "weighted over 49-113 anchors per episode. That re-weighting alone -- no subsampling, "
            "no stochasticity -- is a deterministic offset, and it does not cancel in a ratio."),
        "pass_line_subset200": {k: round(v, 4) for k, v in pass_sub.items()},
        "pass_line_full1397": {k: round(v, 4) for k, v in pass_full.items()},
        "pass_line_recomputed_and_validated": True,
        "pass_line_comparison_rule": "compare the subset MAEs below to pass_line_subset200 ONLY",
    }
    if ref_pred_full is not None and len(ref_pred_full) == len(gt):
        perr_full = np.abs(ref_pred_full[:, :K, :] - gt)
        perr_sub = perr_full[sel]
        p_full_fw, p_full_ew = _weighted(perr_full, eps_full)
        p_sub_fw, p_sub_ew = _weighted(perr_sub, ep_sub)
        out["subset_vs_full_offset"] = {
            "preds": {"mae": round(p_sub_fw - p_full_fw, 4),
                      "arm": round(float(perr_sub[..., :7].mean() - perr_full[..., :7].mean()), 4),
                      "grip": round(float(perr_sub[..., 7].mean() - perr_full[..., 7].mean()), 4)},
            "hold": {"mae": round(sub_fw - full_fw, 4),
                     "arm": round(pass_sub["arm"] - pass_full["arm"], 4),
                     "grip": round(pass_sub["grip"] - pass_full["grip"], 4)},
            "hold_over_preds_ratio": {"full1397": round(full_fw / p_full_fw, 4),
                                      "subset200": round(sub_fw / p_sub_fw, 4)},
            "preds_episode_equal": {"full1397": round(p_full_ew, 4), "subset200": round(p_sub_ew, 4)},
            "note": ("same preds file, zero stochasticity -- population + weighting only. If the "
                     "model and the pass line move in opposite directions, the RATIO moves more "
                     "than either one."),
        }
    return out


def build_summary(args, preds, sel, seeds, gt, eps_full, state, K, cfg, design, seedcheck,
                  subset_path, subset_md5, subset_is_standard, xcheck) -> dict:
    sub_gt = gt[sel]
    per_seed = []
    for si, s in enumerate(seeds):
        err = np.abs(preds[si][:, :K, :] - sub_gt)
        per_seed.append({"seed": int(s), "mae": float(err.mean()),
                         "arm": float(err[..., :7].mean()), "grip": float(err[..., 7].mean())})
    maes = np.array([d["mae"] for d in per_seed])
    arms = np.array([d["arm"] for d in per_seed])
    grips = np.array([d["grip"] for d in per_seed])
    spread = float(np.abs(preds - preds.mean(axis=0, keepdims=True)).mean())
    m_std, a_std, g_std = (float(maes.std(ddof=1)), float(arms.std(ddof=1)),
                           float(grips.std(ddof=1)))
    n = len(sel)
    # Per-DRAW std (NOT std/sqrt(R)): the board row is ONE realization, so the question is "where
    # could this row have landed", not "how well do we know the mean of 5 draws".
    # SUBSET-200 scale. The board row is scored on all 1397 anchors and the noise is drawn per
    # (seed, anchor) independently, so the row's sampling SE is sqrt(200/1397) = 0.378x this.
    # Exponent measured on n15's landed 200x5 npz (m=25..200): -0.4901 vs -0.5 for independent
    # noise. Both scales are emitted; the *_full1397 field is the one to put in quadrature with a
    # jackknife SE_episode, the subset field is the contract artifact.
    sc = float(np.sqrt(n / 1397.0))
    return {
        "tag": args.tag, "backbone": args.tag,
        "artifact_kind": "sampling_variance_footnote", "not_a_leaderboard_row": True,
        "ckpt": args.ckpt_dir, "config": args.config_name,
        "anchors": n, "n_anchors": n, "seeds": [int(s) for s in seeds], "repeats": len(seeds),
        "K_compared": int(K),
        # gr00t-n17 2026-08-05: openpi's q01/q99 are histogram estimates whose bin layout depends
        # on config.batch_size (the norm-stats pass batches by it and drops the tail), so the
        # normalization -- hence these weights -- is a function of this number. Record it.
        "train_batch_size": int(getattr(cfg, "batch_size", -1)),
        "action_horizon": int(cfg.model.action_horizon), "action_dim": int(cfg.model.action_dim),
        "per_seed": per_seed, "per_repeat": per_seed,
        "mae_mean": float(maes.mean()), "mae_std": m_std,
        "arm_mean": float(arms.mean()), "arm_std": a_std,
        "grip_mean": float(grips.mean()), "grip_std": g_std,
        # --- schema aliases (same numbers, other legs' key layouts) ---
        "std_ddof1": {"mae": m_std, "arm": a_std, "grip": g_std},          # n15 layout
        "mean": {"mae": float(maes.mean()), "arm": float(arms.mean()), "grip": float(grips.mean())},
        "mae": {"mean": float(maes.mean()), "std": m_std},                 # act layout
        "arm": {"mean": float(arms.mean()), "std": a_std},
        "grip": {"mean": float(grips.mean()), "std": g_std},
        # Measured on the BYTES, not inferred from `m_std == 0.0`. A field named "bit exact" that
        # is really "a derived float compared to zero" answers a different question, and the gap
        # is REACHABLE, not theoretical: perturb one float32 of one draw by 1e-6 and the MAE moves
        # ~6e-11, which is two orders below the ~2.4e-07 ULP of a float32 mean near 3.07, so every
        # per-draw MAE is unchanged, m_std is exactly 0.0, and the proxy asserts bit-exactness for
        # draws that visibly differ (measured in test_variance_openpi.py section 5b). The opposite
        # direction -- identical draws whose std lands on ~1e-17 and so reads as "not bit-exact" --
        # is plausible from the same arithmetic but I have NOT reproduced it here. Same family as
        # the dead `x.std()==0` guard, which on a constant float32 array returns ~6e-05 and can
        # never fire. The std-derived value is kept alongside so the two can be compared rather
        # than silently substituted for one another.
        "bit_exact_across_draws": bool(all(preds[i].tobytes() == preds[0].tobytes()
                                           for i in range(1, len(preds)))),
        "bit_exact_via_zero_std": bool(m_std == 0.0),
        # --- scale annotation ---
        "std_scale": f"per-draw std on the SUBSET-{n} anchors (NOT the 1397-anchor board row)",
        "SE_sampling_full1397": {"mae": m_std * sc, "arm": a_std * sc, "grip": g_std * sc},
        "SE_sampling_scale_factor": sc,
        "pred_spread_mean_abs_dev": spread,
        # --- provenance of the anchor set (identity, not existence) ---
        "subset_file": str(subset_path) if subset_path else None,
        "subset_md5": subset_md5,
        "subset_is_standard": bool(subset_is_standard),
        "subset_design": design,
        # --- seed provenance: the MEASURED field is the gate, the declared ones are for humans ---
        "seed_convention": "global anchor index",
        "per_draw_seeds": [int(s) for s in seeds],
        "per_anchor_seed_formula": "fold_in(key(noise_seed), global_anchor_index)",
        "seed_stride_exceeds_anchor_count": None,
        "seed_fields_are_declared_not_measured": True,
        "seed_stride_note": ("not applicable: this leg derives each draw from a separate ROOT key "
                             "and folds the anchor index in, so there is no additive stride to "
                             "compare against the anchor count. The measured collision count "
                             "below is the gate; a declared stride field would be meaningless "
                             "here and true-by-construction is exactly the failure mode n17 hit."),
        "per_anchor_noise_collisions_measured": seedcheck["collisions"],
        # Sink-tapped counterpart: the gate above proves the GENERATOR does not alias, these prove
        # the production CALL SITE fed exactly those tensors. n17's defect lived in the gap.
        "driver_noise_collisions_measured": seedcheck.get("driver_noise_collisions_measured"),
        "driver_matches_gate_multiset": seedcheck.get("driver_matches_gate_multiset"),
        "noise_multiset_fingerprint": seedcheck.get("noise_multiset_fingerprint"),
        # Same three aliases as inside seed_check, repeated at TOP level because that is where
        # the shared reader looks (paired_signif.py consults j["..."], not j["seed_check"]["..."]).
        # A reader that misses its key here falls back to accepting the bare 0 above.
        "power_control_A_additive_stride1": seedcheck["power_control_A_additive_stride1"],
        "seed_check_power_control_collisions": seedcheck["power_control_A_additive_stride1"],
        "seed_collision_power_control": seedcheck["power_control_A_additive_stride1"],
        "power_control_collisions": seedcheck["power_control_A_additive_stride1"],
        "power_control_key_aliases": seedcheck["power_control_key_aliases"],
        "seed_check_power_control_B": seedcheck["power_control_B_anchor_not_folded_in"],
        "power_control_ok": seedcheck["power_controls_ok"],
        "seed_check_verdict": seedcheck["verdict"],
        "seed_check": seedcheck,
        "subset_run_is_independent_process": True,
        "seed0_crosscheck": xcheck,
        "provenance_npz": (f"logs/variance/preds_{args.tag}_subset{n}_seed0.npz"
                           if 0 in seeds else None),
        "note": (f"MAE here is on the {n}-anchor shared subset; compare it to "
                 f"pass_line_subset200 only. The board row is the full-1397 preds file."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tag", required=True, help="backbone tag, e.g. pi05")
    ap.add_argument("--subset", default=str(BENCH / "logs" / "variance_subset200.npy"),
                    help="shared anchor-index subset shared by every random backbone")
    ap.add_argument("--subset-n", type=int, default=200, help="only used if --subset is missing")
    ap.add_argument("--allow-nonstandard-subset", action="store_true",
                    help="publish a footnote on a NON-comparable anchor set (default: fatal)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ref-preds", default=None, help="preds/preds_<bb>.npz for the seed-0 cross-check")
    ap.add_argument("--out-dir", default=str(BENCH / "logs" / "variance"),
                    help="dry runs MUST point this at /tmp -- a dry run that writes to the "
                         "production path is not a dry run")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, frames, states, gt = A["episodes"], A["frames"], A["state"], A["gt"]
    total = len(eps)
    K = gt.shape[1]

    # ---- everything that can fail runs BEFORE the 3B policy load ----
    subset_path, sel, subset_md5, subset_std, src = resolve_subset(
        args.subset, args.allow_nonstandard_subset, args.subset_n, total)
    if len(np.unique(sel)) != len(sel) or sel.min() < 0 or sel.max() >= total:
        raise SystemExit(f"bad subset: n={len(sel)} unique={len(np.unique(sel))} "
                         f"range=[{sel.min()},{sel.max()}] vs total {total}")
    n = len(sel)
    _u, _c = np.unique(eps[sel], return_counts=True)
    print(f"variance study: {n} anchors x {len(seeds)} seeds {seeds}", flush=True)
    print(f"  subset source: {src} | covers {len(_u)} val episodes, "
          f"{_c.min()}-{_c.max()} anchors each | first8={sel[:8].tolist()}", flush=True)

    import jax
    import jax.numpy as jnp  # noqa: F401  (pinned_noise imports its own)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                    str(BENCH / "archive")]   # was one flat scripts/ dir
    from predict_openpi import to_hwc_uint8

    cfg = _config.get_config(args.config_name)
    h, adim = cfg.model.action_horizon, cfg.model.action_dim

    seedcheck = measure_noise_collisions(sel, seeds, h, adim)
    seedcheck["backbone"] = args.tag
    seedcheck["n_anchors"] = n
    seedcheck["repeats"] = len(seeds)
    seedcheck["anchor_index_span"] = [int(sel.min()), int(sel.max())]
    print(json.dumps(seedcheck, indent=2), flush=True)
    if not seedcheck["verdict"].startswith("PASS"):
        raise SystemExit(f"[FATAL] noise-collision gate: {seedcheck['verdict']} -- refusing to "
                         f"spend a GPU pass on a footnote whose sampling std would be wrong.")

    ref_pred_full = None
    if args.ref_preds and pathlib.Path(args.ref_preds).is_file():
        _r = np.load(args.ref_preds)["pred"].astype(np.float32)
        if _r.shape[0] == total:
            ref_pred_full = _r
    design = subset_design(sel, eps, states.astype(np.float32), gt.astype(np.float32), K,
                           ref_pred_full)
    print(f"  pass line: subset200 {design['pass_line_subset200']} | "
          f"full1397 {design['pass_line_full1397']} (recomputed, validated)", flush=True)

    # ---- GPU work ----
    policy = _policy_config.create_trained_policy(cfg, args.ckpt_dir)

    preds = np.zeros((len(seeds), n, h, 8), dtype=np.float32)
    # Ledger of the noise actually handed to infer, filled in the loop below and checked against
    # the pre-GPU gate afterwards. Empty strings so an unwritten cell is detectable (np.zeros on a
    # string dtype would give "" too, but being explicit about WHY matters here).
    noise_md5 = np.full((len(seeds), n), "", dtype="<U32")
    sel_eps = eps[sel]
    t0, done = time.monotonic(), 0
    for e in sorted(set(int(x) for x in sel_eps)):
        rows = np.flatnonzero(sel_eps == e)
        ds = LeRobotDataset(SRC_REPO, episodes=[e])
        for r in rows:
            g = int(sel[r])
            t = int(frames[g])
            item = ds[t]
            assert int(item["frame_index"]) == t
            obs = {
                "observation/image": to_hwc_uint8(item["observation.images.cam_high"]),
                "observation/wrist_image": to_hwc_uint8(item["observation.images.cam_wrist"]),
                "observation/state": states[g].astype(np.float32),
                "prompt": PROMPT,
            }
            for si, s in enumerate(seeds):
                noise = pinned_noise(s, g, h, adim)
                noise_md5[si, r] = hashlib.md5(noise.tobytes()).hexdigest()  # tap at the sink
                preds[si, r] = np.asarray(policy.infer(obs, noise=noise)["actions"],
                                          dtype=np.float32)
            done += 1
            if done % 50 == 0:
                el = time.monotonic() - t0
                print(f"  {done}/{n} anchors  {el:.0f}s", flush=True)
        del ds

    # Reconcile what the driver actually fed against what the pre-GPU gate certified. Runs BEFORE
    # anything is written, so a run whose call site aliased leaves no artifact behind to be
    # mistaken later for a good one.
    drivercheck = verify_driver_noise(noise_md5, seedcheck["noise_multiset_fingerprint"])
    seedcheck.update(drivercheck)
    print(f"[noise-ledger] {drivercheck['driver_noise_chunks']} chunks fed to infer, "
          f"{drivercheck['driver_noise_collisions_measured']} collisions, multiset == pre-GPU gate "
          f"({drivercheck['driver_noise_multiset_fingerprint'][:8]}...)", flush=True)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{args.tag}_seeds.npz", pred=preds, anchor_idx=sel.astype(np.int32),
             seeds=np.array(seeds, dtype=np.int32), noise_md5=noise_md5)

    # Per-draw PROVENANCE npz (board policy, gr00t-n17's quadrature work / n15's convention).
    # The all-seeds file above stacks every draw in one array; the reader that bit-verifies a
    # footnote against the board row wants ONE draw in the same layout as preds_<bb>.npz, so it
    # can assert footnote[i] == preds_<bb>[anchor_idx[i]] byte-for-byte. What this catches: a
    # footnote computed off an EARLIER checkpoint attaches a plausible, wrongly-scaled sampling
    # term, and no shape/finiteness/schema check can see it -- only the bytes can.
    # Written here, while the draws are in memory, so a later crash still leaves the artifact.
    # LOCATION: logs/variance/, NOT preds/ -- the reader globs a directory, and preds/ is what
    # score.py globs. Keys are a superset of both existing legs (n15 ships `pred` only).
    if 0 in seeds:
        prov = out_dir / f"preds_{args.tag}_subset{n}_seed0.npz"
        np.savez(prov, pred=preds[seeds.index(0)], anchor_idx=sel.astype(np.int32),
                 noise_md5=noise_md5[seeds.index(0)], seed=np.int32(0))
        print(f"wrote provenance {prov.name}  pred{preds[seeds.index(0)].shape}", flush=True)

    xcheck = "not requested"
    if args.ref_preds:
        if ref_pred_full is not None and 0 in seeds:
            si0 = seeds.index(0)
            d = np.abs(ref_pred_full[sel] - preds[si0]).max()
            xcheck = f"seed0 vs preds/{pathlib.Path(args.ref_preds).name}: max|diff|={d:.3e} " \
                     f"({'IDENTICAL' if d == 0 else 'MISMATCH'})"
        else:
            xcheck = f"skipped (ref {args.ref_preds} missing/wrong shape, seeds {seeds})"

    summary = build_summary(args, preds, sel, seeds, gt.astype(np.float32), eps,
                            states.astype(np.float32), K, cfg, design, seedcheck,
                            subset_path, subset_md5, subset_std, xcheck)
    (out_dir / f"variance_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"seedcheck_{args.tag}.json").write_text(json.dumps(seedcheck, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
