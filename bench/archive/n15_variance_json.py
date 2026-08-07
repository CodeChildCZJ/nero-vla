#!/usr/bin/env python3
"""Summarise N1.5's sampling-variance footnote into the team's canonical JSON.

Convention (team-lead): logs/variance/variance_<bb>.json, NOT preds/ (score.py globs
preds/ and would try to rank a 200-anchor footnote against 1397-anchor rows).

The footnote is scored on the SHARED 200-anchor subset, which has its OWN pass line
(3.198 / 2.591 / 7.449) different from the full-1397 board line (3.072 / 2.535 / 6.829)
-- the JSON carries both so nobody places a subset MAE beside a leaderboard number.
The subset-vs-full bit-exactness result is PARSED from the pipeline log, never
hardcoded: it is a measured invariant and a False is a real finding.
"""
import argparse, hashlib, json, re
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
FULL_PASS = {"mae": 3.072, "arm": 2.535, "grip": 6.829}      # 1397 anchors
SUBSET_PASS = {"mae": 3.198, "arm": 2.591, "grip": 7.449}    # the shared 200
# must mirror predict_gr00t.py PinnedNoise.build: manual_seed(seed + 1000003*repeat + g)
BASE_SEED, SEED_STRIDE = 0, 1000003


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var-npz", default=str(BENCH / "logs/variance/var_gr00t_n15_200x5.npz"))
    ap.add_argument("--subset", default=str(BENCH / "logs/variance_subset200.npy"))
    ap.add_argument("--log", default=str(BENCH / "logs/n15_post_train.log"))
    ap.add_argument("--seed-check", default=str(BENCH / "logs/variance/seedcheck_gr00t_n15.json"))
    ap.add_argument("--callsite-check",
                    default=str(BENCH / "logs/variance/seedcheck_gr00t_n15_callsite.json"))
    ap.add_argument("--out", default=str(BENCH / "logs/variance/variance_gr00t_n15.json"))
    args = ap.parse_args()

    stack = np.load(args.var_npz)["pred"]            # (repeats, N, K', 8)
    sel = np.load(args.subset)
    _A = np.load(BENCH / "data" / "val_anchors.npz")
    gt = _A["gt"][sel]
    K = gt.shape[1]
    R, N = stack.shape[0], stack.shape[1]
    assert N == len(sel), (N, len(sel))

    per_repeat = []
    for r in range(R):
        e = np.abs(stack[r][:, :K, :].astype(np.float64) - gt)
        per_repeat.append({"repeat": r, "mae": float(e.mean()),
                           "arm": float(e[..., :7].mean()), "grip": float(e[..., 7].mean())})
    arr = {k: np.array([p[k] for p in per_repeat]) for k in ("mae", "arm", "grip")}

    # Hard requirement, never a silent default: a missing seed-check would otherwise be
    # indistinguishable from a passing one, which is the exact failure mode this footnote
    # exists to rule out. Run archive/n15_seed_collision_check.py first.
    seedchk_p = Path(args.seed_check)
    if not seedchk_p.exists():
        raise SystemExit(f"missing {seedchk_p} — run archive/n15_seed_collision_check.py "
                         f"(n1d5 venv, CPU) before writing the footnote")
    seedchk = json.loads(seedchk_p.read_text())
    if not seedchk.get("power_control_ok"):
        raise SystemExit(f"{seedchk_p}: power control did not trip — the collision check "
                         f"has no power, refusing to write a footnote that cites it")

    # Layer 3, the production CALL SITE. Layer 1 (above) tests the shipped PinnedNoise CLASS
    # and layer 2 the landed noise TENSORS; NEITHER can see a caller that never passes
    # `repeat` — which is the bug gr00t-n17 actually shipped. Same hard-requirement logic.
    callsite_p = Path(args.callsite_check)
    if not callsite_p.exists():
        raise SystemExit(f"missing {callsite_p} — run archive/n15_callsite_seedcheck.py "
                         f"(CPU, no GPU, no re-inference) before writing the footnote")
    callsite = json.loads(callsite_p.read_text())
    if callsite["verdict"] != "PASS":
        raise SystemExit(f"{callsite_p}: call-site seed export did not pass")
    if not callsite["delivered_script"]["seed_path_identical_to_delivered"]:
        raise SystemExit(f"{callsite_p}: predict_gr00t.py's seed path has changed since "
                         f"delivery — the exported seeds are not evidence about the run "
                         f"that produced the preds")
    # The seed fields below are now taken from that MEASURED export. Cross-check them against
    # the source literals: if the two ever disagree, the constants are lying about the run and
    # that must be loud, not silently papered over by preferring one side.
    if (callsite["measured_per_draw_seeds"] != [BASE_SEED + SEED_STRIDE * r for r in range(R)]
            or callsite["measured_per_draw_seed_stride"] != SEED_STRIDE):
        raise SystemExit(
            f"MEASURED seeds disagree with the source literals: measured "
            f"{callsite['measured_per_draw_seeds']} stride {callsite['measured_per_draw_seed_stride']} "
            f"vs declared base={BASE_SEED} stride={SEED_STRIDE}. One of them is wrong about "
            f"what actually ran — do not write a footnote until this is resolved.")

    log = Path(args.log).read_text(errors="ignore") if Path(args.log).exists() else ""
    m = re.search(r"check-preds vs .*: bit_exact=(True|False)\s+maxdiff=([0-9.e+-]+)", log)
    bit = {"measured": m is not None,
           "bit_exact": (m.group(1) == "True") if m else None,
           "maxdiff": float(m.group(2)) if m else None}

    # --- subset DESIGN + the offset it induces, both MEASURED (gr00t-n15 2026-08-05) ---
    # Rule (a)(b)(c) on a subset artifact -- anchor count / sampling scheme / that subset's
    # OWN pass line -- misses a (d): the WEIGHTING. Half of a subset's offset is not sampling
    # at all but deterministic re-weighting (episode-equal vs frame-length), and it moves the
    # model and the pass line in OPPOSITE directions, so it does NOT cancel in a ratio; it
    # adds. Measured here rather than described, because a design claim ("10 per episode") is
    # exactly the kind of statement a broken selector still satisfies on paper.
    epi = _A["episodes"]
    per_ep = np.bincount(np.searchsorted(np.unique(epi), epi[sel]))
    gt_f, st_f = _A["gt"].astype(np.float32), _A["state"].astype(np.float32)
    hold_err = np.abs(np.broadcast_to(st_f[:, None, :], gt_f.shape) - gt_f)

    def _trip(e):
        return {"mae": float(e.mean()), "arm": float(e[..., :7].mean()), "grip": float(e[..., 7].mean())}

    # The two pass lines were hardcoded above as rounded literals; recompute and require them to
    # agree, so a regenerated val_anchors.npz cannot leave stale constants behind.
    # ⚠ TOLERANCE, NOT ROUNDING EQUALITY (board rule, openpi 2026-08-05). `round(x,3)==const`
    # puts a discontinuity at every .xxx5 boundary, and grip lands on one: this file recomputes
    # 6.8294873, which is 1.27e-5 from the 6.82950 cliff. A float32 summation-order shift of
    # that size flips round() to 6.830 and this gate SystemExits on a perfectly correct file —
    # a false positive on valid input, the exact failure the poisoned-REFERENCE gate exists to
    # avoid causing. |diff| < 1e-3 still means "agrees to 3 dp" and is ~60x tighter than any
    # val_anchors change worth catching; it is 2x looser than round-3 at the boundary, which is
    # the whole point. openpi hit this first; my grip was one summation order away from it.
    PASS_TOL = 1e-3
    for label, computed, declared in (("full1397", _trip(hold_err), FULL_PASS),
                                      ("subset200", _trip(hold_err[sel]), SUBSET_PASS)):
        bad = {k: (computed[k], declared[k]) for k in declared
               if abs(computed[k] - declared[k]) >= PASS_TOL}
        if bad:
            raise SystemExit(f"pass_line_{label} constants are stale vs val_anchors.npz "
                             f"(|recomputed - declared| >= {PASS_TOL}): {bad}")

    design = {
        "sampling_scheme": "stratified by episode",
        "episodes_covered": int(len(np.unique(epi[sel]))),
        "episodes_total": int(len(np.unique(epi))),
        "anchors_per_episode": {"min": int(per_ep.min()), "median": int(np.median(per_ep)),
                                "max": int(per_ep.max())},
        "episode_balanced": bool(per_ep.min() == per_ep.max()),
        "weighting_within_subset": "frame-length-weighted (plain mean over anchors)",
        "weighting_note": ("the subset is episode-BALANCED, so its plain mean is also the "
                           "episode-equal-weighted mean, while the full-1397 board row is "
                           "frame-length-weighted -- that difference alone (no subsampling) "
                           "accounts for ~half of the offset below, and it has OPPOSITE sign "
                           "for the model and for the pass line, so it does not cancel in a ratio"),
    }
    board = BENCH / "preds/preds_gr00t_n15.npz"
    if board.exists():
        pe = np.abs(np.load(board)["pred"].astype(np.float32)[:, :K, :] - gt_f)
        pf, ps = _trip(pe), _trip(pe[sel])
        hf, hs = _trip(hold_err), _trip(hold_err[sel])
        design["subset_vs_full_offset"] = {
            "preds": {k: round(ps[k] - pf[k], 4) for k in pf},
            "hold": {k: round(hs[k] - hf[k], 4) for k in hf},
            "hold_over_preds_ratio": {"full1397": round(hf["mae"] / pf["mae"], 4),
                                      "subset200": round(hs["mae"] / ps["mae"], 4)},
            "note": ("same preds file, zero stochasticity -- population only. Model and pass "
                     "line move in opposite directions, so the RATIO moves more than either."),
        }

    out = {
        "backbone": "gr00t_n15",
        "artifact_kind": "sampling_variance_footnote",
        "not_a_leaderboard_row": True,
        "n_anchors": int(N), "repeats": int(R), "K_compared": int(K),
        "batch_size": 1, "seed_convention": "global anchor index",
        # UPSTREAM evidence that the R draws really differ. A reader that sees std==0.0
        # cannot tell a deterministic decoder from a harness that reused one seed, and
        # `bit_exact_across_draws` cannot settle it either — that field is computed from
        # the same R arrays, i.e. downstream of the bug (gr00t-n17 2026-08-05). Only the
        # seeds are upstream. ⚠ Distinctness of these base seeds is NOT sufficient: the
        # per-anchor seed is base + global_anchor_index, so if the stride were <= the
        # anchor count the draws would ALIAS (draw r anchor g == draw r+1 anchor g-1) and
        # still list distinct base seeds. Measured with stride 1: 796/800 anchor-level
        # noise tensors reused across draws and std 15.1x too small, gate still green.
        # The real condition is stride > n_anchors; 1000003 is prime and >> 1397.
        # These four used to be DECLARED (mirroring the source literal), which is not
        # evidence — n17's leg had this exact (correct) formula and a driver that never
        # passed `repeat`, so the EFFECTIVE stride was 1 while a constant-derived field
        # would still have reported 1000003 / True. They are now read from the seeds that
        # actually reached torch.Generator.manual_seed in the real anchor/repeat loop
        # (--dry-run-seeds, CPU, forward short-circuited), with the delivered seed path
        # proven byte-identical so the export speaks for the run that made the preds.
        "per_draw_seeds": callsite["measured_per_draw_seeds"],
        "per_draw_seed_stride": callsite["measured_per_draw_seed_stride"],
        "per_anchor_seed_formula": "base_seed + %d*repeat + global_anchor_index" % SEED_STRIDE,
        "seed_stride_exceeds_anchor_count": callsite["measured_stride_exceeds_anchor_count"],
        "seed_fields_are_declared_not_measured": False,
        "seed_fields_provenance": {
            "source": "logs/variance/seedcheck_gr00t_n15_callsite.json",
            "measured_at": callsite["measured_at"],
            "layer": "3 = production call site (layers 1/2 = pinner class / landed tensors)",
            "gpu_used": False, "reinference": False,
            "delivered_seed_path_identical": True,
            "seed_load_bearing_statements_compared":
                callsite["delivered_script"]["seed_load_bearing_statements_compared"],
            "call_site_seed_collisions": callsite["measured_seed_collisions"],
            "min_seed_gap_across_draws": callsite["min_seed_gap_across_draws"],
            "power_controls": callsite["power_controls"],
        },
        # MEASURED on the shipped PinnedNoise over the real subset indices, with a power
        # control (n17's bug shape reproduced on the same class) that must trip.
        "per_anchor_noise_collisions_measured": seedchk["collisions"],
        # POWER-CONTROL KEY SUPERSET (board contract, 2026-08-05). n17's cross-reader looked
        # for its own key name, didn't find mine, and SILENTLY accepted my bare 0 — defeating
        # the entire reason the 0 is worth anything. A 0 with no power control behind it is
        # worthless, so emit the same measurement under every key name in circulation rather
        # than make readers guess. Same discipline as schema-superset: cost is bytes, the
        # failure it prevents is silent.
        "seed_check_power_control_collisions": seedchk["power_control_collisions"],
        "seed_collision_power_control": seedchk["power_control_collisions"],   # n17's name
        "power_control_collisions": seedchk["power_control_collisions"],       # generic
        "power_control_key_aliases": ["seed_check_power_control_collisions",
                                      "seed_collision_power_control",
                                      "power_control_collisions"],
        "seed_check_verdict": seedchk["verdict"],
        # `subset_vs_full_bit_exact` only proves ckpt provenance if the subset run was a
        # SEPARATE process that re-loaded the checkpoint; an in-process array slice would
        # satisfy the field name while proving nothing. n15_post_train.sh invokes
        # predict_gr00t.py twice, so it is two processes. The field name does not carry
        # that premise, so state the premise (gr00t-n17's framing).
        "subset_run_is_independent_process": True,
        "subset_file": args.subset, "subset_md5": hashlib.md5(Path(args.subset).read_bytes()).hexdigest(),
        "subset_design": design,
        "per_repeat": per_repeat,
        "mean": {k: float(v.mean()) for k, v in arr.items()},
        "std_ddof1": {k: float(v.std(ddof=1)) for k, v in arr.items()},
        "pass_line_subset200": SUBSET_PASS,
        "pass_line_full1397": FULL_PASS,
        "subset_vs_full_bit_exact": bit,
        "note": ("MAE here is on the 200-anchor shared subset; compare it to "
                 "pass_line_subset200 only. The board row is the full-1397 preds file."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_anchors", "repeats", "mean", "std_ddof1",
                                          "subset_vs_full_bit_exact")}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
