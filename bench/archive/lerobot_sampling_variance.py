#!/usr/bin/env python
"""Sampling-variance footnote for the stochastic LeRobot backbone (SmolVLA).

CONTRACT.md requires it: flow-matching / diffusion policies denoise from random
noise, so a single-sample MAE carries a noise term that deterministic backbones
(ACT, OpenVLA-OFT) simply do not have. Comparing one draw of SmolVLA against ACT
without knowing that spread is comparing a random variable to a constant.

  200 anchors x 5 independent noise draws  ->  MAE mean +- std

Output goes to logs/variance/ ONLY. Nothing here may land in preds/ -- the
leaderboard must stay one row per backbone (CONTRACT.md).

  ./lerobot_sampling_variance.py --ckpt runs/smolvla/checkpoints/last/pretrained_model --device cuda:5

ACT does not need this: its decoder is a deterministic forward pass, so all draws
are byte-identical. Running it on ACT is still a useful *check* of that claim
(std must come out exactly 0.0); --backbone act is allowed for that reason.
"""

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
VENV_PY = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "lerobot" / ".venv" / "bin" / "python")
SUBSET = BENCH / "logs" / "variance_subset200.npy"
PREDICT_SRC = BENCH / "scripts" / "predict" / "lerobot_predict.py"


SUBSET_MD5 = "9007df1fcec5eca2a1f01591c6266b43"


def load_subset():
    """LOAD the shared 200-anchor subset -- never regenerate it.

    Every leg's variance footnote must run on the SAME anchors or the numbers are
    not comparable across backbones. Two legs (openpi, gr00t-n17) each silently
    built their own and ended up sharing only 24/200 anchors with this file, while
    the first 8 indices agreed -- a false-agreement trap that a spot check passes.
    So a missing/altered file is a hard stop, not a cue to rebuild: regenerating
    would produce a plausible run whose mean is quietly incomparable (seed-to-seed
    std is robust to the subset, but the subset MEAN is not).
    """
    if not SUBSET.exists():
        sys.exit(f"missing shared subset {SUBSET} -- do NOT regenerate it, get the "
                 f"file from the bench lead (md5 {SUBSET_MD5})")
    got = hashlib.md5(SUBSET.read_bytes()).hexdigest()
    if got != SUBSET_MD5:
        sys.exit(f"{SUBSET} md5 {got} != shared {SUBSET_MD5} -- refusing to run on a "
                 f"divergent anchor set")
    return np.load(SUBSET)


def seed_multiplier_from_source():
    """Read the per-anchor seed multiplier OUT OF lerobot_predict.py, do not restate it.

    gr00t-n17's own seed-stride gate went green while their sampler reused noise across
    draws, because the gate checked a CONSTANT COPIED INTO THE FOOTNOTE rather than the
    number the sampler actually used. A field restating `1000003` proves nothing about
    lerobot_predict.py; it proves something about this file. So parse the shipped source
    and take the literal from the `torch.manual_seed(args.seed * M + int(glob_idx[n]))`
    call itself -- if anyone edits that line, this check follows it, and if the call
    shape changes so this parse no longer matches, that is a hard stop, not a default.
    """
    tree = ast.parse(PREDICT_SRC.read_text())
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "manual_seed" and node.args):
            continue
        a = node.args[0]
        # want: <seed> * <int literal> + <anchor index>
        if (isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add)
                and isinstance(a.left, ast.BinOp) and isinstance(a.left.op, ast.Mult)
                and isinstance(a.left.right, ast.Constant)
                and isinstance(a.left.right.value, int)):
            hits.append((node.lineno, a.left.right.value))
    if len(hits) != 1:
        sys.exit(f"cannot locate exactly one per-anchor manual_seed(seed*M + idx) in "
                 f"{PREDICT_SRC} (found {hits}) -- the collision gate would be measuring "
                 f"a formula the sampler no longer uses; fix the parse, do not assume M")
    return hits[0][1], hits[0][0]


def measure_noise_collisions(seeds, anchor_idx, mult):
    """COUNT actual (draw, anchor) noise-stream collisions. Do not infer them from stride.

    team-lead replaced the derived `seed_stride_exceeds_anchor_count` field with this
    after n17 measured 254/1000 reused noise tensors while their per-draw seed list was
    perfectly distinct. Collisions bias the std DOWNWARD -- correlated draws look like a
    quiet decoder -- which is the direction that flatters the footnote, so it has to be
    measured.

    Fingerprint = the RNG STATE after `torch.manual_seed(s)`, not a sampled tensor. Two
    seeds collide iff their entire subsequent noise stream is identical, whatever shape
    the model draws; and this also catches seed arithmetic that only collides after
    torch's internal 64-bit masking (a pure `s_a != s_b` comparison would not).
    """
    def grid(m):
        seen, coll = {}, 0
        for d in seeds:
            for g in anchor_idx:
                torch.manual_seed(int(d) * m + int(g))
                h = hashlib.md5(torch.get_rng_state().numpy().tobytes()).hexdigest()
                if h in seen:
                    coll += 1
                else:
                    seen[h] = (d, g)
        return coll
    live = grid(mult)
    # TWO power controls, because the obvious one can be legitimately 0 here.
    #
    # stride 1 is n17's ACTUAL defect (draw index folded into the base seed), but on THIS
    # anchor set it may well collide zero times: `d + g` collides only for anchors closer
    # together than the draw count, and the shared 200-of-1397 subset is sparse. A control
    # that can be innocently 0 cannot gate anything -- reading a 0 there as "we are clean"
    # would be the zero-power trap itself, one level up. So it is REPORTED, not gated.
    #
    # stride 0 is the defect at full strength -- every draw reuses the same per-anchor
    # noise, i.e. "the harness never varied it", the failure that makes a std=0 row
    # indistinguishable from real determinism. It MUST trip: with >1 draw it collides
    # (draws-1) x anchors times by construction, so a 0 there means the measurement is
    # broken (loop not varying, hashes constant), and the live 0 above means nothing.
    ctrl_stride1 = grid(1)
    ctrl_stride0 = grid(0)
    return live, ctrl_stride1, ctrl_stride0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="smolvla", choices=["smolvla", "act"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seeds-from-argv", action="store_true",
                    help="do not re-run inference; tap the per-draw seeds from the launch "
                         "argv and recompute stats from the draw npz files already on disk. "
                         "Only defensible where rng_consumption_<bb>.json MEASURES that the "
                         "predict path consumes no RNG (so a re-run is bit-identical).")
    # openpi's --out board rule: a pre-flight that writes to the production path is not a
    # pre-flight. Default stays the canonical path every reader globs.
    ap.add_argument("--out", default=None,
                    help="output json (default logs/variance/variance_<backbone>.json); "
                         "point at /tmp for a pre-flight")
    args = ap.parse_args()

    # Whether per-draw seeds mean anything on this backbone is a MEASURED fact, not an
    # architectural belief: logs/rng_consumption_<bb>.json hashes the global RNG state
    # around the shipped pre/predict/post call sequence, with a two-sided power control.
    # If the file is absent we do not guess -- an unmeasured stack must take the slow path.
    _rngf = BENCH / "logs" / f"rng_consumption_{args.backbone}.json"
    _rng_consumed = None
    if _rngf.exists():
        _rj = json.loads(_rngf.read_text())
        if not _rj.get("probe_power_control", {}).get("randn_moves_rng_hash"):
            sys.exit(f"{_rngf.name} has no power control on its probe; its verdict is not usable.")
        _rng_consumed = bool(_rj["predict_path_consumes_rng"])
    if args.seeds_from_argv:
        # The shortcut is only sound where a re-run would be bit-identical. On a stack
        # that DOES sample, skipping inference would silently turn a 5-draw variance
        # estimate into 5 reads of one cached file and report std over nothing.
        if _rng_consumed is None:
            sys.exit(f"--seeds-from-argv needs {_rngf.name}: without a measurement that the "
                     f"predict path consumes no RNG, skipping inference may be discarding a "
                     f"real sampling term.")
        if _rng_consumed:
            sys.exit(f"--seeds-from-argv REFUSED: {args.backbone}'s predict path was MEASURED "
                     f"to consume RNG, so the draws are genuinely stochastic and must actually "
                     f"be run. This flag is only for decoders measured to be deterministic.")

    sel = load_subset()
    outdir = BENCH / "logs" / "variance"
    outdir.mkdir(parents=True, exist_ok=True)

    gt = np.load(BENCH / "data" / "val_anchors.npz")["gt"].astype(np.float32)[sel]  # (n,K,8)
    K = gt.shape[1]

    # The per-draw seeds are what makes R draws R DRAWS. Bind them to a name here and
    # emit that same list, so the footnote reports what the loop actually ran rather
    # than what a reader would assume from `--draws`. gr00t-n17's reader needs it: a
    # std of exactly 0.0 has two causes -- a deterministic decoder, or a harness that
    # reused one seed R times -- and BOTH also write `bit_exact_across_draws: true`,
    # so that field cannot discriminate them (it is computed downstream of the same R
    # numbers). Only upstream evidence of DISTINCT seeds can.
    seeds = list(range(args.draws))

    # Measured collision gate, BEFORE any GPU work: if the sampler reuses noise across
    # draws the whole run is worthless, and finding that out after 5 model loads costs
    # the draws for nothing. Seconds on CPU.
    seed_mult, seed_lineno = seed_multiplier_from_source()
    coll_live, coll_s1, coll_s0 = measure_noise_collisions(seeds, sel, seed_mult)
    print(f"[collisions] multiplier {seed_mult} (ast-read from {PREDICT_SRC.name}:{seed_lineno}); "
          f"live {coll_live}/{len(seeds) * len(sel)}  "
          f"control stride1 {coll_s1}  control stride0 {coll_s0}")
    if coll_s0 == 0:
        sys.exit("POWER CONTROL DEAD: injecting stride 0 (every draw reuses each anchor's "
                 "noise) produced 0 collisions, so this measurement cannot detect the defect "
                 "it exists to detect and the live count is not evidence of anything.")
    if coll_live != 0:
        sys.exit(f"{coll_live} per-(draw,anchor) noise-stream COLLISIONS with the shipped "
                 f"multiplier {seed_mult}: draws are correlated, so the reported std is "
                 f"biased DOWN -- it would understate the sampling term, the flattering "
                 f"direction. Fix the seeding in {PREDICT_SRC.name} before footnoting.")

    # --seeds-from-argv: record the seed each draw ACTUALLY receives, tapped at the
    # launch boundary (the argv handed to subprocess), and skip the inference itself.
    #
    # Two reasons this is the right artifact rather than a GPU re-run:
    #   1. Provenance. Hand-writing `per_draw_seeds` into an existing json exports my
    #      memory of how I ran it. Reading the value off the seed EXPRESSION exports a
    #      re-derivation of the formula. Only the argv is what the child process was
    #      really given, and it is produced by the same unmodified statement below that
    #      the live path uses -- there is no second copy of the loop to drift.
    #   2. Information. `logs/rng_consumption_act.json` MEASURES that ACT's predict path
    #      touches the global RNG zero times (power-controlled: randn moves the hash,
    #      arithmetic does not). So for ACT the seeds provably cannot reach any output,
    #      and re-running inference to obtain them would buy a field that is true and
    #      load-bearing on nothing. For SmolVLA the same probe is expected to answer
    #      True, and there the seeds ARE load-bearing -- so this flag is a shortcut only
    #      where it has been measured to be one, never by default.
    #
    # In this mode the per-draw MAEs are recomputed from the seed npz files ALREADY on
    # disk (written by the real run), so the reported numbers are the delivered ones.
    # The json records `inference_rerun_in_this_invocation: false` so no reader can
    # mistake it for a fresh five-draw sample.
    launched_argv = []

    maes, arms, grips = [], [], []
    for s in seeds:
        out = outdir / f"preds_{args.backbone}_seed{s}.npz"
        cmd = [VENV_PY, str(BENCH / "scripts" / "predict" / "lerobot_predict.py"),
               "--backbone", args.backbone, "--ckpt", args.ckpt, "--device", args.device,
               "--subset", str(SUBSET), "--seed", str(s), "--out", str(out)]
        print(f"\n=== draw {s}/{args.draws - 1} ===", flush=True)
        launched_argv.append(list(cmd))
        if args.seeds_from_argv:
            if not out.exists():
                sys.exit(f"--seeds-from-argv recomputes from previously written draws, but "
                         f"{out} does not exist. Run the real thing first.")
            print(f"[seeds-from-argv] not launching; reusing {out.name}", flush=True)
            r = subprocess.CompletedProcess(cmd, 0)
        else:
            r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"draw {s} failed (rc={r.returncode})")
        p = np.load(out)["pred"].astype(np.float32)[:, :K, :]
        err = np.abs(p - gt)
        maes.append(err.mean()); arms.append(err[..., :7].mean()); grips.append(err[..., 7].mean())
        print(f"[draw {s}] MAE={maes[-1]:.4f} arm={arms[-1]:.4f} grip={grips[-1]:.4f}", flush=True)

    maes, arms, grips = map(np.array, (maes, arms, grips))
    print(f"\n=== sampling-variance footnote: {args.backbone}, "
          f"n={len(sel)} anchors x {args.draws} noise draws ===")
    for nm, v in (("MAE", maes), ("arm", arms), ("grip", grips)):
        print(f"  {nm:4s} mean {v.mean():.4f}  std {v.std(ddof=1):.4f}  "
              f"min {v.min():.4f}  max {v.max():.4f}  spread {v.max() - v.min():.4f}")
    print(f"\nper-draw MAE: {' '.join(f'{v:.4f}' for v in maes)}")

    # Aggregate std can be 0 while the predictions differ (rounding), so assert
    # determinism on the raw arrays -- that is the claim, the MAE is a proxy for it.
    ref = np.load(outdir / f"preds_{args.backbone}_seed0.npz")["pred"]
    bit_exact = all(np.array_equal(ref, np.load(outdir / f"preds_{args.backbone}_seed{s}.npz")["pred"])
                    for s in range(1, args.draws))
    print(f"bit-exact across draws: {bit_exact}")
    if args.backbone == "act":
        print("(ACT is deterministic -- std must be exactly 0; anything else means "
              "nondeterminism leaked in, e.g. a cudnn autotune path. Note the "
              "determinism comes from policy.eval() disabling dropout=0.1, not from "
              "the architecture -- in train mode ACT is stochastic.)")

    # ---- PROVENANCE: does draw 0 reproduce the delivered board row bit-for-bit? -----
    # This is the OPPOSITE prediction to bit_exact_across_draws and the two must not be
    # conflated (gr00t-n15's catch). Across DRAWS the same anchor must DIFFER for a
    # stochastic decoder (that is the sampling SE being real); across PROCESS/BATCHING
    # at the same draw it must be IDENTICAL (that is the footnote and the row coming
    # from the same checkpoint). Both hold here because the flow prior is seeded per
    # (draw, GLOBAL anchor index) -- lerobot_predict.py reseeds with
    # `seed * 1_000_003 + glob_idx[n]` and runs one anchor per forward, so an anchor's
    # noise does not depend on which run, which subset, or which batch position it sat
    # in. If a leg instead lets the prior come off the global RNG, a 200-anchor and a
    # 1397-anchor run disagree on every row for pure batching reasons -- and that is
    # indistinguishable from a wrong-checkpoint footnote, which is a real defect.
    full_f = BENCH / "preds" / f"preds_{args.backbone}.npz"
    prov = {"checked": False, "reason": f"{full_f.name} does not exist yet"}
    if full_f.exists():
        full = np.load(full_f)
        fp = full["pred"].astype(np.float32)
        if fp.shape[0] != 1397:
            prov = {"checked": False, "reason": f"{full_f.name} has {fp.shape[0]} rows, not the full 1397"}
        else:
            d0 = np.load(outdir / f"preds_{args.backbone}_seed0.npz")["pred"].astype(np.float32)
            sub_of_full = fp[sel][:, : d0.shape[1], :]
            same = bool(np.array_equal(sub_of_full, d0))
            prov = {
                "checked": True,
                "bit_exact": same,
                "maxdiff": float(np.abs(sub_of_full - d0).max()),
                "against": str(full_f),
                "draw_npz": str(outdir / f"preds_{args.backbone}_seed0.npz"),
                "note": "draw-0 subset rows vs the same global indices of the delivered row",
            }
        print(f"[provenance] subset draw-0 vs {full_f.name}: {prov}")
    else:
        print(f"[provenance] SKIPPED -- {full_f.name} not written yet; re-run this after "
              f"the board row exists, or the footnote ships unverified")

    # Canonical output path agreed across all legs (gr00t-n17 / openpi / team-lead).
    # The subset has its OWN pass line, different from the full-1397 leaderboard's --
    # record it here so the footnote can never be read against the wrong baseline.
    _VA = np.load(BENCH / "data" / "val_anchors.npz")
    gt_a = np.abs(np.broadcast_to(
        _VA["state"].astype(np.float32)[sel][:, None, :], gt.shape) - gt)
    _sub_eps = _VA["episodes"][sel]
    res = {
        "backbone": args.backbone, "ckpt": args.ckpt, "device": args.device,
        "n_anchors": int(len(sel)), "draws": int(args.draws), "batch_size": 1,
        "subset_file": str(SUBSET), "subset_md5": SUBSET_MD5,
        "mae": {"mean": float(maes.mean()), "std": float(maes.std(ddof=1)),
                "min": float(maes.min()), "max": float(maes.max())},
        "arm": {"mean": float(arms.mean()), "std": float(arms.std(ddof=1))},
        "grip": {"mean": float(grips.mean()), "std": float(grips.std(ddof=1))},
        # Same three stds, emitted in ALL THREE key layouts in use on this board. The
        # nested one above is this leg's original and gr00t-n17's reader parses it, so
        # this is not a fix -- it is insurance. Four readers exist across the legs and
        # each RAISES on a layout it does not know (deliberately: a silent 0.0 fallback
        # would make a dropped sampling term look like a deterministic decoder). n17's
        # own writer emitted a fourth layout its own reader rejected, which would have
        # killed the shared `--all` the moment their row landed. A superset costs 3
        # lines and makes this file unable to do that to anyone.
        "std_ddof1": {"mae": float(maes.std(ddof=1)), "arm": float(arms.std(ddof=1)),
                      "grip": float(grips.std(ddof=1))},
        "mae_std": float(maes.std(ddof=1)),
        "arm_std": float(arms.std(ddof=1)),
        "grip_std": float(grips.std(ddof=1)),
        "per_draw_mae": [float(v) for v in maes],
        # TAPPED AT THE SINK, not restated. These are parsed back out of the argv that
        # was actually handed to each child `lerobot_predict.py`, so the field cannot
        # stay green if the loop below ever stops passing the seed it computed. Writing
        # `seeds` (the range) here instead would export the FORMULA a second time --
        # gr00t-n17's leg went green that way while its sampler reused noise.
        "per_draw_seeds": [int(a[a.index("--seed") + 1]) for a in launched_argv],
        "per_draw_seeds_source": "parsed from the argv passed to subprocess (launch boundary)",
        "per_draw_seeds_distinct": len({a[a.index("--seed") + 1] for a in launched_argv}) == len(launched_argv),
        # ...and the honest caveat about what distinct seeds are worth HERE. Measured in
        # logs/rng_consumption_<bb>.json, not assumed from architecture.
        "predict_path_consumes_rng": _rng_consumed,
        "per_draw_seeds_are_load_bearing": _rng_consumed,
        "inference_rerun_in_this_invocation": (not args.seeds_from_argv),
        # gr00t-n17's second audit: distinct per-DRAW seeds do NOT rule out per-(draw,
        # anchor) seed COLLISIONS. With `base + stride*draw + anchor`, a stride below the
        # anchor count makes draw d anchor g reuse draw d+1 anchor g-stride's noise; the
        # draws correlate and the std comes out too SMALL -- noise reported as signal, the
        # dangerous direction. n17 measured 254/1000 reused in their own script while its
        # `per_draw_seeds` list stayed perfectly distinct, so only the stride catches it.
        # Ours is 1_000_003 against 1397 anchors = 716x headroom, but the number has to be
        # in the file: a reader cannot audit a formula it cannot see.
        "per_anchor_seed_formula": f"torch.manual_seed(draw_seed * {seed_mult} + global_anchor_index)",
        "per_draw_seed_stride": int(seed_mult),
        "per_draw_seed_stride_source": f"ast-extracted from {PREDICT_SRC.name}:{seed_lineno}",
        "n_anchors_full": 1397,
        "seed_stride_exceeds_anchor_count": bool(seed_mult > 1397),
        # team-lead's replacement gate: the stride field above is DERIVED, and a derived
        # field is exactly what went green on n17 while their sampler reused noise. These
        # three are MEASURED -- torch.manual_seed is actually called for every (draw,
        # anchor) pair and the resulting RNG states are hashed and counted.
        "per_anchor_noise_collisions_measured": int(coll_live),
        "per_anchor_noise_collisions_grid_size": int(len(seeds) * len(sel)),
        "collision_power_control": {
            "stride0_collisions": int(coll_s0),
            "stride0_must_be_nonzero": True,
            "stride1_collisions": int(coll_s1),
            "stride1_is_reported_not_gated": "n17's real defect, but it can be innocently 0 "
                                             "on a sparse anchor subset (d+g collides only for "
                                             "anchors closer than the draw count), so gating on "
                                             "it would re-create the zero-power trap",
        },
        # Same block under the key gr00t-n17's reader actually looks up. n15 measured the
        # cost of not doing this: a reader that misses a power-control key does NOT fall
        # back loudly, it just silently stops counting the evidence and accepts a bare 0
        # as if it were unaudited. Aliases are three lines; a silently-uncounted control
        # is indistinguishable from no control at all.
        "harness_power_control": {
            "stride0_collisions": int(coll_s0),
            "stride1_collisions": int(coll_s1),
            "live_collisions": int(coll_live),
            "grid_size": int(len(seeds) * len(sel)),
            "control_is_construction_fixed": "stride 0 collides (draws-1)*anchors on ANY anchor "
                                             "set, so the gate is an equality by construction "
                                             "rather than a data-dependent '> 0'",
        },
        "noise_collision_power_control": {"stride0_collisions": int(coll_s0),
                                          "stride1_collisions": int(coll_s1)},
        # Self-advertised alias list (gr00t-n17's request). A hardcoded union of four
        # legs' key names does not extend -- the fifth reader has to be taught again --
        # so the artifact names its own power-control keys and the reader tries those
        # first. This can only ADD names, never waive the check: n17 verified across 11
        # positive controls that a new key WITHOUT an advertised alias still gets noted,
        # and an advertised alias pointing at a key that does not exist still gets noted.
        # So forgetting this field costs a note, it does not open an escape hatch.
        "power_control_key_aliases": ["collision_power_control", "harness_power_control",
                                      "noise_collision_power_control",
                                      "power_control_collisions"],
        "bit_exact_across_draws": bool(bit_exact),
        "subset_vs_full_provenance": prov,
        # A bit_exact claim is UNINTERPRETABLE without the batch shape it was measured at:
        # bf16 matmul is not batch-invariant, so a false could mean "wrong checkpoint" or
        # merely "different batch size", and those two have opposite remedies. gr00t-n17
        # measured maxdiff 0.78 between batch 8 and batch 1 on N1.5 -- large enough to move
        # an MAE. Both sides here are batch_size 1 (the top-level field above).
        "bit_exact_claims_are_at_batch_size": 1,
        # gr00t-n15's third mandatory field. A `bit_exact: true` is satisfiable by a weak
        # test -- slicing one in-memory array twice inside ONE process proves nothing about
        # a separately-launched run. Here every draw AND the delivered row are separate
        # `subprocess.run` invocations of lerobot_predict.py with fresh interpreters, fresh
        # CUDA contexts and freshly loaded weights, so the claim spans process boundaries.
        "subset_run_is_independent_process": True,
        # The std above is measured on 200 anchors but the BOARD row is a mean over 1397,
        # and per-anchor noise is drawn independently (per-anchor reseed), so the spread of
        # the board-scale mean is smaller by sqrt(200/1397). openpi MEASURED the exponent on
        # n15's draws (-0.4901 vs the -0.5 independent-noise prediction) rather than assuming
        # it. Emit BOTH so no reader has to guess which scale a std is on -- mixing them
        # overstates the sampling term by 2.64x. (It changes no verdict on this board:
        # SE_episode is 0.10-0.18, so the sampling term is ~1% of the variance either way.)
        "sampling_se_scaling": {
            "std_above_is_measured_on_n_anchors": int(len(sel)),
            "board_row_is_scored_on_n_anchors": 1397,
            "factor": float(np.sqrt(len(sel) / 1397)),
            "exponent_measured_by_openpi": -0.4901,
            "se_sampling_full1397": {
                "mae": float(maes.std(ddof=1) * np.sqrt(len(sel) / 1397)),
                "arm": float(arms.std(ddof=1) * np.sqrt(len(sel) / 1397)),
                "grip": float(grips.std(ddof=1) * np.sqrt(len(sel) / 1397)),
            },
            "which_to_use": "per-DRAW std (not std/sqrt(R)) -- the board row is ONE realization; "
                            "the unscaled subset-200 value is the conservative choice",
        },
        # Board rule (d) (team-lead, from gr00t-n15's measurement). A subset caliber is
        # unreadable without all four of: how many anchors, how they were stratified, how
        # many per episode, and HOW THEY ARE WEIGHTED -- plus a pass line recomputed on the
        # subset itself. n15 measured that ~half the subset-vs-full shift is not sampling
        # noise at all but a deterministic per-frame -> per-episode REWEIGHTING, and that it
        # moves numerator and denominator in OPPOSITE directions (long episodes are harder
        # for a model, corr +0.34, and easier for hold-state, corr -0.62), so a model/hold
        # ratio moves further than either side: 1.501 -> 1.607 on their leg, +7.1%, from one
        # preds file with zero randomness. More draws cannot average that away.
        "subset_design": {
            "n_anchors": int(len(sel)),
            "n_anchors_full_population": 1397,
            "stratification": "10 per episode x 20 val episodes (balanced; verified below)",
            "per_episode_counts": {str(int(e)): int(c) for e, c in
                                   zip(*np.unique(_sub_eps, return_counts=True))},
            "n_episodes_covered": int(len(np.unique(_sub_eps))),
            "weighting": "frame-length-weighted (unweighted mean over anchors; long episodes "
                         "contribute proportionally more anchors). NOT episode-equal.",
            "effective_episode_count": "200 anchors / ~10 per episode = 20 episodes; an "
                                       "UNSTRATIFIED 200-anchor draw would instead be a ~3-episode "
                                       "eval (n15: block-resample sd 2.3x the iid sd)",
        },
        # Recomputed ON THE SUBSET. A reader who compares a subset MAE against the full-1397
        # pass line gets a systematically flattering answer, because the two sides shift in
        # opposite directions -- so the correct baseline must travel with the subset number.
        "subset_pass_line": {"mae": float(gt_a.mean()), "arm": float(gt_a[..., :7].mean()),
                             "grip": float(gt_a[..., 7].mean())},
        "full_pass_line_do_not_use_with_subset_mae": {"mae": 3.0719962120056152,
                                                      "arm": 2.5352115631103516,
                                                      "grip": 6.829487323760986},
        "warning": "subset MAE is NOT comparable to the full-1397 leaderboard "
                   "(different anchors -> different pass line); footnote only.",
    }
    canonical = outdir / f"variance_{args.backbone}.json"
    out_json = pathlib.Path(args.out) if args.out else canonical
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[write] {out_json}")

    # ---- writer/reader round-trip: can the SHARED consumer read what I just wrote? ---
    # gr00t-n17 found their own variance writer emitting a schema their own reader did
    # not accept -- and they found it from the READER side, i.e. the shared `--all` run
    # would have died the moment their row landed. The writer is the cheap place to
    # catch that: it already knows the file, and it fails before anyone else's run does.
    # `sampling_std` deliberately RAISES on an unknown schema (a silent 0.0 fallback
    # would make a dropped sampling term look identical to a deterministic decoder), so
    # this is a real gate, not a formality. Its std==0.0 / seed-stride audits print
    # warnings here too -- if a warning appears below, the footnote is missing a field
    # the shared reader wants, and I fix it now rather than after the board is built.
    #
    # ⚠ It reads the CANONICAL path, so under --out it would validate the OLD production
    # file and print a green line about a file this run never touched -- a pass that is
    # about the wrong artifact, which is the same shape as the defect --out exists to
    # prevent. Skip loudly instead of passing vacuously.
    if out_json != canonical:
        print(f"[roundtrip] SKIPPED: wrote {out_json}, but the shared reader only looks at "
              f"{canonical}. A 'pass' here would describe the previous production file, not "
              f"this run. Re-run without --out before trusting the schema.")
        return
    sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                    str(BENCH / "archive")]   # was one flat scripts/ dir
    import paired_signif
    for chan, key in paired_signif.VAR_KEY.items():
        got = paired_signif.sampling_std(args.backbone, chan)
        want = res[key]["std"]
        if got is None or abs(got - want) > 0:
            sys.exit(f"WRITER/READER MISMATCH on {key}: paired_signif reads {got}, "
                     f"file says {want}. My row would corrupt or kill the shared --all run.")
    print("[roundtrip] paired_signif.sampling_std reads back all 3 channels bit-exact "
          "(no [warn] lines above = every field its audits want is present)")


if __name__ == "__main__":
    main()
