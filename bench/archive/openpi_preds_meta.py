#!/usr/bin/env python3
"""Provenance + scored-row artifact for an openpi board row (pi0 / pi0.5).

Board rule (adopted 2026-08-05): any script that produces a result artifact takes an explicit
`--out` and writes the PATH + MD5 of every input into its own output. The failure this prevents is
not hypothetical -- `preds/` already holds ACT's and N1.5's arrays with the right schema, the right
units and no NaNs, so a mis-pointed run publishes another leg's numbers under my name and nothing
downstream can tell. Hence:
  * `--out` is REQUIRED and has no default (a dry run that writes to the production path is not a
    dry run);
  * the preds filename must match `preds_<tag>.npz` or the run dies -- that single assertion is
    what makes "wrong file" unrepresentable rather than merely unlikely;
  * every input is recorded by md5, including the norm_stats baked inside the checkpoint.

Reuses the tested helpers from variance_openpi rather than re-deriving the pass line: one
implementation, one test suite (archive/test_variance_openpi.py).

  python openpi_preds_meta.py --tag pi0 --preds preds/preds_pi0.npz \
      --ckpt-dir "$NERO_CKPT/pi0_nero_b2_train/pi0_b2train/29999" \
      --config-name pi0_nero_b2_train --out /tmp/preds_pi0_meta.json
"""
import argparse, json, pathlib, sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir
from variance_openpi import PASS_FULL, _hold_err, _weighted, md5_of  # tested helpers

VAL_EPISODES = [0, 2, 3, 4, 20, 24, 25, 33, 37, 51, 56, 68, 70, 80, 90, 91, 92, 101, 106, 111]
NORM_STATS_MD5 = "9467e876b49b33168d5c2654482f0d09"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="pi0 | pi05")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--out", required=True,
                    help="REQUIRED, no default: point a dry run at /tmp, never at logs/ or preds/")
    ap.add_argument("--allow-normstats-mismatch", action="store_true",
                    help="inspect a ckpt whose baked norm_stats is NOT the leak-free source")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--host", default="")
    ap.add_argument("--wandb", default="")
    args = ap.parse_args()

    preds_path = pathlib.Path(args.preds)
    # The gate that makes "published another leg's numbers" unrepresentable.
    if preds_path.stem != f"preds_{args.tag}":
        raise SystemExit(f"[FATAL] --preds {preds_path.name} is not this tag's file "
                         f"(expected preds_{args.tag}.npz). preds/ holds every leg's arrays with "
                         f"identical schema and units; naming them apart is the only defence.")
    if not preds_path.is_file():
        raise SystemExit(f"[FATAL] {preds_path} does not exist")

    A = np.load(BENCH / "data" / "val_anchors.npz")
    gt, state, eps = (A["gt"].astype(np.float32), A["state"].astype(np.float32), A["episodes"])
    N, K, D = gt.shape

    Z = np.load(preds_path)
    P_full = Z["pred"].astype(np.float32)
    if P_full.shape[0] != N:
        raise SystemExit(f"[FATAL] preds has {P_full.shape[0]} anchors, val_anchors has {N}")
    if "anchor_idx" in Z and not np.array_equal(Z["anchor_idx"], np.arange(N)):
        raise SystemExit("[FATAL] anchor_idx is not 0..N-1: rows are not in val_anchors order, so "
                         "every per-anchor comparison downstream would be silently misaligned")
    if not np.isfinite(P_full).all():
        raise SystemExit("[FATAL] non-finite cells in preds")
    P = P_full[:, :K, :]

    # Pass line: recomputed here, validated against the board constants with the SAME tolerance
    # the variance script uses (round-3 equality has a cliff -- grip lands on 6.82950 exactly).
    herr = _hold_err(state, gt, K)
    pass_full = {"mae": _weighted(herr, eps)[0], "arm": float(herr[..., :7].mean()),
                 "grip": float(herr[..., 7].mean())}
    bad = {k: (pass_full[k], v) for k, v in PASS_FULL.items() if abs(pass_full[k] - v) >= 1e-3}
    if bad:
        raise SystemExit(f"[FATAL] recomputed pass line disagrees with the board constants: {bad}")

    err = np.abs(P - gt)
    mae, arm, grip = float(err.mean()), float(err[..., :7].mean()), float(err[..., 7].mean())

    # The leak-provenance field. A glob that matches nothing would write `md5: null`, which reads
    # exactly like "the check ran and found nothing wrong" -- so it is fatal, not null. Layout is
    # <step>/assets/local/<repo>/norm_stats.json (verified against four landed openpi ckpts).
    ns = sorted((pathlib.Path(args.ckpt_dir)).glob("assets/*/*/norm_stats.json"))
    if not ns:
        raise SystemExit(f"[FATAL] no assets/*/*/norm_stats.json under {args.ckpt_dir}. openpi "
                         f"always bakes the stats it trained on; their absence means this is not "
                         f"a training checkpoint, and a null md5 here would look like a passed "
                         f"check. Refusing to publish a leak-provenance claim I cannot support.")
    ns_md5 = md5_of(ns[0])
    if ns_md5 != NORM_STATS_MD5 and not args.allow_normstats_mismatch:
        raise SystemExit(f"[FATAL] the norm_stats baked into this checkpoint ({ns_md5}) is not the "
                         f"leak-free 111-episode source ({NORM_STATS_MD5}).\n  {ns[0]}\n"
                         f"  This model was normalized with different statistics than the split "
                         f"the board believes it was trained on -- the row's leak-free claim does "
                         f"not hold. --allow-normstats-mismatch to inspect anyway.")

    meta = {
        "backbone": args.tag,
        "stack": "openpi (third_party/openpi-agilex), JAX, flow matching",
        "ckpt": args.ckpt_dir,
        "inputs": {
            "preds": {"path": str(preds_path), "md5": md5_of(preds_path),
                      "shape": list(P_full.shape), "noise_seed": int(Z["noise_seed"])
                      if "noise_seed" in Z else None},
            "val_anchors": {"path": str(BENCH / "data" / "val_anchors.npz"),
                            "md5": md5_of(BENCH / "data" / "val_anchors.npz")},
            "norm_stats_in_ckpt": {"path": str(ns[0]) if ns else None, "md5": ns_md5,
                                   "matches_leak_free_source": ns_md5 == NORM_STATS_MD5},
        },
        "train": {"config": args.config_name, "host": args.host, "gpu": args.gpu,
                  "wandb": args.wandb, "val_episodes_excluded": VAL_EPISODES},
        "eval": {
            "anchors": int(N), "K": int(K),
            "mae": mae, "arm": arm, "grip": grip,
            "per_joint": [float(v) for v in err.mean(axis=(0, 1))],
            "per_step_mae": [float(v) for v in err.mean(axis=(0, 2))],
            "horizon_predicted": int(P_full.shape[1]),
            # CORRECTED after gr00t-n15 caught the earlier wording. The old note said the npz's
            # 10-step chunk lets a K-slope study "use the unscored tail without a re-run" -- but
            # having 10 PREDICTED steps is not the constraint. val_anchors' gt is (N, 8, 8): the
            # TARGET is capped at K=8, so steps 8..9 have nothing to score against in the column a
            # reader would look in, and taking the note literally means regenerating val_anchors
            # and breaking the board's md5 pin. The tail IS recoverable, via a route I did not
            # know: anchors are stride-5, so frame f+k can be read off anchor(i+m) step k-5m.
            "per_step_note": (
                f"scored steps 0..{K-1}. The npz carries {P_full.shape[1]} predicted steps, but "
                f"val_anchors' gt is capped at K={K}, so the tail is NOT scorable from the gt "
                f"column directly -- do not read this as 'no re-run needed'. gr00t-n15's "
                f"archive/kslope_table.py recovers it instead by chaining stride-5 anchors "
                f"(frame f+k from anchor i+m, step k-5m; costs the last anchor of each episode, "
                f"1357/1397), which needs no re-run AND leaves val_anchors byte-identical."),
        },
        # Units label. MEASURED on the delivered file, not read off the config: mean|pred_arm[0] -
        # state| vs mean|pred_arm[0]|. Recording it because gr00t-n15 inferred "arm = DELTA" for
        # this leg from the ckpt's norm_stats (correctly, for the MODEL'S INTERNAL space) and a
        # reader could carry that label onto the npz, where it is false -- openpi's
        # AbsoluteActions transform adds state@t back before anything is written here.
        "action_parameterization": {
            "arm_j0_j6": "absolute (deg)",
            # MEASURED range, not the physical convention. The NERO gripper is 0-76 mm in
            # hardware, but this dataset's action/state channel is on a 0-100 scale (gt spans
            # 0.000..100.908 -- note it exceeds 100, so it is not a clipped percentage either).
            # Writing "mm" here from the robot's spec would have been wrong by 1.33x.
            "gripper_j7": "absolute, 0-100 scale (0=closed); gt range "
                          f"{float(gt[:, :, 7].min()):.3f}..{float(gt[:, :, 7].max()):.3f}",
            "model_internal_arm_space": "delta vs state@t -- see norm_stats action joint means ~0 "
                                        "against state means ~103; converted by "
                                        "model_transforms.outputs -> AbsoluteActions(7 delta + 1 abs)",
            "evidence_mean_abs_pred0_minus_state": float(np.abs(P[:, 0, :7] - state[:, :7]).mean()),
            "evidence_mean_abs_pred0": float(np.abs(P[:, 0, :7]).mean()),
            "evidence_gt_mean_abs_gt0_minus_state": float(np.abs(gt[:, 0, :7] - state[:, :7]).mean()),
            "gated_at_produce_time": "predict_openpi.py refuses to write if subtracting state@t "
                                     "does not shrink the arm chunk (raw-delta output would sit "
                                     "~38 deg from state instead of ~1.3)",
            "scoring_note": "this label does NOT split the board into competing families. Every "
                            "row is scored against the same absolute val_anchors gt, and MAE is "
                            "invariant to a shift both pred and gt share, so a stack that emitted "
                            "deltas and added them back scores identically either way. The label "
                            "matters for comparing absolute QUANTITIES across stacks, not for the "
                            "ranking.",
        },
        "pass_line_full1397": {k: float(v) for k, v in pass_full.items()},
        "pass_line_recomputed_and_validated": True,
        "verdict": ("beats the do-nothing pass line by "
                    f"{pass_full['mae'] - mae:+.3f} MAE ({100*(pass_full['mae']-mae)/pass_full['mae']:+.1f}%)"
                    if mae < pass_full["mae"] else
                    "WORSE THAN DO-NOTHING -- learned nothing"),
        "determinism": {
            "stochastic": True,
            "noise": "normal(fold_in(key(noise_seed), global_anchor_index)) passed explicitly to "
                     "policy.infer(noise=...) -- NOT the Policy's internal rng, which splits per "
                     "call and would make a row depend on iteration order",
            # MEASURED, not assumed. An earlier version of this field claimed the row was a pure
            # function of (ckpt, anchor, seed). That is true WITHIN a process and false ACROSS
            # processes, and the difference is what both legs' seed0 cross-check reports as a
            # "MISMATCH" -- so state the caliber instead of a guarantee the stack does not give.
            "reproducible_within_process": True,       # same noise twice, one process: 0.000e+00
            "reproducible_across_processes": False,
            "across_process_elementwise_max_deg": 8.4e-02,   # same GPU, same ckpt, same noise
            "across_process_elementwise_mean_deg": 1.2e-02,
            "across_process_aggregate_mae_shift": 7.7e-05,   # 200 anchors, seed 0, two processes
            "reproducibility_note": (
                "openpi/JAX is bit-reproducible within a process but not across processes (XLA "
                "picks kernels per process). Elementwise this is ~1e-02 deg; on the AGGREGATE it "
                "cancels to 7.7e-05 MAE, which is 0.01x this leg's own sampling std (6.7e-03) and "
                "~1000x smaller than the smallest board gap (0.067). So the leaderboard ordering "
                "is unaffected, but do NOT expect a re-run to reproduce this npz byte-for-byte, "
                "and do not use bit-equality as an integrity check on these files."),
            "sampling_footnote": f"logs/variance/variance_{args.tag}.json",
        },
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
