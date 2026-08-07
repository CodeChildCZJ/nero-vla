#!/usr/bin/env python3
"""OFT BOUNDS soft-floor: the MAE an ORACLE that reproduces the clipped TRAINING TARGET
would still score on val_anchors, in raw absolute joint units.

Board convention (see gr00t-n17's inverse-clip floor): report the number at score.py's
window K, and say which K.  Unlike N1.7 the clip here lives on the *forward* pass only
(`nero_dataset.normalize` clips to [-1,1]; `unnormalize` / the model's
`_unnormalize_actions` do NOT clip), and the L1 regression head is unbounded -- so this
is an UPPER bound on the structural penalty, not a hard floor.  True penalty in [0, X].

CPU only, no model, no GPU.  Stats come from the same `compute_dataset_statistics` that
finetune_nero.py bakes into the checkpoint's dataset_statistics.json, so this can be
re-run against the trained checkpoint's own file with --ckpt-stats to confirm identity.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parent.parent

# `prismatic.vla.constants` resolves the platform AT IMPORT TIME from $ROBOT_PLATFORM, falling
# back to an argv substring match and finally to LIBERO -- which is BOUNDS_Q99 + ACTION_DIM 7.
# Set it before the import or this script silently computes a q99 floor for a 7-dim robot.
os.environ.setdefault("ROBOT_PLATFORM", "NERO")
sys.path.insert(0, str(BENCH.parent / "third_party" / "openvla-oft"))

from prismatic.vla.constants import (  # noqa: E402
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NormalizationType,
)
from prismatic.vla.datasets.nero_dataset import (  # noqa: E402
    DATASET_NAME,
    compute_dataset_statistics,
    normalize,
    unnormalize,
)

assert ACTION_DIM == 8, f"wrong platform constants: ACTION_DIM={ACTION_DIM}"
assert ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS, (
    f"wrong platform constants: norm={ACTION_PROPRIO_NORMALIZATION_TYPE}")


def _delivery_inverse_identity(a_stats: dict, gt: np.ndarray) -> dict:
    """Prove THE DELIVERY PATH restores absolute units -- the leg's #1 trap.

    Everything else in this file measures the floor through `nero_dataset.unnormalize`, but that is
    NOT the function that produces preds: `vla.predict_action` calls the model's own
    `_unnormalize_actions`. Two implementations of "the inverse" exist and they are not identical --
    the model's has a `mask` branch (`np.where(mask, formula, normalized_actions)`) that returns the
    head's RAW [-1,1] output on any masked-off dim. So a mask with a False entry would ship that dim
    in [-1,1] while this script, measuring the other implementation, reported a clean floor.

    Bind the SHIPPED method to a stub supplying only `get_action_stats` -- no model, no GPU, no
    weights -- and (a) diff the two inverses, (b) round-trip GT through the model's one.
    """
    import types

    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    stub = types.SimpleNamespace(get_action_stats=lambda _k: a_stats)
    model_inv = OpenVLAForActionPrediction._unnormalize_actions.__get__(stub, type(stub))

    x = np.random.default_rng(0).uniform(-1, 1, (5000, gt.shape[-1]))
    impl_dev = float(np.abs(np.asarray(model_inv(x), np.float64)
                            - unnormalize(x.astype(np.float32), a_stats).astype(np.float64)).max())
    flat = gt.reshape(-1, gt.shape[-1])
    rt = np.abs(np.asarray(model_inv(normalize(flat, a_stats)), np.float32) - flat)
    # Controls: the two ways a unit bug ships a plausible number. score.py's do-nothing flag is the
    # only automatic backstop, so record whether it would actually fire for each.
    raw_mae = float(np.abs(normalize(flat, a_stats) - flat).mean())
    q99 = {**a_stats, "min": a_stats["q01"], "max": a_stats["q99"]}
    s2 = types.SimpleNamespace(get_action_stats=lambda _k: q99)
    q99_mae = float(np.abs(np.asarray(OpenVLAForActionPrediction._unnormalize_actions.__get__(
        s2, type(s2))(normalize(flat, a_stats)), np.float32) - flat).mean())
    return {
        "mask_key_present": "mask" in a_stats,
        "mask_all_true": bool(np.asarray(a_stats.get("mask", [True])).all()),
        # Report the MEASURED mask, not just a verdict on it (n17 2026-08-05: an audit should
        # return the branch/state it observed, so a failure says WHICH dims -- a bare bool only
        # says "not all true", and the dims are exactly what you need to debug a unit bug).
        "mask_measured": (np.asarray(a_stats["mask"]).astype(bool).tolist()
                          if "mask" in a_stats else None),
        "mask_false_dims": ([i for i, m in enumerate(np.asarray(a_stats["mask"]).astype(bool))
                             if not m] if "mask" in a_stats else []),
        "two_inverse_impls_max_abs_dev": impl_dev,
        "two_inverse_impls_agree": impl_dev < 1e-4,
        "roundtrip_via_MODEL_inverse": {"mean": float(rt.mean()), "max": float(rt.max()),
                                        "arm": float(rt[:, :7].mean()), "grip": float(rt[:, 7].mean()),
                                        "per_dim": np.round(rt.mean(0), 6).tolist()},
        "controls_would_score_py_catch_it": {
            "raw_head_output_shipped_as_degrees": {"mae": raw_mae, "flag_fires": raw_mae >= 3.0719962},
            "q01_q99_bounds_by_mistake": {"mae": q99_mae, "flag_fires": q99_mae >= 3.0719962},
        },
        "note": "roundtrip mean here MUST equal floor_at_K_score.total -- same quantity via the "
                "delivery code path instead of nero_dataset.unnormalize",
    }


def _leak_verdict(val_g: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Falsify 'val episodes contributed to the norm stats' from the stats themselves.

    BOUNDS bakes the TRUE min/max, so if any val value lies OUTSIDE the baked box, the box
    cannot have been computed over a set containing that value.  This is evidence carried by
    the delivered artifact, independent of whether my episode-filtering code is correct --
    which is the point: it does not trust the code under test.

    Direction matters.  A dim whose val range sits INSIDE the train box proves nothing (that
    is what leak-free data usually looks like); only exceedances are informative, so this
    reports coverage rather than a bare pass.  See docs/METHODOLOGY.md (leak-freedom check).
    """
    vmax, vmin = val_g.max(0), val_g.min(0)
    exceeds_hi = vmax > high
    exceeds_lo = vmin < low
    informative = exceeds_hi | exceeds_lo
    return {
        "test": "val_max > baked_train_max  OR  val_min < baked_train_min  =>  val not in stats",
        "dims_exceeding_high": [int(d) for d in np.flatnonzero(exceeds_hi)],
        "dims_exceeding_low": [int(d) for d in np.flatnonzero(exceeds_lo)],
        "max_exceedance_high": (vmax - high).round(6).tolist(),
        "max_exceedance_low": (low - vmin).round(6).tolist(),
        "n_informative_dims": int(informative.sum()),
        "verdict": (
            "CLEAN"
            if informative.any()
            else "INCONCLUSIVE -- val lies entirely inside the train box on every dim; this test "
                 "cannot distinguish leak from no-leak here, rely on the bit-exact recompute"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(BENCH / "data" / "b2_oft"))
    ap.add_argument("--split", default=str(BENCH / "data" / "split.json"))
    ap.add_argument("--anchors", default=str(BENCH / "data" / "val_anchors.npz"))
    ap.add_argument("--ckpt-stats", default=None,
                    help="path to a checkpoint's dataset_statistics.json; when given, the "
                         "floor is computed from IT and diffed against the recomputed stats")
    # No default, deliberately. A fixed default --out means the LAST run wins the authoritative
    # filename, and the ordering is systematically wrong: the authoritative run happens once, the
    # cheap debug reruns happen after it (gr00t-n17 lost a full-population artifact to an
    # `--n-anchors 200` rerun this way -- the file was internally consistent and silently a subset).
    # Requiring the name forces every run to say which population/stats-source it is.
    ap.add_argument("--out", required=True,
                    help="output json; REQUIRED -- put the stats-source/population in the name")
    args = ap.parse_args()

    split = json.loads(pathlib.Path(args.split).read_text())
    train_eps = split["train_episodes"]

    traj_f = np.load(pathlib.Path(args.data_root) / "traj.npz")
    traj = {k: traj_f[k] for k in traj_f.files}
    recomputed = compute_dataset_statistics(traj, train_eps)[DATASET_NAME]

    stats = recomputed
    stats_source = f"recomputed from split.json train_episodes (n={len(train_eps)})"
    ckpt_diff = None
    if args.ckpt_stats:
        baked = json.loads(pathlib.Path(args.ckpt_stats).read_text())[DATASET_NAME]
        # lerobot's float32/float64 trap: cast BOTH sides to the dtype the inverse uses.
        # BOTH channels, not just action.  `ACTION_PROPRIO_NORMALIZATION_TYPE` normalizes proprio
        # too, and predict feeds it live (`normalize(state, stats[key]["proprio"])` ->
        # ProprioProjector), so proprio stats are an INPUT-side leak channel.  Checking action
        # alone covers 6 of the 12 fields the inference path reads while looking complete --
        # n17 hit the same shape on N1.7 (abs stats clean => "clean", but the relative table,
        # 7 of its 8 dims, was never tested).  Keys stay FLAT and prefixed so the consumer's
        # `not v["bit_identical"]` filter keeps working and simply now spans 12 fields.
        ckpt_diff = {}
        for chan in ("action", "proprio"):
            for key in ("min", "max", "q01", "q99", "mean", "std"):
                a = np.asarray(baked[chan][key], np.float32)
                b = np.asarray(recomputed[chan][key], np.float32)
                ckpt_diff[f"{chan}.{key}"] = {"max_abs_dev": float(np.abs(a - b).max()),
                                              "bit_identical": bool(np.array_equal(a, b))}
        stats = baked
        stats_source = f"checkpoint dataset_statistics.json ({args.ckpt_stats})"

    a_stats = stats["action"]
    low = np.asarray(a_stats["min"], np.float32)
    high = np.asarray(a_stats["max"], np.float32)

    anc = np.load(args.anchors)
    gt = anc["gt"].astype(np.float32)          # (A, K, 8) raw absolute units
    K_score = int(anc["K"])
    A, K_full, D = gt.shape

    # An oracle that perfectly reproduces the clipped training target.
    oracle = unnormalize(normalize(gt, a_stats), a_stats)
    err = np.abs(oracle - gt)                  # (A, K, 8)

    def block(k: int) -> dict:
        e = err[:, :k, :]
        return {
            "K": k,
            "total": float(e.mean()),
            "arm": float(e[:, :, :7].mean()),
            "grip": float(e[:, :, 7].mean()),
            "per_dim": [float(e[:, :, d].mean()) for d in range(D)],
        }

    # Where the penalty comes from: val frames outside the TRAIN box.
    g = gt[:, :K_score, :].reshape(-1, D)
    below = (g < low).sum(0)
    above = (g > high).sum(0)
    n_frames = g.shape[0]
    overshoot_hi = np.where(g > high, g - high, 0.0)
    overshoot_lo = np.where(g < low, low - g, 0.0)

    out = {
        "leg": "openvla-oft",
        "normalization": "BOUNDS (train min/max)",
        "clip_semantics": {
            "forward_normalize_clips": True,
            "inverse_unnormalize_clips": False,
            "action_head": "L1 regression, unbounded MLPResNet",
            "floor_kind": "SOFT -- this is an UPPER bound; true structural penalty in [0, value]",
        },
        "stats_source": stats_source,
        "ckpt_vs_recomputed": ckpt_diff,
        "delivery_inverse_identity": _delivery_inverse_identity(a_stats, gt),
        "n_anchors": int(A),
        "K_score": K_score,
        "K_full": int(K_full),
        "floor_at_K_score": block(K_score),
        "floor_by_K": [block(k) for k in range(1, K_full + 1)],
        "train_box": {
            "min": low.tolist(),
            "max": high.tolist(),
            "num_transitions": int(stats["num_transitions"]),
            "num_trajectories": int(stats["num_trajectories"]),
        },
        "val_oob_at_K_score": {
            "n_frames": int(n_frames),
            "below_min_count": below.tolist(),
            "above_max_count": above.tolist(),
            "below_min_pct": (below / n_frames * 100).round(4).tolist(),
            "above_max_pct": (above / n_frames * 100).round(4).tolist(),
            "max_overshoot_high": overshoot_hi.max(0).round(6).tolist(),
            "max_overshoot_low": overshoot_lo.max(0).round(6).tolist(),
        },
        "self_inflicted_split": {
            "note": "BOUNDS uses TRUE train min/max, so unlike BOUNDS_Q99 / N1.7's q99 gripper "
                    "there is NO self-inflicted truncation component: 100% of this floor is "
                    "genuine val-outside-train-range OOD.",
        },
        "leakage_falsification": _leak_verdict(g, low, high),
        # Second channel.  val_anchors' `state` IS the proprio quantity predict normalizes, so the
        # same exceedance test applies to the proprio box with no new data.  Kept as its own key
        # (not folded into the one above) because a combined verdict could be carried by whichever
        # channel happens to be informative, hiding an untested one -- the exact failure being
        # fixed here.
        "leakage_falsification_proprio": _leak_verdict(
            anc["state"].astype(np.float32),
            np.asarray(stats["proprio"]["min"], np.float32),
            np.asarray(stats["proprio"]["max"], np.float32),
        ),
        "normalization_artifacts_enumerated": {
            "action": "dataset-derived (BOUNDS min/max) -- falsified above",
            "proprio": "dataset-derived (same BOUNDS stats) -- falsified above",
            "image": "NO dataset statistic exists: dataset_statistics.json has no image key at all; "
                     "pixel normalization is a CONSTANT baked into the pretrained OpenVLA-7B "
                     "preprocessor (DINOv2 tower ImageNet 0.485/0.456/0.406, SigLIP tower 0.5) and "
                     "never touches NERO data => structurally zero exposure, nothing to re-check "
                     "per retrain. Stronger than ACT's case, where a dataset image stat IS computed "
                     "and then overwritten by ImageNet constants.",
        },
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))

    f = out["floor_at_K_score"]
    print(f"OFT soft floor @K={K_score} (A={A}): total {f['total']:.6f}  "
          f"arm {f['arm']:.6f}  grip {f['grip']:.6f}")
    print(f"  per-dim: {[round(v, 6) for v in f['per_dim']]}")
    print(f"  as pct of pass line 3.072: {f['total'] / 3.072 * 100:.3f}%")
    print(f"  K-dependence: " + "  ".join(
        f"K{b['K']}={b['total']:.6f}" for b in out["floor_by_K"]))
    print(f"  val OOB frames/{n_frames}: below={below.tolist()} above={above.tolist()}")
    if ckpt_diff:
        print(f"  ckpt vs recomputed: {ckpt_diff}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
