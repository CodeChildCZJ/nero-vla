"""Measure openpi's inverse-transform floor on the leaderboard population -- both recipes.

Every other leg of this bench pays an irreducible "floor": the error a PERFECT model still
scores, because the inverse transform cannot represent the ground truth. N1.7 pays 0.0176
(its `unnormalize_values_minmax` clips to [-1,1] unconditionally), OFT <=0.0239. I have been
reporting openpi's floor as 0 on the strength of a source read: there is no `clip`/`clamp`
anywhere in `transforms.py`, `DeltaActions`/`AbsoluteActions` are pure -=/+=, and `NeroOutputs`
is a slice.

That is an argument, not a measurement, and gr00t-n17 just demonstrated on its own leg why the
two differ (a `--n-anchors 200` default silently reported a +33% wrong floor with no warning).
A source read also cannot catch a floor that comes from somewhere other than a clip -- float32
round-off in the affine, or a dim-padding asymmetry between Normalize (truncates stats) and
Unnormalize (pads them). So: push the real GT through the real chain and measure.

  raw GT --DeltaActions(state)--> delta --Normalize--> normalized
         --Unnormalize--> delta' --AbsoluteActions(state)--> raw GT'
  floor = |raw GT' - raw GT|, grouped exactly as score.py groups (arm j1-j7 / gripper j8).

This is the counterfactual "the model's normalized output is exactly right" -- i.e. the best
score the recipe permits. Run for BOTH recipes off the same md5-identical norm_stats file,
because the two differ in which stats they use: pi0 z-score (mean/std), pi0.5 quantile
(q01/q99). A quantile inverse is the shape that DOES clip on other stacks, so pi0.5 is the one
actually worth measuring; pi0 comes along free.

CPU only, no GPU, no model, seconds.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "third_party" / "openpi-agilex" / "src"))
from openpi import transforms as _tf  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402

ANCHORS = ROOT / "data" / "val_anchors.npz"
STATS_DIR = str(ROOT.parent / "third_party" / "openpi-agilex" / "assets" / "{cfg}" /
                "local" / "pick_pink_sponge_b2_train")
OUT = ROOT / "logs" / "openpi_roundtrip_floor.json"

# score.py's grouping: dims 0-6 are the arm (deg), dim 7 the gripper (0-100).
ARM = slice(0, 7)
GRIP = 7
MODEL_ACTION_DIM = 32  # what the model actually emits; Unnormalize sees this width, not 8


def roundtrip(gt, state, norm_stats, use_quantiles):
    """Push raw absolute GT through forward+inverse exactly as the policy would."""
    n, k, d = gt.shape
    # Forward: absolute -> delta (arm only, gripper stays absolute), then normalize.
    fwd_delta = _tf.DeltaActions(_tf.make_bool_mask(7, -1))
    data = {"state": state.copy(), "actions": gt.copy()}
    delta = fwd_delta(data)["actions"]

    # Training-time Normalize sees both keys (policy_config.py:81, strict default).
    norm = _tf.Normalize(norm_stats, use_quantiles=use_quantiles)
    normalized = norm({"state": state.copy(), "actions": delta.copy()})["actions"]

    # The model emits MODEL_ACTION_DIM; pad so Unnormalize sees the width it sees in production
    # (this is where a Normalize-truncates / Unnormalize-pads asymmetry would show up).
    padded = np.concatenate(
        [normalized, np.zeros((n, k, MODEL_ACTION_DIM - d), dtype=normalized.dtype)], axis=-1)

    # Inverse: the production output chain. strict=False mirrors policy_config.py:86 -- the
    # output dict carries only "actions", so a strict Unnormalize would raise on the "state"
    # selector. Reconstructing this with the default strict=True is not the production path.
    unnorm = _tf.Unnormalize(norm_stats, use_quantiles=use_quantiles, strict=False)
    back = unnorm({"actions": padded})["actions"]
    back = _tf.AbsoluteActions(_tf.make_bool_mask(7, -1))(
        {"state": state.copy(), "actions": back})["actions"]
    back = back[..., :d]  # NeroOutputs

    err = np.abs(back - gt)
    return {
        "total": float(err.mean()),
        "arm": float(err[..., ARM].mean()),
        "grip": float(err[..., GRIP].mean()),
        "max_abs": float(err.max()),
        "max_arm": float(err[..., ARM].max()),
        "max_grip": float(err[..., GRIP].max()),
        "n_exact": int((err == 0).sum()),
        "n_cells": int(err.size),
    }


def out_of_range(gt, state, norm_stats, use_quantiles):
    """How far the normalized GT leaves [-1,1] -- what a clipping inverse WOULD have cost."""
    delta = _tf.DeltaActions(_tf.make_bool_mask(7, -1))(
        {"state": state.copy(), "actions": gt.copy()})["actions"]
    z = _tf.Normalize(norm_stats, use_quantiles=use_quantiles)(
        {"state": state.copy(), "actions": delta.copy()})["actions"]
    over = np.maximum(np.abs(z) - 1.0, 0.0)
    frac = float((np.abs(z) > 1.0).mean())
    # counterfactual: the floor openpi would pay if its inverse clipped like N1.7's
    clipped = _tf.Unnormalize(norm_stats, use_quantiles=use_quantiles, strict=False)(
        {"actions": np.concatenate(
            [np.clip(z, -1.0, 1.0),
             np.zeros((*z.shape[:2], MODEL_ACTION_DIM - z.shape[-1]), dtype=z.dtype)], axis=-1)}
    )["actions"]
    clipped = _tf.AbsoluteActions(_tf.make_bool_mask(7, -1))(
        {"state": state.copy(), "actions": clipped})["actions"][..., : gt.shape[-1]]
    cerr = np.abs(clipped - gt)
    # Per-dim, because "how far outside [-1,1] does the target sit" is a per-joint property and
    # the gripper is the dim the pi0-vs-pi0.5 confound test turns on. A flow model's prior is
    # N(0,1); a target at |z|=12 is a qualitatively harder regression than one at |z|=1.2, so
    # this is a *mechanism* for a recipe-level gripper gap, not just a floor bookkeeping number.
    per_dim_frac = [float((np.abs(z[..., j]) > 1.0).mean()) for j in range(gt.shape[-1])]
    per_dim_p999 = [float(np.quantile(np.abs(z[..., j]), 0.999)) for j in range(gt.shape[-1])]
    return {
        "frac_outside_pm1": frac,
        "per_dim_frac_outside_pm1": per_dim_frac,
        "per_dim_absz_p999": per_dim_p999,
        "arm_frac_outside_pm1": float((np.abs(z[..., ARM]) > 1.0).mean()),
        "grip_frac_outside_pm1": float((np.abs(z[..., GRIP]) > 1.0).mean()),
        "max_overshoot_norm": float(over.max()),
        "counterfactual_clipped_total": float(cerr.mean()),
        "counterfactual_clipped_arm": float(cerr[..., ARM].mean()),
        "counterfactual_clipped_grip": float(cerr[..., GRIP].mean()),
    }


def main():
    A = np.load(ANCHORS)
    gt = A["gt"].astype(np.float32)        # (1397, K=8, 8) raw absolute
    state = A["state"].astype(np.float32)  # (1397, 8)
    print(f"population: gt {gt.shape} state {state.shape} (FULL leaderboard set, no subsample)",
          flush=True)

    res = {"population": {"n_anchors": int(gt.shape[0]), "K": int(gt.shape[1]),
                          "subsampled": False}}
    for cfg, use_q, label in [("pi0_nero_b2_train", False, "pi0 (z-score mean/std)"),
                              ("pi05_nero_b2_train", True, "pi0.5 (quantile q01/q99)")]:
        ns = _normalize.load(STATS_DIR.format(cfg=cfg))
        r = roundtrip(gt, state, ns, use_q)
        o = out_of_range(gt, state, ns, use_q)
        res[cfg] = {"recipe": label, "use_quantiles": use_q, **r, **o}
        print(f"\n--- {label} [{cfg}] ---", flush=True)
        print(f"  floor  total {r['total']:.6g}  arm {r['arm']:.6g}  grip {r['grip']:.6g}",
              flush=True)
        print(f"  max|err| {r['max_abs']:.6g}  (arm {r['max_arm']:.6g} grip {r['max_grip']:.6g})"
              f"   exact cells {r['n_exact']}/{r['n_cells']}", flush=True)
        print(f"  normalized GT outside [-1,1]: {o['frac_outside_pm1'] * 100:.2f}% of cells "
              f"(arm {o['arm_frac_outside_pm1'] * 100:.2f}% / grip "
              f"{o['grip_frac_outside_pm1'] * 100:.2f}%), max overshoot "
              f"{o['max_overshoot_norm']:.4g}", flush=True)
        print("   per-dim %outside : " + " ".join(
            f"j{j + 1}={f * 100:5.2f}" for j, f in enumerate(o["per_dim_frac_outside_pm1"])),
            flush=True)
        print("   per-dim |z| p99.9: " + " ".join(
            f"j{j + 1}={v:5.2f}" for j, v in enumerate(o["per_dim_absz_p999"])), flush=True)
        print(f"  counterfactual IF the inverse clipped (N1.7-style): total "
              f"{o['counterfactual_clipped_total']:.6g} "
              f"(arm {o['counterfactual_clipped_arm']:.6g} "
              f"grip {o['counterfactual_clipped_grip']:.6g})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
