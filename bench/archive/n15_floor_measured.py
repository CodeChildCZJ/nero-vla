#!/usr/bin/env python3
"""MEASURE the N1.5 normalization floor through the SHIPPED transform objects.

Prompted by gr00t-n17 2026-08-05: their `HARD` floor label turned out to rest on a
docstring sentence rather than on read source, and the clip they assumed lives in the
*forward* path only. Their advice: check whether n15's "min_max both sides => floor=0"
was MEASURED off the checkpoint or DERIVED from the config. It was derived. This
measures it.

What is measured: the action round trip that the training target and the model output
actually pass through --
    normalized = StateActionTransform.apply(gt_action)      # shipped forward
    recon      = StateActionTransform.unapply(normalized)   # shipped inverse
    floor      = mean |recon - gt|      (score.py convention: K-prefix, arm=0..6, grip=7)
The transform is the one `nero_data_config.NeroDualCamDataConfig.transform()` builds and
`Gr00tPolicy._load_metadata` parameterises -- constructed here the same way, from the
checkpoint's experiment_cfg/metadata.json. No model is loaded, no GPU is touched.

Power controls (the board META-rule: the test of the gate needs its own negative control,
because a zero-power control looks exactly like a passing gate):
  ctrl_clip  -- inject a clamp(-1,1) on the normalized values, i.e. simulate the
                `clip_outliers=True` that n17 found on their leg. Floor MUST become > 0,
                otherwise this script could not have detected a clip if there were one.
  ctrl_q99   -- switch the shipped Normalizer to mode "q99", which is the one shipped
                branch that really does clamp (state_action.py:134) and whose inverse
                un-maps through q01/q99. Exercises shipped code, not a mock.
  ctrl_noop  -- transparency: re-run the real config, must reproduce the real number.
"""
import argparse, hashlib, json, os, pathlib, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only; must not touch any GPU

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "Isaac-GR00T-n1d5"
CKPT = pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "gr00t_n15_nero_b2" / "checkpoint-10000"
VAL_ANCHORS_MD5 = "cc34dda2a99ade1011c192e2b3e4e343"  # board pin, frozen 2026-08-05 00:03:53

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BENCH / "configs"))

from gr00t.data.schema import DatasetMetadata  # noqa: E402
from gr00t.data.transform.state_action import StateActionTransform  # noqa: E402
from nero_data_config import NeroDualCamDataConfig  # noqa: E402

ACTION_KEYS = ["action.single_arm", "action.gripper"]
SLICES = {"action.single_arm": slice(0, 7), "action.gripper": slice(7, 8)}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_action_transform(mode="min_max"):
    """The shipped StateActionTransform for the action keys, metadata-loaded from the ckpt."""
    cfg = NeroDualCamDataConfig()
    composed = cfg.transform()
    with open(CKPT / "experiment_cfg" / "metadata.json") as f:
        metas = json.load(f)
    meta = DatasetMetadata.model_validate(metas["new_embodiment"])
    composed.set_metadata(meta)
    picked = [t for t in composed.transforms
              if isinstance(t, StateActionTransform) and set(t.apply_to) == set(ACTION_KEYS)]
    assert len(picked) == 1, f"expected exactly 1 action StateActionTransform, got {len(picked)}"
    sat = picked[0]
    assert sat.normalization_modes == {k: "min_max" for k in ACTION_KEYS}, sat.normalization_modes
    if mode != "min_max":
        # rebuild the shipped Normalizer objects in a different shipped mode
        sat.normalization_modes = {k: mode for k in ACTION_KEYS}
        sat.set_metadata(meta)
    return sat, meta


def roundtrip(gt, dtype, clip=False, mode="min_max"):
    """gt (N,K,8) numpy -> shipped apply -> optional clamp -> shipped unapply -> (N,K,8).

    Builds a FRESH transform each call: StateActionTransform memoises the input dtype on
    first apply and asserts consistency afterwards, so one instance cannot serve fp32 and
    fp64. (Reusing one would have silently coupled the two measurements' provenance too.)
    """
    sat, _ = build_action_transform(mode)
    out = np.empty_like(gt, dtype=np.float64)
    for key in ACTION_KEYS:
        x = torch.as_tensor(gt[..., SLICES[key]].astype(dtype))
        d = {key: x}
        d = sat.apply(d)
        if clip:
            d[key] = torch.clamp(d[key], -1.0, 1.0)
        d = sat.unapply(d)
        out[..., SLICES[key]] = d[key].to(torch.float64).numpy()
    return out


def grep_census():
    """Count hits for the two clip switches n17 found on their leg, plus a positive control
    on the same command shape so a zero is evidence of absence rather than of a typo."""
    import subprocess
    def n(pat):
        r = subprocess.run(["grep", "-rn", "--include=*.py", pat, "."],
                           cwd=str(REPO), capture_output=True, text=True)
        return [l for l in r.stdout.splitlines() if "/.venv/" not in l]
    ctrl = n("min_max")  # must hit: it is the mode this leg actually uses
    return dict(
        clip_outliers_hits=len(n("clip_outliers")),
        use_percentiles_hits=len(n("use_percentiles")),
        positive_control_pattern="min_max",
        positive_control_hits=len(ctrl),
        scope="*.py under the N1.5 repo, .venv excluded",
        blind_spot="non-.py config/serialized flags would not be seen by this grep; "
                   "the ckpt-side check is the degeneracy/oob census above",
    )


def score(recon, gt, K):
    err = np.abs(recon[:, :K, :] - gt[:, :K, :])
    return dict(
        total=float(err.mean()),
        arm=float(err[..., :7].mean()),
        grip=float(err[..., 7].mean()),
        max_abs=float(err.max()),
        per_k=[float(err[:, k, :].mean()) for k in range(K)],
        per_joint=[float(err[..., j].mean()) for j in range(8)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH / "logs" / "n15_floor_measured.json"))
    a = ap.parse_args()

    got = md5(BENCH / "data" / "val_anchors.npz")
    if got != VAL_ANCHORS_MD5:
        sys.exit(f"[FATAL] val_anchors.npz md5 {got} != board pin {VAL_ANCHORS_MD5}")

    A = np.load(BENCH / "data" / "val_anchors.npz")
    gt = A["gt"].astype(np.float32)
    N, K, D = gt.shape
    assert (N, K, D) == (1397, 8, 8), (N, K, D)

    sat, meta = build_action_transform("min_max")

    # --- degeneracy census: min==max columns are the ONE lossy branch inside min_max
    # (forward sends them to 0, inverse returns `min`) -- a silent per-column floor.
    degen = {}
    for key in ACTION_KEYS:
        st = sat.normalization_statistics[key]
        mn = np.asarray(st["min"], dtype=np.float64)
        mx = np.asarray(st["max"], dtype=np.float64)
        degen[key] = dict(
            n_dims=int(mn.size),
            n_degenerate=int((mn == mx).sum()),
            degenerate_dims=[int(i) for i in np.where(mn == mx)[0]],
            min=[float(v) for v in mn], max=[float(v) for v in mx],
        )

    # --- out-of-bounds census: how much of val lies outside the train min/max box.
    # This is the population a clip WOULD have damaged; it is what makes the controls
    # non-vacuous. (n17's leg: 0.63% of values crossed q99-as-bounds.)
    oob = {}
    for key in ACTION_KEYS:
        st = sat.normalization_statistics[key]
        mn = np.asarray(st["min"], dtype=np.float64)
        mx = np.asarray(st["max"], dtype=np.float64)
        v = gt[..., SLICES[key]].astype(np.float64)
        below, above = v < mn, v > mx
        oob[key] = dict(
            frac_below=float(below.mean()), frac_above=float(above.mean()),
            frac_outside=float((below | above).mean()),
            max_excess=float(max((v - mx).max(), (mn - v).max(), 0.0)),
        )

    res = {"val_anchors_md5": got, "ckpt": str(CKPT), "N": int(N), "K": int(K),
           "normalization_modes": sat.normalization_modes,
           "degeneracy": degen, "val_out_of_train_bounds": oob}

    # --- REAL floor, float32 (score.py's dtype) and float64 (separates roundoff from clipping)
    g64 = gt.astype(np.float64)
    res["floor_fp32"] = score(roundtrip(gt, np.float32), g64, K)
    res["floor_fp64"] = score(roundtrip(gt, np.float64), g64, K)

    # --- controls
    res["ctrl_clip_fp64"] = score(roundtrip(gt, np.float64, clip=True), g64, K)
    res["ctrl_q99_fp64"] = score(roundtrip(gt, np.float64, mode="q99"), g64, K)
    res["ctrl_noop_fp64"] = score(roundtrip(gt, np.float64), g64, K)

    # --- is the clip that n17 found on their leg even a switch on this one? MEASURED by
    # grep, with a positive control on the SAME invocation -- a bare negative grep is the
    # empty-filtered-query failure mode (0 hits reads identically to "I spelled it wrong").
    res["clip_switch_grep"] = grep_census()

    # --- verdicts. NB every threshold below is a tolerance, not `== 0.0`: the real floor is
    # roundoff-limited, not algebraically zero, so exact equality here would be a guard that
    # can only ever FAIL (the mirror of the `x.std()==0` guard that can only ever PASS).
    real = res["floor_fp64"]["total"]
    per_k = res["floor_fp64"]["per_k"]
    v = {}
    v["floor_fp64_at_roundoff"] = bool(res["floor_fp64"]["max_abs"] < 1e-9)
    v["floor_fp32_negligible_vs_board"] = bool(res["floor_fp32"]["total"] < 1e-4)  # board MAE 2.047
    v["floor_K_variation_at_roundoff"] = bool(max(per_k) - min(per_k) < 1e-12)
    v["ctrl_clip_fires"] = bool(res["ctrl_clip_fp64"]["total"] > max(real, 1e-12) * 1e3)
    v["ctrl_q99_fires"] = bool(res["ctrl_q99_fp64"]["total"] > max(real, 1e-12) * 1e3)
    v["ctrl_noop_transparent"] = bool(res["ctrl_noop_fp64"]["total"] == real)  # same code+dtype: exact is right here
    v["grep_has_power"] = bool(res["clip_switch_grep"]["positive_control_hits"] > 0)
    v["no_clip_switch_in_repo"] = bool(
        res["clip_switch_grep"]["clip_outliers_hits"] == 0
        and res["clip_switch_grep"]["use_percentiles_hits"] == 0)
    res["verdicts"] = v
    # The number that actually carries the claim: what a clip WOULD have cost. "floor=0" is
    # only interesting because this is non-trivially larger than it.
    res["counterfactual_clip_cost_total"] = res["ctrl_clip_fp64"]["total"]
    res["PASS"] = bool(v["floor_fp64_at_roundoff"] and v["ctrl_clip_fires"]
                       and v["ctrl_q99_fires"] and v["ctrl_noop_transparent"]
                       and v["grep_has_power"] and v["no_clip_switch_in_repo"])

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"val_anchors md5 OK  N={N} K={K}  modes={sat.normalization_modes}")
    for key in ACTION_KEYS:
        print(f"  {key}: degenerate dims {degen[key]['degenerate_dims']}  "
              f"val outside train box {oob[key]['frac_outside']*100:.3f}% "
              f"(max excess {oob[key]['max_excess']:.4f})")
    for k in ["floor_fp32", "floor_fp64", "ctrl_clip_fp64", "ctrl_q99_fp64", "ctrl_noop_fp64"]:
        r = res[k]
        print(f"  {k:16s} total {r['total']:.10g}  arm {r['arm']:.10g}  "
              f"grip {r['grip']:.10g}  max|e| {r['max_abs']:.10g}")
    print("  per_k(fp64):", " ".join(f"{x:.3g}" for x in res["floor_fp64"]["per_k"]))
    for k, val in v.items():
        print(f"  {k:34s} {val}")
    print("PASS" if res["PASS"] else "FAIL")
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
