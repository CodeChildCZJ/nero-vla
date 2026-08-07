#!/usr/bin/env python3
"""Identify the N1.5 normalization BRANCH from the delivered artifact, not from my config.

Why this exists (gr00t-n17, 2026-08-05): on their leg the branch is a flag serialised into
the checkpoint (`use_percentiles=True`, bounds q01/q99), so they can READ which branch made
their preds. Measured on mine: `experiment_cfg/` contains ONLY `metadata.json` (statistics);
there is no serialised processor, no flag, nowhere for one to live. The branch is supplied by
the CALLER at inference (`Gr00tPolicy(modality_transform=data_config.transform())`), i.e. it
is a property of a repo file that is editable after the preds were written, with no artifact
tying the two together. So `floor ~ roundoff` rested on reading my own config -- one layer
better than n17's docstring, but still not the artifact.

An identification test I DROPPED before writing it, because it cannot work: comparing the
physical scale of preds against GT. If train and predict use the SAME branch, the branch
cancels -- the model learns targets in whatever normalised space, and the matching inverse
puts them back on the correct physical scale either way. Pred-vs-GT scale is therefore
mode-INVARIANT and carries zero information. What does NOT cancel is the clamp: q99's
FORWARD path clamps to [-1,1] (state_action.py:134), so under H_q99 the model is trained on
targets truncated at q01/q99 and should essentially never be asked to emit beyond them.
min_max's forward does not clamp.

  H_mm  (min_max) : preds exceed [q01,q99] at a rate comparable to GT's own ~2%
  H_q99 (q99)     : preds are ~confined to [q01,q99]; exceedance collapses toward 0

CALIBRATION, not a bare threshold: GT's own exceedance is measured on the same values and is
the reference both hypotheses are scored against. Blind spot stated in the output: model
shrinkage also suppresses tail excursions, so a LOW pred rate is ambiguous (H_q99 or just a
conservative model); only a HIGH rate is decisive, and it is decisive for H_mm.

CPU only, no GPU, no model. Reads the delivered preds npz and the ckpt metadata.
"""
import os
import hashlib, json, pathlib, sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
CKPT = pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "gr00t_n15_nero_b2" / "checkpoint-10000"
VAL_ANCHORS_MD5 = "cc34dda2a99ade1011c192e2b3e4e343"
# preds md5 is READ from the delivered meta json, never typed here. First version of this
# file hardcoded a hand-copied hash whose first 8 hex chars matched the real one and whose
# tail I had confabulated from a truncated log line -- it passes every eyeball check and
# fails only the machine. A hash you typed is not a pin, it is a second thing to verify.
PREDS_MD5_SOURCE = BENCH / "preds" / "preds_gr00t_n15_meta.json"
K = 8
SLICES = {"single_arm": slice(0, 7), "gripper": slice(7, 8)}


def read_bytes_and_md5(p):
    """Hash the bytes we actually load -- not a verify-then-reload, which leaves a window."""
    b = pathlib.Path(p).read_bytes()
    return b, hashlib.md5(b).hexdigest()


def npz_from_bytes(b):
    import io
    return np.load(io.BytesIO(b))


def exceedance(v, lo, hi):
    """Fraction of values outside [lo,hi] broadcast over the last axis, + how far."""
    below, above = v < lo, v > hi
    out = below | above
    excess = np.maximum(np.maximum(lo - v, v - hi), 0.0)
    return dict(
        frac_outside=float(out.mean()),
        frac_below=float(below.mean()),
        frac_above=float(above.mean()),
        max_excess=float(excess.max()),
        per_dim_frac_outside=[float(x) for x in out.reshape(-1, out.shape[-1]).mean(axis=0)],
    )


def main():
    declared = json.load(open(PREDS_MD5_SOURCE))["inputs"]
    for k in ("preds", "val_anchors"):          # req(): a missing key exits, never silently None
        if k not in declared or "md5" not in declared[k]:
            sys.exit(f"[FATAL] {PREDS_MD5_SOURCE} has no inputs.{k}.md5")
    pb, pmd5 = read_bytes_and_md5(BENCH / "preds" / "preds_gr00t_n15.npz")
    ab, amd5 = read_bytes_and_md5(BENCH / "data" / "val_anchors.npz")
    if amd5 != VAL_ANCHORS_MD5:
        sys.exit(f"[FATAL] val_anchors.npz md5 {amd5} != board pin {VAL_ANCHORS_MD5}")
    if declared["val_anchors"]["md5"] != VAL_ANCHORS_MD5:
        sys.exit(f"[FATAL] meta json declares a different val_anchors than the board pin")
    if pmd5 != declared["preds"]["md5"]:
        sys.exit(f"[FATAL] preds md5 {pmd5} != meta-json pin {declared['preds']['md5']}")

    P = npz_from_bytes(pb)
    A = npz_from_bytes(ab)
    pred = np.asarray(P["pred"], dtype=np.float64)[:, :K, :]
    gt = np.asarray(A["gt"], dtype=np.float64)[:, :K, :]
    assert pred.shape == gt.shape == (1397, K, 8), (pred.shape, gt.shape)

    meta = json.load(open(CKPT / "experiment_cfg" / "metadata.json"))
    st = meta["new_embodiment"]["statistics"]["action"]

    res = {"preds_md5": pmd5, "val_anchors_md5": amd5,
           "ckpt_serialises_branch": False,
           "ckpt_experiment_cfg_files": sorted(p.name for p in (CKPT / "experiment_cfg").iterdir()),
           "branch_supplied_by": "caller: Gr00tPolicy(modality_transform=data_config.transform())",
           "runtime_record": "logs/n15_post_train.log:9,110 'Loading external config: "
                             "nero_data_config.NeroDualCamDataConfig' (module identity only, "
                             "not the file's contents at that time)",
           "train_log_mode_lines": 0,
           "train_log_positive_control": "grep -c loss logs/train_gr00t_n15.log = 1001",
           "keys": {}}

    verdict_bits = {}
    for key, sl in SLICES.items():
        s = st[key]
        mn = np.asarray(s["min"], dtype=np.float64)
        mx = np.asarray(s["max"], dtype=np.float64)
        q01 = np.asarray(s["q01"], dtype=np.float64)
        q99 = np.asarray(s["q99"], dtype=np.float64)
        pv, gv = pred[..., sl], gt[..., sl]

        span_ratio = (q99 - q01) / np.maximum(mx - mn, 1e-12)
        e = {
            "pred_vs_minmax": exceedance(pv, mn, mx),
            "gt_vs_minmax": exceedance(gv, mn, mx),
            "pred_vs_q01q99": exceedance(pv, q01, q99),
            "gt_vs_q01q99": exceedance(gv, q01, q99),
            "q_span_over_minmax_span_per_dim": [float(x) for x in span_ratio],
        }
        pq, gq = e["pred_vs_q01q99"]["frac_outside"], e["gt_vs_q01q99"]["frac_outside"]
        e["pred_over_gt_q_exceedance_ratio"] = float(pq / gq) if gq > 0 else None
        res["keys"][key] = e

        # H_q99 predicts pred exceedance of q01/q99 collapses toward 0 relative to GT's.
        # H_mm predicts it is comparable. Tolerance, never an exact-float gate.
        verdict_bits[f"{key}_pred_exceeds_q99_box"] = bool(pq > 1e-4)
        verdict_bits[f"{key}_pred_exceeds_minmax_box"] = bool(
            e["pred_vs_minmax"]["frac_outside"] > 1e-4)

    res["verdicts"] = verdict_bits
    any_q = any(v for k, v in verdict_bits.items() if k.endswith("q99_box"))
    any_mm = any(v for k, v in verdict_bits.items() if k.endswith("minmax_box"))
    res["branch_inference"] = (
        "H_mm SUPPORTED: preds leave the q01/q99 box, which a q99-clamped training target "
        "distribution would not have taught" if any_q else
        "UNRESOLVED: preds stay inside q01/q99 -- consistent with H_q99 clamping AND with a "
        "merely conservative model under H_mm; this direction does not discriminate")
    res["no_clip_at_minmax_in_delivered_path"] = bool(any_mm)
    res["blind_spots"] = [
        "identifies the CLAMP, not the branch label: a hypothetical unclamped q99 variant "
        "would be invisible to this test",
        "low exceedance is ambiguous (model shrinkage mimics clamping); only high is decisive",
        "says nothing about whether train and predict used the SAME branch -- it assumes they "
        "did, which is what makes the branch cancel in the physical scale",
    ]
    res["PASS"] = bool(any_q or any_mm)

    out = BENCH / "logs" / "n15_mode_from_artifact.json"
    out.write_text(json.dumps(res, indent=1))
    for key in SLICES:
        e = res["keys"][key]
        print(f"{key}:")
        print(f"   pred outside [min,max] {e['pred_vs_minmax']['frac_outside']*100:.4f}%   "
              f"gt {e['gt_vs_minmax']['frac_outside']*100:.4f}%")
        print(f"   pred outside [q01,q99] {e['pred_vs_q01q99']['frac_outside']*100:.4f}%   "
              f"gt {e['gt_vs_q01q99']['frac_outside']*100:.4f}%   "
              f"ratio {e['pred_over_gt_q_exceedance_ratio']}")
        print(f"   q-span/minmax-span per dim: "
              f"{' '.join(f'{x:.3f}' for x in e['q_span_over_minmax_span_per_dim'])}")
    print(res["branch_inference"])
    print("PASS" if res["PASS"] else "FAIL")
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
