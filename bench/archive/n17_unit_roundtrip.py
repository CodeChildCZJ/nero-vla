#!/usr/bin/env python3
"""Verify the RELATIVE->ABSOLUTE unit contract on a *trained* checkpoint, CPU only.

The bench contract requires predictions in the dataset's own action units. N1.7's
official new-embodiment recipe makes the arm RELATIVE, and the add-back happens
inside StateActionProcessor.unapply_action -- but only when `use_relative_action`
is set on the processor, a flag that is serialized into the checkpoint and whose
class default is False. A mismatch returns raw deltas with no exception: the MAE
just silently becomes garbage.

This checks that end of the pipeline without loading the 3B model (so it needs no
GPU and no flash-attn): take real absolute actions from the val split, run them
through the checkpoint's own processor forward (absolute -> relative -> normalize)
and back, and see what comes out.

What the numbers mean:
  round-trip error  the error floor every prediction pays before the model is
                    even involved -- normalization is lossy because a val action
                    outside the train split's normalization bounds cannot be
                    represented at all. NOTE the clip that causes this lives in
                    the FORWARD direction (`apply_action`, state_action_processor
                    .py:414) and is gated on the checkpoint's `clip_outliers`
                    flag; `unapply_action` contains no clip. An earlier version
                    of this docstring said "unnormalize_values_minmax clips
                    unconditionally", which had the function, the direction and
                    the conditionality all wrong. The floor's KIND is therefore
                    measured, not assumed -- see n17_contracts.
  gate margin       |arm chunk[0] - state| vs |arm chunk[0]|, i.e. how strongly
                    the predict-side sanity assert can tell "absolute" from
                    "raw deltas". Must be compared as a ratio, never against an
                    absolute threshold -- true per-frame arm motion is ~1 deg.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from transformers import AutoProcessor

BENCH = pathlib.Path(__file__).resolve().parents[1]

import gr00t.model  # noqa: E402,F401  (registers Gr00tN1d7Processor with AutoProcessor)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

from n17_contracts import load_val_anchors, normalization_contract  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="a finetuned checkpoint dir")
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_n17_val"))
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    # 0 = every anchor, and it is the DEFAULT on purpose. This used to default to a
    # 200-anchor evenly-spaced subsample, which is a trap for a floor number: the
    # gripper floor is driven by the ~2.7% of frames that exceed q99, so subsampling
    # a rare-event statistic is badly noisy -- 200 anchors reports grip 0.1746 /
    # total 0.02347 against the true 0.1326 / 0.01757, a 33% overstatement with no
    # warning. A caller who omits the flag must get the leaderboard number, not a
    # cheap approximation of it.
    ap.add_argument("--n-anchors", type=int, default=0,
                    help="0 (default) = all anchors = the leaderboard population")
    ap.add_argument("--score-k", type=int, default=8,
                    help="score.py only compares the first K steps of the chunk; the floor "
                         "is reported for that window as well as for the full action horizon, "
                         "because the RELATIVE arm's clip bounds widen with the horizon index "
                         "(so the two windows give genuinely different floors)")
    ap.add_argument("--out", default=str(BENCH / "logs/n17_unit_roundtrip.json"))
    args = ap.parse_args()

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    processor = AutoProcessor.from_pretrained(args.model_path)
    sap = processor.state_action_processor
    modality = processor.get_modality_configs()[tag.value]
    acfgs = modality["action"].action_configs
    keys = modality["action"].modality_keys

    print("checkpoint:", args.model_path)
    print("convention:", ", ".join(f"{k}={c.rep.value}/{c.type.value}" for k, c in zip(keys, acfgs)))
    print(f"use_relative_action={sap.use_relative_action}  use_percentiles={sap.use_percentiles}")
    # the floor's LABEL depends on three flags, not on this script's prose
    contract = normalization_contract(modality, sap)
    print("normalization contract:", contract)
    if any(c.rep.value == "relative" for c in acfgs):
        assert sap.use_relative_action, (
            "checkpoint declares RELATIVE actions but use_relative_action=False -> "
            "unapply_action would hand back raw deltas"
        )

    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}
    A = load_val_anchors()
    episodes, frames = A["episodes"], A["frames"]
    n_sel = args.n_anchors or len(episodes)
    sel = (np.arange(len(episodes)) if n_sel >= len(episodes)
           else np.linspace(0, len(episodes) - 1, n_sel).astype(int))
    if len(sel) < len(episodes):
        print(f"WARNING: {len(sel)}/{len(episodes)} anchors -- a SUBSET floor, not the "
              f"leaderboard number (the gripper floor is rare-event driven)")

    per_key_err = {k: [] for k in keys}
    d_state, d_zero = [], []
    visit_eps = []          # episode of each visited anchor, in per_key_err order
    for ep in sorted({int(episodes[i]) for i in sel}):
        traj = loader[pos_of_ep[ep]]
        for i in sel:
            if int(episodes[i]) != ep:
                continue
            step = extract_step_data(traj, int(frames[i]), modality, tag, allow_padding=True)
            abs_action = {k: np.asarray(step.actions[k], np.float32) for k in keys}
            state = {k: np.asarray(v, np.float32) for k, v in step.states.items()}

            normalized = sap.apply_action(abs_action, tag.value, state=state)
            recovered = sap.unapply_action(normalized, tag.value, state=state)

            for k in keys:
                per_key_err[k].append(np.abs(recovered[k] - abs_action[k]))
            # the predict-side gate, measured on ground truth
            arm0 = recovered["single_arm"][0]
            d_state.append(np.abs(arm0 - state["single_arm"][-1]).mean())
            d_zero.append(np.abs(arm0).mean())
            visit_eps.append(ep)
        del traj

    print(f"\nround-trip over {len(d_state)} anchors (absolute -> normalized -> absolute):")
    summary = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "n_anchors": len(d_state),
        "use_relative_action": bool(sap.use_relative_action),
        "use_percentiles": bool(sap.use_percentiles),
        **contract,
        "convention": {k: f"{c.rep.value}/{c.type.value}" for k, c in zip(keys, acfgs)},
        "per_key": {},
    }
    K = args.score_k
    summary["score_k"] = K
    for win, cut in (("full", None), (f"k{K}", K)):
        total = []
        bucket = summary["per_key"] if win == "full" else summary.setdefault(f"per_key_{win}", {})
        print(f"  -- window={win}" + ("" if cut is None else f" (first {cut} steps, score.py's)"))
        for k in keys:
            stack = np.stack(per_key_err[k])[:, :cut] if cut else np.stack(per_key_err[k])
            e = stack.reshape(-1)
            total.append(e)
            pj = stack.mean(axis=(0, 1))
            bucket[k] = {"mae": float(e.mean()), "max": float(e.max()),
                         "per_joint": [float(v) for v in pj]}
            print(f"     {k:10s} MAE {e.mean():.4f}  max {e.max():.4f}   per-joint "
                  + " ".join(f"{v:.4f}" for v in pj))
        key = "floor_mae" if win == "full" else f"floor_mae_{win}"
        summary[key] = float(np.concatenate(total).mean())
        print(f"     {'TOTAL':10s} MAE {summary[key]:.4f}"
              f"   = the {summary['floor_kind']} floor "
              f"(bounds {summary['bounds_source']}), before the model contributes")
    print(f"\n  score.py-comparable floor (first {K} steps) = {summary[f'floor_mae_k{K}']:.5f}"
          f"   [full-horizon {summary['floor_mae']:.5f} is NOT the leaderboard number]")

    # --- board rule (d.2): the floor under BOTH weightings ----------------------
    # gr00t-n15 measured that per-frame vs per-episode weighting moves a model and
    # the do-nothing line in OPPOSITE directions, so it is a real term. The floor is
    # a rare-event statistic (2.7% of frames supply 100% of the gripper error), which
    # is exactly the shape most sensitive to how episodes are weighted -- so report
    # both rather than asserting the choice does not matter.
    veps = np.asarray(visit_eps)
    ep_ids = sorted(set(veps.tolist()))
    weighting = {}
    per_anchor_all = []
    for k in keys:
        stack = np.stack(per_key_err[k])[:, :K]                 # (A, K, dim)
        pa = stack.reshape(len(stack), -1).mean(axis=1)          # per-anchor MAE
        per_anchor_all.append(stack.reshape(len(stack), -1))
        fw = float(pa.mean())
        ew = float(np.mean([pa[veps == e].mean() for e in ep_ids]))
        weighting[k] = {"frame_weighted": fw, "episode_equal": ew, "delta": ew - fw,
                        "pct": 100.0 * (ew - fw) / fw if fw else float("nan")}
    pa_tot = np.concatenate(per_anchor_all, axis=1).mean(axis=1)
    fw = float(pa_tot.mean())
    ew = float(np.mean([pa_tot[veps == e].mean() for e in ep_ids]))
    weighting["TOTAL"] = {"frame_weighted": fw, "episode_equal": ew, "delta": ew - fw,
                          "pct": 100.0 * (ew - fw) / fw if fw else float("nan")}
    summary[f"floor_weighting_k{K}"] = weighting
    summary["n_episodes"] = len(ep_ids)
    print(f"\n  floor under both weightings (k{K}, {len(ep_ids)} episodes):")
    for k, v in weighting.items():
        print(f"     {k:10s} frame-weighted {v['frame_weighted']:.5f} -> episode-equal "
              f"{v['episode_equal']:.5f} ({v['pct']:+.2f}%)")

    ds, dz = float(np.mean(d_state)), float(np.mean(d_zero))
    summary.update(gate_dist_to_state=ds, gate_dist_to_zero=dz, gate_ratio=dz / max(ds, 1e-9))
    print(f"\ngate margin: mean|arm chunk[0] - state|={ds:.3f}  mean|arm chunk[0]|={dz:.3f}"
          f"  ratio={dz / max(ds, 1e-9):.1f}x")
    assert ds < dz, "gate cannot discriminate -- the add-back did not happen"
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nUNIT CONTRACT OK -- wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
