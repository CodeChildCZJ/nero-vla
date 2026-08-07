#!/usr/bin/env python3
"""Decompose N1.7's HARD floor into irreducible-OOD vs self-inflicted-q99, CPU only.

`n17_unit_roundtrip.py` measures the floor the checkpoint ACTUALLY pays
(bounds = q01/q99, because the N1.7 new-embodiment recipe sets
use_percentiles=True).  It cannot say how much of that is unavoidable.  The
counterfactual that answers it is the SAME round-trip with the bounds swapped
to the train split's true min/max -- i.e. what the floor would be if the only
loss were "a val action genuinely outside the train envelope".

  floor(q01/q99)  =  floor(min/max)  +  the price of use_percentiles

The swap is done through the processor's own supported path -- set the flag and
call `_compute_normalization_parameters()`, which is where bounds are chosen
(state_action_processor.py:157) and cached into `norm_params`.  Nothing is
monkeypatched, so the counterfactual runs the shipped arithmetic.

Two controls ride along, both on a REAL checkpoint rather than a synthetic one:

  POSITIVE  the as-is arm goes through the identical loop and must reproduce
            n17_unit_roundtrip's number.  If it does not, this script's copy of
            the loop has drifted from the validated one and every number below
            is void.
  MUST-FAIL `normalization_contract()` is called again AFTER the flip.  Its job
            is to guarantee "floor and preds came from the same normalization
            config", so it MUST raise here.  A gate that only ever sees inputs
            it approves of has never been shown to be able to refuse.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from transformers import AutoProcessor

BENCH = pathlib.Path(__file__).resolve().parents[1]

import gr00t.model  # noqa: E402,F401
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

from n17_contracts import load_val_anchors, normalization_contract  # noqa: E402


def roundtrip(sap, loader, pos_of_ep, modality, tag, keys, episodes, frames, K):
    """Identical to n17_unit_roundtrip's loop; returns per-key MAE over the first K steps."""
    per_key_err = {k: [] for k in keys}
    for ep in sorted({int(e) for e in episodes}):
        traj = loader[pos_of_ep[ep]]
        for i in range(len(episodes)):
            if int(episodes[i]) != ep:
                continue
            step = extract_step_data(traj, int(frames[i]), modality, tag, allow_padding=True)
            abs_action = {k: np.asarray(step.actions[k], np.float32) for k in keys}
            state = {k: np.asarray(v, np.float32) for k, v in step.states.items()}
            normalized = sap.apply_action(abs_action, tag.value, state=state)
            recovered = sap.unapply_action(normalized, tag.value, state=state)
            for k in keys:
                per_key_err[k].append(np.abs(recovered[k] - abs_action[k])[:K])
        del traj
    out = {}
    for k in keys:
        e = np.stack(per_key_err[k])
        out[k] = {"mae": float(e.mean()), "max": float(e.max()), "n": int(e.shape[0])}
    tot = np.concatenate([np.stack(per_key_err[k]).reshape(len(episodes), -1) for k in keys], axis=1)
    out["_total_mae"] = float(np.abs(tot).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_n17_val"))
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--score-k", type=int, default=8)
    ap.add_argument("--expect-grip-asis", type=float, default=0.13255,
                    help="the positive control: as-is gripper floor from n17_unit_roundtrip")
    ap.add_argument("--out", default=str(BENCH / "logs/n17_floor_counterfactual.json"))
    args = ap.parse_args()

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    processor = AutoProcessor.from_pretrained(args.model_path)
    sap = processor.state_action_processor
    modality = processor.get_modality_configs()[tag.value]
    keys = modality["action"].modality_keys

    contract_asis = normalization_contract(modality, sap)
    print("as-is contract:", contract_asis)
    assert contract_asis["use_percentiles"] is True, "this ckpt is not the q01/q99 config"

    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}
    A = load_val_anchors()
    episodes, frames = A["episodes"], A["frames"]
    K = args.score_k
    print(f"anchors: {len(episodes)}  (full leaderboard population)  K={K}")

    asis = roundtrip(sap, loader, pos_of_ep, modality, tag, keys, episodes, frames, K)
    print(f"\n[as-is  q01/q99] total {asis['_total_mae']:.5f}  "
          + "  ".join(f"{k} {asis[k]['mae']:.5f}" for k in keys))

    # POSITIVE CONTROL: this loop must agree with the validated script's loop.
    got = asis["gripper"]["mae"]
    if abs(got - args.expect_grip_asis) > 1e-4:
        sys.exit(f"POSITIVE CONTROL FAILED: gripper {got:.5f} != {args.expect_grip_asis:.5f} "
                 f"-- this script's loop has drifted from n17_unit_roundtrip's; numbers void")
    print(f"positive control OK: gripper reproduces {args.expect_grip_asis} to <1e-4")

    # ---- the counterfactual: bounds = train min/max instead of q01/q99 --------
    sap.use_percentiles = False
    sap._compute_normalization_parameters()

    # MUST-FAIL CONTROL: the contract gate has to notice.
    try:
        normalization_contract(modality, sap)
    # BaseException, not Exception: n17_contracts refuses via sys.exit(), and SystemExit
    # does NOT inherit from Exception. The first version of this control caught Exception
    # and so the gate's refusal -- the very event being tested -- killed the harness
    # instead of being recorded as a pass. A control that the expected outcome destroys
    # is not a control.
    except BaseException as e:                               # noqa: BLE001
        gate_refused, gate_msg = True, str(e)[:200]
    else:
        gate_refused, gate_msg = False, ""
    print(f"must-fail control: gate refused the flipped config = {gate_refused}")
    if not gate_refused:
        sys.exit("MUST-FAIL CONTROL FAILED: normalization_contract accepted a config it "
                 "was written to reject -- the gate cannot protect the floor/preds pairing")

    cf = roundtrip(sap, loader, pos_of_ep, modality, tag, keys, episodes, frames, K)
    print(f"[cf     min/max] total {cf['_total_mae']:.5f}  "
          + "  ".join(f"{k} {cf[k]['mae']:.5f}" for k in keys))

    print("\ndecomposition (K=%d, %d anchors):" % (K, len(episodes)))
    for k in list(keys) + ["_total_mae"]:
        a = asis[k]["mae"] if k in keys else asis[k]
        c = cf[k]["mae"] if k in keys else cf[k]
        share = (a - c) / a * 100 if a > 0 else float("nan")
        print(f"  {k:12s} paid {a:.5f} = irreducible {c:.5f} + q99-truncation "
              f"{a - c:.5f}  ({share:.1f}% self-inflicted)")

    summary = {
        "model_path": args.model_path,
        "n_anchors": int(len(episodes)),
        "score_k": K,
        "contract_asis": contract_asis,
        "gate_refused_flipped_config": gate_refused,
        "gate_message": gate_msg,
        "floor_asis_q01q99": asis,
        "floor_cf_minmax": cf,
    }
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
