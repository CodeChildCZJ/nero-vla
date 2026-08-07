#!/usr/bin/env python3
"""Does the per-anchor reseed ACTUALLY change SmolVLA's sampled noise?

Why this exists BEFORE SmolVLA trains: the variance footnote reports seed-to-seed
std, and a std of exactly 0.0 has TWO possible causes that look identical in the
output JSON -- a genuinely deterministic decoder (true for ACT) or a broken seeding
path that hands every draw the same noise. gr00t-n17 flagged the general form: a
dropped uncertainty term is indistinguishable from a legitimately-zero one. So the
mechanism has to be established while a zero would still be surprising, not after.

What the naive version gets wrong: `lerobot_predict.py --random --seed S` reseeds
at line 87, BEFORE `make_policy`, so two seeds give different random WEIGHTS as
well as different noise. Its predictions differ under both hypotheses => zero power.
This probe builds the policy ONCE and varies only the per-anchor seed, which is
exactly what `lerobot_sampling_variance.py` does across draws (same ckpt, `--seed s`).

Design: SmolVLA is the test, ACT is the DETERMINISM CONTROL run through the same
code. If both came out "identical" the probe would be broken rather than informative,
so the pair is what makes either reading trustworthy.

  smolvla  expect: seed0 != seed1  (noise varies)  AND  seed0 == seed0-repeat  (reproducible)
  act      expect: seed0 == seed1  (no noise consumed at all)

CPU-only, one anchor, no GPU, no checkpoint needed.
"""

import os
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
ROOT = os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench"))
REPO = "local/pick_pink_sponge_b2_bench"
PROMPT = "pick the pink sponge and place it in the blue bucket"
WEIGHT_SEED = 12345  # fixed: weights must be IDENTICAL across the seeds under test


def run(backbone: str, item, ds_meta, seeds) -> dict:
    from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

    torch.manual_seed(WEIGHT_SEED)  # weights drawn once, from a seed we then never reuse
    cfg = make_policy_config(backbone, device="cpu")
    policy = make_policy(cfg, ds_meta=ds_meta)
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds_meta.stats)
    policy.eval()

    outs = []
    for s in seeds:
        policy.reset()
        torch.manual_seed(s * 1_000_003)  # same formula as lerobot_predict.py, anchor 0
        with torch.no_grad():
            outs.append(post(policy.predict_action_chunk(pre(item))).squeeze(0).float().numpy())
        print(f"  [{backbone}] seed={s} first row {np.round(outs[-1][0, :3], 5)}", flush=True)
    return {"outs": outs}


def main() -> None:
    # argparse for TWO reasons, only one of which is the obvious one.
    # (1) openpi's --out board rule: a pre-flight must be redirectable off the production path.
    # (2) Without argparse, `--help` is just an ignored sys.argv entry and this script RUNS.
    #     That is not hypothetical -- a `for s in scripts/*.py; do $s --help; done` coverage
    #     sweep of mine launched a second copy of a long CPU probe onto the same fixed output
    #     paths as the live one, and orphaned it when the sweep timed out. The verification
    #     step was the side effect. argparse makes --help inert.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="output json (default logs/stochasticity_probe.json)")
    args = ap.parse_args()
    sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                    str(BENCH / "archive")]   # was one flat scripts/ dir
    from lerobot_predict import build_anchor_index

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    A = np.load(BENCH / "data" / "val_anchors.npz")
    ep, fr = A["episodes"][:1], A["frames"][:1]
    ds = LeRobotDataset(REPO, root=ROOT, episodes=[int(ep[0])])
    item = ds[build_anchor_index(ds, ep, fr)[0]]
    item["task"] = item.get("task", PROMPT)
    print(f"[anchor] episode {int(ep[0])} frame {int(fr[0])}")

    res = {}
    for bb, expect_varies in (("act", False), ("smolvla", True)):
        print(f"\n=== {bb} (weights fixed at seed {WEIGHT_SEED}; only the per-anchor seed varies) ===")
        outs = run(bb, item, ds.meta, [0, 1, 0])["outs"]
        d01 = float(np.abs(outs[0] - outs[1]).max())
        d00 = float(np.abs(outs[0] - outs[2]).max())
        varies, reproducible = d01 > 0.0, d00 == 0.0
        ok = (varies == expect_varies) and reproducible
        res[bb] = {"maxdiff_seed0_vs_seed1": d01, "maxdiff_seed0_vs_seed0repeat": d00,
                   "varies_with_seed": varies, "reproducible_at_same_seed": reproducible,
                   "expected_to_vary": expect_varies, "PASS": ok}
        print(f"  seed0 vs seed1        maxdiff {d01:.6e}   varies={varies} (expected {expect_varies})")
        print(f"  seed0 vs seed0-repeat maxdiff {d00:.6e}   reproducible={reproducible}")
        print(f"  => {'PASS' if ok else 'FAIL'}")

    res["verdict"] = (
        "SmolVLA's noise IS driven by the per-anchor torch.manual_seed, so a std of 0.0 in its "
        "variance footnote would be a BUG, not determinism -- and ACT through the identical code "
        "path stays bit-identical, which is what gives the SmolVLA reading its power."
        if res["smolvla"]["PASS"] and res["act"]["PASS"] else
        "MECHANISM NOT ESTABLISHED -- do not trust a std=0.0 footnote until this passes")
    p = pathlib.Path(args.out) if args.out else BENCH / "logs" / "stochasticity_probe.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n{res['verdict']}\n[write] {p}")


if __name__ == "__main__":
    main()
