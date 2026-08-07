#!/usr/bin/env python
"""Does the DELIVERY predict path consume RNG at all?  (CPU, no GPU, no writes to preds/)

Why this exists
---------------
`variance_act.json` reports sampling std = 0.0.  Two very different worlds produce
that number:

  (a) the decoder is deterministic -- there is no noise to vary, so 5 draws MUST agree;
  (b) the harness reused one seed -- 5 "draws" were literally the same computation.

An audit cannot tell them apart from the std alone, which is why the board asks for a
list of DISTINCT per-draw seeds.  But openvla-oft raised the sharper point: on a stack
with no RNG consumer, a list of distinct seeds is a field that EXISTS AND PROVES
NOTHING -- the seeds are real, they were really passed, and they cannot have changed
any output.  Recording them would satisfy the audit's letter while leaving its actual
question ("were these five independent samples?") unanswered.

So measure the thing that separates (a) from (b) directly: does anything between
`torch.manual_seed(...)` and the returned chunk *touch the global RNG*?  Torch's RNG
state is a byte buffer that advances iff a consumer draws from it, so hashing it
before and after the shipped call sequence answers this exactly, with no reliance on
what I believe ACT's architecture does at inference.

The probe carries its own power control: a `torch.randn(1)` must move the hash.  A
probe that reports "no consumption" because it cannot detect consumption at all is
the zero-power control this bench keeps re-learning about.

Run:  CUDA_VISIBLE_DEVICES= .venv/bin/python archive/lerobot_rng_consumption.py --backbone act --ckpt <dir>
"""

import os
import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROOT = os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench"))
DEFAULT_REPO_ID = "local/pick_pink_sponge_b2_bench"
PROMPT = "pick the pink sponge and place it in the blue bucket"


def rng_hash():
    """Fingerprint the GLOBAL CPU RNG state (a byte tensor that advances when drawn from)."""
    return hashlib.md5(torch.get_rng_state().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["smolvla", "act"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--n-anchors", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ---- power control FIRST: if the probe cannot see a known consumer, nothing
    # it says about the model is worth reading.  Run it before the expensive part
    # so a dead probe costs nothing.
    h0 = rng_hash()
    _ = torch.randn(1)
    h1 = rng_hash()
    probe_has_power = (h0 != h1)
    if not probe_has_power:
        raise SystemExit("[FATAL] zero-power probe: torch.randn did not move the RNG hash.")
    # ...and the converse control: a pure-arithmetic op must NOT move it, else the
    # probe would report "consumed" for everything and be equally uninformative.
    h2 = rng_hash()
    _ = torch.ones(4) * 2.0
    probe_no_false_positive = (rng_hash() == h2)
    if not probe_no_false_positive:
        raise SystemExit("[FATAL] probe fires on a non-RNG op; it cannot discriminate.")
    print(f"[power control] randn moves hash: {probe_has_power};  arithmetic does not: {probe_no_false_positive}")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    anchors = np.load(BENCH / "data" / "val_anchors.npz")
    episodes, frames = anchors["episodes"], anchors["frames"]
    val_eps = sorted(set(int(e) for e in episodes))
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=val_eps)

    ep_col = ds.hf_dataset.data.column("episode_index").to_numpy()
    fr_col = ds.hf_dataset.data.column("frame_index").to_numpy()
    key2rel = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep_col, fr_col, strict=True))}

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cpu"
    rename_map = None
    tc = pathlib.Path(args.ckpt) / "train_config.json"
    if tc.exists():
        rename_map = json.loads(tc.read_text()).get("rename_map") or None
    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=rename_map)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=args.ckpt, dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.eval()

    rows = []
    for n in range(args.n_anchors):
        ri = key2rel[(int(episodes[n]), int(frames[n]))]
        item = ds[ri]
        item["task"] = item.get("task", PROMPT)
        policy.reset()
        torch.manual_seed(12345)
        a = rng_hash()
        batch = pre(item)
        b = rng_hash()
        chunk = policy.predict_action_chunk(batch)
        c = rng_hash()
        chunk = post(chunk)
        d = rng_hash()
        rows.append({"anchor": n,
                     "pre_consumed": a != b,
                     "predict_consumed": b != c,
                     "post_consumed": c != d,
                     "any_consumed": a != d})
        print(f"  anchor {n}: pre={a != b}  predict={b != c}  post={c != d}")

    any_consumed = any(r["any_consumed"] for r in rows)

    # Second, independent line of evidence on the same question, from the artifacts
    # rather than from the RNG: if the 5 delivered seed runs are byte-identical, the
    # seed cannot have reached anything that affects output.  This does NOT by itself
    # separate (a) from (b) -- one seed used 5 times looks the same -- which is
    # exactly why it is reported alongside the RNG measurement, not instead of it.
    vd = BENCH / "logs" / "variance"
    seed_files = sorted(vd.glob(f"preds_{args.backbone}_seed*.npz"))
    md5s = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in seed_files}
    pairwise_identical = len(set(md5s.values())) == 1 if md5s else None

    verdict = {
        "backbone": args.backbone,
        "ckpt": args.ckpt,
        "probe_power_control": {"randn_moves_rng_hash": probe_has_power,
                                "arithmetic_does_not": probe_no_false_positive},
        "n_anchors_probed": args.n_anchors,
        "per_anchor": rows,
        "predict_path_consumes_rng": any_consumed,
        "delivered_seed_run_md5": md5s,
        "delivered_seed_runs_byte_identical": pairwise_identical,
        "interpretation": (
            "predict path consumes NO RNG -> sampling std 0.0 is a property of the "
            "decoder, not evidence about the harness. A per_draw_seeds list would be "
            "true but non-load-bearing here: distinct seeds cannot change an output "
            "that never reads them. Report the seeds AND this measurement, or the "
            "audit's question stays unanswered either way."
            if not any_consumed else
            "predict path DOES consume RNG -> distinct per-draw seeds are load-bearing "
            "evidence and sampling std 0.0 would be a real finding about the decoder."
        ),
    }
    p = pathlib.Path(args.out) if args.out else BENCH / "logs" / f"rng_consumption_{args.backbone}.json"
    p.write_text(json.dumps(verdict, indent=2))
    print(f"\n[RESULT] {args.backbone}: predict path consumes RNG = {any_consumed}")
    print(f"[RESULT] delivered seed runs byte-identical = {pairwise_identical}")
    print(f"[write] {p}")


if __name__ == "__main__":
    main()
