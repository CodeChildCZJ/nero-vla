#!/usr/bin/env python
"""Does the DELIVERY predict path consume RNG?  -- v2, two orthogonal covers.

Why a v2
--------
v1 (`lerobot_rng_consumption.py`) hashed `torch.get_rng_state()` and reported
`predict_path_consumes_rng: False` for ACT.  gr00t-n15 pointed out the hole and she is
right: that call fingerprints the GLOBAL CPU Mersenne stream ONLY.  It is blind to

  * explicitly constructed `torch.Generator` objects (a policy that does
    `g = torch.Generator(device); torch.randn(..., generator=g)` never touches it),
  * the CUDA generator -- which is exactly where a flow-matching / diffusion decoder
    draws its noise when the policy runs on GPU,
  * `numpy.random`, and Python's `random`.

So a CPU-only state hash can return False on a stack that samples heavily, i.e. it can
manufacture the very answer v1 reported.  That matters twice over: it weakens the ACT
claim, and it would silently destroy the SmolVLA control (the whole point of running the
same probe on a sampling backbone is to see it say True; a probe that CANNOT say True
there proves nothing when it says False on ACT).

Two covers, each blind where the other sees
-------------------------------------------
COVER A -- state hash over FOUR global streams (torch CPU, torch CUDA all-devices,
numpy, python `random`).  Sees any draw from a global stream, including draws made
deep in C++ that never pass through a Python name.  Blind to constructed Generators.

COVER B -- call counter: wrap the torch sampling entry points and count invocations.
Sees a draw no matter WHICH generator it uses, including an explicitly constructed one.
Blind to sampling that bypasses these Python names (fused kernels, aten calls from C++).

Neither is a superset.  Reported separately, never merged into one boolean, and the
patched-name list is written into the output so the residual blind spot is explicit
instead of implied.

The probe carries its own power controls, per source, and a negative control: a
pure-arithmetic op must move nothing.  A probe that cannot detect a known consumer is
the zero-power control this bench keeps re-learning about.

Run (GPU is the load-bearing configuration -- CUDA noise is invisible on CPU):
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python archive/lerobot_rng_consumption2.py \
      --backbone smolvla --ckpt <dir> --device cuda
"""

import os
import argparse
import hashlib
import json
import pathlib
import random

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROOT = os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench"))
DEFAULT_REPO_ID = "local/pick_pink_sponge_b2_bench"
PROMPT = "pick the pink sponge and place it in the blue bucket"

# COVER B -- the sampling entry points to wrap.  Written out in the verdict so a reader
# can see exactly which surface was watched (and therefore what was NOT watched).
PATCH_TARGETS = [
    (torch, "randn"), (torch, "rand"), (torch, "randint"), (torch, "randperm"),
    (torch, "normal"), (torch, "bernoulli"), (torch, "multinomial"),
    (torch, "randn_like"), (torch, "rand_like"), (torch, "randint_like"),
    (torch.Tensor, "normal_"), (torch.Tensor, "uniform_"), (torch.Tensor, "random_"),
    (torch.Tensor, "bernoulli_"), (torch.Tensor, "exponential_"),
    (torch.nn.functional, "dropout"), (torch.nn.functional, "dropout2d"),
]


class CallCounter:
    """COVER B.  Counts calls to the sampling API regardless of `generator=`."""

    def __init__(self):
        self.counts = {}
        self.dropout_args = set()   # (p, training) actually passed -- see note below
        self._saved = []

    def install(self):
        for obj, name in PATCH_TARGETS:
            orig = getattr(obj, name, None)
            if orig is None:
                continue
            key = f"{getattr(obj, '__name__', obj.__class__.__name__)}.{name}"
            self._saved.append((obj, name, orig))

            def make(fn, k):
                def wrapper(*a, **kw):
                    self.counts[k] = self.counts.get(k, 0) + 1
                    # A COUNT IS NOT A DRAW.  Measured on ACT: 16 F.dropout calls per
                    # predict, and the state hash never moved -- because the calls carry
                    # training=False (and p=0.0 would do it too: verified in isolation,
                    # F.dropout draws RNG only at p>0 AND training=True).  Recording the
                    # arguments is what separates "a stochastic site exists here" from
                    # "a stochastic site fired here"; without it cover B is a
                    # false-positive machine on any net that merely CONTAINS dropout.
                    if "dropout" in k:
                        p = kw.get("p", a[1] if len(a) > 1 else None)
                        tr = kw.get("training", a[2] if len(a) > 2 else None)
                        self.dropout_args.add((p, tr))
                    return fn(*a, **kw)
                return wrapper

            setattr(obj, name, make(orig, key))
        return self

    def restore(self):
        for obj, name, orig in self._saved:
            setattr(obj, name, orig)
        self._saved = []

    def snapshot(self):
        return dict(self.counts)

    @staticmethod
    def delta(before, after):
        return {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0) > 0}


def rng_state(device: str) -> dict:
    """COVER A.  Fingerprint every GLOBAL stream the predict path could draw from."""
    st = {
        "torch_cpu": hashlib.md5(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "numpy": hashlib.md5(repr(np.random.get_state()).encode()).hexdigest(),
        "python_random": hashlib.md5(repr(random.getstate()).encode()).hexdigest(),
    }
    if device == "cuda" and torch.cuda.is_available():
        blob = b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())
        st["torch_cuda"] = hashlib.md5(blob).hexdigest()
    return st


def moved(a: dict, b: dict) -> list:
    """Which streams advanced between two fingerprints."""
    return sorted(k for k in a if a[k] != b.get(k))


def power_controls(device: str) -> dict:
    """Each cover must be shown to FIRE on a known consumer, and to STAY SILENT on a
    non-consumer.  Run before the expensive part so a dead probe costs nothing."""
    res = {}
    checks = [
        ("torch_cpu", lambda: torch.randn(1)),
        ("numpy", lambda: np.random.rand()),
        ("python_random", lambda: random.random()),
    ]
    if device == "cuda" and torch.cuda.is_available():
        checks.append(("torch_cuda", lambda: torch.randn(1, device="cuda")))

    for stream, fn in checks:
        a = rng_state(device)
        fn()
        b = rng_state(device)
        mv = moved(a, b)
        res[f"{stream}_detected"] = stream in mv
        res[f"{stream}_moved_streams"] = mv

    # negative control: a probe that fires on arithmetic would report "consumed" for
    # everything and be exactly as uninformative as one that never fires.
    a = rng_state(device)
    _ = torch.ones(4) * 2.0
    res["arithmetic_moves_nothing"] = moved(a, rng_state(device)) == []

    # COVER B power control: a draw through an EXPLICITLY CONSTRUCTED generator must be
    # counted -- this is the case COVER A cannot see, so it is the one that matters.
    cc = CallCounter().install()
    try:
        before = cc.snapshot()
        g = torch.Generator(device="cpu")
        g.manual_seed(7)
        state_before = rng_state(device)
        torch.randn(2, generator=g)
        state_after = rng_state(device)
        res["counter_sees_explicit_generator"] = CallCounter.delta(before, cc.snapshot()) != {}
        res["state_hash_blind_to_explicit_generator"] = moved(state_before, state_after) == []
    finally:
        cc.restore()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["smolvla", "act"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--n-anchors", type=int, default=3)
    # The counterfactual that turns "cover B fired" into a mechanism.  Cover B counts
    # CALLS, and a call is not a draw: `F.dropout(x, p, training=False)` returns x and
    # consumes nothing, so a counter alone cannot tell an inert site from a live one.
    # Running the identical probe with the policy left in train() answers it: if the
    # same sites then move the state hash, determinism was gated by `policy.eval()`
    # rather than absent from the graph.  NEVER used for a delivered row.
    ap.add_argument("--policy-mode", default="eval", choices=["eval", "train"])
    ap.add_argument("--cover-a-power-on-network", action="store_true",
                    help="after measuring, make this net's own dropout sites live and re-run: "
                         "turns 'cover A saw nothing' into 'cover A saw nothing AND would have "
                         "seen something'. Mutates the policy; never use on a delivery run.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pc = power_controls(args.device)
    dead = [k for k, v in pc.items() if k.endswith("_detected") and not v]
    if dead:
        raise SystemExit(f"[FATAL] zero-power probe: no detection on {dead}. Nothing it says is readable.")
    if not pc["arithmetic_moves_nothing"]:
        raise SystemExit("[FATAL] probe fires on a non-RNG op; it cannot discriminate.")
    if not pc["counter_sees_explicit_generator"]:
        raise SystemExit("[FATAL] COVER B cannot see a draw through an explicit Generator -- that is its only reason to exist.")
    print(f"[power controls] {json.dumps(pc, indent=2)}")

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
    cfg.device = args.device
    rename_map = None
    tc = pathlib.Path(args.ckpt) / "train_config.json"
    if tc.exists():
        rename_map = json.loads(tc.read_text()).get("rename_map") or None
    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=rename_map)
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=args.ckpt, dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.eval() if args.policy_mode == "eval" else policy.train()
    assert policy.training == (args.policy_mode == "train"), "policy mode did not take"

    rows = []
    counter = CallCounter().install()
    try:
        for n in range(args.n_anchors):
            ri = key2rel[(int(episodes[n]), int(frames[n]))]
            item = ds[ri]
            item["task"] = item.get("task", PROMPT)
            policy.reset()
            torch.manual_seed(12345)

            a, ca = rng_state(args.device), counter.snapshot()
            batch = pre(item)
            b, cb = rng_state(args.device), counter.snapshot()
            chunk = policy.predict_action_chunk(batch)
            c, cc_ = rng_state(args.device), counter.snapshot()
            chunk = post(chunk)
            d, cd = rng_state(args.device), counter.snapshot()

            rows.append({
                "anchor": n,
                "state_streams_moved": {
                    "pre": moved(a, b), "predict": moved(b, c), "post": moved(c, d),
                },
                "sampling_calls": {
                    "pre": CallCounter.delta(ca, cb),
                    "predict": CallCounter.delta(cb, cc_),
                    "post": CallCounter.delta(cc_, cd),
                },
            })
            print(f"  anchor {n}: predict state-moved={moved(b, c)}  predict calls={CallCounter.delta(cb, cc_)}")
    finally:
        counter.restore()

    cover_a = any(r["state_streams_moved"]["predict"] for r in rows)
    cover_b = any(r["sampling_calls"]["predict"] for r in rows)

    # ---- COVER A power control ON THIS NETWORK (not on a synthetic torch.randn).
    # `torch.randn moves the hash` only shows the probe works on torch; it does not show
    # the probe would have caught THIS graph sampling.  Here the graph is held fixed and
    # exactly one property is changed: the nn.Dropout modules are made live.  If the hash
    # then moves and the outputs become seed-dependent, the negative result above is a
    # measurement rather than a dead probe -- the same probe, same weights, same anchors,
    # opposite answer, with the flip supplied by the model's own config (dropout > 0)
    # rather than by me.  MUTATES the policy, so it runs last and never near a delivery.
    net_power = None
    if args.cover_a_power_on_network:
        item = ds[key2rel[(int(episodes[0]), int(frames[0]))]]
        item["task"] = item.get("task", PROMPT)
        n_flipped = 0
        # ACTPolicy.predict_action_chunk re-asserts self.eval() (modeling_act.py:128), so
        # flipping the modules is not enough -- the call site would undo it. That guard is
        # itself the finding: ACT's inference determinism does NOT depend on the caller.
        policy.eval = lambda *a_, **k_: policy
        for m in policy.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
                n_flipped += 1
        torch.manual_seed(999)
        s0 = rng_state(args.device)
        y1 = policy.predict_action_chunk(pre(item))
        s1 = rng_state(args.device)
        torch.manual_seed(999)
        y2 = policy.predict_action_chunk(pre(item))
        torch.manual_seed(1000)
        y3 = policy.predict_action_chunk(pre(item))
        net_power = {
            "dropout_modules_flipped_live": n_flipped,
            "state_streams_moved": moved(s0, s1),
            "same_seed_outputs_identical": bool(torch.equal(y1, y2)),
            "different_seed_outputs_identical": bool(torch.equal(y1, y3)),
            "max_abs_diff_across_seeds": float((y1 - y3).abs().max()),
            "note": ("if state_streams_moved is non-empty here while the delivered path above "
                     "moved nothing, cover A has demonstrated power on this exact network."),
        }
        print(f"[cover-A power on network] {json.dumps(net_power)}")

    vd = BENCH / "logs" / "variance"
    seed_files = sorted(vd.glob(f"preds_{args.backbone}_seed*.npz"))
    md5s = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in seed_files}
    pairwise_identical = len(set(md5s.values())) == 1 if md5s else None

    if cover_a or cover_b:
        interp = ("predict path DOES consume RNG (cover A=%s, cover B=%s) -> distinct per-draw "
                  "seeds are load-bearing, and a sampling std of 0.0 would be a real finding "
                  "about the decoder rather than a property of a deterministic graph."
                  % (cover_a, cover_b))
    else:
        interp = ("neither cover detected consumption on the predict path. Both covers have "
                  "declared blind spots (A: constructed Generators; B: sampling that bypasses "
                  "the patched Python names), so this is 'not seen by two independent probes', "
                  "not 'proved absent'. The load-bearing evidence remains the byte-identical "
                  "delivered seed runs, which is source-agnostic.")

    verdict = {
        "probe_version": 2,
        "policy_mode": args.policy_mode,
        "supersedes": "lerobot_rng_consumption.py (v1, torch CPU state only)",
        "backbone": args.backbone,
        "ckpt": args.ckpt,
        "device": args.device,
        "n_anchors_probed": args.n_anchors,
        "power_controls": pc,
        "cover_a_streams_watched": sorted(rng_state(args.device).keys()),
        "cover_b_names_patched": [f"{getattr(o, '__name__', o.__class__.__name__)}.{n}" for o, n in PATCH_TARGETS],
        "per_anchor": rows,
        "predict_consumes_rng_cover_a_state_hash": cover_a,
        "predict_consumes_rng_cover_b_call_count": cover_b,
        "cover_b_dropout_args_seen": sorted(str(t) for t in counter.dropout_args),
        "declared_blind_spots": {
            "cover_a": "explicitly constructed torch.Generator objects",
            "cover_b": "draws that bypass the patched Python names (fused/aten C++ paths)",
            "both": "a stack that samples on a device whose state is not enumerated",
        },
        "cover_a_power_on_this_network": net_power,
        "delivered_seed_run_md5": md5s,
        "delivered_seed_runs_byte_identical": pairwise_identical,
        "interpretation": interp,
    }
    p = pathlib.Path(args.out) if args.out else BENCH / "logs" / f"rng_consumption2_{args.backbone}_{args.device}_{args.policy_mode}.json"
    p.write_text(json.dumps(verdict, indent=2))
    print(f"\n[RESULT] {args.backbone} on {args.device}: cover A (state hash) = {cover_a}, cover B (call count) = {cover_b}")
    print(f"[RESULT] delivered seed runs byte-identical = {pairwise_identical}")
    print(f"[write] {p}")


if __name__ == "__main__":
    main()
