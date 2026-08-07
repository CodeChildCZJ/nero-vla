#!/usr/bin/env python3
"""Sampling-variance footnote for GR00T N1.7 (bench contract section 'random backbone').

N1.7's action head is a flow-matching sampler, so a single draw pays a noise
penalty that deterministic backbones (act, openvla_oft) do not. This quantifies
that penalty: MAE over a fixed 200-anchor subset, repeated with 5 different RNG
seeds, reported as mean +/- std.

Deliberately writes to logs/variance/, NOT preds/ -- it must not reach the
leaderboard.

The noise is pinned per (seed, GLOBAL anchor index) rather than by seeding the
global RNG, so seed 0 of this subset reproduces the corresponding rows of the
full 1397-anchor preds run bit for bit; --check-preds verifies that instead of
asserting it.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
# Read the stride OFF PinnedNoise (below) rather than restating it: the footnote
# must report the stride the noise actually used, not a second copy that can drift.
BASE_SEED = 0
N_ANCHORS_FULL = 1397
# Identity of the board-shared 200-anchor subset. Pinned, not just "the file at that
# path": see resolve_subset() for why existence is not enough.
EXPECTED_SUBSET_MD5 = "9007df1fcec5eca2a1f01591c6266b43"
sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.utils import parse_observation_gr00t  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

from predict_gr00t_n17 import PinnedNoise  # noqa: E402  (same pinning contract)

from n17_contracts import load_val_anchors  # noqa: E402

SEED_STRIDE = PinnedNoise.STRIDE


def resolve_subset(primary, allow_nonstandard):
    """Locate the board-shared 200-anchor subset AND prove it is that exact set.

    Returns (path, md5), or (None, None) when the caller explicitly opted into a
    locally derived subset.

    Two ways the footnote silently stops being comparable, both of which this
    script answered with a `print(WARNING)` and a full run until 2026-08-05:

    (a) the file MOVES. openpi is consolidating the subset artifacts into
        logs/variance/ right now, and the index .npy is a plausible casualty. The
        old code then fell through to `np.linspace(0, 1396, 200)`, which overlaps
        the shared set by **24/200** (measured) -- so the +/- would be quoted on
        88% different anchors while the json still reads "n_anchors": 200.
    (b) the file is REGENERATED with a different seed. It exists, so an
        existence check passes, and the anchors are still not the other legs'.

    Hence: search the known locations, but gate on md5, and make failure fatal.
    A wrong footnote is worse than a missing one -- the variance step is already
    non-fatal in n17_post_train.sh (preds ship first), so dying here costs the
    footnote and nothing else, whereas continuing puts a non-comparable number on
    the board wearing a comparable label.
    """
    primary = pathlib.Path(primary)
    tried = []
    for cand in (primary, BENCH / "logs/variance" / primary.name, BENCH / "logs" / primary.name):
        if cand in tried:
            continue
        tried.append(cand)
        if not cand.is_file():
            continue
        md5 = hashlib.md5(cand.read_bytes()).hexdigest()
        if md5 == EXPECTED_SUBSET_MD5:
            if cand != primary:
                print(f"variance subset: not at {primary}, found {cand} with the board md5 "
                      f"-- continuing (logs/ reorg)")
            print(f"variance subset: {cand} (md5 {md5}, board-shared)")
            return cand, md5
        print(f"variance subset: {cand} exists but md5 {md5} != board {EXPECTED_SUBSET_MD5} "
              f"-- NOT the shared anchor set")
    if allow_nonstandard:
        print("WARNING: --allow-nonstandard-subset -- falling back to a locally derived "
              "subset. This footnote is NOT comparable to the other backbones'.")
        return None, None
    raise SystemExit(
        "[FATAL] board-shared subset not found with md5 " + EXPECTED_SUBSET_MD5 + "; looked in "
        + ", ".join(str(t) for t in tried) + ". Refusing to fall back silently: the local "
        "fallback overlaps the shared set by 24/200, which would publish the sampling "
        "footnote on different anchors than every other leg. Pass "
        "--allow-nonstandard-subset if you really want an incomparable number.")


def require_seedcheck(path):
    """Refuse to publish a sampling std without MEASURED non-aliasing evidence.

    The `seed_stride_exceeds_anchor_count` field below is computed from
    `PinnedNoise.STRIDE`, and that constant was ALREADY 1000003 on the day the
    caller passed the draw index in as the base seed and 254/1000 noise tensors
    were reused. So the declared field is not evidence and cannot be made into
    evidence by sourcing the constant more carefully -- the fault lives in the
    caller. `n17_seed_collision_check.py` produces the measurement (it drives the
    shipped pinner and counts collisions in the tensors it actually produces) plus
    a power control that must trip. This gate is why that file cannot quietly go
    stale: missing, unpowered or failing all stop the run.

    Deliberately NOT tolerant. A missing seedcheck and a passing seedcheck must
    not both yield exit 0, or the gate degrades into the same "silence looks like
    success" shape it exists to prevent (gr00t-n15).
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise SystemExit(
            f"[FATAL] no seed-collision measurement at {p}. The declared stride fields "
            "are derived from constants and stayed green through a real 254/1000 "
            "aliasing bug, so they cannot stand in for it. Run "
            "archive/n17_seed_collision_check.py (CPU, ~15 s, no model).")
    j = json.loads(p.read_text())
    if not j.get("power_control_ok"):
        raise SystemExit(
            f"[FATAL] {p} reports power_control_ok={j.get('power_control_ok')}: its "
            "bug-shaped control did NOT produce collisions, so its "
            f"`collisions={j.get('per_anchor_noise_collisions_measured')}` has no "
            "discriminating power and must not be cited.")
    if not j.get("caller_audit_power_control_ok"):
        raise SystemExit(
            f"[FATAL] {p}: the caller-audit power control did not trip; the structural "
            "half of the check cannot distinguish the two call shapes.")
    n = j.get("per_anchor_noise_collisions_measured")
    if n != 0 or not j.get("caller_audit_ok"):
        raise SystemExit(
            f"[FATAL] {p}: collisions={n}, caller_audit_ok={j.get('caller_audit_ok')}. "
            "Draws alias, so the sampling std would be biased LOW -- i.e. noise would "
            "be published as signal. Fix the caller before measuring variance.")
    # A json written by an EARLIER version of the checker satisfies every clause
    # above, so the gate has to demand the newer evidence by name or the upgrade
    # is silently optional. `.get()` returning None reads as absent, not as pass.
    if not j.get("anchor_set_immune_power_controls_ok"):
        raise SystemExit(
            f"[FATAL] {p} carries no passing anchor-set-immune power control. The "
            "stride-1 control's 254 is a property of this anchor set, not of the "
            "defect (lerobot-setup); re-run n17_seed_collision_check.py.")
    dlv = j.get("delivery_path") or {}
    if not dlv.get("ok"):
        raise SystemExit(
            f"[FATAL] {p}: delivery-path collisions="
            f"{dlv.get('collisions_measured')}, control="
            f"{dlv.get('power_control_within_episode_index')}. Layers A/B cover the "
            "class and the VARIANCE caller; the shipped npz is written by a "
            "different caller and needs its own measurement.")
    print(f"seed-collision check: {p.name} PASS "
          f"(0/{j.get('n_noise_tensors')} collisions, control {j.get('power_control_collisions')})")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_n17_val"))
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--n-anchors", type=int, default=200)
    ap.add_argument("--subset", default=str(BENCH / "logs/variance_subset200.npy"),
                    help="npy of anchor indices shared by every stack, so the per-seed spread "
                         "is quoted on the same anchors everywhere; also searched under "
                         "logs/variance/, and gated on md5 -- absent/changed is FATAL unless "
                         "--allow-nonstandard-subset")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="DRAW indices, not raw seeds: each is multiplied by the prime "
                         "stride before the anchor index is added, so draws cannot alias")
    ap.add_argument("--check-preds", default=str(BENCH / "preds/preds_gr00t_n17.npz"),
                    help="verify that seed 0 reproduces these preds' rows bit for bit; "
                         "skipped if the file is absent")
    ap.add_argument("--out", default=str(BENCH / "logs/variance/variance_gr00t_n17.json"))
    ap.add_argument("--allow-nonstandard-subset", action="store_true",
                    help="permit a locally derived subset when the board-shared one is "
                         "missing/changed; the footnote is then NOT cross-stack comparable")
    ap.add_argument("--seedcheck", default=str(BENCH / "logs/variance/seedcheck_gr00t_n17.json"))
    args = ap.parse_args()

    # Resolve BEFORE loading the 3B model: a fatal subset problem should cost seconds,
    # not a checkpoint load and 1000 forward passes.
    sub_path, sub_md5 = resolve_subset(args.subset, args.allow_nonstandard_subset)
    seedcheck = require_seedcheck(args.seedcheck)

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    obs_modality = deepcopy(modality)
    obs_modality.pop("action", None)
    pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}
    action_keys = modality["action"].modality_keys

    A = load_val_anchors()
    episodes, frames, gt = A["episodes"], A["frames"], A["gt"]
    # The shared subset, so this footnote is comparable across backbones. Deriving it
    # locally is not equivalent: linspace-truncate and linspace-round give sets that
    # overlap only ~half, and both overlap the shared file by 24/200.
    if sub_path is not None:
        sel = np.load(sub_path).astype(int)
        print(f"variance subset: {len(sel)} anchors, shared across stacks")
    else:
        sel = np.linspace(0, len(episodes) - 1, args.n_anchors).astype(int)
    assert sel.max() < len(episodes), "subset indexes past the anchor list"

    # Build every observation once; only the sampling noise varies across seeds.
    by_ep = defaultdict(list)
    for i in sel:
        by_ep[int(episodes[i])].append(int(i))
    obs_cache = {}
    for ep in sorted(by_ep):
        traj = loader[pos_of_ep[ep]]
        for i in by_ep[ep]:
            step = extract_step_data(traj, int(frames[i]), obs_modality, embodiment_tag)
            obs = {}
            for k, v in step.states.items():
                obs[f"state.{k}"] = v
            for k, v in step.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lang_key in modality["language"].modality_keys:
                obs[lang_key] = step.text
            obs_cache[i] = parse_observation_gr00t(obs, modality)
        del traj
    print(f"cached {len(obs_cache)} observations", flush=True)

    K = gt.shape[1]
    rows = []
    seed0_chunks = {}
    for repeat in args.seeds:
        # Noise is a function of (repeat, global anchor index) -- NOT of the call
        # order -- so these rows are the same ones the full 1397-anchor run
        # produces. batch=1 for the same reason (a bf16 forward is not
        # batch-invariant, so batching would break the match even with pinning).
        #
        # The draw index goes in through `repeat`, which PinnedNoise multiplies by
        # the prime stride 1000003, NOT through the base seed. Passing it as the
        # base seed (what this script did until 2026-08-05) makes the per-anchor
        # seed `draw + anchor_idx`, so draw d anchor g and draw d+1 anchor g-1 get
        # the SAME seed and the SAME noise tensor: measured 254/1000 of this
        # subset's anchor-level draws were reused across the 5 draws. The
        # per-draw seed list stayed perfectly distinct [0,1,2,3,4] the whole time,
        # so a "are the draw seeds distinct" gate -- including my own -- passes it
        # green. The invariant that actually matters is STRIDE > anchor count
        # (gr00t-n15). Base 0 + repeat 0 is exactly predict's seed, so the
        # bit-exact provenance check below still compares like with like.
        pinner = PinnedNoise(BASE_SEED)
        errs = []
        for i in sel:
            i = int(i)
            if pinner.want is None:
                pinner.probe(lambda: policy.get_action(obs_cache[i]))
                print(f"noise pinned per global anchor index: prior {pinner.want}", flush=True)
            pinner.build(i, repeat=repeat)
            with pinner:
                chunk_dict, _ = policy.get_action(obs_cache[i])
            chunk = np.concatenate([np.asarray(chunk_dict[k])[0] for k in action_keys], axis=-1)
            if repeat == 0:
                seed0_chunks[i] = chunk
            errs.append(np.abs(chunk[:K] - gt[i]))
        e = np.stack(errs)
        rows.append({"repeat": repeat, "seed": SEED_STRIDE * repeat + BASE_SEED,
                     "mae": float(e.mean()),
                     "arm": float(e[..., :7].mean()), "grip": float(e[..., 7].mean())})
        print(f"draw {repeat}: MAE {rows[-1]['mae']:.4f} "
              f"arm {rows[-1]['arm']:.4f} grip {rows[-1]['grip']:.4f}", flush=True)

    # --- does the pinning contract actually hold? --------------------------------
    # Claiming "subset rows bit-match the full run" is cheap; checking it is one
    # array comparison. A mismatch means the noise leaked in through some path the
    # pinner does not cover (extra RNG draw, non-deterministic kernel, batching).
    reproduces = None
    check = pathlib.Path(args.check_preds)
    if seed0_chunks and check.is_file():
        full = np.load(check)["pred"]
        idx = np.array(sorted(seed0_chunks))
        mine = np.stack([seed0_chunks[i] for i in idx])
        Kp = min(mine.shape[1], full.shape[1])
        exact = bool(np.array_equal(mine[:, :Kp], full[idx][:, :Kp]))
        maxdiff = float(np.abs(mine[:, :Kp] - full[idx][:, :Kp]).max())
        reproduces = {"preds": str(check), "bit_exact": exact, "max_abs_diff": maxdiff}
        print(f"[pin check] seed 0 vs {check.name}: bit_exact={exact} maxdiff={maxdiff:.3g}"
              + ("" if exact else "  <-- pinning does NOT fully determine the output"))
    else:
        print(f"[pin check] skipped ({check} absent) -- seed-0/full-run equality unverified")

    summary = build_summary(args, sel, sub_path, sub_md5, reproduces, rows, seedcheck)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nMAE {summary['mae_mean']:.4f} +/- {summary['mae_std']:.4f} "
          f"(arm {summary['arm_mean']:.4f} +/- {summary['arm_std']:.4f}, "
          f"grip {summary['grip_mean']:.4f} +/- {summary['grip_std']:.4f})")
    print(f"wrote {out}")


def build_summary(args, sel, sub_path, sub_md5, reproduces, rows, seedcheck):
    """The footnote dict. Module level ON PURPOSE: this is the one part of the run
    that executes only AFTER the GPU work is finished, i.e. the most expensive place
    in the script to discover a typo. Being importable means it can be exercised on
    dummy rows with no GPU and no checkpoint (see the emit test), rather than
    validated by eyeballing a copy of it -- a copy proves nothing about what ships.
    """
    summary = {
        "model_path": args.model_path,
        "n_anchors": int(len(sel)),
        # Both fields come from resolve_subset's ONE read of the file the run actually
        # loaded -- not a second read here, which could hash a file that changed in
        # between and certify anchors the run never used.
        "subset": str(sub_path) if sub_path is not None else "LOCAL FALLBACK -- not comparable",
        "subset_md5": sub_md5,
        "denoising_steps": args.denoising_steps,
        "seeds": args.seeds,
        # Distinct-seed evidence, but distinctness of THESE is not sufficient: the
        # seed that matters is per (draw, anchor), so the stride is the real
        # invariant (gr00t-n15 measured a naive stride reusing 796/800 draws while
        # this very list stayed distinct; my own pre-fix stride of 1 reused
        # 254/1000). Report the formula and the check, not just the list.
        "per_draw_seeds": [BASE_SEED + SEED_STRIDE * r for r in args.seeds],
        "per_draw_seed_stride": SEED_STRIDE,
        "per_anchor_seed_formula": f"{BASE_SEED} + {SEED_STRIDE}*repeat + global_anchor_index",
        "seed_stride_exceeds_anchor_count": bool(SEED_STRIDE > N_ANCHORS_FULL),
        # ^ The four fields above are DECLARED, not measured: every one is arithmetic
        # over BASE_SEED/STRIDE, and STRIDE was already 1000003 while the effective
        # stride was 1. They stayed green through the real bug. Kept because other
        # legs' readers consume them, but flagged so nobody mistakes them for the
        # evidence -- the load-bearing field is the measured collision count below
        # (gr00t-n15's correction to my "read STRIDE off the class" fix).
        "seed_fields_are_declared_not_measured": True,
        "per_anchor_noise_collisions_measured": seedcheck["per_anchor_noise_collisions_measured"],
        # Board rule (team-lead 2026-08-05): the measured collision count is
        # canonically named, but every leg invented a DIFFERENT name for its power
        # control, and a reader that knows only one name silently accepts a bare 0
        # -- reopening the exact hole the measurement exists to close. I did this to
        # n15's json. The endorsement does not travel with the number unless the
        # writer emits the union of known names, so emit all four.
        **{k: seedcheck["power_control_collisions"] for k in (
            "seed_collision_power_control",          # gr00t_n17 (mine)
            "seed_check_power_control_collisions",   # gr00t_n15
            "power_control_collisions",              # generic
            "collision_power_control",               # lerobot-setup
        )},
        "power_control_key_aliases": [
            "seed_collision_power_control", "seed_check_power_control_collisions",
            "power_control_collisions", "collision_power_control",
        ],
        "power_control_semantics": (
            "collisions produced by the v20260804 bug shape (effective stride 1) on "
            "the shared 200-anchor subset; nonzero == the measurement has "
            "discriminating power. Construction-fixed controls that do not depend "
            "on the anchor set are in the seed_collision_check block."
        ),
        "seed_collision_check": {
            "path": args.seedcheck,
            "verdict": seedcheck["verdict"],
            "n_noise_tensors": seedcheck["n_noise_tensors"],
            "caller_audit_ok": seedcheck["caller_audit_ok"],
            "collision_count_shape_invariant": seedcheck["collision_count_shape_invariant"],
        },
        "n_anchors_full": N_ANCHORS_FULL,
        "batch_size": 1,                # lerobot: the bit-exactness below only holds
        # at the SAME batch shape as the board run -- a different shape picks other
        # cuDNN kernels and degrades to "close but not bit-exact", which looks exactly
        # like a checkpoint mismatch. So the shape is part of the claim, not context.
        "noise": "pinned per (seed, global anchor index); batch=1",
        "reproduces_full_run": reproduces,
        # gr00t-n15's point: `bit_exact` only proves PROVENANCE if the subset numbers
        # came from an independent reload, not from slicing the board run's array in
        # the same process (two views of one buffer are trivially equal). This is a
        # separate `Gr00tPolicy.from_pretrained` in a separate process, so state it.
        "subset_run_is_independent_process": True,
        "per_seed": rows,
    }
    for f in ("mae", "arm", "grip"):
        vals = [r[f] for r in rows]
        summary[f"{f}_mean"] = float(np.mean(vals))
        summary[f"{f}_std"] = float(np.std(vals, ddof=1))
    # Emit all three key layouts on the board (flat *_std = mine + openpi, std_ddof1 =
    # n15, nested = act). Same numbers, three doors: a reader that knows only one
    # schema and falls back to 0.0 silently DROPS the sampling term, and a dropped
    # term is indistinguishable from a genuinely deterministic stack. Writing the
    # superset is the cheap side of that trade -- the strict-reader side raises, and
    # my own strict reader did exactly that on this file until 2026-08-05.
    summary["std_ddof1"] = {f: summary[f"{f}_std"] for f in ("mae", "arm", "grip")}
    for f in ("mae", "arm", "grip"):
        summary[f] = {"mean": summary[f"{f}_mean"], "std": summary[f"{f}_std"]}
    summary["std_scale"] = ("per-draw std on the shared SUBSET-200 anchors, NOT the "
                            "1397-anchor board row; unscaled = conservative")
    return summary


if __name__ == "__main__":
    main()
