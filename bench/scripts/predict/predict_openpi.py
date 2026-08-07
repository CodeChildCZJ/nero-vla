#!/usr/bin/env python3
"""Produce preds_<bb>.npz for an openpi (pi0 / pi0.5) checkpoint on the b2 val anchors.

Reads val_anchors.npz (episodes, frames, state) and, for each anchor, feeds the model the
cam_high + cam_wrist frames and the 8-dim state from the ORIGINAL b2 dataset (val episodes are
absent from the b2_train dataset the model was fine-tuned on).

Unit contract: openpi's Policy output_transforms are
    model_transforms.outputs -> Unnormalize -> AbsoluteActions(mask 7 delta + 1 abs) -> NeroOutputs
so `policy.infer(...)["actions"]` already comes back in the *dataset action space*
(joints deg, gripper 0..100), which is exactly the space of val_anchors' gt. No manual
inverse transform is applied here on purpose -- doing it by hand is the classic double-inverse bug.

Determinism (CONTRACT.md "采样方差脚注"): pi0/pi0.5 are flow-matching, so `sample_actions` draws
a Gaussian noise chunk per call. We do NOT rely on Policy's internal rng (which is seeded at
key(0) but then *split per infer call*, making a row's noise depend on iteration order). Instead
each anchor gets `noise = normal(fold_in(key(noise_seed), global_anchor_index))`, so row i's
noise is independent of iteration order and of which subset is being run.

What that buys, and what it does NOT (measured 2026-08-05 -- an earlier version of this docstring
claimed both, and the second claim is false):
  - WITHIN one process, identical (obs, noise) gives bit-identical actions (max|diff| 0.000e+00).
  - ACROSS processes it does NOT: same ckpt, same GPU, same noise gives max|diff| ~8.4e-02 deg
    (mean ~1.2e-02), because XLA picks kernels per process. So the seed-0 variance draw does NOT
    reproduce this file bit-for-bit, and `--ref-preds` correctly reports MISMATCH on both rows.
    The effect on the published aggregate is 7.7e-05 MAE -- 0.01x this leg's sampling std and
    ~1000x below the smallest board gap -- so rankings are unaffected, but never use bit-equality
    as an integrity check on these npz files; use the recorded md5 of the specific file.

Usage:
  # main preds (fixed seed 0, all 1397 anchors)
  python predict_openpi.py --config-name pi05_nero_b2_train \
      --ckpt-dir "$NERO_CKPT/pi05_nero_b2_train/bench/29999" \
      --out "$NERO_ROOT/bench/preds/preds_pi05.npz"
  # variance footnote (200 anchors, one run per seed; NOT written under preds/)
  python predict_openpi.py --config-name pi05_nero_b2_train --ckpt-dir ... \
      --subset-n 200 --noise-seed 3 --out .../variance/pi05_s3.npz
"""
import argparse, pathlib, time

import numpy as np

# 仓库自包含:从本文件位置推导。bench/scripts/predict/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
PREDS_DIR = BENCH / "preds"
SRC_REPO = "local/pick_pink_sponge_b2"  # full dataset: holds the val episodes
PROMPT = "pick the pink sponge and place it in the blue bucket"
# config-name -> the ONE filename stem that config is allowed to produce under preds/. Unknown
# configs are fatal rather than waved through: adding a third backbone costs one line here, which
# is cheaper than the failure below.
CFG_TAG = {"pi05_nero_b2_train": "pi05", "pi0_nero_b2_train": "pi0"}
# The normalization type each config is EXPECTED to resolve to. Asserting `use_quantile_norm ==
# (model_type != PI0)` would be tautological -- that IS how config.py:187 computes it. The useful
# assertion pins the expectation per config, so an upstream change to that line, or a config edit,
# turns "the board rows silently switched normalization" into a hard stop. openvla-oft measured
# the cost of getting this wrong: a wrong bounds/units choice produced MAE 3.71 against a 3.072
# pass line -- only 0.63 above it, indistinguishable from undertraining, and it ranks.
EXPECTED_NORM = {"pi05_nero_b2_train": "quantile", "pi0_nero_b2_train": "zscore"}


def assert_norm_convention(cfg, config_name: str, ckpt_dir) -> dict:
    """Hard-stop if the normalization this run would use is not the one the board row assumes.

    Note what this does NOT establish: pi0 and pi0.5 CANNOT be made to share a normalization type
    in this stack (config.py:187 welds it to model_type), so passing this assert confirms each row
    is self-consistent, not that the pi0-vs-pi0.5 comparison is a clean backbone A/B. It is not --
    see logs/openpi_gripper_confound.json, where the normalization-only null for the gripper
    ratio-of-ratios is 1.961 against a measured 1.072.
    """
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    got = "quantile" if dc.use_quantile_norm else "zscore"
    want = EXPECTED_NORM.get(config_name)
    if want is None:
        raise SystemExit(f"[norm-assert] {config_name} has no pinned normalization type. Add it to "
                         f"EXPECTED_NORM before letting it write a row; resolved '{got}'.")
    if got != want:
        raise SystemExit(f"[norm-assert] {config_name} resolves to '{got}' normalization but this "
                         f"leg's rows were produced under '{want}'. A units/bounds change of this "
                         f"kind yields a PLAUSIBLE wrong MAE that still ranks -- refusing to run.")
    # Read the keys from the CHECKPOINT-BAKED norm_stats.json, not from `dc.norm_stats`.
    # `dc.norm_stats` is None here: DataConfigFactory._load_norm_stats resolves against
    # cfg.assets_dirs and swallows a miss, while the inference path loads the copy baked into the
    # checkpoint. Asserting on the former would have failed on every production config (it did,
    # in test) while telling us nothing about what the model actually normalizes with.
    stats = sorted(pathlib.Path(ckpt_dir).glob("assets/*/*/norm_stats.json"))
    if not stats:
        raise SystemExit(f"[norm-assert] no baked assets/*/*/norm_stats.json under {ckpt_dir}. "
                         f"openpi always bakes what it trained on; absence means this is not a "
                         f"training checkpoint.")
    import json as _json
    ns = _json.loads(stats[0].read_text())
    keys = sorted(ns.get("norm_stats", ns))
    if keys != ["actions", "state"]:
        raise SystemExit(f"[norm-assert] expected baked norm_stats over exactly "
                         f"['actions','state'] (both are consumed: state on the way in, actions "
                         f"on the way out); got {keys}. The leak falsification covers those two.")
    print(f"[norm-assert] {config_name}: normalization='{got}' (pinned), baked norm_stats keys="
          f"{keys} from {stats[0].parent.name}", flush=True)
    return {"normalization": got, "norm_stats_keys": keys, "norm_stats_path": str(stats[0])}


def guard_production_out(out, config_name, subset_n, limit, noise_seed, state_source):
    """Refuse to write a non-contract npz into preds/. Raises SystemExit; returns None otherwise.

    Fires only for paths under preds/ -- /tmp, logs/variance/ and friends stay unrestricted, so a
    pre-flight or a variance run is never inconvenienced. Runs BEFORE `import jax`, so a rejected
    invocation costs ~0.2s instead of dying 40 min in (or worse: not dying at all).

    What it actually defends against, in order of nastiness:
      * a full-1397 run of the WRONG ckpt landing on the other backbone's row. score.py cannot see
        this -- shape, anchor count and finiteness are all perfect -- so it gets RANKED. The
        pipeline takes $CFG and $BB as separate argv positions, so this is one typo away.
      * a non-canonical --noise-seed: also a perfect 1397 rows, also silently ranked, and it
        additionally falsifies the footnote's "seed 0 reproduces preds bit-for-bit" claim.
      * a partial run (--limit/--subset-n): score.py DOES catch this via its anchor-count skip, so
        the cost is a destroyed delivery rather than a wrong board -- still worth 40 min.
    """
    p = pathlib.Path(out).resolve()
    try:
        p.relative_to(PREDS_DIR.resolve())
    except ValueError:
        return  # not a delivery path; caller is on their own
    bad = []
    if subset_n:
        bad.append(f"--subset-n {subset_n} (partial run)")
    if limit is not None:
        bad.append(f"--limit {limit} (partial run)")
    if noise_seed != 0:
        bad.append(f"--noise-seed {noise_seed} (preds/ is the canonical seed-0 row)")
    if state_source != "anchors":
        bad.append(f"--state-source {state_source} (contract states come from val_anchors)")
    want = CFG_TAG.get(config_name)
    if want is None:
        bad.append(f"--config-name {config_name} is absent from CFG_TAG; add its stem mapping "
                   f"before letting it write a leaderboard row")
    elif p.stem != f"preds_{want}":
        # exact equality, NOT startswith: 'preds_pi0' is a prefix of 'preds_pi05', so a prefix
        # test would wave through the exact swap this gate exists to stop.
        bad.append(f"--config-name {config_name} may only write preds_{want}.npz, not {p.name}")
    if bad:
        raise SystemExit("[out-gate] refusing to write a leaderboard row:\n  - "
                         + "\n  - ".join(bad)
                         + f"\n  target: {p}\n  (write it somewhere outside preds/ instead)")


def to_hwc_uint8(x) -> np.ndarray:
    """LeRobot gives torch float32 CHW in [0,1]; NeroInputs wants something it can parse."""
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if np.issubdtype(a.dtype, np.floating):
        a = (255.0 * a).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N anchors")
    ap.add_argument("--subset-n", type=int, default=0,
                    help="variance study: evenly-spaced subset of N anchors (0 = all)")
    ap.add_argument("--noise-seed", type=int, default=0,
                    help="flow-matching noise seed; 0 = the canonical seed used for preds/")
    ap.add_argument("--state-source", choices=["anchors", "dataset"], default="anchors")
    args = ap.parse_args()
    guard_production_out(args.out, args.config_name, args.subset_n, args.limit,
                         args.noise_seed, args.state_source)

    import jax
    import jax.numpy as jnp
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, frames, states = A["episodes"], A["frames"], A["state"]
    total = len(eps) if args.limit is None else min(args.limit, len(eps))
    if args.subset_n and args.subset_n < total:
        # deterministic, spread across all val episodes (no RNG involved)
        sel = np.unique(np.linspace(0, total - 1, args.subset_n).round().astype(int))
    else:
        sel = np.arange(total)
    N = len(sel)
    print(f"anchors: {N} (of {len(eps)}), noise_seed={args.noise_seed}", flush=True)

    cfg = _config.get_config(args.config_name)
    assert_norm_convention(cfg, args.config_name, args.ckpt_dir)
    policy = _policy_config.create_trained_policy(cfg, args.ckpt_dir)
    horizon = cfg.model.action_horizon
    adim = cfg.model.action_dim
    key0 = jax.random.key(args.noise_seed)
    print(f"policy ready: {args.config_name} h={horizon} adim={adim} ckpt={args.ckpt_dir}", flush=True)

    preds = np.zeros((N, horizon, 8), dtype=np.float32)
    t0 = time.monotonic()
    done = 0
    sel_eps = eps[sel]
    # Group anchors by episode so each episode's video is opened once.
    for e in sorted(set(int(x) for x in sel_eps)):
        rows = np.flatnonzero(sel_eps == e)
        ds = LeRobotDataset(SRC_REPO, episodes=[e])
        # ds is re-indexed 0..L-1 for this single episode, so ds[t] == frame t of episode e.
        for r in rows:
            g = int(sel[r])  # global anchor index -> keys the noise, so order can't matter
            t = int(frames[g])
            item = ds[t]
            assert int(item["frame_index"]) == t, f"frame mismatch ep{e} t={t}"
            if args.state_source == "dataset":
                state = np.asarray(item["observation.state"], dtype=np.float32)
            else:
                state = states[g].astype(np.float32)
                ds_state = np.asarray(item["observation.state"], dtype=np.float32)
                assert np.allclose(state, ds_state, atol=1e-4), (
                    f"anchor/dataset state mismatch ep{e} t={t}: {state} vs {ds_state}")
            obs = {
                "observation/image": to_hwc_uint8(item["observation.images.cam_high"]),
                "observation/wrist_image": to_hwc_uint8(item["observation.images.cam_wrist"]),
                "observation/state": state,
                "prompt": PROMPT,
            }
            noise = np.asarray(
                jax.random.normal(jax.random.fold_in(key0, g), (horizon, adim), dtype=jnp.float32)
            )
            out = policy.infer(obs, noise=noise)
            act = np.asarray(out["actions"], dtype=np.float32)
            assert act.shape == (horizon, 8), act.shape
            preds[r] = act
            done += 1
            if done % 100 == 0:
                el = time.monotonic() - t0
                print(f"  {done}/{N}  {el:.0f}s  ({el/done*1000:.0f} ms/anchor)", flush=True)
        del ds

    # --- delta/absolute gate: check the space at runtime, never infer it from config ---------
    # The arm dims are trained as deltas vs state@t and only become absolute joint targets via the
    # AbsoluteActions output transform. If that transform is ever skipped -- e.g. a DataConfig flag
    # serialized into the checkpoint disagreeing with this config -- preds come out as 0-centred
    # deltas, every MAE is garbage, and nothing raises. Measured on the GT: a correct absolute pred
    # sits 1.27 deg from state@t, a raw-delta pred would sit ~38.6 deg away (= mean|state_arm|).
    # Gate on that ~30x gap *before* writing, so a bad npz never reaches preds/.
    # The test is *relative* on purpose: "subtracting state@t must shrink it". An absolute
    # threshold would conflate a representation bug with a merely bad model -- a poorly trained
    # backbone can legitimately predict large deltas, and must still be allowed onto the
    # leaderboard. GT ratio here is 1.269/38.327 = 0.033, and the bug flips it to 38.860/1.269.
    arm0 = preds[:, 0, :7]
    st_arm = states[sel][:, :7].astype(np.float32)
    d_state = float(np.abs(arm0 - st_arm).mean())
    d_zero = float(np.abs(arm0).mean())
    print(f"[delta-gate] mean|pred_arm[0]-state| = {d_state:.3f} deg (GT ref 1.269) | "
          f"mean|pred_arm[0]| = {d_zero:.3f} deg (GT ref 38.327) | ratio {d_state/max(d_zero,1e-9):.3f} "
          f"(GT ref 0.033)", flush=True)
    if not d_state < d_zero:
        raise SystemExit(
            f"[delta-gate] FAILED: subtracting state@t did NOT shrink arm chunk[0] "
            f"({d_state:.2f} vs {d_zero:.2f} deg) -- these look like raw deltas, not absolute "
            f"joint targets (AbsoluteActions skipped). Refusing to write {args.out}.")
    if d_state > 0.25 * d_zero:
        print(f"[delta-gate] WARNING: ratio {d_state/d_zero:.3f} is far above the GT 0.033. "
              f"Gate passed (representation looks absolute) but chunk[0] is unusually far from "
              f"state@t -- suspect model quality, not units.", flush=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pred=preds, anchor_idx=sel.astype(np.int32), noise_seed=np.int32(args.noise_seed))
    print(f"wrote {out_path}  pred{preds.shape}", flush=True)
    gt = A["gt"][sel]
    K = gt.shape[1]
    err = np.abs(preds[:, :K, :] - gt)
    print(f"self-check MAE overall {err.mean():.3f} | arm(j0-6) {err[..., :7].mean():.3f} "
          f"| grip(j7) {err[..., 7].mean():.3f}")
    print("per-joint:", " ".join(f"{v:.2f}" for v in err.mean(axis=(0, 1))))


if __name__ == "__main__":
    main()
