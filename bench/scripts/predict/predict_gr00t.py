#!/usr/bin/env python3
"""Produce preds/preds_gr00t_n15.npz for the NERO b2 backbone bench.

For every anchor in val_anchors.npz we rebuild the exact observation the policy
would see on the robot (cam_high + cam_wrist frame @t, 8-D follower state @t,
task prompt) and record the predicted 16-step action chunk.

Gr00tPolicy.get_action() already runs modality_transform.unapply(), so what
comes back is in the dataset's raw action units (arm joints in degrees, gripper
in its own 0-100 unit) -- the same space as val_anchors' gt. The script asserts
the state it feeds the model equals val_anchors' recorded state, which catches
any episode/frame misalignment before it silently poisons the MAE.
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

# 仓库自包含:从本文件位置推导。bench/scripts/predict/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH / "configs"))

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

PROMPT = "pick the pink sponge and place it in the blue bucket"


class PinnedNoise:
    """Force the flow-matching prior to be a function of the GLOBAL anchor index.

    flow_matching_action_head.get_action() draws exactly one torch.randn of shape
    (B, action_horizon, action_dim) (line ~364); the denoising loop after it is a
    deterministic ODE. Seeding the global RNG would tie the noise to an anchor's
    position inside its batch, so a 200-anchor subset run would not reproduce the
    same rows as the 1397-anchor run. Injecting per-anchor noise instead makes a
    row depend only on (seed, repeat, global anchor index) -- subset rows then
    bit-match full-run rows no matter how the anchors are batched.

    Exactly-once is enforced in BOTH directions (gr00t-n17 caught the second half):
    a prior-shaped draw that never happens is a silent fall-through to the global
    RNG, and so is a *second* prior-shaped draw -- which is precisely what an
    upstream ODE->SDE change would look like. Zero draws raises with the shapes we
    actually observed (so the diagnostic names the right shape without costing a
    probe forward); two draws raises immediately.
    """

    def __init__(self, horizon, dim, seed):
        self.horizon, self.dim, self.seed = horizon, dim, seed
        self._pending = None
        self._fired = 0
        self._seen = []
        self._orig = torch.randn

    def build(self, global_idx, repeat):
        rows = []
        for g in global_idx:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.seed) + 1000003 * int(repeat) + int(g))
            rows.append(torch.randn(self.horizon, self.dim, generator=gen))
        self._pending = torch.stack(rows)

    def __enter__(self):
        self.want = (self._pending.shape[0], self.horizon, self.dim)
        self._fired = 0
        self._seen = []

        def patched(*a, **kw):
            size = kw.get("size", a[0] if a else None)
            try:
                shape = tuple(size)
            except TypeError:                       # torch.randn(3, 4) varargs form
                shape = tuple(a)
            self._seen.append(shape)
            if shape == self.want:
                if self._pending is None:
                    raise RuntimeError(
                        f"upstream drew a SECOND prior-shaped tensor {shape}; the "
                        "sampling path is no longer a single draw + deterministic ODE, "
                        "so pinning only the first draw would silently understate variance"
                    )
                out = self._pending.to(device=kw.get("device"), dtype=kw.get("dtype"))
                self._pending = None
                self._fired += 1
                return out
            return self._orig(*a, **kw)

        torch.randn = patched
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig
        if exc[0] is None and self._fired != 1:
            raise RuntimeError(
                f"pinned noise fired {self._fired}x (want exactly 1). want={self.want}, "
                f"torch.randn shapes actually seen={self._seen}"
            )
        return False


class SeedTap:
    """Record every seed that ACTUALLY reaches the RNG, by shimming torch.Generator.

    Deliberately taps the SINK, not the source: PinnedNoise.build stays byte-identical
    to the delivered file, so what gets exported is what the shipped code fed to
    manual_seed -- not a re-derivation of the formula, which would only re-test the
    formula. Subclassing torch.Generator is transparent (same tensors, verified).

    This closes layer 3 of the collision check (the production CALL SITE), which a
    check that re-writes the draw loop cannot reach: gr00t-n17's real bug was a caller
    that never passed `repeat`, and a class-level check passes that bug.
    """

    def __init__(self):
        self.seeds = []
        self._orig = torch.Generator

    def __enter__(self):
        rec, orig = self.seeds, self._orig

        class _G(orig):
            def manual_seed(self, s):
                rec.append(int(s))
                return orig.manual_seed(self, s)

        torch.Generator = _G
        return self

    def __exit__(self, *exc):
        torch.Generator = self._orig
        return False


def dry_head_cfg(model_path):
    """action_horizon/action_dim from the ckpt's own config.json -- lets --dry-run-seeds
    build the real PinnedNoise without loading 3B of weights. Seeds do not depend on
    these two numbers (only the noise SHAPE does), but reading them from the ckpt keeps
    the dry run's pinner identical to the on-card one instead of hardcoding 16/32."""
    import json
    import types
    c = json.load(open(pathlib.Path(model_path) / "config.json"))
    h = c.get("action_head_cfg", {})
    horizon = h.get("action_horizon", c.get("action_horizon"))
    dim = h.get("action_dim", c.get("action_dim"))
    assert horizon and dim, f"no action_horizon/action_dim in {model_path}/config.json"
    return types.SimpleNamespace(action_horizon=int(horizon), action_dim=int(dim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_gr00t_val"))
    ap.add_argument("--data-config", default="nero_data_config:NeroDualCamDataConfig")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N anchors (pipeline smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the flow-matching noise")
    ap.add_argument("--repeats", type=int, default=1,
                    help=">1: re-sample the same anchors with different noise to quantify "
                         "sampling variance. repeat 0 is the one written to --out.")
    ap.add_argument("--var-out", default="", help="npz for the full (repeats,N,K,8) stack (footnote only)")
    ap.add_argument("--every", type=int, default=1,
                    help=">1: every-k anchor subset spanning all val episodes (footnote only, "
                         "the result is NOT a valid preds/ deliverable)")
    ap.add_argument("--subset", default="",
                    help="npy of anchor indices -- use logs/variance_subset200.npy so every "
                         "stochastic backbone reports its sampling std on the same anchors "
                         "(footnote only, NOT a valid preds/ deliverable)")
    ap.add_argument("--out", default=str(BENCH / "preds/preds_gr00t_n15.npz"))
    ap.add_argument("--check-preds", default=str(BENCH / "preds/preds_gr00t_n15.npz"),
                    help="with --subset: assert the seed-0 subset rows are BIT-identical to the "
                         "corresponding rows of this deliverable. Turns 'subset rows bit-match "
                         "the full run' from a claim in the footnote into a measured boolean, "
                         "which is what catches a stray RNG draw or a non-deterministic kernel "
                         "when predict and variance run as two separate processes. (gr00t-n17)")
    ap.add_argument("--dry-run-seeds", default="",
                    help="CPU-only provenance export (no GPU, no model, no preds written): run "
                         "THIS script's real anchor/repeat loop, short-circuited just before the "
                         "model forward, and write every seed that actually reached the RNG. "
                         "Measures the delivered CALL SITE, which is where a seed bug lives; a "
                         "checker that re-writes the draw loop only re-tests the pinner class.")
    args = ap.parse_args()
    dry = args.dry_run_seeds
    if (args.every > 1 or args.limit or args.subset) and pathlib.Path(args.out).parent.name == "preds":
        ap.error("a subset run must not be written into preds/ -- it would poison the leaderboard")

    data_config = load_data_config(args.data_config)
    policy = None if dry else Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    dataset = None if dry else LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # policy applies (and un-applies) them itself
        embodiment_tag=args.embodiment_tag,
    )

    A = np.load(BENCH / "data" / "val_anchors.npz")
    episodes, frames, ref_state = A["episodes"], A["frames"], A["state"]
    gt_all = A["gt"]
    gidx = np.arange(len(episodes))   # position in the full 1397 -- this is what seeds the noise
    subset_sel = None
    if args.subset:
        sel = subset_sel = np.load(args.subset)
        assert sel.ndim == 1 and len(set(sel.tolist())) == len(sel) and sel.max() < len(episodes)
        print(f"subset {args.subset}: {len(sel)} anchors over "
              f"{len(set(episodes[sel].tolist()))} val episodes")
        episodes, frames, ref_state, gt_all, gidx = (
            episodes[sel], frames[sel], ref_state[sel], gt_all[sel], gidx[sel])
    if args.every > 1:
        # footnote-only: an every-k subset still spans all 20 val episodes, unlike
        # --limit N which would only cover the first few. Never write this to preds/.
        sel = np.arange(0, len(episodes), args.every)
        episodes, frames, ref_state, gt_all, gidx = (
            episodes[sel], frames[sel], ref_state[sel], gt_all[sel], gidx[sel])
    n = args.limit if args.limit else len(episodes)
    print(f"{n} anchors over {len(set(episodes[:n].tolist()))} val episodes")

    obs_keys = data_config.video_keys + data_config.state_keys + data_config.language_keys
    head_cfg = dry_head_cfg(args.model_path) if dry else policy.model.action_head.config
    pinner = PinnedNoise(head_cfg.action_horizon, head_cfg.action_dim, args.seed)
    print(f"noise pinned to global anchor index (horizon={head_cfg.action_horizon}, "
          f"dim={head_cfg.action_dim}, seed={args.seed})")
    preds = None
    t0 = time.time()
    tap = SeedTap()
    if dry:
        tap.__enter__()
    for lo in range(0, n, args.batch_size):
        hi = min(lo + args.batch_size, n)
        obs = None
        if not dry:
            batch = {k: [] for k in obs_keys}
            for i in range(lo, hi):
                step = dataset.get_step_data(int(episodes[i]), int(frames[i]))
                # hard alignment check: raw follower state must equal the anchor's
                got = np.concatenate([step["state.single_arm"][0], step["state.gripper"][0]])
                assert np.allclose(got, ref_state[i], atol=1e-3), (
                    f"anchor {i} ep{episodes[i]} f{frames[i]} state mismatch:\n"
                    f"  dataset={got}\n  anchors={ref_state[i]}"
                )
                for k in obs_keys:
                    batch[k].append(step[k])
            obs = {k: np.stack(v) if k not in data_config.language_keys else np.array(v)
                   for k, v in batch.items()}
        for r in range(args.repeats):
            pinner.build(gidx[lo:hi], r)
            if dry:            # short-circuit BEFORE the forward; the seed is already drawn
                continue
            with pinner:
                out = policy.get_action(obs)
            # (B, 16, 7) arm + (B, 16, 1) gripper -> (B, 16, 8), raw joint units
            chunk = np.concatenate([out["action.single_arm"], out["action.gripper"]], axis=-1)
            if preds is None:
                preds = np.zeros((args.repeats, n) + chunk.shape[1:], np.float32)
            preds[r, lo:hi] = chunk
        if lo % (args.batch_size * 20) == 0:
            el = time.time() - t0
            print(f"  {hi}/{n}  {el:.0f}s elapsed, eta {el / max(hi, 1) * (n - hi):.0f}s",
                  flush=True)

    if dry:
        tap.__exit__(None, None, None)
        R, want = args.repeats, args.repeats * n
        assert len(tap.seeds) == want, (
            f"seed tap saw {len(tap.seeds)} manual_seed calls, expected repeats*anchors={want} "
            "-- a call site that skips, reuses or duplicates draws is exactly what this exports"
        )
        seeds = np.zeros((R, n), np.int64)
        it = iter(tap.seeds)
        for lo in range(0, n, args.batch_size):    # same nesting as the loop above. The seed
            for r in range(R):                     # VALUES are observed at the RNG; only this
                for j in range(lo, min(lo + args.batch_size, n)):   # reshape assumes the order,
                    seeds[r, j] = next(it)         # and the count assert above pins the total.
        dpath = pathlib.Path(args.dry_run_seeds)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dpath, seeds=seeds, global_idx=gidx[:n].astype(np.int64),
                 base_seed=np.int64(args.seed), repeats=np.int64(R),
                 batch_size=np.int64(args.batch_size), n_anchors=np.int64(n),
                 argv=np.array(" ".join(sys.argv)))
        uniq = len(set(seeds.ravel().tolist()))
        print(f"[dry-run-seeds] wrote {dpath}  seeds{seeds.shape}  "
              f"distinct={uniq}/{seeds.size}  collisions={seeds.size - uniq}")
        return

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pred=preds[0].astype(np.float32))
    print(f"wrote {out_path}  shape={preds[0].shape}  seed={args.seed}")

    # measured invariant, not a claimed one: seed-0 subset rows must be bit-identical
    # to the deliverable's corresponding rows. If they are not, the pinning did not
    # actually decouple a row from its batch context and the footnote is meaningless.
    if subset_sel is not None and args.check_preds and pathlib.Path(args.check_preds).exists():
        ref = np.load(args.check_preds)["pred"]
        mine = preds[0][:, : ref.shape[1], :]
        sub = ref[subset_sel][:, : mine.shape[1], :]
        exact = bool(np.array_equal(mine, sub))
        print(f"check-preds vs {args.check_preds}: bit_exact={exact}  "
              f"maxdiff={np.abs(mine.astype(np.float64) - sub.astype(np.float64)).max():.3e}")
        if not exact:
            print("  !! subset rows differ from the full run -- the variance footnote below "
                  "is NOT comparable to the deliverable; investigate before reporting")
    elif subset_sel is not None:
        print(f"check-preds: skipped ({args.check_preds} not present)")

    gt = gt_all[:n]
    K = gt.shape[1]
    maes = []
    for r in range(args.repeats):
        err = np.abs(preds[r, :, :K, :] - gt)
        maes.append((err.mean(), err[..., :7].mean(), err[..., 7].mean()))
        tag = "self-check" if r == 0 else f"  repeat {r}"
        print(f"{tag} MAE={err.mean():.3f}  arm(j0-6)={err[..., :7].mean():.3f}  "
              f"grip(j7)={err[..., 7].mean():.3f}")
        if r == 0:
            print("per-joint:", " ".join(f"{v:.2f}" for v in err.mean(axis=(0, 1))))
    if args.repeats > 1:
        m = np.array(maes)
        print(f"sampling-variance footnote over {args.repeats} noise draws on {n} anchors: "
              f"MAE {m[:, 0].mean():.4f} +- {m[:, 0].std(ddof=1):.4f}  "
              f"arm {m[:, 1].mean():.4f} +- {m[:, 1].std(ddof=1):.4f}  "
              f"grip {m[:, 2].mean():.4f} +- {m[:, 2].std(ddof=1):.4f}")
        if args.var_out:
            np.savez(args.var_out, pred=preds.astype(np.float32))
            print("wrote variance stack", args.var_out)


if __name__ == "__main__":
    main()
