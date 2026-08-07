#!/usr/bin/env python3
"""Produce preds/preds_gr00t_n17.npz for the NERO b2 backbone bench.

For every anchor in val_anchors.npz we rebuild the exact observation the policy
would see on the robot (cam_high + cam_wrist frame @t, 8-D follower state @t,
task prompt) and record the predicted action chunk.

Unit contract: the modality config declares `single_arm` as a RELATIVE action,
but StateActionProcessor.unapply() adds the reference state back before
returning, so `get_action()` hands back absolute joint targets in the dataset's
raw units (arm ~degrees, gripper 0-100) -- the same space as val_anchors' gt.
The script asserts the state it feeds the model equals val_anchors' recorded
state, which catches any episode/frame misalignment before it silently poisons
the MAE.

N1.7 API differences vs the N1.5 script (scripts/predict_gr00t.py):
  - Gr00tPolicy takes no data-config; the modality config + transforms are baked
    into the processor saved with the checkpoint.
  - Episodes come from LeRobotEpisodeLoader, which indexes POSITIONALLY. Our val
    set keeps b2's original (non-contiguous) episode ids, so we map through
    meta/episodes.jsonl rather than indexing with the raw episode number.
  - loader[i] decodes a whole episode's video, so anchors are grouped by episode
    and each episode is loaded exactly once.
"""
import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy
import pathlib
import sys
import time

import numpy as np
import torch

# 仓库自包含:从本文件位置推导。bench/scripts/predict/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# n17_contracts.py 与本脚本原本同目录;公开版把一次性诊断脚本收进 bench/archive/,
# 而这个契约模块仍是交付路径的一部分,所以显式把 archive/ 加进 sys.path。
sys.path.insert(0, str(BENCH / "archive"))

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.utils import parse_observation_gr00t  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

from n17_contracts import (EXPECTED_ANCHORS_MD5, load_val_anchors,  # noqa: E402
                           normalization_contract)


def delivery_gate(n_written, n_expected, used_seeds, into_preds):
    """Refuse to put a file into preds/ that is not the deliverable.

    Checks what was WRITTEN, not the flags that were passed. `--limit` is already
    refused up front, but a short run can also come from a truncated dataset or an
    exception that left the loop early, and neither carries a flag once the array
    is on disk. OFT hit the same hazard from the other side: a scaffold run with
    the DEFAULT anchor count and the DEFAULT --out produced a full-length,
    correctly-shaped npz of noise that score.py ranked and that overwrote the real
    delivery. Raises SystemExit; a no-op when the target is outside preds/.
    """
    if not into_preds:
        return
    if n_written != n_expected:
        raise SystemExit(
            f"refusing to write {n_written}/{n_expected} anchors into preds/ "
            f"-- a short run has no flag on it once it is on disk")
    dup = len(used_seeds) - len(set(int(s) for s in used_seeds))
    if dup:
        raise SystemExit(
            f"refusing to write into preds/: {dup} of {len(used_seeds)} per-anchor "
            f"noise seeds were REUSED, so some anchors share a prior draw")
    if len(used_seeds) != n_written:
        raise SystemExit(
            f"refusing to write into preds/: {len(used_seeds)} seeds recorded for "
            f"{n_written} anchors -- the pinner was bypassed on some anchors, which "
            f"is the one failure the seed list itself cannot show by being distinct")


class PinnedNoise:
    """Make the flow-matching prior a function of the GLOBAL anchor index.

    gr00t_n1d7.py:349 draws exactly one torch.randn of shape
    (B, action_horizon, max_action_dim); the denoising loop after it is a
    deterministic ODE (no re-noising), so that single draw is the only source of
    run-to-run variation. Seeding the global RNG instead would tie an anchor's
    noise to its position in the call order, so the 200-anchor variance subset
    would not reproduce the corresponding rows of the 1397-anchor run.

    Differences from the N1.5 version (scripts/predict_gr00t.py), both hardening:
      - the prior's shape is LEARNED by probing one get_action() call rather than
        read off head attributes, so a renamed/padded action_dim cannot silently
        turn pinning into a no-op;
      - a second prior-shaped draw inside one get_action() raises instead of
        falling through to the real RNG, which would pin only the first draw.
    """

    # Stride between draws in per-anchor seed space. MUST exceed the anchor count
    # (1397) or draw d anchor g and draw d+1 anchor g-stride collide on one seed and
    # therefore on one noise tensor, which correlates the draws and shrinks the
    # measured sampling std. Exposed as a class attribute so the variance script
    # reports the stride it actually gets rather than a second copy of the number --
    # a duplicated constant is the same two-source split that hides in this stack's
    # config objects (see the N1.7 launcher traps).
    STRIDE = 1000003

    def __init__(self, seed):
        self.seed = int(seed)
        self.want = None          # set by probe()
        self._pending = None
        self._orig = torch.randn
        # Every seed this object actually hands to manual_seed, in call order.
        # Recorded HERE rather than recomputed by the caller: a caller that
        # restates the formula is a second copy that can drift from this one, and
        # the whole point is to carry what happened rather than what should have.
        self.used_seeds = []

    def probe(self, run_once):
        """Run one get_action() with a recording patch to learn the prior shape."""
        seen = []
        orig = self._orig

        def recorder(*a, **kw):
            size = kw.get("size", a[0] if a else None)
            if isinstance(size, (tuple, list, torch.Size)):
                seen.append(tuple(size))
            return orig(*a, **kw)

        torch.randn = recorder
        try:
            run_once()
        finally:
            torch.randn = orig
        prior = [s for s in seen if len(s) == 3 and s[0] == 1]
        assert len(prior) == 1, (
            f"expected exactly one (1,H,D) torch.randn per get_action, saw {seen}. "
            f"The head is no longer a single-draw ODE -- pinning would cover only "
            f"part of the sampling noise and the variance footnote would be wrong."
        )
        self.want = prior[0]
        return self.want

    def build(self, global_idx, repeat=0):
        gen = torch.Generator(device="cpu")
        seed = self.seed + self.STRIDE * int(repeat) + int(global_idx)
        gen.manual_seed(seed)
        self.used_seeds.append(seed)
        self._pending = torch.randn(self.want, generator=gen)

    def __enter__(self):
        assert self._pending is not None, "call build() before entering"

        def patched(*a, **kw):
            size = kw.get("size", a[0] if a else None)
            if isinstance(size, (tuple, list, torch.Size)) and tuple(size) == self.want:
                if self._pending is None:
                    raise RuntimeError(
                        "second prior-shaped randn inside one get_action -- pinning "
                        "would only cover the first draw"
                    )
                out = self._pending.to(device=kw.get("device"), dtype=kw.get("dtype"))
                self._pending = None
                return out
            return self._orig(*a, **kw)

        torch.randn = patched
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig
        if exc[0] is None:
            assert self._pending is None, (
                "pinned noise was never consumed -- the prior's shape changed, so "
                "this run fell back to the global RNG and is not reproducible"
            )
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_n17_val"))
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="fixed RNG seed (bench contract)")
    ap.add_argument("--limit", type=int, default=0, help="only first N anchors (smoke test)")
    ap.add_argument("--out", default=str(BENCH / "preds/preds_gr00t_n17.npz"))
    args = ap.parse_args()

    # A --limit run covers only the first N anchors (i.e. the first few episodes),
    # so it must never masquerade as the deliverable. Refuse to write a partial
    # result into preds/, where score.py would pick it up.
    if args.limit and pathlib.Path(args.out).resolve().parent == (BENCH / "preds").resolve():
        sys.exit(
            f"refusing to write a --limit {args.limit} subset into preds/ "
            f"(score.py would score it as the real thing); pass --out elsewhere"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Training writes the model to <run>/checkpoint-N/ but the processor to
    # <run>/processor/. Gr00tPolicy looks for the processor next to the model, so
    # link it in rather than making every caller remember.
    model_dir = pathlib.Path(args.model_path)
    if not (model_dir / "processor_config.json").exists() and not (model_dir / "processor").exists():
        run_processor = model_dir.parent / "processor"
        if run_processor.is_dir():
            (model_dir / "processor").symlink_to(run_processor)
            print(f"linked processor: {model_dir / 'processor'} -> {run_processor}")

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    modality = policy.get_modality_config()

    # --- unit-contract guard -------------------------------------------------
    # unapply_action() only adds the reference state back when BOTH the key is
    # declared RELATIVE *and* the processor's use_relative_action flag is set
    # (state_action_processor.py:478). The flag defaults to False and is baked
    # into the checkpoint, so a train/infer mismatch would silently hand back
    # raw deltas -- no exception, just a garbage MAE. Assert it here.
    sap = policy.processor.state_action_processor
    acfgs = modality["action"].action_configs
    print("action convention:", ", ".join(
        f"{k}={c.rep.value}/{c.type.value}" for k, c in zip(modality["action"].modality_keys, acfgs)
    ), f"| use_relative_action={sap.use_relative_action}")
    if any(c.rep.value == "relative" for c in acfgs):
        assert sap.use_relative_action, (
            "checkpoint declares RELATIVE actions but its processor has "
            "use_relative_action=False -> deltas would NOT be converted back to "
            "absolute joint targets and preds would be garbage vs the contract gt"
        )
    norm_contract = normalization_contract(modality, sap)
    print("normalization contract:", norm_contract)
    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)

    # Observations must not carry the action modality (it is what we predict).
    obs_modality = deepcopy(modality)
    obs_modality.pop("action", None)

    # LeRobotEpisodeLoader indexes positionally; b2 episode ids are not contiguous.
    pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}

    A = load_val_anchors()
    episodes, frames, ref_state, gt = A["episodes"], A["frames"], A["state"], A["gt"]
    n = args.limit if args.limit else len(episodes)

    by_ep = defaultdict(list)
    for i in range(n):
        by_ep[int(episodes[i])].append(i)
    print(f"{n} anchors over {len(by_ep)} val episodes", flush=True)

    state_keys = modality["state"].modality_keys
    action_keys = modality["action"].modality_keys

    preds = None
    done = 0
    # One anchor per get_action call (batch=1). Episodes are grouped only so each
    # video is decoded once. gr00t-n15 measured that a bf16 forward is NOT
    # batch-invariant (batch 8 vs 1 -> maxdiff 0.78 on N1.5), so batching would
    # make the variance subset's rows differ from the full run's even with the
    # noise pinned; batch=1 removes that failure mode entirely.
    pinner = PinnedNoise(args.seed)
    t0 = time.time()
    for ep in sorted(by_ep):
        traj = loader[pos_of_ep[ep]]  # decodes this episode's video once
        for i in by_ep[ep]:
            step = extract_step_data(traj, int(frames[i]), obs_modality, embodiment_tag)

            # hard alignment check: raw follower state must equal the anchor's
            got = np.concatenate([step.states[k][0] for k in state_keys])
            assert np.allclose(got, ref_state[i], atol=1e-3), (
                f"anchor {i} ep{ep} f{frames[i]} state mismatch:\n"
                f"  dataset={got}\n  anchors={ref_state[i]}"
            )

            obs = {}
            for k, v in step.states.items():
                obs[f"state.{k}"] = v
            for k, v in step.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lang_key in modality["language"].modality_keys:
                obs[lang_key] = step.text

            if pinner.want is None:
                # one throwaway forward on the first anchor to learn the prior's
                # shape from the model itself instead of from attribute names
                pinner.probe(lambda: policy.get_action(parse_observation_gr00t(obs, modality)))
                print(f"noise pinned per global anchor index: prior {pinner.want}, "
                      f"seed {args.seed}, {args.denoising_steps} denoising steps", flush=True)
            pinner.build(i)
            with pinner:
                chunk_dict, _ = policy.get_action(parse_observation_gr00t(obs, modality))
            # (1, H, 7) arm + (1, H, 1) gripper -> (H, 8), raw absolute joint units
            chunk = np.concatenate([np.asarray(chunk_dict[k])[0] for k in action_keys], axis=-1)

            if preds is None:
                preds = np.zeros((n,) + chunk.shape, np.float32)
            preds[i] = chunk
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  {done}/{n}  {el:.0f}s elapsed, eta {el / done * (n - done):.0f}s",
                      flush=True)
        del traj

    # Every anchor must have been written at its own index: preds is pre-zeroed, so a
    # skipped anchor would ship as an all-zero row rather than raise.
    assert done == n, f"only filled {done}/{n} anchors -- the rest would ship as zeros"
    assert preds is not None and np.isfinite(preds).all(), "non-finite values in predictions"

    # --- relative->absolute sanity gate --------------------------------------
    # The arm barely moves between consecutive frames, so the first predicted
    # step must land near the current state. If the state add-back were missed
    # the arm would come back as a small delta centred on 0 instead.
    arm0 = preds[:n, 0, :7]
    d_state = np.abs(arm0 - ref_state[:n, :7]).mean()
    d_zero = np.abs(arm0).mean()
    print(f"[rel->abs gate] mean|arm chunk[0] - state|={d_state:.3f}  "
          f"mean|arm chunk[0]|={d_zero:.3f}")
    assert d_state < d_zero, (
        f"arm chunk[0] is closer to 0 ({d_zero:.3f}) than to the current state "
        f"({d_state:.3f}) -> looks like raw deltas, the relative->absolute "
        f"add-back did not happen"
    )

    K = gt.shape[1]
    assert preds.shape[1] >= K, f"predicted horizon {preds.shape[1]} < contract K={K}"

    out_path = pathlib.Path(args.out)
    into_preds = out_path.resolve().parent == (BENCH / "preds").resolve()

    # The seeds this run actually used, read off the pinner, not recomputed. This
    # is what turns the caller-level non-aliasing claim from "someone read the
    # source and re-implemented the loop" into a property of the delivered file
    # (gr00t-n15's durable fix, board-endorsed). n17_seed_collision_check.py's
    # layer C still drives a re-written loop; this is the artifact that outranks it.
    used = np.asarray(pinner.used_seeds, dtype=np.int64)

    # Two things a board file must not be, checked against what was WRITTEN rather
    # than against the flags that were passed. --limit is already refused up front,
    # but a short run can also come from a truncated dataset or an exception that
    # left the loop early, and those carry no flag. OFT's live case was the same
    # shape from the other side: a scaffold run with DEFAULT anchor count and
    # DEFAULT --out produced a full-length, correctly-shaped file of noise that
    # score.py ranked and that overwrote the real delivery.
    delivery_gate(len(preds), len(gt), used, into_preds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp name and os.replace() it into place. preds/ is inside
    # the syncthing folder shared 145<->147 (.stignore excludes `data/raw`, `**/log/`
    # and refs/ -- NOT vla_backbone_bench), and the board's monitors rerun score.py
    # and paired_signif.py the moment a preds_*.npz appears. A direct np.savez is
    # visible to inotify while still short, so the other host (and the monitors) can
    # observe a truncated file that has the right name. os.replace is atomic within
    # a filesystem, so a reader sees either the old file or the whole new one.
    # The temp name is a DOTFILE, and that detail is load-bearing: np.savez appends
    # `.npz` to any name that lacks it, so the obvious `preds_gr00t_n17.npz.partial`
    # lands on disk as `preds_gr00t_n17.npz.partial.npz` -- which MATCHES the exact
    # `preds_*.npz` glob that score.py:31 and paired_signif.py:295 use, i.e. the
    # "safe" temp file would have appeared on the board as an extra phantom row.
    # `.preds_...` is skipped by glob (dotfiles are excluded by default). Measured,
    # not assumed -- the first version of this comment claimed the opposite.
    tmp_path = out_path.with_name("." + out_path.name + ".partial")
    np.savez(tmp_path, pred=preds.astype(np.float32), noise_seeds=used,
             # provenance travelling with the numbers: a full-length file from the
             # wrong checkpoint is otherwise indistinguishable from the deliverable
             model_path=np.array(str(pathlib.Path(args.model_path).resolve())),
             base_seed=np.array(args.seed), denoising_steps=np.array(args.denoising_steps),
             seed_formula=np.array(f"{args.seed} + {PinnedNoise.STRIDE}*repeat + global_idx"),
             # the flags the floor number's HARD/SOFT label rests on, measured off
             # this checkpoint's own processor -- so a later reader never has to
             # take the label on trust. json string: 0-d unicode array, no pickle.
             normalization_contract=np.array(json.dumps(norm_contract, sort_keys=True)),
             val_anchors_md5=np.array(EXPECTED_ANCHORS_MD5))
    # np.savez appends .npz when the name lacks it -- resolve what it actually wrote
    # rather than assuming, or the replace() silently targets a nonexistent file.
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if not written.exists():
        raise SystemExit(f"np.savez wrote neither {tmp_path} nor {written}")
    os.replace(written, out_path)
    print(f"wrote {out_path}  shape={preds.shape}  "
          f"{len(set(used.tolist()))}/{len(used)} distinct noise seeds")

    err = np.abs(preds[:n, :K, :] - gt[:n])
    print(f"self-check MAE={err.mean():.3f}  arm(j0-6)={err[..., :7].mean():.3f}  "
          f"grip(j7)={err[..., 7].mean():.3f}")
    print("per-joint:", " ".join(f"{v:.2f}" for v in err.mean(axis=(0, 1))))


if __name__ == "__main__":
    main()
