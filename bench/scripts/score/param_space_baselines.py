#!/usr/bin/env python3
"""How much does the ACTION PARAMETERIZATION give a model for free?

The two GR00T rows on this board are meant to differ by backbone, but they also
differ in action space, because each leg follows its own stack's official recipe:

    N1.5  action = ABSOLUTE joint targets, min_max normalized
    N1.7  action = arm RELATIVE (delta from current state) + gripper ABSOLUTE
          (`launch_finetune.py` hardcodes config.model.use_relative_action = True)

That is not a free choice, but it IS a confound: a relative-arm model is handed
the current joint position and only has to predict a small correction, while an
absolute model has to reproduce the position itself. Any N1.7-vs-N1.5 gap is
therefore part backbone, part parameterization, and the board should say so.

This quantifies the parameterization head start with NO model at all, by scoring
constant predictors -- the best a model can do while learning only the marginal
distribution of its target, never looking at the image:

    hold-state          pred = current state                    (the pass line)
    const-absolute      pred = per-(step,joint) TRAIN mean action
    const-relative      pred = state + per-(step,joint) TRAIN mean delta

Constants come from the 111 TRAIN episodes only and are scored on the same 1397
val anchors and K=8 prefix as score.py, so the numbers sit on the leaderboard's
own scale. A large const-relative advantage means part of any N1.7 win is the
action space, not the backbone.
"""

import glob
import json
import os
import pathlib

import numpy as np
import pandas as pd

# 仓库自包含:从本文件位置推导。bench/scripts/score/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
DS = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
H = 16  # training horizon; scored on the first K


def load_episodes():
    """episode_index -> (state (L,8), action (L,8)) straight from parquet."""
    out = {}
    for p in sorted(glob.glob(str(DS / "data" / "chunk-*" / "*.parquet"))):
        df = pd.read_parquet(p, columns=["episode_index", "frame_index",
                                         "observation.state", "action"])
        for ep, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            st = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
            ac = np.stack(g["action"].to_numpy()).astype(np.float64)
            out[int(ep)] = (st, ac)
    return out


def train_constants(eps, train_ids):
    """Per-(horizon step, joint) mean of the absolute action and of the delta.

    Uses every legal training anchor (the same half-open windowing the trainer's
    effective episode length implies), TRAIN episodes only.
    """
    abs_sum = np.zeros((H, 8))
    rel_sum = np.zeros((H, 8))
    n = 0
    for ep in train_ids:
        st, ac = eps[ep]
        L = len(ac)
        for t in range(0, L - H + 1):
            chunk = ac[t:t + H]                 # (H,8)
            abs_sum += chunk
            rel_sum += chunk - st[t][None, :]   # delta from the anchor state
            n += 1
    assert n > 0
    return abs_sum / n, rel_sum / n, n


def main():
    split = json.loads((BENCH / "data" / "split.json").read_text())
    train_ids = sorted(split["train_episodes"])
    val_ids = set(split["val_episodes"])

    A = np.load(BENCH / "data" / "val_anchors.npz")
    gt = A["gt"].astype(np.float64)        # (N,K,8)
    state = A["state"].astype(np.float64)  # (N,8)
    N, K, D = gt.shape
    print(f"val anchors: N={N} K={K} D={D}")

    eps = load_episodes()
    assert not (set(train_ids) & val_ids)
    missing = [e for e in train_ids if e not in eps]
    assert not missing, f"train episodes absent from parquet: {missing}"

    mean_abs, mean_rel, n_anchor = train_constants(eps, train_ids)
    print(f"constants fitted on {len(train_ids)} TRAIN episodes / {n_anchor} anchors "
          f"(val episodes never touched)")

    def score(pred, name):
        err = np.abs(pred[:, :K, :] - gt)
        mae, arm, grip = err.mean(), err[..., :7].mean(), err[..., 7].mean()
        pj = " ".join(f"{v:5.2f}" for v in err.mean(axis=(0, 1)))
        print(f"{name:<24}{mae:8.3f}{arm:10.3f}{grip:9.3f}   {pj}")
        return mae

    print(f"\n{'predictor':<24}{'MAE':>8}{'arm(0-6)':>10}{'grip(7)':>9}   per-joint j0..j7")
    hold = np.broadcast_to(state[:, None, :], (N, K, D))
    m_hold = score(hold, "hold-state (pass line)")
    m_abs = score(np.broadcast_to(mean_abs[None, :K, :], (N, K, D)), "const-absolute")
    m_rel = score(state[:, None, :] + mean_rel[None, :K, :], "const-relative")

    # The N1.7 recipe is relative ARM + absolute GRIPPER, so the mixed row is the
    # one that actually corresponds to a marginal-only N1.7.
    mixed = np.empty((N, K, D))
    mixed[..., :7] = state[:, None, :7] + mean_rel[None, :K, :7]
    mixed[..., 7] = np.broadcast_to(mean_abs[None, :K, 7], (N, K))
    m_mix = score(mixed, "const-N1.7-mixed")

    print(f"\nparameterization head start (lower MAE = easier target space):")
    print(f"  const-absolute / hold-state : {m_abs / m_hold:6.2f}x")
    print(f"  const-relative / hold-state : {m_rel / m_hold:6.2f}x")
    print(f"  const-absolute / const-relative = {m_abs / m_rel:.2f}x")
    print("\nA marginal-only model in ABSOLUTE space scores "
          f"{m_abs:.3f}; the same model in the N1.7 mixed space scores {m_mix:.3f}. "
          "That gap is available to N1.7 without any visual competence, so an "
          "N1.7-over-N1.5 margin smaller than it is NOT evidence about backbones.")


if __name__ == "__main__":
    main()
