#!/usr/bin/env python3
"""CPU smoke test for the explicit-noise path used by predict_openpi.py.

Checks, on a single real b2 observation:
  1. policy.infer(obs, noise=n) accepts an explicit (horizon, action_dim) noise chunk;
  2. same noise -> bit-identical actions (so preds/ is reproducible from the seed alone);
  3. different noise -> different actions (so the 5-seed variance footnote measures something);
  4. no-noise infer twice -> different actions (confirms the default path is stochastic,
     i.e. the footnote is necessary rather than decorative).

Runs on CPU on purpose: GPU5 is busy with the pi05_nero_b2_train run and must not be disturbed.
"""
import os
import pathlib, time

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
CFG_NAME = "pi05_nero_b2"  # leaky 131-ep delta ckpt, used only as a plumbing fixture
CKPT = os.path.join(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints")), "pi05_nero_b2", "b2_delta", "29999")
PROMPT = "pick the pink sponge and place it in the blue bucket"

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

import sys
sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir
from predict_openpi import to_hwc_uint8  # noqa: E402

A = np.load(BENCH / "data" / "val_anchors.npz")
g = 0
e, t = int(A["episodes"][g]), int(A["frames"][g])
ds = LeRobotDataset("local/pick_pink_sponge_b2", episodes=[e])
item = ds[t]
obs = {
    "observation/image": to_hwc_uint8(item["observation.images.cam_high"]),
    "observation/wrist_image": to_hwc_uint8(item["observation.images.cam_wrist"]),
    "observation/state": A["state"][g].astype(np.float32),
    "prompt": PROMPT,
}
print(f"obs ready: ep{e} t={t}", flush=True)

cfg = _config.get_config(CFG_NAME)
policy = _policy_config.create_trained_policy(cfg, CKPT)
h, adim = cfg.model.action_horizon, cfg.model.action_dim
print(f"policy ready h={h} adim={adim} devices={jax.devices()}", flush=True)


def noise_for(seed, idx):
    return np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(seed), idx), (h, adim), dtype=jnp.float32))


def run(tag, **kw):
    t0 = time.monotonic()
    a = np.asarray(policy.infer(obs, **kw)["actions"], dtype=np.float32)
    print(f"  {tag}: {time.monotonic()-t0:.0f}s shape={a.shape} first={a[0][:4]}", flush=True)
    return a


n0, n1 = noise_for(0, g), noise_for(3, g)
a0 = run("seed0 #1", noise=n0)
a0b = run("seed0 #2", noise=n0)
a1 = run("seed3", noise=n1)
d0 = run("no-noise #1")
d1 = run("no-noise #2")

print("\n--- verdict ---")
print(f"[1] explicit noise accepted        : OK, actions {a0.shape}")
print(f"[2] same noise reproducible        : max|diff|={np.abs(a0-a0b).max():.3e}"
      f"  -> {'OK' if np.array_equal(a0, a0b) else 'NOT BIT-IDENTICAL'}")
print(f"[3] different noise -> different   : mean|diff|={np.abs(a0-a1).mean():.4f}"
      f"  -> {'OK' if np.abs(a0-a1).mean() > 1e-4 else 'SUSPICIOUS (noise ignored?)'}")
print(f"[4] default path stochastic        : mean|diff|={np.abs(d0-d1).mean():.4f}"
      f"  -> {'stochastic (footnote needed)' if np.abs(d0-d1).mean() > 1e-4 else 'deterministic'}")
