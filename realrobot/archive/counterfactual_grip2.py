#!/usr/bin/env python3
"""反事实自检 (参数化 config+ckpt): 验 mask 是否破了 chunk[0] 复读 state.grip.
v6(masked): 注入 state.grip {5,35,65} 应**完全不影响**输出 (spread→0) = 破了复读 ✅
v5(unmasked对照): 输出随注入 1:1 变 (spread~55) = 还在复读

用法: CUDA_VISIBLE_DEVICES=3 uv run python counterfactual_grip2.py <config_name> <ckpt_dir>
"""
import os, sys
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import numpy as np, pandas as pd
from openpi.training import config as _config
from openpi.policies import policy_config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

CONFIG = sys.argv[1]
CKPT = sys.argv[2]
PROMPT = "pick the pink sponge and place it in the blue bucket"
GRIP = 7
step = CKPT.rstrip("/").split("/")[-1]
print(f"[反事实自检] config={CONFIG} ckpt={CKPT} step={step}", flush=True)
cfg = _config.get_config(CONFIG)
print(f"  mask_gripper_state = {getattr(cfg.data, 'mask_gripper_state', 'n/a')}", flush=True)
policy = policy_config.create_trained_policy(cfg, CKPT)
ds = lerobot_dataset.LeRobotDataset("local/pick_pink_sponge_v3")
df = pd.read_parquet(f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000/episode_000025.parquet")
gidx = df["index"].values
act = np.stack(df["action"].values).astype(float)

FRAMES = {"approach(rec~70开)": [135, 140, 145], "grasp(rec~13该闭)": [168, 171, 174]}
INJECT = [5.0, 35.0, 65.0]

def to_img(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)

spreads = []
for label, frames in FRAMES.items():
    print(f"\n=== {label} ===", flush=True)
    for f in frames:
        frame = ds[int(gidx[f])]
        base = {
            "observation/image": to_img(frame["observation.images.cam_high"]),
            "observation/wrist_image": to_img(frame["observation.images.cam_wrist"]),
            "observation/state": np.asarray(frame["observation.state"], dtype=np.float32),
            "prompt": PROMPT,
        }
        rec = act[f, GRIP]
        preds = []
        for inj in INJECT:
            st = base["observation/state"].copy(); st[GRIP] = inj
            obs = dict(base); obs["observation/state"] = st
            preds.append(float(np.asarray(policy.infer(obs)["actions"])[0, GRIP]))
        spread = max(preds) - min(preds); spreads.append(spread)
        tag = "✅破复读(注入无关)" if spread < 8 else "❌还在复读" if spread > 25 else "半破"
        print(f"  f{f}(rec={rec:.0f}): 注入5→{preds[0]:.0f} 35→{preds[1]:.0f} 65→{preds[2]:.0f} | spread={spread:.0f} {tag}", flush=True)

ms = float(np.mean(spreads))
print(f"\n[判据] 平均spread={ms:.1f}mm", flush=True)
print(f"[判据] {'✅✅ MASK生效: 输出不随注入state.grip变, chunk[0]复读已破' if ms<8 else '❌ MASK没生效, 还在复读 state.grip' if ms>25 else '⚠️ 半破, 需查'}", flush=True)
print(f"[对比] v5未mask时此值~55 (1:1复读)。现在={ms:.1f}", flush=True)
