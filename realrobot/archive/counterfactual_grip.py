#!/usr/bin/env python3
"""反事实判决: causal confusion (复读state.grip) vs vision-driven (从视觉闭爪).
取训练帧 (视觉明确), 篡改输入 state.grip 成 {5,35,65}, 看 pred_grip0 跟谁:
- pred ≈ 注入的 state.grip (不管图)  → 复读 state = CAUSAL CONFUSION 铁证
- pred ≈ 视觉该有的值 (grasp帧≈13 不管注入)  → vision driven, 另寻因
"""
import os
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import numpy as np, pandas as pd
from openpi.training import config as _config
from openpi.policies import policy_config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

CKPT = f"{os.environ.get('NERO_CKPT', 'checkpoints')}/pi05_nero_pick_pink_sponge_v5/v5/4000"
PROMPT = "pick the pink sponge and place it in the blue bucket"
GRIP = 7
cfg = _config.get_config("pi05_nero_pick_pink_sponge_v5")
policy = policy_config.create_trained_policy(cfg, CKPT)
ds = lerobot_dataset.LeRobotDataset("local/pick_pink_sponge_v3")
df = pd.read_parquet(f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000/episode_000025.parquet")
gidx = df["index"].values
act = np.stack(df["action"].values).astype(float)

# 选两类帧: grasp帧(视觉=该闭, rec~13) + approach帧(视觉=开/接近, rec~70)
FRAMES = {
    "approach(视觉=开,rec~70)": [135, 140, 145],
    "grasp(视觉=该闭,rec~13)": [168, 171, 174],
}
INJECT = [5.0, 35.0, 65.0]   # 篡改 state.grip 成这些值

def to_img(x):
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)

print(f"反事实测试 ckpt_4000: 每帧把输入 state.grip 篡改成 {INJECT}, 看 pred_grip0\n")
for label, frames in FRAMES.items():
    print(f"=== {label} ===")
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
            st = base["observation/state"].copy()
            st[GRIP] = inj
            obs = dict(base); obs["observation/state"] = st
            p = float(np.asarray(policy.infer(obs)["actions"])[0, GRIP])
            preds.append(p)
        # 判: pred 是否随注入值线性变 (复读) 还是稳定在视觉值 (driven)
        spread = max(preds) - min(preds)
        tag = "→复读state(随注入大幅变)" if spread > 25 else "→视觉driven(注入无关,稳定)" if spread < 8 else "→混合"
        print(f"  f{f} (rec_grip={rec:.0f}): 注入5→{preds[0]:.0f}  注入35→{preds[1]:.0f}  注入65→{preds[2]:.0f}  | spread={spread:.0f} {tag}")
    print()
print("[判据] grasp帧若 pred 随注入从~5变到~65 = 模型无视'视觉该闭'只复读state = CAUSAL CONFUSION 铁证")
