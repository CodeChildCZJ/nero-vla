#!/usr/bin/env python3
"""决定性测试: 把真机帧亮度匹配到训练帧, 看v6预测grip是否更闭.
若调亮后该闭的帧从'开'变'闭' = 光照是病B主因 = 加灯即可修(不重训).
用法: CUDA_VISIBLE_DEVICES=3 uv run python brighten_test.py <obs_dump.npz>
"""
import os, sys
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import numpy as np, pandas as pd
from openpi.training import config as _config
from openpi.policies import policy_config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

NPZ = sys.argv[1]
CKPT = f"{os.environ.get('NERO_CKPT', 'checkpoints')}/pi05_nero_pick_pink_sponge_v6_gmask/v6_gmask/7999"
PROMPT = "pick the pink sponge and place it in the blue bucket"
z = np.load(NPZ, allow_pickle=True)
highs, wrists, states = np.asarray(z["cam_high"]), np.asarray(z["cam_wrist"]), np.asarray(z["state"])
T = len(highs)

# 训练 grasp 帧 ep25 f164 的每通道均值 (目标亮度)
ds = lerobot_dataset.LeRobotDataset("local/pick_pink_sponge_v3")
df = pd.read_parquet(f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000/episode_000025.parquet")
tf = ds[int(df["index"].values[164])]
def to_hwc(x):
    a = np.asarray(x.numpy() if hasattr(x,"numpy") else x)
    if a.ndim==3 and a.shape[0]==3: a=np.transpose(a,(1,2,0))
    return ((255*a).astype(np.uint8) if (a.dtype!=np.uint8 and a.max()<=1.01) else a.astype(np.uint8))
th, tw = to_hwc(tf["observation.images.cam_high"]), to_hwc(tf["observation.images.cam_wrist"])
tgt_h = [th[...,c].mean() for c in range(3)]; tgt_w = [tw[...,c].mean() for c in range(3)]
print(f"训练grasp帧目标亮度 high{[round(x) for x in tgt_h]} wrist{[round(x) for x in tgt_w]}")

def match(img, tgt):
    out = img.astype(np.float32)
    for c in range(3):
        out[...,c] += (tgt[c] - out[...,c].mean())
    return np.clip(out, 0, 255).astype(np.uint8)

cfg = _config.get_config("pi05_nero_pick_pink_sponge_v6_gmask")
policy = policy_config.create_trained_policy(cfg, CKPT)

# 选代表帧: 全程均匀 + arm最深(j4最小)几帧 (最可能grasp时刻)
j4 = states[:,3]
deep = np.argsort(j4)[:5].tolist()
sample = sorted(set(list(range(0,T,max(1,T//8))) + deep))
print(f"\n{'帧':>5} {'j4':>5} {'state.grip':>9} | {'pred原图':>8} {'pred调亮':>8}  判")
for f in sample:
    base_state = states[f].astype(np.float32)
    def infer(hi, wr):
        obs={"observation/image":hi,"observation/wrist_image":wr,
             "observation/state":base_state,"prompt":PROMPT}
        return float(np.asarray(policy.infer(obs)["actions"])[0,7])
    p_raw = infer(highs[f], wrists[f])
    p_br  = infer(match(highs[f],tgt_h), match(wrists[f],tgt_w))
    tag = "调亮后更闭" if p_br < p_raw-8 else "调亮后更开" if p_br>p_raw+8 else "≈"
    print(f"{f:>5} {j4[f]:>5.0f} {states[f,7]:>9.0f} | {p_raw:>8.0f} {p_br:>8.0f}  {tag}", flush=True)
print("\n[判据] 若多帧'调亮后更闭'(尤其低j4深扎帧) = 光照是病B主因 = 加灯匹配训练亮度即可修, 不用重训/重采数据")
print("[判据] 若调亮没让它更闭 = 不只光照(还有位置/画面内容OOD) = 需重采真机条件数据")
