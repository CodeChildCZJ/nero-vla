import json, glob, os, numpy as np, pandas as pd, pathlib
# 仓库自包含:从本文件位置推导。bench/scripts/prepare/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
B2 = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
K = 8          # common eval horizon (min across all backbones)
STRIDE = 5     # subsample anchors within each val episode
split = json.loads((BENCH/"data"/"split.json").read_text())
val_eps = split["val_episodes"]

def ep_parquet(e):
    return B2/f"data/chunk-000/episode_{e:06d}.parquet"

eps, frames, gts, states = [], [], [], []
for e in val_eps:
    df = pd.read_parquet(ep_parquet(e))
    act = np.stack(df["action"].to_numpy())      # (L,8)
    st  = np.stack(df["observation.state"].to_numpy())  # (L,8)
    L = len(act)
    for t in range(0, L-K, STRIDE):
        eps.append(e); frames.append(t)
        gts.append(act[t:t+K])       # (K,8) future action targets
        states.append(st[t])          # (8,) current state at anchor
eps = np.array(eps, np.int32); frames = np.array(frames, np.int32)
gts = np.stack(gts).astype(np.float32)      # (A,K,8)
states = np.stack(states).astype(np.float32) # (A,8)
np.savez(BENCH/"data"/"val_anchors.npz", episodes=eps, frames=frames, gt=gts, state=states, K=K, stride=STRIDE)
print("anchors:", len(eps), "| val episodes:", len(val_eps), "| gt", gts.shape, "| K", K)
print("action range per joint (min..max over gt):")
for j in range(8):
    print(f"  j{j}: {gts[...,j].min():8.2f} .. {gts[...,j].max():8.2f}")
