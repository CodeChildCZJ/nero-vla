#!/usr/bin/env python3
"""B 离线 grasp-close 真测 (修正夹爪约定: 0mm=闭 / 高 mm=张)。
锚到【真抓取帧 = mm 张开后首次掉回低带 @~48%】, 喂真 demo 帧给模型, 看 chunk 夹爪输出闭(→0)还是张(→高)。
也喂 approach-open 帧(~35%)对照。区分: 模型在 demo 流形上会不会 grasp-close
  (会闭=真机失败=闭环漂离流形; 不会闭=根本没学会 grasp-close)。
用法: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
      HF_HUB_OFFLINE=1 ${NERO_ROOT}/third_party/openpi-agilex/.venv/bin/python3 tf_grip_b.py <ckpt> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd, av
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else "b1"
CFG = "pi05_nero_b1"
DS  = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, J7, GRIP = 3, 6, 7
N_EP = 20

def decode(mp4, idxs):
    want = set(int(i) for i in idxs); out = {}; i = 0
    c = av.open(mp4)
    for fr in c.decode(c.streams.video[0]):
        if i in want:
            out[i] = fr.to_ndarray(format="rgb24"); want.discard(i)
            if not want: break
        i += 1
    c.close(); return out

print(f"[load] {CKPT}")
policy = pc.create_trained_policy(_config.get_config(CFG), CKPT, default_prompt=PROMPT)
print("[ready]")
eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))[:N_EP]
rows = []
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    g = A[:, GRIP]; T = len(g); lo, hi = float(g.min()), float(g.max()); rng = max(1e-6, hi - lo)
    op = np.where(g > lo + 0.75 * rng)[0]; cl = np.where(g < lo + 0.25 * rng)[0]
    if len(op) == 0: continue
    t_o = int(op[0]); t_g = -1
    for t in cl:
        if t > t_o: t_g = int(t); break
    if t_g < 0: continue
    hi_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wr_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    H = decode(hi_mp4, {t_o, t_g}); W = decode(wr_mp4, {t_o, t_g})
    for name, t in [("open", t_o), ("grasp", t_g)]:
        if t not in H or t not in W: continue
        ch = np.asarray(policy.infer({"observation/image": H[t], "observation/wrist_image": W[t],
            "observation/state": S[t], "prompt": PROMPT})["actions"])
        rows.append({"ep": epi, "kind": name, "t": t, "train_grip": float(A[t, GRIP]),
            "pred_grip0": float(ch[0, GRIP]), "pred_grip_min": float(ch[:, GRIP].min()),
            "pred_grip_max": float(ch[:, GRIP].max()), "train_j4": float(A[t, J4]), "pred_j4": float(ch[0, J4])})
    print(f"ep{epi} t_o={t_o} t_g={t_g} done")

R = pd.DataFrame(rows)
R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_grip_{TAG}.json", orient="records")
print("\n" + "=" * 68)
print(f"  B 离线 grasp-close 真测  tag={TAG}  (夹爪 0=闭 / 高=张)")
print("=" * 68)
for k in ["open", "grasp"]:
    sub = R[R.kind == k]
    if len(sub) == 0: continue
    print(f"  [{k:5s}] n={len(sub)}  train_grip mean={sub.train_grip.mean():5.1f}  "
          f"pred_grip0 mean={sub.pred_grip0.mean():5.1f}  pred_min={sub.pred_grip_min.mean():5.1f}  "
          f"pred_max={sub.pred_grip_max.mean():5.1f}")
print("=" * 68)
print("  ★判读: grasp 帧 train_grip≈0(闭). pred_grip0 也≈0 → 模型流形上会闭(真机失败=闭环漂);")
print("         pred_grip0 高(→张) → 模型根本没学会 grasp-close.")
print("=" * 68)
