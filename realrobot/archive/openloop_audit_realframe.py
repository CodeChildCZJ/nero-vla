#!/usr/bin/env python3
"""真机帧对照 audit: 把真机失败run的逐帧 obs (NPZ: 图+state) 喂 ckpt,
看预测 grip 闭不闭。对照训练帧版 (openloop_audit.py):
  训练帧→闭(13mm) + 真机帧→张着不闭(>40)  ==>  直接坐实 OOD。

用法: CUDA_VISIBLE_DEVICES=3 uv run python openloop_audit_realframe.py <npz> [ckpt_dir]
NPZ 期望含: cam_high/cam_wrist 图序列 (T,H,W,3) 或 (T,3,H,W); state (T,8) 或 (T,16)。
key 名自动探测 (high/wrist/state 子串), 探不到就打印所有 key 让我改。
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import sys, json
import numpy as np

NPZ = sys.argv[1]
CKPT = sys.argv[2] if len(sys.argv) > 2 else f"{os.environ.get('NERO_CKPT', 'checkpoints')}/pi05_nero_pick_pink_sponge_v6_gmask/v6_gmask/7999"
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "pi05_nero_pick_pink_sponge_v6_gmask"
PROMPT = "pick the pink sponge and place it in the blue bucket"
GRIP, J4 = 7, 3
step = CKPT.rstrip("/").split("/")[-1]

z = np.load(NPZ, allow_pickle=True)
keys = list(z.keys())
print(f"[realframe] NPZ keys: {[(k, np.asarray(z[k]).shape) for k in keys]}", flush=True)


def pick(*subs):
    for k in keys:
        kl = k.lower()
        if all(s in kl for s in subs):
            return k
    return None

k_high = pick("high") or pick("cam_high") or pick("base") or pick("top")
k_wrist = pick("wrist")
k_state = pick("state") or pick("qpos") or pick("joint")
print(f"[realframe] mapped: high={k_high} wrist={k_wrist} state={k_state}", flush=True)
if not (k_high and k_wrist and k_state):
    print("[realframe] !! key 探测失败, 上面是全部 key, 手动改脚本 pick() 即可。", flush=True)
    sys.exit(1)

highs = np.asarray(z[k_high]); wrists = np.asarray(z[k_wrist]); states = np.asarray(z[k_state], dtype=np.float32)
T = len(highs)
print(f"[realframe] T={T}  high{highs.shape} wrist{wrists.shape} state{states.shape}", flush=True)

from openpi.training import config as _config
from openpi.policies import policy_config
cfg = _config.get_config(CONFIG)
policy = policy_config.create_trained_policy(cfg, CKPT)
print(f"[realframe] policy loaded (ckpt step {step})", flush=True)


def fix_img(a):
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] == 3:  # CHW->HWC
        a = np.transpose(a, (1, 2, 0))
    if np.issubdtype(a.dtype, np.floating):
        a = (255 * a).astype(np.uint8) if a.max() <= 1.01 else a.astype(np.uint8)
    return a


STRIDE = int(os.environ.get("STRIDE", "1"))
rows = []
for f in range(0, T, STRIDE):
    st = states[f]
    if st.shape[0] < 16:  # pad to 16 (_NeroInputs 取前8, 但 state key 期望够长)
        st = np.concatenate([st, np.zeros(16 - st.shape[0], np.float32)])
    obs = {
        "observation/image": fix_img(highs[f]),
        "observation/wrist_image": fix_img(wrists[f]),
        "observation/state": st.astype(np.float32),
        "prompt": PROMPT,
    }
    pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
    rows.append({
        "f": f,
        "state_grip": round(float(states[f][GRIP]), 1) if states.shape[1] > GRIP else None,
        "pred_grip0": round(float(pred[0, GRIP]), 1),
        "pred_grip_min": round(float(pred[:, GRIP].min()), 1),
        "state_j4": round(float(states[f][J4]), 1) if states.shape[1] > J4 else None,
        "pred_j4_0": round(float(pred[0, J4]), 1),
    })
    if f % 20 == 0:
        print(f"  {f}/{T}", flush=True)

print(f"\n{'f':>4} {'st_grip':>8} {'pred_g0':>8} {'pred_gmin':>10} | {'st_j4':>7} {'pred_j4':>8}", flush=True)
for r in rows:
    print(f"{r['f']:>4} {str(r['state_grip']):>8} {r['pred_grip0']:>8} {r['pred_grip_min']:>10} | {str(r['state_j4']):>7} {r['pred_j4_0']:>8}", flush=True)

pg0 = np.array([r["pred_grip0"] for r in rows])
pgm = np.array([r["pred_grip_min"] for r in rows])
print(f"\n[判据] 真机帧喂模型: 预测chunk0 grip 全程 min={pg0.min():.1f} max={pg0.max():.1f} 均值={pg0.mean():.1f}", flush=True)
print(f"[判据] chunk内最小 grip 全程 min={pgm.min():.1f}", flush=True)
verdict = "真机帧也闭(<=20)→OOD不成立, 另寻因" if pg0.min() <= 20 else "真机帧张着不闭(>=35)→训练闭/真机不闭, **OOD坐实**" if pg0.min() >= 35 else "真机帧半闭(20~35)→部分OOD"
print(f"[判据] {verdict}", flush=True)

outp = f"{os.environ.get('NERO_ROOT', '.')}/realrobot/openloop_realframe_{step}.json"
with open(outp, "w") as f:
    json.dump(rows, f)
print(f"[realframe] 落 {outp}", flush=True)
