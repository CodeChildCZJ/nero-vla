#!/usr/bin/env python3
"""决定性测试: grasp_frames 真机帧, (a)state.j4原样 vs (b)state.j4改102, 看v6预测grip闭不闭.
(b)闭 = 闭爪被关节pose gate住, descend辅助压到102就能抓, 不用重采数据.
(b)不闭 = 正确pose真机视觉也不触发 = 必须diverse数据.
注: v6 mask了夹爪state(dim7→0), 但j4(dim3)不mask, 所以改j4真影响模型.
用法: CUDA_VISIBLE_DEVICES=3 uv run python inject_j4_test.py <grasp_frames.npz>
"""
import os, sys
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config

NPZ = sys.argv[1]
CKPT = f"{os.environ.get('NERO_CKPT', 'checkpoints')}/pi05_nero_pick_pink_sponge_v6_gmask/v6_gmask/7999"
PROMPT = "pick the pick the pink sponge and place it in the blue bucket".replace("pick the pick the", "pick the")
J4, GRIP = 3, 7
z = np.load(NPZ, allow_pickle=True)
keys = list(z.keys())
def pick(*subs):
    for k in keys:
        if all(s in k.lower() for s in subs): return k
    return None
kh, kw, ks = pick("high") or pick("base"), pick("wrist"), pick("state")
kr = pick("raw")
print(f"[inject] NPZ keys={[(k,np.asarray(z[k]).shape) for k in keys]}")
highs, wrists, states = np.asarray(z[kh]), np.asarray(z[kw]), np.asarray(z[ks], dtype=np.float32)
raw = np.asarray(z[kr], dtype=np.float32) if kr else None
T = len(highs)

cfg = _config.get_config("pi05_nero_pick_pink_sponge_v6_gmask")
policy = policy_config.create_trained_policy(cfg, CKPT)
print(f"[inject] policy loaded, {T}帧")

def fix_img(a):
    a=np.asarray(a)
    if a.ndim==3 and a.shape[0]==3: a=np.transpose(a,(1,2,0))
    if np.issubdtype(a.dtype,np.floating): a=(255*a).astype(np.uint8) if a.max()<=1.01 else a.astype(np.uint8)
    return a

def infer(hi, wr, st):
    obs={"observation/image":fix_img(hi),"observation/wrist_image":fix_img(wr),
         "observation/state":st.astype(np.float32),"prompt":PROMPT}
    return float(np.asarray(policy.infer(obs)["actions"])[0,GRIP])

print(f"\n{'帧':>4} {'真j4':>6} {'raw.grip':>8} | {'(a)原样':>8} {'(b)j4=102':>10}  判")
a_list=[]; b_list=[]
for f in range(T):
    st=states[f]
    rg = raw[f,GRIP] if raw is not None else float('nan')
    pa = infer(highs[f], wrists[f], st.copy())
    st_b = st.copy(); st_b[J4]=102.0
    pb = infer(highs[f], wrists[f], st_b)
    a_list.append(pa); b_list.append(pb)
    tag = "(b)闭<25" if pb<25 else "(b)半" if pb<40 else "(b)开"
    print(f"{f:>4} {st[J4]:>6.0f} {rg:>8.0f} | {pa:>8.0f} {pb:>10.0f}  {tag}", flush=True)

a=np.array(a_list); b=np.array(b_list)
print(f"\n[结果] (a)原样state.j4~90: grip min{a.min():.0f} max{a.max():.0f} mean{a.mean():.0f}")
print(f"[结果] (b)state.j4=102:    grip min{b.min():.0f} max{b.max():.0f} mean{b.mean():.0f}")
print(f"[结果] j4=102后 平均变化: {b.mean()-a.mean():+.1f}mm, (b)闭合(<25)帧: {(b<25).sum()}/{T}")
if (b<25).mean()>0.5:
    print("[判决] ✅✅ (b)多数闭 → 闭爪被j4 pose gate住! descend压到102就能抓, **今天能成, 不用重采数据**")
elif b.mean() < a.mean()-15:
    print("[判决] 🟡 (b)明显更闭但不全 → j4 pose部分gate, descend辅助+可能微调")
else:
    print("[判决] ❌ (b)也不闭 → 正确pose真机视觉也不触发闭爪 → 必须diverse数据重训")
