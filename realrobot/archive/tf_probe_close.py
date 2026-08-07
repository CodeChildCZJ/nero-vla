#!/usr/bin/env python3
"""离线探针: 喂训练'张爪后即将闭爪'的obs给ckpt, 看模型chunk会不会命令闭爪(grip绝对维下穿50)+肘推(j2)。
判别真机'张爪后卡住不闭'是: 腕OOD/闭环(离线在分布内会闭) 还是 闭信号没学好(离线也不闭)。
用法: CUDA_VISIBLE_DEVICES=3 XLA_..MEM_FRACTION=0.25 uv run python tf_probe_close.py <ckpt_dir> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd, av
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else "v7"
CFG="pi05_nero_pick_pink_sponge_v7"; DS=f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v7"
PROMPT="pick the pink sponge and place it in the blue bucket"; J2,GRIP=1,7

def dec(mp4, idxs):
    want=set(int(i) for i in idxs); out={}; i=0; c=av.open(mp4)
    for f in c.decode(c.streams.video[0]):
        if i in want: out[i]=f.to_ndarray(format="rgb24"); want.discard(i)
        if not want: break
        i+=1
    c.close(); return out

print(f"[load] {CKPT}")
policy=pc.create_trained_policy(_config.get_config(CFG), CKPT, default_prompt=PROMPT)
eps=sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
rows=[]
for ep in eps:
    epi=int(ep.split("episode_")[-1].split(".")[0])
    df=pd.read_parquet(ep); A=np.stack(df["action"].values).astype(np.float32)
    S=np.stack(df["observation.state"].values).astype(np.float32); g=A[:,GRIP]
    pk=int(np.argmax(g)); after=g[pk:]; dn=np.where(after<50)[0]
    if not len(dn): continue
    cf=pk+int(dn[0])  # 训练闭爪帧
    # 喂'张爪保持期, 闭爪前': cf-8, cf-4, cf-1 (此时训练obs下模型应命令闭)
    samples={f"close-{k}":max(pk,cf-k) for k in (8,4,1)}
    hi=f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wr=f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    H=dec(hi,samples.values()); W=dec(wr,samples.values())
    for lab,t in samples.items():
        if t not in H or t not in W: continue
        ch=np.asarray(policy.infer({"observation/image":H[t],"observation/wrist_image":W[t],
            "observation/state":S[t],"prompt":PROMPT})["actions"])
        rows.append({"ep":epi,"lab":lab,"t":t,
            "state_grip":float(S[t,GRIP]),"train_grip_now":float(A[t,GRIP]),
            "train_grip_+15":float(A[min(t+15,len(A)-1),GRIP]),
            "pred_grip_chunk_min":float(ch[:,GRIP].min()),"pred_grip_chunk_end":float(ch[-1,GRIP]),
            "pred_j2_motion":float(ch[-1,J2]-ch[0,J2])})
R=pd.DataFrame(rows); R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_probe_close_{TAG}.json",orient="records")
print("="*70); print(f"  闭爪探针 tag={TAG}  ckpt={CKPT.split('/')[-1]}  n={len(R)}")
print("="*70)
print(f"  喂'训练张爪保持期(闭爪前1-8帧)'obs:")
print(f"  state_grip(当前张着)均={R.state_grip.mean():.0f}  训练此刻grip={R.train_grip_now.mean():.0f}→+15帧后={R['train_grip_+15'].mean():.0f}(训练在闭)")
print(f"  ★模型chunk grip最小值均={R.pred_grip_chunk_min.mean():.0f} (若<50=模型命令闭爪commit; 若~90=模型也不闭只张着)")
print(f"  模型命令'闭爪'(chunk有grip<50)的样本占比 = {100*(R.pred_grip_chunk_min<50).mean():.0f}%")
print(f"  模型chunk j2(肘)净动={R.pred_j2_motion.mean():+.1f} (训练张→闭要+6.8)")
print("="*70)
