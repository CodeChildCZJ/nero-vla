#!/usr/bin/env python3
"""离线探针: 喂训练approach帧, 看6k模型命令的腕j7(idx6)是否本身带偏(往高roll命令) vs 纯闭环累积。
若命令j7运动方向≈训练=模型在分布内不带偏(真机漂=闭环累积/图像OOD); 若命令系统性比训练高=模型bias。
用法: CUDA_VISIBLE_DEVICES=3 ...MEM_FRACTION=0.25 uv run python tf_probe_j7.py <ckpt> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd, av
from openpi.policies import policy_config as pc
from openpi.training import config as _config
CKPT=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else "v7"
CFG="pi05_nero_pick_pink_sponge_v7"; DS=f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v7"
PROMPT="pick the pink sponge and place it in the blue bucket"; J7,GRIP=6,7
def dec(mp4,idxs):
    want=set(int(i) for i in idxs);out={};i=0;c=av.open(mp4)
    for f in c.decode(c.streams.video[0]):
        if i in want:out[i]=f.to_ndarray(format="rgb24");want.discard(i)
        if not want:break
        i+=1
    c.close();return out
policy=pc.create_trained_policy(_config.get_config(CFG),CKPT,default_prompt=PROMPT)
eps=sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet")); rows=[]
for ep in eps:
    epi=int(ep.split("episode_")[-1].split(".")[0])
    df=pd.read_parquet(ep);A=np.stack(df["action"].values).astype(np.float32)
    S=np.stack(df["observation.state"].values).astype(np.float32);g=A[:,GRIP]
    pk=int(np.argmax(g))  # 张爪峰=approach终点附近
    # approach: 从开始到张爪, 每25帧采
    fr=[t for t in range(10,pk,25)]
    if len(fr)<2: continue
    hi=f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wr=f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    H=dec(hi,fr);W=dec(wr,fr)
    for t in fr:
        if t not in H or t not in W: continue
        ch=np.asarray(policy.infer({"observation/image":H[t],"observation/wrist_image":W[t],
            "observation/state":S[t],"prompt":PROMPT})["actions"])
        rows.append({"ep":epi,"t":t,"state_j7":float(S[t,J7]),
            "train_j7_now":float(A[t,J7]),"pred_j7_0":float(ch[0,J7]),
            "train_j7_motion":float(A[min(t+15,len(A)-1),J7]-A[t,J7]),
            "pred_j7_motion":float(ch[-1,J7]-ch[0,J7])})
R=pd.DataFrame(rows);R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_probe_j7_{TAG}.json",orient="records")
print("="*68);print(f"  腕j7 approach探针 tag={TAG} ckpt={CKPT.split('/')[-1]} n={len(R)}");print("="*68)
print(f"  approach帧 j7绝对: 训练={R.train_j7_now.mean():.1f} 模型chunk[0]={R.pred_j7_0.mean():.1f} Δ={R.pred_j7_0.mean()-R.train_j7_now.mean():+.1f}")
print(f"  ★j7运动方向(chunk内, +=继续roll高): 训练={R.train_j7_motion.mean():+.1f} 模型={R.pred_j7_motion.mean():+.1f}")
d=R.pred_j7_motion.mean()-R.train_j7_motion.mean()
print(f"  模型-训练 j7运动偏差={d:+.1f}  →  {'模型系统性多roll(bias, 多训步未必够要数据)' if d>5 else ('模型少roll' if d<-5 else '模型j7运动≈训练(分布内不带偏→真机漂=闭环累积, 多训步可能够)')}")
print("="*68)
