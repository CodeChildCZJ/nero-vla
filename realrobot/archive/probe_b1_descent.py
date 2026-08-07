#!/usr/bin/env python3
"""探测 B-3k 下扎驱动: 喂 state.j4 低(approach/未下扎)的帧, 看模型 chunk 会不会命令 j4 上升到~93.
关键诊断: online j4 卡 46. 若模型在低-j4 obs 下命令 j4↑ → 下扎驱动 OK, online 失败=闭环/OOD;
         若模型在低-j4 obs 下 j4 不升 → 模型没学会从 approach 发起下扎。"""
import os
import sys, glob
import numpy as np, pandas as pd, av
from openpi.policies import policy_config as pc
from openpi.training import config as _config
CKPT=sys.argv[1]; DS=f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"
PROMPT="pick the pink sponge and place it in the blue bucket"; J4,J7,GRIP=3,6,7
def dec(mp4,idxs):
    want=set(int(i) for i in idxs); out={}; i=0; c=av.open(mp4)
    for fr in c.decode(c.streams.video[0]):
        if i in want: out[i]=fr.to_ndarray(format="rgb24"); want.discard(i)
        if not want: break
        i+=1
    c.close(); return out
pol=pc.create_trained_policy(_config.get_config("pi05_nero_b1"),CKPT,default_prompt=PROMPT)
print("[ready]")
eps=sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))[:15]
rows=[]
for ep in eps:
    epi=int(ep.split("episode_")[-1].split(".")[0]); df=pd.read_parquet(ep)
    S=np.stack(df["observation.state"].values).astype(np.float32); A=np.stack(df["action"].values).astype(np.float32); T=len(df)
    # 找 state.j4 最低的帧 (最 approach/未下扎) + state.j4≈46 附近的帧
    j4s=S[:,J4]
    t_low=int(np.argmin(j4s))                      # 最低 j4
    # 也找一个 state.j4 在 40-60 之间的帧 (匹配 online 卡的 46)
    cand=np.where((j4s>=40)&(j4s<=60))[0]
    t_mid=int(cand[0]) if len(cand) else t_low
    hi=f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wr=f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    H=dec(hi,{t_low,t_mid}); W=dec(wr,{t_low,t_mid})
    for tag,t in [("min_j4",t_low),("j4~46",t_mid)]:
        if t not in H or t not in W: continue
        ch=np.asarray(pol.infer({"observation/image":H[t],"observation/wrist_image":W[t],"observation/state":S[t],"prompt":PROMPT})["actions"])
        rows.append({"ep":epi,"tag":tag,"state_j4":float(S[t,J4]),"future_train_j4":float(A[min(t+9,T-1),J4]),
                     "pred_j4_0":float(ch[0,J4]),"pred_j4_end":float(ch[-1,J4]),"pred_j4_max":float(ch[:,J4].max())})
    print(f"ep{epi} done")
R=pd.DataFrame(rows)
print("\n"+"="*78)
for tag in ["min_j4","j4~46"]:
    g=R[R.tag==tag]
    if not len(g): continue
    print(f"[{tag}] 输入state_j4={g.state_j4.mean():.1f} | 训练该帧后9步 j4={g.future_train_j4.mean():.1f} | 模型 chunk[0]={g.pred_j4_0.mean():.1f} chunk末={g.pred_j4_end.mean():.1f} chunk_max={g.pred_j4_max.mean():.1f}")
    print(f"        → 模型从此 obs 命令 j4 升幅(chunk末-state)={g.pred_j4_end.mean()-g.state_j4.mean():+.1f}° (正大=会下扎驱动)")
print("="*78)
