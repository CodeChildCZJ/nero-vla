#!/usr/bin/env python3
"""Tail v4 trace during force-close test. Track gripper + joint reaction.
Key question: after force-close (state.grip drops but model still commands open),
does the model REACT (chunk joints diverge from hover = sees grasp) or keep hovering?"""
import os
import json, time

P = f"{os.environ.get('NERO_ROOT', '.')}/realrobot/server_trace_v4_4999.jsonl"
with open(P) as f:
    f.seek(0, 2); pos = f.tell()

real_n = 0; hb = 0; prev_grip = None
print(f"[forceclose-watch] armed at byte {pos}", flush=True)
while True:
    with open(P) as f:
        f.seek(pos); lines = f.readlines(); pos = f.tell()
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        try:
            d = json.loads(ln); s = d["state_8"]; c = d["chunk_8"]
        except Exception: continue
        if all(abs(x) < 1e-6 for x in s): continue  # skip zero-obs ping
        real_n += 1
        sg = s[7]; j2 = s[1]; j4 = s[3]
        cg0 = c[0][7]
        # 模型意图: chunk 末帧 vs 当前 state (chunk 是 absolute, 末帧=8帧后想到哪)
        head_j2 = c[-1][1] - j2; head_j4 = c[-1][3] - j4; head_g = c[-1][7] - sg
        forced = sg < 8 and cg0 > 30        # 爪闭了但model还想开 = 强制闭爪态
        react = abs(head_j2) > 5 or abs(head_j4) > 5   # model 想动关节 >5°
        gjit = prev_grip is not None and abs(sg - prev_grip) > 15
        prev_grip = sg
        emit = forced or react or gjit or (real_n - hb >= 15)
        if emit:
            hb = real_n
            tag = []
            if forced: tag.append("FORCED-CLOSE(爪闭model还想开)")
            if react: tag.append(f"JOINT-REACT(想动j2{head_j2:+.0f}/j4{head_j4:+.0f})")
            if gjit: tag.append("GRIP-JUMP")
            t = " ".join(tag) if tag else "(hb)"
            print(f"#{real_n} grip={sg:.0f} j2={j2:+.0f} j4={j4:+.0f} | chunk头grip={cg0:.0f} 意图Δj2={head_j2:+.0f} Δj4={head_j4:+.0f} Δg={head_g:+.0f} {t}", flush=True)
    time.sleep(0.5)
