#!/usr/bin/env python3
"""解析 train_b1.log 的 'Step N: grad_norm=.., loss=.., param_norm=..' 行 → 写 B 训练 loss 的
TensorBoard events。B loss 一直在 log 里(openpi-agilex 用 tqdm_loggable 把 pbar.write 走 logging
进 log 文件, 不是没记)。tags 用 loss/grad_norm/param_norm = 跟主仓 v8 train.py 完全一致 → 同一 chart
上 v8 与 b1 两条线可叠加对比。重跑先清旧 events 保证单条干净曲线(B 训练续跑后重跑刷新即可)。
用法: ${NERO_ROOT}/third_party/openpi-agilex/.venv/bin/python3 b_loss_to_tb.py [logfile] [outdir]
"""
import re, sys, os, glob
from torch.utils.tensorboard import SummaryWriter

LOG = sys.argv[1] if len(sys.argv) > 1 else f"{os.environ.get('NERO_ROOT', '.')}/realrobot/train_b1.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tb_compare/b1"
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "events.out.tfevents*")):
    os.remove(f)

pat = re.compile(r"^Step (\d+): grad_norm=([0-9.]+), loss=([0-9.]+), param_norm=([0-9.]+)")
rows = []
for line in open(LOG):
    m = pat.match(line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
rows.sort()
if not rows:
    print(f"[!] no 'Step N: ...' lines in {LOG}"); sys.exit(1)

w = SummaryWriter(OUT)
for step, gn, loss, pn in rows:
    w.add_scalar("loss", loss, step)
    w.add_scalar("grad_norm", gn, step)
    w.add_scalar("param_norm", pn, step)
w.close()
print(f"[tb] wrote {len(rows)} pts to {OUT} "
      f"(step {rows[0][0]}..{rows[-1][0]}, loss {rows[0][2]:.4f}..{rows[-1][2]:.4f})")
