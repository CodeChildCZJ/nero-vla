"""Decide if v4 has plateaued at step 4999. Plateau = last 1000 step loss decrease < 10%."""
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc = EventAccumulator(f'{os.environ.get("NERO_CKPT", "checkpoints")}/pi05_nero_pick_pink_sponge_v4/v4/tb/', size_guidance={'scalars': 1000})
acc.Reload()
pts = acc.Scalars('loss')
if len(pts) < 20:
    print("INSUFFICIENT_DATA", flush=True)
    sys.exit(2)
# Compare last loss vs loss at step ~4000 (20 scalars before end since log_interval=50)
last = pts[-1].value
n_back = 20  # 20 * 50 = 1000 step back
ref = pts[-1 - n_back].value if len(pts) > n_back else pts[0].value
ratio = (ref - last) / ref if ref > 0 else 0
print(f"last_step={pts[-1].step} loss={last:.4f}")
print(f"ref_step={pts[-1 - n_back].step} loss={ref:.4f}")
print(f"reduction_last_1k_step={ratio*100:.1f}%")
if ratio < 0.10:
    print("PLATEAU=YES")
    sys.exit(0)
else:
    print("PLATEAU=NO_keep_training")
    sys.exit(1)
