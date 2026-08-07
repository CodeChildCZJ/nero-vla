#!/usr/bin/env python3
"""Print the latest train/loss from a run's tensorboard events.

The finetune script's stdout is block-buffered, so the {'loss': ...} lines only
appear when the process exits. TensorBoard events are flushed continuously, so
this is the only live view of the loss curve.
"""
import pathlib
import os
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints")), "gr00t_n15_nero_b2")
dirs = sorted(glob.glob(f"{run}/runs/*"))
if not dirs:
    print("no tensorboard runs yet")
    sys.exit(0)
ea = EventAccumulator(dirs[-1])
ea.Reload()
if "train/loss" not in ea.Tags()["scalars"]:
    print("no loss logged yet")
    sys.exit(0)
ev = ea.Scalars("train/loss")
tail = " ".join(f"{e.step}:{e.value:.4f}" for e in ev[-4:])
first = ev[0]
print(f"step={ev[-1].step} loss={ev[-1].value:.4f} (first {first.step}:{first.value:.4f}) recent[{tail}]")
