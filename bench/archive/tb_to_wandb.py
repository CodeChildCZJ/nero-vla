#!/usr/bin/env python3
"""Live-mirror a HF Trainer tensorboard run into wandb (online).

Used because the gr00t_n15 finetune was already ~2900/10000 steps in when wandb was
requested; restarting to get --report-to wandb would have thrown away ~1h of GPU4.
This backfills every scalar point already written and then keeps polling the
tfevents file, so the wandb curve is identical to what --report-to wandb would show.
"""
import argparse, glob, os, time, subprocess

p = argparse.ArgumentParser()
p.add_argument("--logdir", required=True)
p.add_argument("--project", default="nero_backbone_bench")
p.add_argument("--name", required=True)
p.add_argument("--watch-pid", type=int, default=0, help="stop once this pid exits")
p.add_argument("--poll", type=float, default=60.0)
p.add_argument("--config", default="", help="k=v,k=v attached to the wandb run")
a = p.parse_args()

os.environ.setdefault("WANDB_MODE", "online")
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

cfg = {}
for kv in filter(None, a.config.split(",")):
    k, v = kv.split("=", 1)
    try:
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError:
            pass
    cfg[k] = v

run = wandb.init(project=a.project, name=a.name, config=cfg, resume="never")
print("WANDB_URL", run.url, flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "wandb_url_%s.txt" % a.name), "w") as f:
    f.write(run.url + "\n")

sent = 0  # highest step already logged


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


idle_rounds = 0
while True:
    files = sorted(glob.glob(os.path.join(a.logdir, "**", "events.out.tfevents.*"), recursive=True))
    new = 0
    if files:
        ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        by_step = {}
        for t in tags:
            for s in ea.Scalars(t):
                by_step.setdefault(s.step, {})[t] = s.value
        for step in sorted(by_step):
            if step > sent:
                wandb.log(by_step[step], step=step)
                sent = step
                new += 1
    alive = pid_alive(a.watch_pid)
    print(f"[mirror] step={sent} new={new} train_alive={alive}", flush=True)
    if not alive:
        idle_rounds += 1
        if idle_rounds >= 2:  # one extra sweep after the trainer exits to catch the tail
            break
    time.sleep(a.poll)

wandb.finish()
print("mirror done at step", sent, flush=True)
