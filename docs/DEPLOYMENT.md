# Real-robot deployment

## Read this first

> **None of the seven backbones on the [leaderboard](LEADERBOARD.md) has ever been run on a
> robot.** The board is 100% offline, open-loop evaluation.
>
> **榜上 7 个 backbone 全部只做了离线开环评测,一个都没上过真机。**

Only the **openpi (pi0 / pi0.5) path has a working NERO serving route**, and it was built for
the `realrobot/` deployment line, not for the benchmark. The other five stacks
(`openvla_oft`, `gr00t_n17`, `gr00t_n15`, `smolvla`, `act`) have **no serving integration at
all** here — each framework has its own inference server, observation packing, image
preprocessing and action decoding, and each would have to be wired up separately.

Nothing on the leaderboard should be read as validated on hardware. If you need a number that
predicts closed-loop success, this repository does not contain one — see
[CAVEATS.md §4](CAVEATS.md#4-low-open-loop-mae-does-not-mean-the-policy-works-on-a-robot).

What follows is the pi0.5 loop as actually operated, so that the shape of the problem — and
the ways it silently breaks — are on record.

---

## The closed loop, in six stages

### 1. Pair the checkpoint with its normalization statistics

An openpi checkpoint carries its own `assets/<asset_id>/norm_stats.json`. **The config you
serve with selects which `asset_id` is loaded**, so serving a checkpoint under a config whose
`asset_id` points somewhere else loads the wrong constants — and the policy still runs,
producing confident, wrong actions.

This is the first failure we hit on this project (a NERO checkpoint served against another
robot's stats) and it cost a full debugging cycle because nothing errors. Check the stats file
that actually got loaded, not the one you expect.

### 2. Start the inference server

```bash
source env.sh
SERVE_GPU=<gpu> SERVE_MEM=0.25 bash realrobot/serve/serve_b2_ref.sh <step> <port>
```

The scripts under `realrobot/serve/` are thin wrappers over openpi's
`scripts/serve_policy.py --env ALOHA`, parameterized by checkpoint step and port. Each one
pins `CUDA_VISIBLE_DEVICES` and sets `XLA_PYTHON_CLIENT_MEM_FRACTION` (pi0.5 inference needs
roughly 15–19 GB; the scripts leave headroom for cold-start compilation so the process does
not OOM against itself).

Each script also sets `OPENPI_TRACE` to a per-run JSONL path, so every inference call is
logged server-side. That trace is the only artifact that lets you reconcile "what the model
was asked" against "what the client thinks it sent".

### 3. Collect the observation on the robot side

Per control tick, the client gathers:

- `cam_high` — the overhead camera, `(H, W, 3)` uint8 **RGB**
- `cam_wrist` — the wrist camera, same format
- the 8-dim follower joint state (arm j0–j6 + gripper)

### 4. Pack, POST, infer

The client sends flat keys:

```
observation/image        <- cam_high
observation/wrist_image  <- cam_wrist
observation/state        <- (8,)
prompt                   <- "pick the pink sponge and place it in the blue bucket"
```

The server repacks, applies its input transforms, normalizes, runs the flow-matching
denoiser, and returns an action chunk of shape `[horizon, 8]` (`action_horizon = 10` for
every NERO pi0.5 config in `patches/openpi-agilex.patch`).

### 5. Decode the chunk — units and parameterization

**The wire units are a property of the config you launched, not of the checkpoint.** Get this
from the config, and verify it against a real response, before trusting any comment.

For the `b1` / `b2_abs` / `b2_delta` configs the wire contract is **absolute degrees +
absolute gripper millimetres**, in both directions:

- `use_delta_joint_actions=False` (`b1`, `b2_abs`) → the model emits absolute joint targets.
- `use_delta_joint_actions=True` (`b2_delta`) → the model emits arm deltas internally, but
  `AbsoluteActions` on the output side adds `state[t]` back **inside the server**, so the
  client still receives absolute values and needs no shim. This is exactly the inversion the
  benchmark's prediction contract also requires.

For the production `b2_ref` model there is one extra layer. The checkpoint was trained on a
dataset re-encoded into **radians + gripper `[0, 1]`** (0 = closed, 1 = 76 mm open), with the
conversion baked into the data. `serve_b2_ref.sh` therefore launches the config
`pi05_nero_b2_ref_serve`, whose `serve_convert_degmm=True` inserts
`NeroRefDegToRad` before normalization and `NeroRefRadToDeg` after un-normalization:

```
client sends deg + mm  ->  deg2rad, mm/76  ->  normalize -> model -> un-normalize
                       ->  rad2deg, ×76    ->  client receives deg + mm
```

So the **external contract stays deg + mm** and the rad/`[0,1]` space is entirely internal.

> ⚠️ **Known documentation conflict.** The header comment inside
> `realrobot/serve/serve_b2_ref.sh` describes the *older* contract — client sends radians and
> gripper `[0,1]`, client converts on its own side. That comment contradicts the config the
> same script launches. The code is authoritative:
> `NeroRefDegToRad.__call__` multiplies the incoming state by `π/180` and divides the gripper
> by `76.0`, which is only correct if the wire carries degrees and millimetres. Treat the
> comment as stale and confirm against a live response before wiring a new client.

Gripper polarity, for the robot side: **0 mm = closed, 76 mm = fully open.** (The dataset's
own j7 channel is a 0–100 scale — a different convention; see [DATA.md §3](DATA.md#3-units).)

### 6. Execute the first few steps, then re-observe

The client executes a prefix of the returned chunk at the control frequency and then loops
back to stage 3. Executing the whole 10-step chunk open-loop before re-observing makes the
policy blind for a third of a second; executing only step 0 wastes most of the inference. The
split is a tuning knob, and it interacts with everything in stage 5.

---

## Known failure modes

Ordered by how much time each one cost.

| failure | symptom | how to catch it |
|---|---|---|
| **Wrong `norm_stats` paired with the checkpoint** | Policy runs, motion is confidently wrong | Print the resolved `assets/<asset_id>/norm_stats.json` path at startup and diff its md5 against the one used at training |
| **delta vs absolute confusion** | Arm drifts, or jumps by ~the magnitude of the joint angles | Compare `chunk[0]` against the state you just sent: an absolute contract puts them ~1° apart, a mis-decoded delta puts them tens of degrees apart |
| **Camera mapping swapped** | Plausible but useless behaviour, no error anywhere | Assert on the client that `observation/image` is the overhead stream; no downstream check can detect this |
| **Unit convention (deg/rad, mm/[0,1])** | Motion scaled by 57× or 76× | The conflict documented in stage 5 — read the config, not the comment |
| **Server killed after GPU idle timeout** | Client hangs / connection refused mid-session | `realrobot/serve/supervise_ref.sh` (foreground supervisor loop) and `realrobot/serve/selfheal_ref.sh` (idempotent restart, safe to call on a timer) |

Two notes on the last one. `selfheal_ref.sh` checks both the listening port *and* the process
before restarting, because a server that is still compiling holds neither yet — a naive
"port is closed, restart" check spawns a second server onto the same GPU. And the supervisor's
outer loop is deliberately plain `bash` so that a `pkill -f serve_policy.py` does not take the
supervisor down with the server.

---

## Serving scripts in this repository

| script | model | action space | notes |
|---|---|---|---|
| `serve_v8.sh` | `pi05_nero_pick_pink_sponge_v8` | delta arm | pre-rebuild line — **this config is not in the pinned fork**, kept for provenance only |
| `serve_b1.sh` | `pi05_nero_b1` | delta arm, absolute gripper | first fully-official-stack rebuild |
| `serve_b1_abs.sh` | `pi05_nero_b1_abs` | absolute | b1 data, absolute convention |
| `serve_b2_abs.sh` | `pi05_nero_b2_abs` | absolute | b2 recovery data |
| `serve_b2_delta.sh` | `pi05_nero_b2` | delta arm (added back server-side) | b2 recovery data |
| `serve_b2_ref.sh` | `pi05_nero_b2_ref_serve` | absolute, rad/[0,1] internal | **production**, step 29999 |
| `supervise_ref.sh` / `selfheal_ref.sh` | — | — | keep the production server alive |

All of them require `source env.sh` first (they hard-fail on an unset `NERO_ROOT`) and expect
openpi-agilex at `$NERO_ROOT/third_party/openpi-agilex` (run `tools/bootstrap_upstreams.sh openpi-agilex`) with its virtualenv built.

---

## If you want to serve one of the benchmark backbones

You would need, per stack: an inference entry point, an observation-packing contract matching
the stack's expected keys and image preprocessing, the same normalization the checkpoint was
trained with, and the inverse transform back to absolute joint targets. The benchmark's
predict scripts (`bench/scripts/predict/`) already contain the last two for each stack — they
are the closest thing to a starting point that exists here.

That work has not been done, and until it is, the leaderboard makes no claim about any of
these policies on hardware.
