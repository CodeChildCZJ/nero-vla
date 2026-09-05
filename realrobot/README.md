# `realrobot/` — the NERO pi0.5 deployment line

Getting a pi0.5 policy to actually pick up a pink sponge and drop it in a bucket, on a real
7-DoF arm. This is where the dataset behind [`bench/`](../bench/README.md) came from, and it
is a different kind of engineering: **there is no held-out validation set here.** Every
conclusion below was reached by an A/B on the physical robot.

For the closed-loop protocol, the serving contract and the unit conventions, see
[`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).

---

## Current production model

**`pi05_nero_b2_ref` @ step 29999**, served by `realrobot/serve/serve_b2_ref.sh` under the
config `pi05_nero_b2_ref_serve`. Since 2026-09-05 **`pi05_nero_b2_abs` @ 29999 is co-mainline**
— the on-robot comparison could not separate the two (see [The 2×2 result](#the-22-result)).

Trained from `pi05_base` on the b2 recovery dataset re-encoded into radians + gripper `[0,1]`,
with absolute action targets (`use_delta_joint_actions=False`), `action_horizon=10`,
batch 16, 30000 steps, cosine 2.5e-5 → 2.5e-6 with 1000 warmup steps, EMA 0.99. (Config body:
`patches/openpi-agilex.patch`.)

---

## Timeline: what each version fixed

Each entry names the *one thing* that version changed and what it bought. The hyper-parameters
quoted are from the launcher comments in `realrobot/train/`.

| version | change | outcome |
|---|---|---|
| **v1** | first single-task set (31 episodes), pi0.5 fine-tune | failed — the policy replayed episode 0's trajectory mechanically regardless of where the sponge was. Classic causal confusion from a single-task, single-layout set. |
| **v2** | `action_horizon=16`, 5000 steps | still failed; **the normalization statistics were wrong** — the config pointed at another robot's asset id |
| **v3** | identical to v2 **except** the correct `norm_stats` asset (`local/pick_pink_sponge_v1`) | isolated the norm-stats bug as a single variable |
| **v4** | clean 50-episode set, horizon 16, 5000 steps | undertrained — 5k steps had not converged |
| **v5** | 51 episodes, **20000 steps** | trained to convergence; no sharpening, no extra layouts (deliberately held back so the next variable would be clean) |
| **v6** | `mask_gripper_state=True`, everything else identical to v5, 8000 steps | single-variable test of the gripper **copycat** hypothesis: was the policy just echoing the gripper state it was fed? |
| **v7** | mask off again; **copycat fixed at the data source** + `ema_decay=0.99`, 23 diverse episodes, 10000 steps, recomputed norm stats | the recorder had been writing `qpos[7]` — the *follower's* gripper readback — into the action column instead of the *leader's* command. The two are not the same signal, and the lead-lag check showed the action leads by 3 frames. With `state ≡ action` on that channel, copying the input was a perfect solution to the training loss. |
| **v8** | 62 episodes covering sponge position × approach direction, `action_horizon` 16 → 10, `ema_decay` None → 0.99, 30000 steps | attacked position OOD: sponge position correlates with grasp configuration at r ≈ 0.93–0.95 in the data, so coverage of positions is coverage of grasp poses |
| **b1** | full rebuild on the **stock upstream openpi-agilex stack** (32 clean episodes), hyper-parameters frozen from v8 | removed the custom-stack variable entirely; the remaining failure was *not* a stack bug |
| **b2** | 131 episodes including **recovery** demonstrations — deliberate mis-approaches followed by a visual correction back to a graspable pose | the dataset the benchmark also uses |
| **b2 ×2 ablation** | `{b1 clean, b2 recovery} × {absolute, delta}` action convention | see below |
| **b2_ref** | b2 recovery data, absolute targets, rad + gripper `[0,1]` baked into the dataset | **production** |

## The 2×2 result

Four checkpoints, run on the physical robot on 2026-06-20. **Read the two axes separately —
they are not backed by the same strength of evidence.** An earlier version of this section
published a single ordering, `ref > b2_abs > b2_delta >> b1_abs`; the middle comparison in that
chain does not survive the numbers below and has been retracted.

### Data axis — established

Same action convention, only the dataset differs:

```
b2_abs (131 ep, recovery)  grasps        b1_abs (32 ep, clean)  does not
```

`b1_abs` fails at 21k steps and again at 29999, so more optimizer steps do not rescue clean
data. The mechanism was read off the trajectory logs: clean demonstrations contain no "the
wrist drifted, pull it back" samples, so in closed loop the policy cannot hold j7 inside the
training grasp manifold (−15° to 60°) and issues the close command at j7 = 90–96°. Recovery
episodes supply exactly those corrective samples. This is the single largest intervention
across the whole v1→b2 sequence — larger than any hyper-parameter or parameterization change.

### Convention axis — mostly NOT established

The controlled protocol (same five sponge placements per model, success rate recorded) was
specified but its results were never written down. The only cross-model number that survives is
the grasp-trigger rate, measured over **unaligned** test positions:

| checkpoint | rollouts that triggered a grasp | vs `b2_ref`, Fisher exact |
|---|---|---|
| `b2_ref` | 8/10 | — |
| `b2_abs` | 5/8 | **p = 0.61 — not separated** |
| `b2_delta` | 1/8 | p = 0.015 |
| `b1_abs` (clean) | 2/5 | p = 0.25 |

`b2_delta` being worst is real. **`ref > b2_abs` is not a measured result and must not be cited
as one** — at 21k both were described at the time as "both grasp, no visible difference", and
at 29999 the two are 0.61 apart on the only metric that exists. Since 2026-09-05 both are
treated as co-mainline; `b2_abs` is in fact the simpler deployment (native deg+mm, no
conversion layer), and is the fallback if the `b2_ref` serve-time conversion ever misbehaves.

Also note that trigger rate is not task success: closing the gripper is necessary but not
sufficient for picking the sponge up and placing it.

One mechanism worth recording because it was proposed and then falsified: `b2_delta` was
thought to fail by error accumulation. It does not — commanded step size stays around 1° and
*shrinks* across a rollout (1.0° → 0.3°). It fails by stalling short of a graspable
configuration, which is why 6 of its 8 rollouts never close the gripper at all. The ordering
survived the correction; the explanation did not.

Note what this implies for [`bench/`](../bench/README.md): the benchmark scores fits to
*clean* demonstration actions. The intervention that mattered most on hardware was adding a
*different kind of demonstration*. Those are not the same axis.

---

## Still unsolved

Stated plainly, because the production model is not a solved task:

- **The gripper-close trigger is decoupled from vision.** Fed a grasp-configuration frame
  offline, the model does emit a close command — so the capability is there and the offline
  error looks fine. During a rollout, when the arm sits slightly off the demonstration
  manifold, the close simply does not fire. The trigger appears to key off cues correlated
  with grasping *within the demonstrations* rather than off the grasp configuration itself,
  and that correlation breaks the moment the trajectory leaves the manifold.
- **Grasping fails at off-center and near-singular arm positions**, even after v8's position
  coverage and b2's recovery episodes.

Both are closed-loop failures that an open-loop action-MAE benchmark **cannot** detect. That
is [caveat 4](../docs/CAVEATS.md#4-low-open-loop-mae-does-not-mean-the-policy-works-on-a-robot),
and this line is where it was learned.

---

## Layout

```
realrobot/
├─ train/       train_v2..v8.sh, train_b1.sh — one launcher per version above
├─ serve/       serve_*.sh — one per served model, plus supervise_ref.sh / selfheal_ref.sh
└─ archive/     the diagnostic scripts behind every conclusion on this page, with INDEX.md
```

> **There is no launcher for the b2 generation, including the production model.** `train/`
> stops at `train_b1.sh`. The `b2`, `b2_abs`, `b2_delta` and `b2_ref` runs were started from
> the command line rather than from a committed script, so no such file exists to publish and
> none has been written after the fact — a launcher invented now would claim a provenance it
> does not have. The invocation is the same two steps `train_b1.sh` performs, with the config
> name swapped (all four configs are in `patches/openpi-agilex.patch`):
>
> ```bash
> cd "$NERO_ROOT/third_party/openpi-agilex"
> HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
>   .venv/bin/python3 scripts/compute_norm_stats.py --config-name pi05_nero_b2_ref
> CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python3 scripts/train.py pi05_nero_b2_ref \
>   --exp-name b2_ref --checkpoint-base-dir "$NERO_CKPT" --no-wandb-enabled --overwrite
> ```
>
> Treat that block as reconstructed from `train_b1.sh` plus the config, not as a transcript of
> the run that produced the published checkpoint.

All scripts require `source env.sh` first (see `env.example.sh` at the repository root) and
hard-fail on an unset `NERO_ROOT` rather than silently writing somewhere unexpected. They
expect openpi-agilex at `$NERO_ROOT/third_party/openpi-agilex` (via `tools/bootstrap_upstreams.sh`) with its virtualenv
built; the training configs live in that fork and are exported as a diff in
`patches/openpi-agilex.patch`.

> **The two generations come from two different upstreams.** `patches/openpi-agilex.patch`
> defines the eight `b1`/`b2`-generation `TrainConfig`s — `pi05_nero_b1`, `b1_abs`, `b2`,
> `b2_abs`, `b2_ref`, `b2_ref_serve`, `b2_train` and `pi0_nero_b2_train`. The configs the
> v2–v8 launchers name (`pi05_nero_pick_pink_sponge_v2` … `_v8`) live in a **different**
> upstream — Physical-Intelligence/openpi, exported as `patches/openpi.patch` — because the
> real-robot line migrated stacks partway through. Both patches ship, so both generations are
> reproducible; you just have to apply each one to its own fork. See
> [`third_party/UPSTREAM.md`](../third_party/UPSTREAM.md).

`archive/` is kept **verbatim**. With no held-out validation set, those one-off scripts —
trajectory diffs, gripper lead-lag checks, grasp-configuration correlations, server-trace
audits — *are* the evidence for the timeline above. `realrobot/archive/INDEX.md` says what
each one measured.

Neither the datasets nor the checkpoints are published.
