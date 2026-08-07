# Leaderboard

Offline open-loop action-chunk error on 20 held-out episodes of one real-robot manipulation
dataset. Lower is better. Reproduce it with numpy and nothing else:

```bash
python bench/scripts/score/score.py
```

> **Before citing anything on this page, read [CAVEATS.md](CAVEATS.md).** This is a
> recipe-level comparison, not a controlled architecture A/B; no row has been validated on a
> robot; and adjacent rows are not necessarily separated by more than this val set can
> resolve.

---

## The board

```
=== NERO b2 held-out (1397 anchors, K=8, per-joint MAE, raw joint units) ===
backbone               MAE  arm(0-6)  grip(7)   per-joint j0..j7
pi05                 1.265     1.164    1.971    0.82  1.68  0.81  1.12  1.01  1.08  1.62  1.97
pi0                  1.332     1.209    2.194    0.87  1.76  0.90  1.11  1.05  1.11  1.65  2.19
openvla_oft          1.642     1.564    2.184    1.39  2.08  1.42  1.42  1.41  1.37  1.84  2.18
gr00t_n17            1.648     1.511    2.601    1.11  2.27  1.10  1.45  1.30  1.31  2.03  2.60
smolvla              1.929     1.855    2.453    1.26  2.54  1.40  1.71  1.63  1.70  2.76  2.45
gr00t_n15            2.047     1.972    2.565    1.75  2.70  1.76  1.81  1.75  1.71  2.32  2.57
act                  2.727     2.676    3.088    2.31  3.34  2.17  2.34  2.39  2.66  3.52  3.09
[hold-state]         3.072     2.535    6.829    1.35  6.37  1.38  2.84  1.40  1.34  3.07  6.83  <-- pass line (do-nothing baseline)

(lower=better; arm in deg, gripper in its unit; open-loop first-K prediction vs GT)
(hold-state pass line: MAE 3.072 — a real backbone must beat this)
```

Verbatim output of `bench/scripts/score/score.py`. Columns are defined in
[METHODOLOGY.md §5](METHODOLOGY.md#5-scoring): `MAE` over 1397 anchors × 8 horizon steps ×
8 joints, `arm(0-6)` and `grip(7)` the same mean restricted to those dims.

`[hold-state]` is not a model — it is the do-nothing baseline (predict that the action equals
the current state). Any backbone above it on this list has learned nothing useful.

---

## What each row was actually trained with

Filled in **only** from run records that exist in this repository. Where a value is marked
*(training-script default)* it was read from `bench/scripts/train/`, which is what the run
*would* have used with no overrides — it is **not** a record of that specific run. Where
nothing is recoverable it says **not recorded**, and no value has been inferred.

| backbone | framework | base model | steps | batch | learning rate |
|---|---|---|---|---|---|
| `pi05` | openpi (JAX), flow matching | `pi05_base` | **30000** (ckpt step 29999) | **16** | cosine, warmup 1000, peak 2.5e-5 → 2.5e-6 |
| `pi0` | openpi (JAX), flow matching | `pi0_base` | **30000** (ckpt step 29999) | **16** | cosine, warmup 1000, peak 2.5e-5 → 2.5e-6 |
| `openvla_oft` | OpenVLA-OFT, PyTorch, L1 regression head | `openvla-7b` + LoRA r32 | **not recorded** | 8 × 4 grad-accum = 32 effective *(training-script default)* | 5e-4 *(training-script default)* |
| `gr00t_n17` | Isaac-GR00T N1.7, PyTorch, flow matching | `GR00T-N1.7-3B` | **10000** (ckpt `checkpoint-10000`) | 64 *(training-script default)* | 1e-4 *(training-script default)* |
| `smolvla` | HF LeRobot 0.6.2, flow matching | `lerobot/smolvla_base` | **not recorded** — 20000 *(training-script default)* | 64 *(training-script default)* | **not recorded** |
| `gr00t_n15` | Isaac-GR00T N1.5, PyTorch, flow matching | GR00T N1.5 | **10000** (ckpt `checkpoint-10000`) | 64 *(training-script default)* | 1e-4 *(training-script default)* |
| `act` | HF LeRobot 0.6.2 | from scratch (ACT has no base model) | **100000** | **8** | 1e-5 *(training-script comment citing the LeRobot ACT config default)* |

Provenance, per cell:

- `pi0` / `pi05` — the run's config **name** is recorded in `bench/preds/preds_pi0*_meta.json`
  (`pi0_nero_b2_train` / `pi05_nero_b2_train`); the config **body** with those hyper-parameters
  is pinned in `patches/openpi-agilex.patch`. Both also set `action_horizon=10`,
  `ema_decay=0.99`, `use_delta_joint_actions=True`.
- `gr00t_n15` — step count from the checkpoint path in `preds_gr00t_n15_meta.json`.
- `gr00t_n17` — step count from the `model_path` field **inside** `bench/preds/preds_gr00t_n17.npz`
  (`.../checkpoint-10000`); it also records `denoising_steps = 4` and `base_seed = 0`.
- `openvla_oft` — the checkpoint path recorded in the npz `contract` field carries **no step
  number**, and there is no meta file. Its recipe (parallel decoding, action chunking,
  continuous actions, L1 regression, 2 cameras, proprio, LoRA r32, FiLM off) is documented in
  `bench/scripts/train/oft_train.sh`, but the step count that produced this row is lost.
- `smolvla` — no meta file and no checkpoint path in the npz. `20000` is the value hardcoded
  in `bench/scripts/train/lerobot_train_bb.sh` for the `smolvla` branch, and
  `bench/scripts/predict/lerobot_post_train.sh` independently hardcodes `STEP=020000` as the
  checkpoint it consumes. Two scripts agreeing is consistent with a 20000-step run, but both
  are *script constants*, not a logged run record — so the cell stays "not recorded".
- `act` — the only leg with a complete `train` block: `steps`, `batch`, `epochs 21.04`,
  `wall 3h05m`, `final_train_l1_normalized 0.059`. Its learning rate is not in that block;
  the `1e-5` above comes from a comment in the training script and is therefore a
  script-level default, not a run record.

  > **That block was typed by hand, not written by a program.** `bench/archive/INDEX.md`
  > records this against `lerobot_budget_block.py`, the tool written to derive budget blocks
  > from a checkpoint's own `train_config.json` precisely because a hand-typed one is the
  > path by which an adjacent row gets labelled with another backbone's budget. So the most
  > complete provenance cell on this page is also the least machine-verified one. Two
  > consequences: the `epochs 21.04` figure is *not* an independent check of the
  > steps × batch ÷ frames formula below (it was typed from the same two numbers it would
  > verify), and nothing here rules out a transcription error in `steps` or `batch`.

### Derived scale (not logged anywhere — computed here)

`epochs = steps × batch / 38025 train frames`. The formula is validated against the two
values that *were* logged: it reproduces ACT's recorded `21.04` and the `33.7` in the
SmolVLA branch comment.

| backbone | epochs over the train split |
|---|---|
| `act` | 21.0 |
| `smolvla` | 33.7 *(rests on the unrecorded 20000)* |
| `gr00t_n15` / `gr00t_n17` | 16.8 |
| `pi0` / `pi05` | 12.6 |
| `openvla_oft` | not computable — steps not recorded |

**The step column and the epoch column rank the runs differently.** ACT has 10× the steps of
the GR00T legs and *fewer* effective passes than SmolVLA. This is the confound described in
[CAVEATS.md §1](CAVEATS.md#1-this-is-a-recipe-level-comparison-not-a-controlled-architecture-ab).

---

## Delivery properties

| backbone | horizon in the npz | decoding | action parameterization | `anchor_idx` shipped |
|---|---|---|---|---|
| `pi05` | 10 | stochastic (flow), `noise_seed = 0` | delta arm + absolute gripper | yes |
| `pi0` | 10 | stochastic (flow), `noise_seed = 0` | delta arm + absolute gripper | yes |
| `openvla_oft` | 8 | **deterministic** (`do_sample=False`, L1 head, consumes no seed) | absolute | no (`contract` field instead) |
| `gr00t_n17` | 16 | stochastic (flow), seeds recorded per anchor | delta arm + absolute gripper | no (`noise_seeds` array instead) |
| `smolvla` | 50 | stochastic (flow) | absolute | yes |
| `gr00t_n15` | 16 | stochastic (flow), pinned per-anchor generator | absolute | no |
| `act` | 100 | **deterministic** (verified bit-exact across 5 draws, maxdiff 0.0) | absolute | yes |

All rows are scored on the shared `K = 8` prefix. The parameterization column is why
`grip(7)` is the least confounded column on the board — see
[CAVEATS.md §5](CAVEATS.md#5-delta-parameterization-makes-the-pass-line-cheap).

`openvla_oft` sets `num_actions_chunk = 8`, which is what fixes `K = 8` for everyone.

---

## Significance

Computed by `bench/scripts/score/paired_signif.py` — delete-one-episode cluster jackknife on
the **paired** difference, over the 20 val episodes.

| comparison | gap | paired SE | σ | resolved at 2σ? |
|---|---|---|---|---|
| `act` vs `[hold-state]` (total) | +0.3448 | 0.18284 | 1.9 | **no** |
| `act` vs a random-init control | +17.4761 | 0.90126 | 19.4 | yes |

(per-episode error correlation for the first row: ρ = −0.16, which *inflates* the paired SE
rather than shrinking it.)

Only these two comparisons are recorded in a shipped artifact
(`bench/preds/preds_act_meta.json`). To get the full pairwise table:

```bash
python bench/scripts/score/paired_signif.py --all
```

Note this will print `[no variance footnote]` for the stochastic legs, because the
`bench/logs/variance/` JSONs are not shipped. On this board that changes no verdict —
episode composition dominates seed-to-seed variance by roughly 10× — but the tool says so
rather than silently substituting zero.

---

## Training-run tracking

All seven runs logged to a **private** Weights & Biases project. This page deliberately does
**not** tabulate the run ids: the project is not readable from outside, so an id gives a reader
nothing to check while implying there is something checkable behind it. (One id does survive,
in `bench/preds/preds_act_meta.json` — that file is a provenance record written by the run
itself, and editing a recorded field to tidy up a document would be the wrong trade.)

Everything this leaderboard actually rests on — the split, the frozen anchors, every
backbone's predictions, and the scorer — is committed to this repository instead.

---

## Reproducing the board

Everything needed is committed:

```
bench/data/split.json          the 111/20 episode split
bench/data/val_anchors.npz     1397 frozen anchors, K=8 (md5 cc34dda2a99ade1011c192e2b3e4e343)
bench/preds/preds_*.npz        seven backbones' predictions
bench/scripts/score/score.py   the scorer (numpy only, derives its root from __file__)
```

No GPU, no dataset, no checkpoint, and no upstream repositories are required.
