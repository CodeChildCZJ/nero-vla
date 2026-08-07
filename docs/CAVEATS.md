# Caveats — what this benchmark is *not*

**Read this before quoting any number from [LEADERBOARD.md](LEADERBOARD.md).**
**引用榜单上任何数字之前,先读完这一页。**

The board is a real, reproducible measurement of one specific thing: how well seven
fine-tuning recipes fit held-out demonstration actions on one real-robot dataset, open-loop,
at a horizon of 8 steps. Everything below is a way that measurement can be over-read.

The six limitations are listed in rough order of how badly they bite.

---

## 1. This is a recipe-level comparison, not a controlled architecture A/B

**这是 recipe 级对比,不是受控的架构 A/B。**

Each backbone was trained with **its own framework's official fine-tuning recipe**, because
that is what a practitioner would actually run. The consequence is that the runs differ in
almost everything at once:

| | steps | batch | source |
|---|---|---|---|
| `act` | 100000 | 8 | run record |
| `pi0` / `pi05` | 30000 | 16 | pinned TrainConfig |
| `smolvla` | 20000 (training-script default) | 64 | training script |
| `gr00t_n15` / `gr00t_n17` | 10000 | 64 | checkpoint name / training script |
| `openvla_oft` | not recorded | 8 × 4 accum = 32 effective | training script |

That is a **10× spread in optimizer steps** and an 8× spread in batch size, on top of
different optimizers, LR schedules, base checkpoints, augmentation and freezing strategies.
Any ranking difference is a joint effect of backbone *and* recipe, and this benchmark cannot
separate them.

Note also that **steps are not comparable across stacks** — the meaningful unit is epochs
over the 38025 training frames. `act`'s 100000 × 8 is 21.0 epochs; `smolvla`'s 20000 × 64 is
33.7. The stack with 5× the steps saw *fewer* passes over the data.

One in-family comparison is closer to controlled and still isn't clean: `pi0` vs `pi05` share
every knob their config file sets (batch 16 / 30k / cosine 2.5e-5 / EMA 0.99 / horizon 10 /
same data), but openpi ties normalization to the model type — `pi0` gets z-score, `pi05` gets
quantile — and that cannot be varied independently in this stack. So even that row pair is a
comparison of two *(backbone, normalization)* pairs.

**Do not cite this board as evidence that architecture X is better than architecture Y.**

---

## 2. There is no held-out validation loss — only training loss

**全 7 栈只有 train loss,没有 held-out val loss。**

None of the seven runs logged a validation loss during training. The val MAE on the board is
the *only* held-out signal any of these rows has, and it was computed once, after training
finished.

Practical consequences: nobody watched a val curve, nobody could have detected overfitting
while it was happening, and there is no learning-curve evidence that any run was trained for
the right length. Where a train-loss number exists it is recorded in the per-leg meta files
(e.g. ACT's final normalized L1 was 0.059) — but a training loss says nothing about
generalization.

---

## 3. No validation-based selection, no early stopping — every row is the final checkpoint

**不做 val 选优,不 early-stopping,榜上是每个 run 的最终 ckpt。**

This is deliberate: picking the best-of-N checkpoint by val MAE would be model selection on
the leaderboard set, which biases the row optimistically in proportion to the number of
candidates and breaks comparability with legs that report their final checkpoint. Where
intermediate checkpoints exist (SmolVLA saved every 5000 steps) they were treated as a
diagnostic curve, never a selection pool.

The cost is visible on the board. **ACT at 100000 steps is plainly overfit:**

- final training L1 ≈ 0.059 (normalized units)
- train-split MAE **1.185** vs val-split MAE **2.727** — a generalization ratio of **2.30×**,
  measured like-for-like (same checkpoint, eval mode, raw units, K = 8)
- its `arm` column, 2.676, is essentially the hold-state arm line of 2.535

Both splits were checked to be of near-identical intrinsic difficulty (hold-state 3.0799 on
train vs 3.0720 on val, 0.26% apart), so "val is just harder" is not available as an excuse.
And this is **collapse, not a floor**: on the train split ACT's arm beats hold-state by ~1.5
(2.3× better) under both weighting calibers, so the architecture *can* represent this task —
it just does not transfer.

A reader who wants a "best checkpoint per stack" board will have to run one. This is not it.

---

## 4. Low open-loop MAE does not mean the policy works on a robot

**开环 MAE 低 ≠ 闭环能成。**

Every number here is open-loop: the model is fed a *recorded* observation and its prediction
is compared to the *recorded* action. In closed loop the model's own errors move the robot,
which changes the next observation, which compounds. Nothing in this protocol measures that.

We have direct evidence from the deployment line in `realrobot/` that the gap is not
academic. The real failure mode of the production pi0.5 policy on this task is that the
**gripper-close trigger decouples from what the camera sees**: offline, feeding the model a
grasp-configuration frame produces a close command, so the offline error looks fine; during a
rollout the arm drifts slightly off the demonstration manifold and the close simply never
fires. An open-loop MAE over an 8-step window, averaged across 1397 anchors and 8 joints,
cannot see this at all.

Worth stating plainly: **none of the seven backbones on this board has ever been run on the
robot.** See [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 5. Delta parameterization makes the pass line cheap

**delta 参数化让 pass line 变便宜 —— 混合参数化的栈按总 MAE 排名会偏袒 delta 家族。**

The hold-state pass line **is** the delta-zero predictor. For a stack whose training target
is a delta from the current state, the pass line sits at the origin of the model's own output
space and costs nothing to reach.

Measured with no model at all (constants fitted on the 111 train episodes,
`bench/scripts/score/param_space_baselines.py`): a constant predictor that has learned only
the marginal distribution of its target **ties the pass line in delta space (3.064 vs 3.072)
and misses it by 6.9× in absolute space.** So:

- "Beats the do-nothing baseline" is real evidence of learning for an **absolute** stack and
  nearly vacuous for a **delta** stack. It is not the same claim.
- The arm is 7 of 8 dims, so the headline MAE systematically favors the delta-arm family.
- Board split: **absolute** = `act`, `smolvla`, `gr00t_n15`, `openvla_oft`;
  **delta arm + absolute gripper** = `pi0`, `pi05`, `gr00t_n17`.
- No leg on this board has a *delta gripper*, so the gripper column — which is also the
  channel with the most room over the pass line (6.829) — is unconfounded board-wide. If you
  want the least biased single number on this board, read `grip(7)`.

Two things this caveat does **not** say. It does not void the ranking: scoring is
shift-invariant (`MAE(p+s, g+s) = MAE(p, g)`), all rows are scored against the same absolute
ground truth, and the model ordering was checked to be identical at every horizon step
k = 0…7 — only the *pass line's* position moves. And it is not a claim about the delivered
files: every `preds_*.npz` in this repo stores absolute actions, including the delta-trained
ones, which add the state back before writing. The free pass comes from the **training**
space, not the storage format.

---

## 6. Significance, and the fine print

**显著性: 榜上相邻两名的差距不一定稳。**

Uncertainty here means "would this gap survive a different set of 20 held-out episodes?" —
estimated by a delete-one-episode cluster jackknife on the **paired** difference
(`bench/scripts/score/paired_signif.py`).

The headline result:

| comparison | gap | paired SE | σ | verdict |
|---|---|---|---|---|
| `act` vs hold-state (total) | +0.3448 | 0.18284 | **1.9** | not resolved at 2σ |
| `act` vs a random-init model | +17.4761 | 0.90126 | **19.4** | resolved |

ACT unambiguously learned something (19.4σ), and yet **cannot be said to beat doing nothing**
at 2σ on this val set. Four things follow:

- **Pairing does not always shrink the SE.** Against the hold-state line, per-episode errors
  are *negatively* correlated (ρ = −0.16 for ACT), because episodes that are static and easy
  for do-nothing are relatively hard for a model. So the paired SE (0.183) is *larger* than
  either marginal. There is no global σ threshold; compute it per pair.
- **The 20 val episodes are the irreducible uncertainty.** Per-episode MAE spans roughly 2×
  across them. Episode composition dominates decoder sampling noise by ~10×, so more seeds
  buy nothing.
- **Small gaps are weighting-dependent.** Frame-weighted and episode-equal averaging are both
  legitimate and move a model and its baseline in opposite directions. On this board exactly
  one comparison flips sign under reweighting (ACT's arm margin vs hold-state,
  −0.1404 → +0.0029), and it is also the one that is statistically unresolved. A claim that
  flips between two defensible weightings is not a claim.
- **Every margin is a `K = 8` margin.** The pass line degrades much faster with horizon than
  a trained model does (hold-state 1.62 → 4.52 over k = 0…7; ACT 2.34 → 3.20), so
  `act − hold` actually changes sign at k = 3. A row that "barely passes" is a statement
  about K = 8 specifically.

Significance is an **interpretation layer, not a gate**: no row is removed for failing it.
But if you are about to write "X beats Y" about two adjacent rows, check the pair first.

---

## Two smaller things worth knowing

- **The normalization floor is not zero, and it is a config choice.** Each stack's inverse
  transform can clip predictions at the edge of its normalization box, putting a floor under
  its achievable MAE. The measured spread across legs is small in absolute terms (under 0.8%
  of the pass line) but it is not a common offset that cancels — it is driven by whether a
  stack's inverse clips and whether its gripper bounds come from true min/max or from
  quantiles, not by the backbone.
- **Reproducibility of the artifacts is aggregate, not bitwise.** openpi/JAX is bit-exact
  within a process but not across processes; re-running the pi0/pi05 predictions gives
  elementwise differences of ~0.01° and an aggregate MAE shift of 7.7e-05 — about 1000×
  smaller than the smallest gap on the board. Do not use bit-equality of a `preds_*.npz` as
  an integrity check.
