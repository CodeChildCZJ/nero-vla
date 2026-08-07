# Methodology

How every number on the [leaderboard](LEADERBOARD.md) is produced. This document is the
protocol; `bench/scripts/score/score.py` is its executable form and wins any disagreement
with this text.

Before citing any result, also read [CAVEATS.md](CAVEATS.md) — it says what this benchmark
is *not*.

---

## 1. What is measured

**Offline, open-loop, action-chunk error on held-out episodes.**

Each backbone is fine-tuned on the same real-robot demonstration set, then asked — at a set
of frozen observation timestamps it never trained on — to predict the next `K = 8` actions.
The score is the mean absolute error against the recorded teleoperator actions, in the
dataset's own raw joint units.

There is **no simulator, no closed loop and no real-robot rollout anywhere in this
benchmark.** A row on the board says "this recipe fits held-out demonstration actions this
well", and nothing more.

---

## 2. Dataset and split

The training data is a single-task real-robot set (`pick the pink sponge and place it in the
blue bucket`) recorded on a dual-camera 7-DoF arm plus gripper. Full schema in
[DATA.md](DATA.md). **The dataset itself is not published**; the split, the anchors and every
backbone's predictions are.

| | value | source |
|---|---|---|
| total episodes | 131 | `bench/data/split.json` |
| train episodes | 111 | `split.json:train_episodes` |
| val episodes | 20 | `split.json:val_episodes` |
| train frames | 38025 | measured, `bench/preds/preds_act_meta.json` |
| fps | 30 | `split.json` |
| split seed | 42 | `bench/scripts/prepare/make_split.py` |

The split is **episode-level**, drawn once by `make_split.py` with
`numpy.random.default_rng(42).permutation(131)`, and is identical for all seven backbones:

```
val_episodes = [0, 2, 3, 4, 20, 24, 25, 33, 37, 51,
                56, 68, 70, 80, 90, 91, 92, 101, 106, 111]
```

**Leak discipline.** No frame of any val episode may enter training — *including through the
normalization statistics*. Each stack filters episodes differently, and the traps are
stack-specific:

- LeRobot's `--dataset.episodes=` filters **frames only**; `meta/stats.json` is *not*
  filtered by it (measured: bit-identical across 111-, 131- and 20-episode loads). The ACT
  and SmolVLA legs are clean because `bench/scripts/prepare/prepare_b2_v30.py` rewrote
  `meta/stats.json` to the train-111 aggregate — not because of the `episodes=` kwarg.
- openpi bakes `norm_stats.json` into the checkpoint; the bench configs point at a separate
  `local/pick_pink_sponge_b2_train` repo id whose stats were computed over the 111 train
  episodes only.
- GR00T and OFT consume a physically separate train-only dataset built by
  `bench/scripts/prepare/build_gr00t_dataset.py` / `oft_extract_frames.py`.

A cheap falsification test that does not require trusting the filtering code: if any val
frame exceeds a stat's recorded `max`, that stat could not have seen it. This is recorded
per-leg in the `leak_check` block of the `*_meta.json` files that have one.

---

## 3. Validation anchors

`bench/scripts/prepare/build_val_anchors.py` produces the frozen evaluation set
`bench/data/val_anchors.npz`:

| key | shape | meaning |
|---|---|---|
| `episodes` | `(1397,)` int32 | which val episode this anchor is in |
| `frames` | `(1397,)` int32 | frame index `t` inside that episode |
| `gt` | `(1397, 8, 8)` float32 | recorded `action[t : t+8]` — the target |
| `state` | `(1397, 8)` float32 | `observation.state[t]` — the proprioceptive input |
| `K` | scalar | 8 |
| `stride` | scalar | 5 |

Anchors are taken every 5th frame of every val episode, for `t` in `range(0, L-8, 5)`.
That gives **A = 1397 anchors over 20 episodes**, 49–113 anchors per episode (median 63).

`K = 8` is the largest horizon *every* backbone on the board can supply — it is a choice
forced by the shortest chunk (OFT's `num_actions_chunk = 8`), not a physical constant.
Every margin quoted anywhere in this repo is a `K = 8` margin; see
[CAVEATS.md](CAVEATS.md#6-significance-and-the-fine-print).

This file is **frozen**. Regenerating it invalidates every `preds_*.npz` in the repo, and
`score.py` carries a hard exit for the one way that would silently corrupt the whole board
(a non-finite cell in `gt`/`state` would make the pass line NaN, disabling the
"learned nothing" flag for *every* row at once).

---

## 4. The prediction contract

This is the entire interface between a backbone and the leaderboard. Anything that satisfies
it is scorable; nothing else about the stack matters.

For anchor `i` the model observes, at frame `frames[i]` of episode `episodes[i]`:

- `observation.state` — the 8-dim joint vector (`state[i]` in the anchor file)
- `observation.images.cam_high` and `observation.images.cam_wrist` — both 480×640 RGB
- the prompt `pick the pink sponge and place it in the blue bucket`

and emits an action chunk. Deliverable:

```
bench/preds/preds_<backbone>.npz
  key "pred", shape (1397, K', 8), K' >= 8, float
  row i corresponds to anchor i of val_anchors.npz, in order
```

**The number-one pitfall: predictions must be in the dataset's absolute action space.**
If the model internally predicts deltas, or normalized values, the predict script must invert
that before writing. A raw-delta file scored against this absolute `gt` lands at arm MAE
≈ 38 (that is `mean|gt_arm|`) instead of ≈ 1–3, so the failure is loud — but a *partially*
inverted one is not. Legs that emit deltas internally gate on this at write time: openpi's
`predict_openpi.py` refuses to write unless subtracting `state[t]` shrinks the arm chunk.

Two optional-but-recommended extras, both of which exist on some rows and not others:

- `anchor_idx` — an explicit `(1397,)` array asserting the row ordering. `act`, `pi0`,
  `pi05` and `smolvla` ship it; `gr00t_n15`, `gr00t_n17` and `openvla_oft` do not. For a leg
  without it, the ordering claim was instead demonstrated by re-running a 200-anchor subset
  in a separate process and requiring bit-exact agreement with the corresponding rows of the
  full file. `anchor_idx` should be mandatory for anything delivered from here on.
- A **write-side finiteness gate**: refuse to write the file at all if `pred[:, :8, :]` has
  any NaN/Inf. Scope it to the scored prefix, not the whole array, or it false-fires on
  padding past the window.

---

## 5. Scoring

`bench/scripts/score/score.py` globs `bench/preds/preds_*.npz`, truncates each to the shared
`K = 8` prefix and computes, over all anchors and all 8 horizon steps:

| column | definition |
|---|---|
| `MAE` | `mean(abs(pred - gt))` over anchors × 8 steps × 8 dims |
| `arm(0-6)` | the same mean restricted to dims 0–6 |
| `grip(7)` | the same mean restricted to dim 7 |
| `per-joint j0..j7` | the same mean per dim |

Rows are sorted by `MAE`, lower is better. Its only dependency is numpy, and it derives its
root from `__file__`, so a fresh clone reproduces the board with no environment set up:

```bash
python bench/scripts/score/score.py
```

Two ingestion gates are load-bearing and deliberately fail loudly:

- Any preds file with a non-finite cell in the `K = 8` prefix is **excluded**, not warned
  about. A NaN row does not just lose its own line — `mae >= HOLD_MAE` is `False` for NaN so
  the "learned nothing" flag never fires, and a NaN sort key breaks the sort's transitivity,
  so the final *ranking* depends on that file's position in glob order.
- A non-finite cell in `val_anchors.npz` itself is a hard `sys.exit`. The pass-line row is
  appended unconditionally and passes no gate, so there is no degraded mode without it.

---

## 6. The pass line

The board's baseline is **hold-state**: predict that the action equals the current
proprioceptive state, broadcast over all 8 horizon steps. It is not a trained model and
takes no input beyond `state`.

```
[hold-state]   MAE 3.072   arm 2.535   grip 6.829
```

(exactly `3.0719962120056152` in float32; the board prints 3 decimals.)

A backbone that does not beat 3.072 has learned nothing useful and `score.py` flags it
inline. Two things about this line that are easy to get wrong, both expanded in
[CAVEATS.md](CAVEATS.md):

- It is **much** harder to beat on the arm (2.535) than on the gripper (6.829), because the
  arm moves ~1°/frame. The arm is 7 of 8 dims, so the headline MAE dilutes the one
  discriminative channel roughly 7×. Always read the `grip(7)` column too.
- Hold-state **is** the delta-zero predictor, so "beats the pass line" is strong evidence of
  learning for an absolute-action stack and nearly vacuous for a delta-action one.

---

## 7. Sampling-variance protocol

Five of the seven backbones (`pi0`, `pi05`, `gr00t_n15`, `gr00t_n17`, `smolvla`) decode by
sampling a flow/diffusion prior, so a single draw carries noise that the deterministic legs
(`act`, `openvla_oft`) do not.

- The **board row** is one draw at a fixed, recorded seed. Seeds are per-`(draw, anchor)`,
  derived from the *global* anchor index so a row cannot depend on iteration order.
- The **spread** is quantified separately on a shared **200-anchor subset × 5 draws** and
  reported as a footnote. It is deliberately kept out of `bench/preds/` so it can never enter
  the board.
- The 200-anchor subset is shared across legs and identified by md5
  `9007df1fcec5eca2a1f01591c6266b43` (stratified, 10 anchors × 20 episodes).
- Deterministic legs report "deterministic, no sampling variance" — and demonstrate it,
  either with distinct per-draw seeds or with a power control showing the comparator would
  have noticed a difference. A hardcoded `std: 0.0` is not evidence.

Two traps worth restating because both were hit here:

- Use the **per-draw std**, not `std/sqrt(R)`. The board row is one realization.
- The seed *stride* must exceed the anchor count. With `base + stride*draw + anchor_index`
  and a small stride, draw `d` anchor `g` reuses draw `d+1` anchor `g-stride`'s noise, and
  the measured spread comes out too small — while a distinctness check on the per-draw seeds
  passes green.

Measured on this board the sampling term is a formality: episode composition dominates
seed-to-seed variance by roughly 10×, and folding a real footnote into a significance verdict
moved it by ~0.35%.

---

## 8. Significance layer

`bench/scripts/score/paired_signif.py` is hung off the side of the frozen scorer — it reads
the same `preds_*.npz` afterwards and never modifies `score.py`.

- The resampling unit is the **episode** (delete-one-episode cluster jackknife, 20 clusters).
  Anchors within an episode are correlated, so a per-anchor SE would overstate `n`.
- Jackknife the **paired difference** per pair, never quote a global threshold. All rows are
  scored on the same 20 episodes, so shared episode difficulty partly cancels — but only if
  the two rows' per-episode errors are positively correlated. Against the hold-state line
  they are *negatively* correlated (measured ρ = −0.16 for ACT), so pairing **amplifies** the
  SE rather than shrinking it. Measured shrink factors on this board span 0.92× to 4.01×.
- Report **per channel** (total / arm / grip), because the total hides the arm/grip split.
- Report both **frame-weighted** (the caliber `score.py` ranks by) and **episode-equal**
  weighting. Reweighting is a deterministic shift that moves a model and its baseline in
  *opposite* directions, so it threatens small differences and leaves large ratios alone.
  The tool flags a pair whose sign or whose 2σ verdict is caliber-dependent.

`paired_signif.py` reads sampling footnotes from `bench/logs/variance/variance_<bb>.json`.
Those log files are **not shipped in this repository**, so a fresh clone gets the episode
term only and prints `[no variance footnote]`. Given the ~10× ratio above, that changes no
verdict — but the column is honest about it rather than silently substituting 0.

---

## 9. Parameterization baselines

`bench/scripts/score/param_space_baselines.py` prices what the *action space* hands a model
for free, with no model at all: it fits per-`(horizon step, joint)` constants on the 111
train episodes and scores them on the same 1397 anchors.

- `hold-state` — the pass line
- `const-absolute` — the mean absolute action
- `const-relative` — `state + mean delta`
- `const-N1.7-mixed` — relative arm + absolute gripper, matching the N1.7 recipe

This is the measurement behind [caveat 5](CAVEATS.md#5-delta-parameterization-makes-the-pass-line-cheap).
Note it needs the (unpublished) dataset, so it cannot be re-run from a bare clone.

---

## 10. What is deliberately *not* in the protocol

- **No validation-based model selection and no early stopping.** Each row is the *final*
  checkpoint of its run, committed before anyone looked at val. Intermediate checkpoints,
  where they exist, are a diagnostic curve and never a selection pool — best-of-N by val MAE
  is model selection on the leaderboard set.
- **No held-out val *loss*.** Only training loss was logged. The board's val MAE is the only
  held-out signal any of these rows has.
- **No real-robot evaluation.** See [DEPLOYMENT.md](DEPLOYMENT.md).
- **No matched training budget.** Each leg follows its own framework's official fine-tuning
  recipe, which is the thing being compared. See
  [caveat 1](CAVEATS.md#1-this-is-a-recipe-level-comparison-not-a-controlled-architecture-ab).

---

## 11. Adding an eighth backbone

`score.py` globs, so a new row needs no code change:

1. Train on `split.json:train_episodes` only — and check that your normalization statistics
   were built from those 111 episodes too.
2. Run inference at every anchor in `bench/data/val_anchors.npz`, in order.
3. Invert whatever internal parameterization you use back to the dataset's absolute action
   space.
4. Write `bench/preds/preds_<yourbb>.npz` with key `pred`, shape `(1397, K'>=8, 8)`, plus an
   `anchor_idx` array.
5. `python bench/scripts/score/score.py`.

Ship a `preds_<yourbb>_meta.json` alongside recording at minimum: checkpoint, training steps,
batch size, learning rate, action parameterization (per channel: `delta` or `absolute`), and
whether decoding is stochastic. The four meta files that exist today are the template — and
the three that do not exist are why [LEADERBOARD.md](LEADERBOARD.md) has "not recorded" cells.
