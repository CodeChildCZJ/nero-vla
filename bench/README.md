# `bench/` — the 7-backbone offline benchmark

One dataset, one split, one prediction contract, seven VLA backbones, one number each.

Everything needed to **reproduce the leaderboard** is committed here. Everything needed to
**regenerate** it (dataset, checkpoints, GPUs) is not — see [Retraining from scratch](#retraining-from-scratch).

---

## Reproduce the board

```bash
pip install numpy
python bench/scripts/score/score.py
```

That is the whole dependency list. `score.py` locates itself from `__file__`, reads
`bench/data/val_anchors.npz` and every `bench/preds/preds_*.npz`, and prints the board. No
GPU, no dataset, no checkpoint, no upstream repositories, no environment variables.

This is also the repository's main regression test: any change that alters the printed board
is either a bug or a deliberate, documented result change.

---

## Layout

```
bench/
├─ data/
│  ├─ split.json          111 train / 20 val episode ids (seed 42, episode-level)
│  └─ val_anchors.npz     1397 frozen anchors: episodes, frames, gt (1397,8,8), state, K=8, stride=5
├─ preds/
│  ├─ preds_<bb>.npz      one per backbone, key "pred", shape (1397, K'>=8, 8)
│  └─ preds_<bb>_meta.json   run + audit record — exists for pi05, pi0, gr00t_n15, act only
├─ scripts/
│  ├─ prepare/            split, anchors, and the per-stack derived datasets
│  ├─ train/              one launcher per stack
│  ├─ predict/            one predict script per stack + the post-training pipelines
│  └─ score/              score.py (the board) + paired_signif.py + param_space_baselines.py
├─ configs/               GR00T modality/data configs (nero_data_config.py, nero_n17_config.py)
└─ archive/               89 one-off audit scripts, verbatim, with an annotated INDEX.md
```

The protocol these files implement is specified in
[`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md). Its limits are in
[`docs/CAVEATS.md`](../docs/CAVEATS.md).

---

## Retraining from scratch

You need: the dataset (not published — schema in [`docs/DATA.md`](../docs/DATA.md)), the
the six upstream repositories with their virtualenvs built, and a GPU per leg.

```bash
cp env.example.sh env.sh && $EDITOR env.sh   # NERO_DATA, NERO_CKPT, WANDB_ENTITY
source env.sh
```

### 1. Prepare

```bash
python bench/scripts/prepare/make_split.py           # -> bench/data/split.json
python bench/scripts/prepare/build_val_anchors.py    # -> bench/data/val_anchors.npz
```

> Both of these **overwrite the frozen artifacts** that make the board reproducible. Running
> them regenerates the split and the anchors; every committed `preds_*.npz` was produced
> against `val_anchors.npz` md5 `cc34dda2a99ade1011c192e2b3e4e343` and becomes
> incomparable if you change it. If you only want to re-train, skip this step.

Then the per-stack derived datasets — each one filters to the 111 train episodes at
materialization time, so no stack can leak val statistics through its normalizer:

```bash
python bench/scripts/prepare/make_b2_train_dataset.py   # openpi
python bench/scripts/prepare/prepare_b2_v30.py          # LeRobot (v3.0 copy, train-only stats)
python bench/scripts/prepare/build_gr00t_dataset.py     # GR00T (+ H.264 transcode)
python bench/scripts/prepare/oft_extract_frames.py      # OpenVLA-OFT (256x256 JPEGs)
```

### 2. Train

| backbone | command |
|---|---|
| `pi05` | `bash bench/scripts/train/train_openpi_bench.sh pi05_nero_b2_train <gpu> pi05_b2train` |
| `pi0` | `bash bench/scripts/train/train_openpi_bench.sh pi0_nero_b2_train <gpu> pi0_b2train` |
| `act` | `bash bench/scripts/train/lerobot_train_bb.sh act <gpu>` |
| `smolvla` | `bash bench/scripts/train/lerobot_train_bb.sh smolvla <gpu>` |
| `gr00t_n15` | `CUDA_VISIBLE_DEVICES=<gpu> bash bench/scripts/train/train_gr00t_n15.sh` |
| `gr00t_n17` | `CUDA_VISIBLE_DEVICES=<gpu> bash bench/scripts/train/train_gr00t_n17.sh` |
| `openvla_oft` | `bash bench/scripts/train/oft_train.sh <gpu> [steps] [batch] [accum] [save_freq]` |

Every launcher pins a single GPU on purpose. For the JAX legs this is not optional: without
`CUDA_VISIBLE_DEVICES` set, JAX builds its mesh over *every* visible device and then dies on
`batch_size % n_devices`.

The exact hyper-parameters each row was trained with — and the cells where nothing was
recorded — are in [`docs/LEADERBOARD.md`](../docs/LEADERBOARD.md).

### 3. Predict

```bash
python bench/scripts/predict/predict_openpi.py --config-name pi05_nero_b2_train \
    --ckpt-dir "$NERO_CKPT/pi05_nero_b2_train/pi05_b2train/29999" \
    --out "$NERO_ROOT/bench/preds/preds_pi05.npz"
python bench/scripts/predict/predict_gr00t.py      ...    # gr00t_n15
python bench/scripts/predict/predict_gr00t_n17.py  ...    # gr00t_n17
python bench/scripts/predict/lerobot_predict.py    ...    # act, smolvla
python bench/scripts/predict/oft_predict.py        ...    # openvla_oft
```

The `*_post_train.sh` / `post_train_pipeline.sh` scripts chain wait-for-training → predict →
score → variance footnote → strip optimizer state, and are what actually produced the rows.
They put predict and score **first** so that a hanging audit costs its own artifact and never
the leaderboard row.

### 4. Score

```bash
python bench/scripts/score/score.py                       # the board
python bench/scripts/score/paired_signif.py --all         # uncertainty on every pair
python bench/scripts/score/param_space_baselines.py       # what the action space gives for free
```

`paired_signif.py` reads sampling footnotes from `bench/logs/variance/variance_<bb>.json`,
which are not shipped; it prints `[no variance footnote]` rather than assuming zero.
`param_space_baselines.py` needs the dataset.

---

## Adding an eighth backbone

`score.py` globs `preds_*.npz`, so a new row needs **no code change** — only a file that
honours the contract:

```
bench/preds/preds_<yourbb>.npz
  pred        (1397, K'>=8, 8)   float, absolute action units, row i == anchor i
  anchor_idx  (1397,)            int, strongly recommended
```

Four things to get right:

1. **Train on `split.json:train_episodes` only** — and check that your *normalization
   statistics* came from those 111 episodes too. Filtering the sampler is not always enough:
   LeRobot's `--dataset.episodes=` filters frames but does **not** filter `meta/stats.json`.
2. **Invert everything back to the dataset's absolute action space.** If your model emits
   deltas or normalized values, undo that before writing. This is the single most common way
   a row comes out wrong. Cheap self-check: `mean|pred[:,0,:7] − state|` should be ~1–3
   (degrees), not ~38 — 38 is `mean|gt_arm|`, i.e. what you get when raw deltas are scored
   against absolute ground truth.
3. **Gate on finiteness before writing**, scoped to `pred[:, :8, :]`. A single NaN does not
   just spoil your row: it silently misorders the *whole* board.
4. **Ship a `preds_<yourbb>_meta.json`** with checkpoint, steps, batch, learning rate,
   per-channel action parameterization, and whether decoding is stochastic. Three of the
   seven existing rows do not have one, and that is exactly why
   [`docs/LEADERBOARD.md`](../docs/LEADERBOARD.md) has "not recorded" cells.

Then re-read [`docs/CAVEATS.md`](../docs/CAVEATS.md) before comparing your number to anyone
else's — your recipe will differ from all seven of these, and that difference is part of your
result.

---

## `archive/`

`bench/archive/` holds the one-off scripts written during this benchmark to answer a specific
objection: leakage falsification, normalization-floor measurements, seed-collision audits,
weighting-caliber sweeps, video integrity checks, and so on. They are kept **verbatim**, not
cleaned up, so that every claim in the docs has code behind it. `bench/archive/INDEX.md`
lists what each one measured and what it found.

They are not on the reproduction path. Most of them need the dataset or a checkpoint.
