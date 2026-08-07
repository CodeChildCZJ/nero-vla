# Data specification

**The dataset itself is not published.** What is published is the split, the frozen
evaluation anchors, every backbone's predictions, and the exact schema below — enough to
reproduce the leaderboard byte-for-byte, and enough to run the same protocol on your own
robot data.

This page describes `pick_pink_sponge_b2`, the set every row of the
[leaderboard](LEADERBOARD.md) was trained and scored on.

---

## 1. Format

**HF LeRobot dataset, `codebase_version` v2.1.**

```
$NERO_DATA/
├─ meta/
│  ├─ info.json              # features, fps, robot_type, episode/frame counts
│  ├─ episodes.jsonl         # one line per episode (length, tasks)
│  ├─ episodes_stats.jsonl   # per-episode min/max/mean/std per feature
│  └─ tasks.jsonl            # task-index -> natural-language task string
├─ data/
│  └─ chunk-000/
│     └─ episode_%06d.parquet
└─ videos/
   └─ chunk-000/
      ├─ observation.images.cam_high/episode_%06d.mp4
      └─ observation.images.cam_wrist/episode_%06d.mp4
```

| property | value |
|---|---|
| episodes | 131 |
| frames | 45131 |
| fps | 30 |
| `robot_type` | `cobot_magic` |
| video codec | AV1 |

> **AV1 is a practical trap.** Neither `decord` nor `torchcodec` 0.1 can decode it, so the
> GR00T legs consume a shared H.264 transcode produced by
> `bench/scripts/prepare/build_gr00t_dataset.py`. If your own data is AV1, expect to
> transcode.

---

## 2. Columns

Per frame, in the parquet:

| column | shape | meaning |
|---|---|---|
| `observation.state` | `(8,)` | **follower** arm joints j0–j6 + gripper j7 |
| `action` | `(8,)` | **leader** arm joints j0–j6 + gripper j7 |
| `episode_index`, `frame_index`, `index`, `timestamp`, `task_index` | scalars | LeRobot bookkeeping |

`action` is the quantity every backbone is trained to predict and the quantity the benchmark
scores. It is the teleoperator's leader-arm command, not the follower's readback — the two
are not the same signal, and confusing them is how the real-robot line produced a
gripper-copycat policy (see [`realrobot/README.md`](../realrobot/README.md)).

Images, decoded from the videos at the corresponding frame:

| key | resolution | note |
|---|---|---|
| `observation.images.cam_high` | 480×640 RGB | overhead / scene camera |
| `observation.images.cam_wrist` | 480×640 RGB | wrist camera |

**The two cameras are not interchangeable.** Every stack has its own name for them, and a
swap is silent: it produces a plausible-looking model that simply does not work. The video
integrity check shipped with the ACT leg explicitly records this as its blind spot — it can
prove all 1397×2 decoded frames are distinct and non-degenerate, and it *cannot* detect a
`cam_high`/`cam_wrist` label swap.

**Prompt / task string** (single task, identical for every frame):

```
pick the pink sponge and place it in the blue bucket
```

---

## 3. Units

This is the single most error-prone part of the schema. There are **two different gripper
conventions in this project** and they belong to two different places.

**Inside the dataset (what the benchmark scores):**

| dims | unit | measured range over the val ground truth |
|---|---|---|
| j0–j6 (arm) | degrees | −100.876 … 125.897 |
| j7 (gripper) | a 0–100 scale | 0.000 … 100.908 |

Note the gripper range *exceeds* 100, so it is not a clipped percentage either. This was
measured off `bench/data/val_anchors.npz`, not read off a spec sheet — an earlier annotation
that called it "mm, 0 = closed / 76 = open" was wrong by 1.33× in the one field whose only
purpose is cross-stack comparison.

**On the physical robot (what the serving contract speaks):** the gripper is in
**millimetres, 0 mm = closed and 76 mm = fully open**. The production pi0.5 checkpoint was
trained on a re-encoded copy in radians + gripper `[0, 1]` (= mm / 76), with that conversion
baked into the dataset rather than the code. See [DEPLOYMENT.md](DEPLOYMENT.md) — do not
carry the benchmark's units onto the robot.

---

## 4. Derived datasets

Nothing trains on `$NERO_DATA` directly. Each stack materializes what it needs, and the
train/val filtering is done at *materialization* time so that normalization statistics cannot
leak:

| script | output | used by |
|---|---|---|
| `bench/scripts/prepare/make_split.py` | `bench/data/split.json` | everything |
| `bench/scripts/prepare/build_val_anchors.py` | `bench/data/val_anchors.npz` | scoring, all predict scripts |
| `bench/scripts/prepare/make_b2_train_dataset.py` | v2.1 111-episode subset (parquet rewritten, videos symlinked) | openpi (`pi0`, `pi05`) |
| `bench/scripts/prepare/prepare_b2_v30.py` | v3.0 working copy with `meta/stats.json` re-aggregated over the 111 train episodes | LeRobot (`act`, `smolvla`) |
| `bench/scripts/prepare/build_gr00t_dataset.py` | H.264 train/val datasets | GR00T (`n15`, `n17`) |
| `bench/scripts/prepare/oft_extract_frames.py` | 256×256 JPEGs + per-episode state/action arrays | `openvla_oft` |

All of them treat `$NERO_DATA` as read-only.

The v3.0 conversion is worth calling out because it is **in-place and destructive** in
upstream LeRobot (`lerobot.scripts.convert_dataset_v21_to_v30` rewrites the directory and
removes the old aggregate stats), which is why `prepare_b2_v30.py` converts a copy.

---

## 5. Running this protocol on your own robot data

You do not need our dataset. You need a LeRobot-style set with:

1. an 8-dim (or N-dim — the scripts assume 8, adjust `score.py`'s `arm`/`grip` slices)
   `observation.state` and a same-shaped `action`, in a consistent absolute unit;
2. at least one image stream per frame;
3. a single task string, or per-episode task strings your stack can consume.

Then:

```bash
source env.sh                                     # see env.example.sh at the repo root

python bench/scripts/prepare/make_split.py        # episode-level split, seed 42
python bench/scripts/prepare/build_val_anchors.py # frozen anchors, K=8, stride 5

# ... train your backbone on split.json:train_episodes ONLY, then:
#     produce bench/preds/preds_<yourbb>.npz per the prediction contract

python bench/scripts/score/score.py
```

Three things to get right, in order of how much damage they do:

- **Absolute action space.** Predictions must be inverted back into the same space as
  `action`. See [METHODOLOGY.md §4](METHODOLOGY.md#4-the-prediction-contract).
- **Normalization statistics must be train-only.** Filtering the *sampler* is not enough in
  every framework — LeRobot's `episodes=` kwarg does not filter `meta/stats.json`.
- **Row order.** Row `i` of your npz must be anchor `i`. Ship an `anchor_idx` array so the
  claim is checkable instead of assumed.

`make_split.py` hardcodes `N = 131`; change it to your episode count. Everything downstream
reads `split.json`.
