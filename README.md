# NERO-VLA

Seven VLA backbones benchmarked on the same batch of real-robot manipulation data, plus the
real-robot deployment line that batch came from.

## Reproduce the leaderboard in 30 seconds — no GPU, no dataset, no checkpoints

```bash
git clone https://github.com/CodeChildCZJ/nero-vla && cd nero-vla
pip install numpy                                    # Python >= 3.9; numpy is the only dependency
python bench/scripts/score/score.py

# and to check the board you get is the one this repo claims:
python bench/scripts/score/score.py | diff - bench/golden_board.txt && echo "reproduced"
```

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

You do not need any of the upstream repositories for this. The frozen evaluation anchors and all
seven backbones' predictions are committed, and `score.py` depends on numpy alone — so the
board above is a regression test anyone can run, not a claim you have to take on faith.

> ### ⚠️ Read [docs/CAVEATS.md](docs/CAVEATS.md) before citing any number above
>
> This is a **recipe-level comparison, not a controlled architecture A/B** — the runs differ
> by 10× in optimizer steps and 8× in batch size, because each backbone follows its own
> framework's official fine-tuning recipe. And every number here is **open-loop**: low action
> MAE on held-out demonstrations does not mean the policy works on a robot. **None of these
> seven checkpoints has ever been run on hardware.**
>
> **引用上面任何数字之前,先读 [docs/CAVEATS.md](docs/CAVEATS.md)。** 这是 recipe 级对比,
> 不是受控的架构 A/B;而且全部是离线开环评测 —— 开环 MAE 低不代表真机能成,
> **榜上 7 个 backbone 一个都没上过真机。**

---

## Two lines of work in one repository

| | `bench/` | `realrobot/` |
|---|---|---|
| Question | which of 7 backbones fits this data best | get pi0.5 to actually pick up the sponge |
| Split | leak-free 111 / 20 episode split | **no train/val split at all** |
| Validation signal | held-out open-loop action MAE | real-robot closed-loop success |
| Status | 7/7 complete | production model = pi0.5 `b2_ref` @ step 29999 |

They share a dataset and a base model family and nothing else. The benchmark answers a
question the deployment line could not afford to ask (it has no held-out set); the deployment
line demonstrates a failure mode the benchmark structurally cannot see.

---

## Where things are

| path | what |
|---|---|
| [`bench/`](bench/README.md) | the 7-backbone offline benchmark — data prep, training, prediction, scoring |
| `bench/data/` | `split.json` + the frozen `val_anchors.npz` (1397 anchors, K = 8) |
| `bench/preds/` | seven `preds_*.npz` + the meta files that exist |
| `bench/scripts/score/score.py` | the scorer. numpy only, self-locating, 67 lines |
| `bench/archive/` | one-off audit and diagnostic scripts, kept verbatim, with an annotated `INDEX.md` |
| [`realrobot/`](realrobot/README.md) | the NERO pi0.5 deployment line — training and serving |
| `realrobot/archive/` | the diagnostic scripts behind every conclusion on that line |
| `third_party/` | six upstream repositories, pinned — see [`third_party/UPSTREAM.md`](third_party/UPSTREAM.md) |
| `patches/` | our diffs against those upstreams, exported for audit and replay |
| `env.example.sh` | copy to `env.sh` and fill in — **only needed to re-train or re-predict** |

## Documentation

| | |
|---|---|
| [docs/CAVEATS.md](docs/CAVEATS.md) | **start here** — six ways this board can be over-read |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | split, anchors, the prediction contract, scoring, the pass line, significance |
| [docs/LEADERBOARD.md](docs/LEADERBOARD.md) | the board plus each stack's actual training config |
| [docs/DATA.md](docs/DATA.md) | dataset schema, unit conventions, how to run this protocol on your own data |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | the real-robot closed loop and its failure modes |

## What this benchmark measures, in one paragraph

131 episodes of a single task (`pick the pink sponge and place it in the blue bucket`) on a
7-DoF arm with a gripper and two cameras. 20 episodes are held out at the episode level, and
1397 anchor frames are sampled from them every 5th frame. Each backbone is fine-tuned on the
other 111 episodes with its own framework's official recipe, then asked to predict the next
8 actions at every anchor. The score is the mean absolute error against the recorded
teleoperator actions, in raw joint units. The `[hold-state]` row — predict that the action
equals the current joint state — is the do-nothing pass line every real model must beat.

The dataset itself is **not published**; the split, the anchors, and all predictions are.

## License

Apache-2.0 for this repository's own code and documents (see [LICENSE](LICENSE)). Everything
under `third_party/` is governed by its own upstream license, unchanged.

## Citing

See [CITATION.cff](CITATION.cff).
