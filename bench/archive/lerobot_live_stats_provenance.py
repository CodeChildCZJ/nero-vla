#!/usr/bin/env python3
"""Which normalization stats does the LIVE predict-path processor actually hold?

Motivation (team-lead's 3rd leak channel + gr00t-n17's live-object lesson): the
earlier leak check compared the stats FILE baked into the checkpoint against a
train-only recompute. That proves the file clean; it says nothing about which
stats the processor object built at predict time ends up holding.

`lerobot_predict.py` calls

    make_pre_post_processors(cfg, pretrained_path=ckpt, dataset_stats=ds.meta.stats, ...)

i.e. it hands the factory a dataset_stats kwarg *alongside* the checkpoint path.
Reading `policies/factory.py` says the non-Groot branch drops that kwarg when
`pretrained_path` is set -- a source read, not a measurement.

WHY THE OBVIOUS TEST HAS NO POWER HERE (and is therefore not the test run):
predict opens `pick_pink_sponge_b2_bench`, whose `meta/stats.json` was already
re-aggregated over the 111 train episodes by prepare_b2_v30.py. So
`ds.meta.stats` == the train-only stats == what the checkpoint should hold.
Comparing live-vs-ds.meta would print 0.0 whether the kwarg is honoured or
ignored: identical output under both hypotheses = zero power, and its silence
would mean nothing.

So the kwarg is tested by INJECTION instead: build the processors a second time
with a deliberately poisoned dataset_stats and check the live tensors do not
move. That is a check that can fail, and the poison magnitude is reported so the
reader can see the failure would have been visible.

Three things get measured:
  1. INERTNESS  -- live(normal stats) vs live(poisoned stats).  Want bit-equal.
  2. PROVENANCE -- live vs TRAIN-111 recompute (want ~0, float32 artifact scale)
                   vs FULL-131 recompute (the leak control; want >> that).
  3. POWER      -- the poison offset and the train/full control spacing, printed,
                   so a PASS is readable as "would have caught it" not just "quiet".

Residuals print as a triple against the control spacing rather than a
BIT-EXACT/DIFFERS boolean -- a 1e-6 float32 artifact and a real ~1.0 leak must
not render as the same string (gr00t-n15).

Comparisons are done in the dtype the LIVE object holds: LeRobot bakes float32
safetensors, and widening one side to float64 while leaving the other float32
manufactures ~2e-6 phantom diffs on mean/std only (and, per gr00t-n15, narrowing
the wrong side manufactures them on quantiles). Never cast only one side.

CPU only, no GPU, no model weights.
"""

import os
import argparse
import json
import pathlib
import sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]


def per_episode_stats(root: pathlib.Path) -> dict[int, dict]:
    """Read meta/episodes/**.parquet and un-flatten the `stats/<key>/<stat>` columns."""
    import pandas as pd

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        sys.exit(f"no meta/episodes parquet under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = {}
    stat_cols = [c for c in df.columns if c.startswith("stats/")]
    for _, row in df.iterrows():
        d: dict[str, dict] = {}
        for c in stat_cols:
            _, key, stat = c.split("/", 2)
            # aggregate_stats hard-requires numpy arrays ("Stats must be composed of
            # numpy array ... is of type list") -- parquet hands back lists/ndarrays.
            v = np.asarray(row[c], dtype=np.float32)
            # Image stats are per-channel (3,1,1); parquet flattens them to (3,) and
            # aggregate_stats hard-rejects the flat form. Restore the channel axes
            # rather than dropping the image keys -- they are normalized too, so
            # excluding them would silently shrink this check's coverage.
            if key.startswith("observation.images") and stat != "count" and v.ndim == 1:
                v = v.reshape(-1, 1, 1)
            d.setdefault(key, {})[stat] = v
        out[int(row["episode_index"])] = d
    return out


def agg(per_ep: dict[int, dict], eps: list[int], label: str) -> dict:
    """Aggregate using LeRobot's OWN shipped function -- comparing against a
    reimplementation would test my arithmetic, not the training path's."""
    from lerobot.datasets.compute_stats import aggregate_stats

    chosen = [per_ep[e] for e in eps if e in per_ep]
    if len(chosen) != len(eps):
        sys.exit(f"[{label}] wanted {len(eps)} episodes, found {len(chosen)}")
    print(f"[{label}] aggregated {len(chosen)} episodes")
    return aggregate_stats(chosen)


def as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def grab(pipeline) -> dict[str, dict]:
    """Pull the stats dicts out of whatever (un)normalize steps the pipeline holds."""
    found = {}
    for step in pipeline.steps:
        name = type(step).__name__
        if "ormaliz" not in name:
            continue
        for attr in ("stats", "_stats", "dataset_stats"):
            st = getattr(step, attr, None)
            if st:
                found[name] = st
                break
    return found


def flatten(live: dict[str, dict]) -> dict[tuple, np.ndarray]:
    return {
        (step, key, stat): as_np(v)
        for step, st in live.items()
        for key, d in st.items()
        for stat, v in d.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="<...>/checkpoints/last/pretrained_model")
    ap.add_argument("--repo-id", default="local/pick_pink_sponge_b2_bench")
    ap.add_argument("--root", default=os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench")))
    ap.add_argument("--split", default=str(BENCH / "data" / "split.json"))
    ap.add_argument("--poison", type=float, default=1000.0)
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    split = json.loads(pathlib.Path(args.split).read_text())
    val_eps = sorted(split["val_episodes"])
    train_eps = sorted(split["train_episodes"])
    all_eps = sorted(train_eps + val_eps)

    # Rebuild the predict-time dataset EXACTLY as lerobot_predict.py does.
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=val_eps)
    print(f"[dataset] {ds.num_episodes} eps / {ds.num_frames} frames (episodes=val)")

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cpu"

    def build(stats):
        return make_pre_post_processors(
            cfg,
            pretrained_path=args.ckpt,
            dataset_stats=stats,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    # --- run 1: the real call predict makes -------------------------------------
    pre_a, post_a = build(ds.meta.stats)
    live_a = flatten({**grab(pre_a), **grab(post_a)})
    if not live_a:
        sys.exit("FAIL: no normalize step holding stats -- pipeline layout changed, test is blind")
    steps = sorted({k[0] for k in live_a})
    print(f"[live] stat-holding steps: {steps}  ({len(live_a)} tensors)")

    # --- run 2: same call, deliberately poisoned dataset_stats -------------------
    poisoned = {
        k: {s: (np.asarray(v, dtype=np.float64) + args.poison) for s, v in d.items()}
        for k, d in ds.meta.stats.items()
    }
    pre_b, post_b = build(poisoned)
    live_b = flatten({**grab(pre_b), **grab(post_b)})

    moved = []
    for k, va in live_a.items():
        vb = live_b.get(k)
        if vb is None:
            moved.append((k, float("inf")))
            continue
        d = float(np.abs(as_np(va).astype(np.float64) - as_np(vb).astype(np.float64)).max())
        if d != 0.0:
            moved.append((k, d))

    print(f"\n=== 1. INERTNESS: live(normal) vs live(+{args.poison} poison) ===")
    print(f"tensors compared : {len(live_a)}")
    print(f"tensors that MOVED: {len(moved)}")
    for k, d in moved[:10]:
        print(f"   MOVED {k} by {d:.6e}")
    inert = not moved

    # --- provenance --------------------------------------------------------------
    per_ep = per_episode_stats(pathlib.Path(args.root))
    TRAIN = agg(per_ep, train_eps, "TRAIN-111")
    FULL = agg(per_ep, all_eps, "FULL-131")

    # Keys whose stats are actually derived from the dataset, i.e. the real leak
    # surface. Image keys are excluded on evidence, not assumption: LeRobot
    # OVERWRITES image stats with ImageNet constants when
    # `cfg.dataset.use_imagenet_stats` is set (datasets/factory.py:137), so they
    # carry no dataset information and are verified separately below.
    DATA_KEYS = {"action", "observation.state"}

    print("\n=== 2. PROVENANCE: live vs recomputed aggregates ===")
    hdr = f"{'step':<26}{'key':<22}{'stat':<6}{'dev_vs_TRAIN111':>18}{'dev_vs_FULL131':>17}  note"
    print(hdr)
    print("-" * len(hdr))
    worst_train, best_full, rows, zero_power = 0.0, float("inf"), 0, 0
    for (step, key, stat), v in sorted(live_a.items()):
        tv = TRAIN.get(key, {}).get(stat)
        fv = FULL.get(key, {}).get(stat)
        if tv is None or fv is None:
            continue
        lv = as_np(v)
        if np.asarray(tv).size != lv.size or np.asarray(fv).size != lv.size:
            continue
        tv, fv = np.asarray(tv).reshape(lv.shape), np.asarray(fv).reshape(lv.shape)
        dt = lv.dtype  # compare in the LIVE object's storage dtype, both sides
        a, b, c = (np.asarray(x).astype(dt).astype(np.float64) for x in (lv, tv, fv))
        dtr, dfu = float(np.abs(a - b).max()), float(np.abs(a - c).max())
        if key not in DATA_KEYS:
            note = "not-dataset-derived" if key.startswith("observation.images") else "bookkeeping"
            print(f"{step:<26}{key:<22}{stat:<6}{dtr:>18.6e}{dfu:>17.6e}  {note} (excluded)")
            continue
        # A row where TRAIN-111 and FULL-131 agree cannot discriminate: its silence
        # is not evidence. Count these instead of letting them into min()/max().
        if dfu == 0.0:
            zero_power += 1
            print(f"{step:<26}{key:<22}{stat:<6}{dtr:>18.6e}{dfu:>17.6e}  ZERO-POWER slot")
            continue
        print(f"{step:<26}{key:<22}{stat:<6}{dtr:>18.6e}{dfu:>17.6e}")
        worst_train, best_full = max(worst_train, dtr), min(best_full, dfu)
        rows += 1
    print("-" * len(hdr))

    if rows == 0:
        sys.exit("FAIL: zero DISCRIMINATING rows -- the provenance test had no power, ignore its silence")

    # Images: confirm the constant-substitution story rather than just excluding them.
    from lerobot.utils.constants import IMAGENET_STATS

    img_const = []
    for (step, key, stat), v in sorted(live_a.items()):
        if not key.startswith("observation.images") or stat not in IMAGENET_STATS:
            continue
        ref = np.asarray(IMAGENET_STATS[stat]).ravel()
        got = as_np(v).ravel()
        if got.size == ref.size:
            img_const.append((key, stat, float(np.abs(got - ref).max())))

    print("\n=== 2b. IMAGE KEYS: constant-substituted, not dataset-derived ===")
    for key, stat, d in img_const:
        print(f"  {key:<32}{stat:<6}max|live - IMAGENET_STATS| = {d:.3e}")
    imgs_ok = bool(img_const) and all(d < 1e-6 for _, _, d in img_const)
    print(f"  => image normalization carries NO dataset statistics "
          f"({'confirmed' if imgs_ok else 'NOT confirmed -- investigate'})")

    print("\n=== 3. POWER ===")
    print(f"discriminating rows        : {rows} (score-relevant keys {sorted(DATA_KEYS)})")
    print(f"zero-power slots skipped   : {zero_power}")
    print(f"worst dev vs TRAIN-111     : {worst_train:.6e}   <- want ~0 (float32 artifact scale)")
    print(f"closest dev vs FULL-131    : {best_full:.6e}   <- leak control spacing")
    snr = (best_full / worst_train) if worst_train > 0 else float("inf")
    print(f"SNR (control / residual)   : {snr:.3e}")
    print(f"poison offset injected     : {args.poison:.1f} (would be plainly visible if honoured)")

    ok = inert and best_full > worst_train and rows > 0 and imgs_ok
    print()
    if not inert:
        print("FAIL: the dataset_stats kwarg IS live -- predict-time stats can be overridden by the")
        print("      dataset object, which for a val-restricted dataset is a leak channel.")
    if best_full <= worst_train:
        print("FAIL: live stats sit closer to the FULL-131 aggregate than to TRAIN-111.")
    if not imgs_ok:
        print("FAIL: image stats are neither ImageNet constants nor a recognised aggregate.")
    if ok:
        print(f"PASS: live processor holds the TRAIN-111 stats and is INERT to the dataset_stats")
        print(f"      kwarg (poison of {args.poison:.0f} moved 0/{len(live_a)} tensors). predict does not")
        print(f"      self-normalize with held-out statistics; 3rd leak channel closed for this leg.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
