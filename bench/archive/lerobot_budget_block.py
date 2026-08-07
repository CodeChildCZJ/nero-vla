#!/usr/bin/env python
"""Derive the training-budget block for a board row FROM THE CHECKPOINT, never by hand.

Why
---
`preds_act_meta.json` carries `train.steps=100000 / batch=8 / epochs=21.04`.  Nothing
wrote those: they were typed.  The SmolVLA row needs the same block, and hand-copying a
block whose neighbour reads `steps: 100000` is precisely how a row gets labelled with the
other backbone's budget -- a wrong number that is plausible, self-consistent, and
invisible to every structural check on the board (same family as the `--out` swap that
GUARD 1b now blocks: the artifact is perfect, the provenance is wrong).

So read `train_config.json` out of the checkpoint being delivered and compute the block.
`epochs` is derived, not stored, so it also cross-checks the frame count: if the dataset
under the checkpoint ever changes, epochs moves and the row shows it.

Self-test: `--verify-against preds/preds_act_meta.json` re-derives ACT's block and
requires it to match the hand-written one field for field.  That is a real must-pass --
if the deriver and the typed block disagree, ONE OF THEM IS WRONG and the board has been
reading it.  The delivery does not get to skip it.

  python archive/lerobot_budget_block.py --ckpt runs/act/checkpoints/last/pretrained_model \
      --verify-against preds/preds_act_meta.json
"""

import argparse
import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parents[1]


def budget(ckpt: pathlib.Path) -> dict:
    tc_path = ckpt / "train_config.json"
    if not tc_path.exists():
        raise SystemExit(f"[FATAL] no train_config.json under {ckpt} -- refusing to guess a budget.")
    tc = json.loads(tc_path.read_text())

    missing = [k for k in ("steps", "batch_size") if tc.get(k) is None]
    if missing:
        raise SystemExit(f"[FATAL] train_config.json lacks {missing}; a partial budget block is worse than none.")

    steps, batch = int(tc["steps"]), int(tc["batch_size"])
    root = pathlib.Path(tc["dataset"]["root"])
    eps = tc["dataset"].get("episodes")

    # Frame count from the dataset's own metadata, restricted to the episodes this run
    # actually trained on -- NOT the whole repo (the val episodes live in the same tree).
    ep_meta = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    n_frames = None
    if ep_meta:
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(ep_meta[0]) if len(ep_meta) == 1 else None
            if tbl is None:
                import pyarrow as pa
                tbl = pa.concat_tables([pq.read_table(p) for p in ep_meta])
            idx = tbl.column("episode_index").to_pylist()
            ln = tbl.column("length").to_pylist()
            keep = set(int(e) for e in eps) if eps else None
            n_frames = sum(l for e, l in zip(idx, ln, strict=True) if keep is None or int(e) in keep)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[warn] frame count unavailable ({type(exc).__name__}: {exc}); epochs will be null")

    out = {
        "steps": steps,
        "batch": batch,
        "samples": steps * batch,
        "train_episodes": len(eps) if eps else None,
        "train_frames": n_frames,
        "epochs": round(steps * batch / n_frames, 2) if n_frames else None,
        "seed": tc.get("seed"),
        "derived_from": str(tc_path),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--verify-against", default=None,
                    help="an existing *_meta.json whose train block must match the derived one")
    args = ap.parse_args()

    b = budget(pathlib.Path(args.ckpt))
    print(json.dumps(b, indent=2))

    if args.verify_against:
        meta = json.loads(pathlib.Path(args.verify_against).read_text())
        old = meta.get("train", {})
        checks = {k: (old.get(k), b.get(k)) for k in ("steps", "batch", "epochs") if k in old}
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        if not checks:
            raise SystemExit(f"[FATAL] {args.verify_against} has no train block to verify against.")
        if bad:
            raise SystemExit(f"[FATAL] derived budget disagrees with the shipped meta: {bad} "
                             f"(hand-written value first). One of them is wrong and the board reads it.")
        print(f"[verify] {len(checks)} field(s) match the hand-written block: {checks}")


if __name__ == "__main__":
    main()
