#!/usr/bin/env python3
"""Produce preds/preds_openvla_oft.npz for the NERO b2 backbone bench.

For each of the 1397 val anchors in val_anchors.npz, run the fine-tuned OpenVLA-OFT policy on
(cam_high, cam_wrist, state, prompt) at that (episode, frame) and write the predicted action chunk
in RAW joint units (absolute joint targets, same space as the dataset `action` column).

Model loading and the forward path mirror `experiments/robot/openvla_utils.py::get_vla /
get_vla_action` (that file is what the official ALOHA/LIBERO eval uses); unnormalization is the
model's own `_unnormalize_actions` driven by the checkpoint's `dataset_statistics.json`, which was
computed on the 111 TRAIN episodes only.

OFT with an L1-regression action head is deterministic (do_sample=False, no diffusion) -> no
sampling variance, so no variance footnote is needed for this backbone.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

# 仓库自包含:从本文件位置推导。bench/scripts/predict/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
REPO = BENCH.parent / "third_party" / "openvla-oft"
PROMPT = "pick the pink sponge and place it in the blue bucket"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="Fine-tuned OFT checkpoint dir (merged model + heads + stats)")
    ap.add_argument("--data-root", default=str(BENCH / "data" / "b2_oft"))
    ap.add_argument("--out", default=str(BENCH / "preds" / "preds_openvla_oft.npz"))
    ap.add_argument("--anchors", default=str(BENCH / "data" / "val_anchors.npz"),
                    help="anchor npz to predict over; default is the frozen val harness. Point it "
                         "at a train-split anchor file for the generalization-ratio footnote.")
    ap.add_argument("--unnorm-key", default="nero_b2")
    ap.add_argument("--num-images-in-input", type=int, default=2)
    ap.add_argument("--use-proprio", action="store_true", default=True)
    ap.add_argument("--use-film", action="store_true", default=False)
    ap.add_argument("--allow-anchor-change", action="store_true",
                    help="Skip the md5 pin on the frozen val_anchors.npz. Only for a deliberate, "
                         "board-wide re-freeze -- every leg must then re-run.")
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N anchors (smoke test)")
    ap.add_argument("--subset-idx", default="",
                    help="npy of anchor indices to run (board-shared logs/variance_subset200.npy, "
                         "md5 9007df1f..., spans all 20 val episodes). Prefer this over --limit for "
                         "the determinism footnote: --limit 200 is a contiguous PREFIX and touches "
                         "only ~3 episodes, so it cannot be compared with the other legs' subsets.")
    ap.add_argument("--device", default="cuda", help="cuda (real run) or cpu (dry-run of the predict path)")
    ap.add_argument(
        "--shape-only",
        action="store_true",
        help="Skip the model entirely and emit a correctly-shaped random preds npz (scaffolding check)",
    )
    args = ap.parse_args()

    # IDENTITY-gate the frozen harness, not just its existence (gr00t-n17's variance_subset200
    # lesson). val_anchors.npz is the row-alignment contract between this npz and score.py, and it
    # is a SHARED file on a box where another leg runs its own anchor rebuilder. A regenerated or
    # reordered file still has 1397 rows and still loads, so every other gate here passes: predict
    # would align rows to one version and score.py would grade them against another, yielding a
    # plausible wrong MAE with no exception anywhere. Existence-checking cannot see that; md5 can.
    # Checked BEFORE the 7B load so a mismatch costs a second, not a checkpoint load.
    VAL_ANCHORS_MD5 = "cc34dda2a99ade1011c192e2b3e4e343"   # frozen 2026-08-05 00:03:53, 414998 B
    is_frozen = pathlib.Path(args.anchors).resolve() == (BENCH / "data" / "val_anchors.npz").resolve()
    if is_frozen and not args.allow_anchor_change:
        # NB: hashlib is imported at module level, NOT here. It used to be a local import inside this
        # branch, which meant the name only existed when the anchors were the frozen val file -- the
        # contract dump at the end of main() then NameError'd on exactly the paths that skip this
        # branch (stage 5's train anchors, and --allow-anchor-change), i.e. after all the GPU work.
        got = hashlib.md5(pathlib.Path(args.anchors).read_bytes()).hexdigest()
        if got != VAL_ANCHORS_MD5:
            sys.exit(
                f"[FATAL] val_anchors.npz md5 {got} != pinned {VAL_ANCHORS_MD5}. The frozen harness "
                f"changed under this leg: preds built against it would be scored against a DIFFERENT "
                f"anchor set, producing a plausible wrong MAE. Every leg must re-run against the new "
                f"file and the board must re-pin. Override only deliberately: --allow-anchor-change")

    anchors = np.load(args.anchors)
    eps, frames, K = anchors["episodes"], anchors["frames"], int(anchors["K"])
    A = len(eps)
    print(f"{A} anchors, K={K}  (from {args.anchors})"
          + ("  [md5 OK]" if is_frozen and not args.allow_anchor_change else ""))
    # A non-default anchor file is by definition not the frozen val harness, so its predictions are
    # not a leaderboard row. Same reasoning as the --limit guard: keep it out of score.py's glob.
    if not is_frozen:
        assert pathlib.Path(args.out).parent.name != "preds", (
            f"predictions over a non-val anchor set must not land in preds/ (got {args.out})")

    # Resolve WHICH anchors to run, and gate the output path, BEFORE loading a 7B model -- these are
    # the two ways a run can produce a file that scores cleanly while describing the wrong thing,
    # and both are decidable from argv alone. Fail in a second, not after a model load.
    sel = None
    if args.subset_idx:
        sel = np.load(args.subset_idx)
        assert sel.ndim == 1 and int(sel.max()) < A, (sel.shape, sel.max(), A)
        print(f"subset: {len(sel)} anchors over {len(set(eps[sel]))} episodes "
              f"(from {args.subset_idx})")
    n = args.limit if args.limit else A
    rows = np.asarray(sel if sel is not None else range(n), dtype=int)
    # Any run that is not the full anchor set produces a row count score.py cannot align, so it must
    # never land where score.py globs. (It no longer zero-pads -- a partial run emits only its own
    # rows -- so the hazard is a wrong-length array, not silent zeros.)
    if len(rows) < A:
        assert pathlib.Path(args.out).parent.name != "preds", (
            f"a partial run ({len(rows)} of {A} anchors) must not land in preds/ where score.py "
            f"globs it; send it elsewhere (got {args.out})")

    # Third path to a wrong-but-scoreable file, and the only one the two guards above MISS:
    # a mode that emits the RIGHT anchor count over the RIGHT anchor file with content that never
    # touched the model. `--shape-only` is random noise and `--device cpu` is a plumbing dry-run
    # (sdpa/CPU numerics != the on-card forward), yet both sail past the anchor-file check and the
    # partial-run check, so `--out` defaults them straight into preds/ where score.py globs.
    # Board rule (openpi's near-miss, relayed by team-lead 2026-08-05): a dry run that writes to the
    # production output path is not a dry run. The other two guards catch a file score.py would
    # SKIP (wrong row count); this one catches a file score.py would happily RANK.
    dryrun_modes = [n for n, on in (("--shape-only", args.shape_only),
                                    ("--device cpu", args.device == "cpu")) if on]
    if dryrun_modes:
        assert pathlib.Path(args.out).parent.name != "preds", (
            f"{' and '.join(dryrun_modes)} is a dry run of the predict path, not a leaderboard row, "
            f"and it emits the full {A}-anchor shape so score.py would RANK it rather than skip it; "
            f"send it outside preds/ (got {args.out})")

    if args.shape_only:
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((A, K, 8)).astype(np.float32)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, pred=pred)
        print(f"[shape-only] wrote {args.out} pred{pred.shape}")
        return

    assert args.ckpt, "--ckpt required unless --shape-only"

    # Constants must be resolved before importing anything under prismatic.*
    os.environ.setdefault("ROBOT_PLATFORM", "NERO")
    sys.path.insert(0, str(REPO))

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import (
        ACTION_DIM,
        ACTION_PROPRIO_NORMALIZATION_TYPE,
        NUM_ACTIONS_CHUNK,
        PROPRIO_DIM,
        ROBOT_PLATFORM,
        NormalizationType,
    )
    from prismatic.vla.datasets.nero_dataset import load_frame, normalize

    print(f"constants: chunk={NUM_ACTIONS_CHUNK} action_dim={ACTION_DIM} proprio_dim={PROPRIO_DIM} "
          f"norm={ACTION_PROPRIO_NORMALIZATION_TYPE}")
    assert NUM_ACTIONS_CHUNK >= K, f"model chunk {NUM_ACTIONS_CHUNK} < contract K {K}"
    assert ACTION_DIM == 8, ACTION_DIM
    # ACTION_DIM==8 already excludes LIBERO/BRIDGE(7)/ALOHA(14), but assert the normalization
    # explicitly: it is what maps the head's [-1,1] output back to raw joint units, and getting
    # it wrong (q01/q99 instead of min/max) yields plausible-looking, silently wrong degrees.
    assert ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS, (
        ACTION_PROPRIO_NORMALIZATION_TYPE)

    device = args.device
    # bf16 matmul is unusably slow on CPU; the dry-run only checks shapes/plumbing, so use fp32 there.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    ckpt = args.ckpt.rstrip("/")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    vla.vision_backbone.set_num_images_in_input(args.num_images_in_input)
    vla.eval()
    vla = vla.to(device)

    # Norm stats from training (train episodes only) -> drives unnormalization back to raw joints
    stats = json.loads((pathlib.Path(ckpt) / "dataset_statistics.json").read_text())
    vla.norm_stats = stats
    assert args.unnorm_key in stats, f"{args.unnorm_key} not in {list(stats)}"
    proprio_stats = stats[args.unnorm_key]["proprio"]

    # `_unnormalize_actions` reads the mask as `action_norm_stats.get("mask", ones_like(min))`, i.e.
    # a masked-off dim is returned in NORMALIZED [-1,1] units while its neighbours are in degrees --
    # a per-dim unit error that no shape/finiteness check can see. This ckpt ships no "mask" key, so
    # today the all-True default is correct by luck; assert it rather than inherit it, and report
    # WHICH dims fail so a hit names the dims instead of just refusing.
    _astats = stats[args.unnorm_key]["action"]
    _mask = np.asarray(_astats.get("mask", np.ones(len(_astats["min"]), dtype=bool)))
    assert _mask.shape == (ACTION_DIM,) and _mask.all(), \
        f"unnormalize mask not all-True: measured={_mask.tolist()} false_dims={np.where(~_mask)[0].tolist()}"
    # Norm stats must come from the 111 TRAIN episodes only; 131 would mean the 20 val episodes
    # leaked into the bounds that map the head's output back to joint units.
    _ntraj = stats[args.unnorm_key]["num_trajectories"]
    assert _ntraj == 111, f"norm stats built from {_ntraj} trajectories, expected 111 (train-only)"

    action_head = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    action_head.load_state_dict(_component(ckpt, "action_head"))
    action_head = action_head.to(dtype).to(device).eval()

    proprio_projector = None
    if args.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        proprio_projector.load_state_dict(_component(ckpt, "proprio_projector"))
        proprio_projector = proprio_projector.to(dtype).to(device).eval()

    traj = np.load(pathlib.Path(args.data_root) / "traj.npz")
    frames_dir = pathlib.Path(args.data_root) / "frames"
    prompt = f"In: What action should the robot take to {PROMPT.lower()}?\nOut:"

    # A subset run emits ONLY those rows, shape (len(idx), K', 8) -- the board convention set by
    # logs/variance/preds_gr00t_n15_subset200_seed0.npz, so it is compared against full[idx].
    out = np.zeros((len(rows), NUM_ACTIONS_CHUNK, ACTION_DIM), np.float32)
    t0 = time.time()
    for j, i in enumerate(rows):
        i = int(i)
        ep, t = int(eps[i]), int(frames[i])
        primary = load_frame(frames_dir, ep, "cam_high", t, 224)
        wrist = load_frame(frames_dir, ep, "cam_wrist", t, 224)

        inputs = processor(prompt, primary).to(device, dtype=dtype)
        if args.num_images_in_input > 1:
            wrist_inputs = processor(prompt, wrist).to(device, dtype=dtype)
            inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)

        proprio = None
        if args.use_proprio:
            proprio = normalize(traj[f"ep{ep:03d}_state"][t], proprio_stats)

        with torch.inference_mode():
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=args.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                use_film=args.use_film,
            )
        out[j] = np.asarray(action, np.float32)
        if j % 100 == 0:
            el = time.time() - t0
            print(f"{j}/{len(rows)}  {el:.0f}s  ({el / max(j, 1):.2f} s/anchor)", flush=True)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Embed the RESOLVED normalization contract next to the numbers. This matters more here than on
    # a stack that serializes the flag into the checkpoint: ACTION_PROPRIO_NORMALIZATION_TYPE is a
    # module-level global fixed at import time from $ROBOT_PLATFORM or a substring match on argv, so
    # it lives in neither the weights nor the stats file -- the same ckpt run by a differently-named
    # script yields different units with no error. Written as a json STRING so downstream `np.load`
    # needs no allow_pickle. Readers index by key, so this is inert for score.py.
    contract = json.dumps({
        "branch": str(ACTION_PROPRIO_NORMALIZATION_TYPE.value),
        "robot_platform": str(ROBOT_PLATFORM),
        "flag_resolved_from": ("env:ROBOT_PLATFORM" if os.environ.get("ROBOT_PLATFORM")
                               else "argv-substring-match"),
        "bounds_source": "action_norm_stats[min]/[max]",
        "inverse_clips": False,
        "forward_clips": True,
        "unnormalize_mask": _mask.tolist(),
        "action_dim": int(ACTION_DIM),
        "num_actions_chunk": int(NUM_ACTIONS_CHUNK),
        "norm_stats_num_trajectories": int(_ntraj),
        "units": "arm deg (absolute), gripper 0-100 (absolute)",
        "val_anchors_md5": hashlib.md5(pathlib.Path(args.anchors).read_bytes()).hexdigest(),
        "ckpt": str(ckpt),
    }, sort_keys=True)
    np.savez(args.out, pred=out, contract=np.array(contract))
    print(f"wrote {args.out} pred{out.shape} in {time.time() - t0:.0f}s")
    print(f"contract: {contract}")

    gt = anchors["gt"][np.asarray(rows, int)]
    mae = np.abs(out[:, :K] - gt).mean()
    print(f"quick MAE over {len(rows)} anchors: {mae:.4f} "
          f"(arm {np.abs(out[:, :K, :7] - gt[..., :7]).mean():.4f}, "
          f"grip {np.abs(out[:, :K, 7] - gt[..., 7]).mean():.4f})")


def _component(ckpt: str, name: str):
    """Load a saved OFT component state dict, stripping DDP prefixes.

    finetune_nero.py names these `<name>--<step>_checkpoint.pt`, or `<name>--latest_checkpoint.pt`
    when save_latest_checkpoint_only=True (what we train with) -- so "latest" must sort last, and a
    plain lexicographic sort would rank step 9000 above 10000.
    """
    import re

    import torch

    def step_of(p):
        m = re.search(rf"{name}--(\d+)_checkpoint", p.name)
        return int(m.group(1)) if m else float("inf")  # "latest" is always the newest

    cands = list(pathlib.Path(ckpt).glob(f"{name}--*_checkpoint.pt"))
    assert cands, f"no {name} checkpoint in {ckpt}"
    cands.sort(key=step_of)
    print(f"  {name}: {cands[-1].name}" + (f"  (of {len(cands)})" if len(cands) > 1 else ""))
    sd = torch.load(cands[-1], weights_only=True, map_location="cpu")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


if __name__ == "__main__":
    main()
