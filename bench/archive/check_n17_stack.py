#!/usr/bin/env python3
"""Smoke-check the GR00T N1.7 stack before committing GPU hours to a 10k-step run.

N1.7 counterpart of check_gr00t_stack.py. Verifies, in order: torch sees the
Blackwell GPU -> flash-attn's sm_120 kernels actually launch -> gr00t_n1d7
imports -> the NERO modality config registers under NEW_EMBODIMENT and is
joint-space (NON_EEF) -> the train/val episode loaders yield the shapes the
model expects -> the val loader's state matches val_anchors frame-for-frame.

Deliberately does NOT touch the VLM backbone repo, so it passes with or without
access to the gated nvidia/Cosmos-Reason2-2B.
"""
import pathlib
import sys

BENCH = pathlib.Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402
import torch  # noqa: E402

print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "| capability", torch.cuda.get_device_capability(0))

    # flash-attn on sm_120: the wheel claims sm_120 cubins, prove they launch
    from flash_attn import flash_attn_func  # noqa: E402

    q, k, v = (torch.randn(2, 64, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    print("flash_attn ok:", out.shape, out.dtype, "finite:", torch.isfinite(out).all().item())
else:
    print("!! no CUDA visible -- flash-attn check skipped")

import gr00t.model  # noqa: E402,F401  (registers the model classes)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.types import ActionRepresentation, ActionType  # noqa: E402

sys.path.insert(0, str(BENCH / "configs"))
import nero_n17_config  # noqa: E402,F401  (registers NEW_EMBODIMENT)

TAG = EmbodimentTag.NEW_EMBODIMENT
modality = MODALITY_CONFIGS[TAG.value]  # registry is keyed by the tag's string value
print("\nregistered modality config for", TAG)
for name, cfg in modality.items():
    print(f"  {name:9s} keys={cfg.modality_keys} n_delta={len(cfg.delta_indices)}")

# The whole point of the config: joint space, not the relative-EEF space N1.7's
# built-in humanoid embodiments use.
act = modality["action"]
for key, acfg in zip(act.modality_keys, act.action_configs):
    print(f"  action.{key}: rep={acfg.rep.value} type={acfg.type.value} format={acfg.format.value}")
    assert acfg.type is ActionType.NON_EEF, f"action.{key} must be joint-space (NON_EEF)"
assert act.action_configs[act.modality_keys.index("gripper")].rep is ActionRepresentation.ABSOLUTE
K_CONTRACT = 8
assert len(act.delta_indices) >= K_CONTRACT, (
    f"action horizon {len(act.delta_indices)} < contract K={K_CONTRACT}")

for split, path in [("train", BENCH / "data/b2_n17_train"), ("val", BENCH / "data/b2_n17_val")]:
    loader = LeRobotEpisodeLoader(dataset_path=str(path), modality_configs=modality)
    print(f"\n{split}: {len(loader)} episodes")
    traj = loader[0]
    step = extract_step_data(traj, 0, modality, TAG)
    for k, v in step.states.items():
        print(f"  state.{k}", np.asarray(v).shape)
    for k, v in step.actions.items():
        print(f"  action.{k}", np.asarray(v).shape)
    for k, v in step.images.items():
        print(f"  video.{k}", np.asarray(v[0]).shape, np.asarray(v[0]).dtype)
    print("  prompt:", step.text)

# val loader must line up with the bench anchors frame-for-frame
A = np.load(BENCH / "data" / "val_anchors.npz")
loader = LeRobotEpisodeLoader(dataset_path=str(BENCH / "data/b2_n17_val"), modality_configs=modality)
pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}
state_keys = modality["state"].modality_keys

checked = 0
for i in (0, len(A["episodes"]) // 2, len(A["episodes"]) - 1):
    ep, fr = int(A["episodes"][i]), int(A["frames"][i])
    traj = loader[pos_of_ep[ep]]
    # allow_padding: 28 of the 1397 anchors sit within 16 frames of their episode's
    # end, so the full action horizon runs off the end here. Only gt[0] is compared
    # below, and the contract's K=8 window is in range for every anchor, so clamping
    # the tail is harmless. Training never hits this (ShardedSingleStepDataset caps
    # start indices at length - action_horizon + 1); the predict path never hits it
    # either (it drops the action modality and reads only delta index 0).
    step = extract_step_data(traj, fr, modality, TAG, allow_padding=True)
    got = np.concatenate([step.states[k][0] for k in state_keys])
    print(f"\nanchor {i} ep{ep} f{fr}")
    print("  dataset state:", np.round(got, 3))
    print("  anchors state:", np.round(A["state"][i], 3))
    assert np.allclose(got, A["state"][i], atol=1e-3), "STATE MISALIGNED"
    # gt is the absolute action chunk; RELATIVE arm actions must round-trip to it
    gt0 = A["gt"][i][0]
    act0 = np.concatenate([step.actions[k][0] for k in act.modality_keys])
    assert np.allclose(act0, gt0, atol=1e-3), f"ACTION MISALIGNED\n {act0}\n {gt0}"
    checked += 1
print(f"\n{checked} anchors verified against val_anchors (state + absolute action)")
print("\nALL CHECKS PASSED")
