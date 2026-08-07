#!/usr/bin/env python3
"""Smoke-check the GR00T N1.5 stack before committing GPU hours to a 10k-step run.

Verifies, in order: torch sees the Blackwell GPU -> flash-attn's sm_120 kernels
actually run -> gr00t imports -> the NERO data config loads -> the train dataset
yields a transformed sample with the shapes the model expects.
"""
import pathlib
import sys

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "configs"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))

# flash-attn on sm_120: the wheel claims sm_120 cubins, prove they launch
from flash_attn import flash_attn_func  # noqa: E402

q, k, v = (torch.randn(2, 64, 8, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3))
out = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
print("flash_attn ok:", out.shape, out.dtype, "finite:", torch.isfinite(out).all().item())

from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402

cfg = load_data_config("nero_data_config:NeroDualCamDataConfig")
ds = LeRobotSingleDataset(
    dataset_path=str(BENCH / "data/b2_gr00t_train"),
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag="new_embodiment",
    video_backend="decord",
)
print("train dataset len:", len(ds), "| episodes:", len(ds.trajectory_lengths))
sample = ds[0]
for k_, v_ in sample.items():
    print("  ", k_, getattr(v_, "shape", v_ if not isinstance(v_, (list, np.ndarray)) else type(v_)))

raw = LeRobotSingleDataset(
    dataset_path=str(BENCH / "data/b2_gr00t_val"),
    modality_configs=cfg.modality_config(),
    transforms=None,
    embodiment_tag="new_embodiment",
    video_backend="decord",
)
A = np.load(BENCH / "data" / "val_anchors.npz")
i = 0
step = raw.get_step_data(int(A["episodes"][i]), int(A["frames"][i]))
got = np.concatenate([step["state.single_arm"][0], step["state.gripper"][0]])
print(f"val anchor0 ep{A['episodes'][i]} f{A['frames'][i]}")
print("  dataset state:", np.round(got, 3))
print("  anchors state:", np.round(A["state"][i], 3))
assert np.allclose(got, A["state"][i], atol=1e-3), "STATE MISALIGNED"
print("  cam_high", step["video.cam_high"].shape, step["video.cam_high"].dtype,
      "cam_wrist", step["video.cam_wrist"].shape)
print("  prompt:", step["annotation.human.task_description"])
print("\nALL CHECKS PASSED")
