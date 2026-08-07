#!/usr/bin/env python
"""Post-install smoke test for the HF-LeRobot bench venv.

Checks, in the order they can break:
  1. versions import at all
  2. Blackwell (sm_120) actually runs a matmul -- a torch built without sm_120
     imports fine and only dies at the first kernel launch
  3. AV1 video decode through whatever backend LeRobot picked (b2 is av1-coded,
     which is the codec most likely to be missing from a bundled ffmpeg)
"""

import os
import sys

import torch

print("=== 1. versions ===")
import lerobot
import torchvision
import transformers

print("torch       ", torch.__version__, "| built for CUDA", torch.version.cuda)
print("torchvision ", torchvision.__version__)
print("lerobot     ", lerobot.__version__)
print("transformers", transformers.__version__)
try:
    import torchcodec

    print("torchcodec  ", torchcodec.__version__)
except Exception as e:  # noqa: BLE001
    print("torchcodec   UNAVAILABLE:", e)
print("cuda avail  ", torch.cuda.is_available(), "| n_dev", torch.cuda.device_count())
print("arch list   ", torch.cuda.get_arch_list())

print("\n=== 2. Blackwell kernel launch (cuda:4) ===")
dev = "cuda:4"
print("device name ", torch.cuda.get_device_name(dev))
cap = torch.cuda.get_device_capability(dev)
print("capability  ", f"sm_{cap[0]}{cap[1]}")
x = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
torch.cuda.synchronize(dev)
print("bf16 matmul  OK, checksum finite:", bool(abs(y) < float("inf")))

print("\n=== 3. AV1 decode via LeRobot's backend ===")
from lerobot.utils.import_utils import get_safe_default_video_backend

backend = get_safe_default_video_backend()
print("backend     ", backend)
from lerobot.datasets.video_utils import decode_video_frames

vid = os.path.join(
    os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")),
    "videos/chunk-000/observation.images.cam_high/episode_000000.mp4",
)
frames = decode_video_frames(vid, [0.0, 13.6], tolerance_s=0.04, backend=backend)
print("decoded     ", tuple(frames.shape), frames.dtype,
      f"range [{frames.min():.3f}, {frames.max():.3f}]")
assert frames.shape[0] == 2 and frames.shape[-2:] == (480, 640), frames.shape

print("\nSMOKE_OK")
sys.exit(0)
