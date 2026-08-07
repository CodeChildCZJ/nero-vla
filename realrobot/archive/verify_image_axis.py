#!/usr/bin/env python3
"""严谨实查图像轴: LeRobot返回啥shape → _nero_parse_image输出啥 → resize后啥。
不推断, 打印真实tensor shape。czj要求。"""
import os
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

# 1. LeRobot 训练时返回的图像 shape/dtype/range
import lerobot.common.datasets.lerobot_dataset as lds
ds = lds.LeRobotDataset("local/pick_pink_sponge_v3")
s = ds[0]
for k in s:
    if "image" in k.lower():
        a = np.asarray(s[k])
        print(f"[LeRobot返回] {k}: shape={a.shape} dtype={a.dtype} min={float(a.min()):.3f} max={float(a.max()):.3f}")

# 2. 过 _nero_parse_image
from openpi.training.config import _nero_parse_image
img_key = [k for k in s if "image" in k.lower() and "cam_high" in k.lower()] or [k for k in s if "image" in k.lower()]
raw = np.asarray(s[img_key[0]])
print(f"\n[输入_nero_parse_image] shape={raw.shape}")
parsed = _nero_parse_image(raw)
print(f"[_nero_parse_image输出] shape={parsed.shape} dtype={parsed.dtype} (应为 HWC=(H,W,3) uint8)")
print(f"  → H={parsed.shape[0]} W={parsed.shape[1]} C={parsed.shape[2]}; 480x640原图应是 (480,640,3) 不能H/W颠倒成(640,480,3)")

# 3. resize_with_pad 后
from openpi.shared import image_tools
resized = image_tools.resize_with_pad(parsed, 224, 224)
print(f"\n[resize_with_pad(224,224)后] shape={np.asarray(resized).shape} (应 (224,224,3))")

# 4. 确认 resize_with_pad 期望的输入轴 (看它怎么取H/W)
import inspect
src = inspect.getsource(image_tools.resize_with_pad)
print(f"\n[resize_with_pad 源码前25行]")
for ln in src.splitlines()[:25]:
    print("  " + ln)
