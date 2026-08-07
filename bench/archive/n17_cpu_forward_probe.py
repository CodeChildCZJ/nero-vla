#!/usr/bin/env python3
"""Run predict_gr00t_n17.py end to end on CPU, so the forward path can be cleared
without holding a GPU.

The checkpoint bakes `use_flash_attention`, and FlashAttention2 raises on CPU
("not available on CPU"). qwen3_backbone.py:167 already has a documented fallback
-- it switches to sdpa when `import flash_attn` fails -- so this makes that import
fail rather than patching the model. sdpa vs flash_attn changes the attention
kernel (numerics differ slightly), so this probe validates PLUMBING, not values:
checkpoint loads, processor attaches, the noise pinning finds its single draw, the
output dict has the expected keys/shapes, and the relative->absolute gate fires.
The MAE it produces is NOT comparable to a GPU run and must never reach preds/.

usage: n17_cpu_forward_probe.py --model-path <ckpt> [--limit 1]
"""
import pathlib
import runpy
import sys

sys.modules["flash_attn"] = None  # make `import flash_attn` raise -> sdpa fallback

if "--out" not in sys.argv:
    sys.argv += ["--out", "/tmp/n17_cpu_probe.npz"]
if "--limit" not in sys.argv:
    sys.argv += ["--limit", "1"]
sys.argv[0] = str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "predict" / "predict_gr00t_n17.py")

print("CPU probe: flash_attn disabled -> sdpa; values are NOT GPU-comparable")
runpy.run_path(sys.argv[0], run_name="__main__")
