#!/usr/bin/env python3
"""CPU smoke test for the NERO OFT data pipeline: norm stats, chunking, batch contract, leakage."""
import json
import os
import pathlib
import sys

os.environ.setdefault("ROBOT_PLATFORM", "NERO")
BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH.parent / "third_party" / "openvla-oft"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets.nero_dataset import (
    DATASET_NAME,
    NeroBatchTransform,
    NeroChunkDataset,
    load_split,
    normalize,
    unnormalize,
)

BASE = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                    "hub", "models--openvla--openvla-7b", "snapshots")
base = next(pathlib.Path(BASE).iterdir())

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
print("processor ok:", type(processor).__name__)

split = load_split()
bt = NeroBatchTransform(
    ActionTokenizer(processor.tokenizer),
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
    use_wrist_image=True,
    use_proprio=True,
)
ds = NeroChunkDataset(
    data_root=str(BENCH / "data" / "b2_oft"),
    episodes=split["train_episodes"],
    batch_transform=bt,
    image_size=224,
    image_aug=True,
)
print(f"train anchors: {len(ds)}  (episodes {len(split['train_episodes'])})")

# --- leakage check: no val episode may appear in the training index ---
idx_eps = {e for e, _ in ds.index}
leak = idx_eps & set(split["val_episodes"])
assert not leak, f"LEAK: val episodes in train index: {leak}"
print(f"leakage check OK: train index covers {len(idx_eps)} episodes, 0 val episodes")

# --- stats sanity + normalize/unnormalize round trip ---
st = ds.dataset_statistics[DATASET_NAME]
print("num_transitions", st["num_transitions"], "num_trajectories", st["num_trajectories"])
print("action min ", np.round(st["action"]["min"], 2))
print("action max ", np.round(st["action"]["max"], 2))
print("proprio min", np.round(st["proprio"]["min"], 2))
print("proprio max", np.round(st["proprio"]["max"], 2))
raw = ds.traj["ep001_action"][:50]
rt = unnormalize(normalize(raw, st["action"]), st["action"])
print("norm round-trip max abs err:", float(np.abs(rt - raw).max()))
assert np.abs(rt - raw).max() < 1e-3

# --- chunking matches val_anchors convention (act[t:t+K]) ---
e, t = ds.index[123]
chunk_raw = ds.traj[f"ep{e:03d}_action"][t : t + NUM_ACTIONS_CHUNK]
assert chunk_raw.shape == (NUM_ACTIONS_CHUNK, ACTION_DIM)

# --- batch contract ---
dl = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    ),
)
batch = next(iter(dl))
for k, v in batch.items():
    print(f"  {k}: {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}", v.dtype if torch.is_tensor(v) else "")
assert batch["pixel_values"].shape[1] == 12, batch["pixel_values"].shape  # 2 imgs x (dino 3 + siglip 3) x ...
assert batch["actions"].shape == (4, NUM_ACTIONS_CHUNK, ACTION_DIM), batch["actions"].shape
assert batch["proprio"].shape == (4, PROPRIO_DIM), batch["proprio"].shape
assert float(batch["actions"].abs().max()) <= 1.0 + 1e-6, "actions not normalized to [-1,1]"

# --- label masking: exactly NUM_ACTIONS_CHUNK*ACTION_DIM action tokens (+ stop) supervised ---
n_sup = (batch["labels"][0] != -100).sum().item()
print(f"supervised label tokens: {n_sup} (expect {NUM_ACTIONS_CHUNK * ACTION_DIM} action tokens + 1 stop)")
assert n_sup == NUM_ACTIONS_CHUNK * ACTION_DIM + 1, n_sup

print("\nSMOKE OK")
