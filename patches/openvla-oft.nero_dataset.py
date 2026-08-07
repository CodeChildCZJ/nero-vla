"""
nero_dataset.py

Map-style PyTorch Dataset for the NERO b2 real-robot dataset, replacing the RLDS/TFDS pipeline.

The official OFT loader (`RLDSDataset`) requires the dataset to be converted to RLDS and pulls in
TensorFlow; `finetune.py` itself documents the "bring your own torch Dataset" path. This module
takes that path but keeps every downstream contract identical:

  * the sample dict handed to `PaddedCollatorForActionPrediction` is produced by the *unmodified*
    `RLDSBatchTransform` logic (copied verbatim below, minus the TF-only imports),
  * action/proprio normalization reproduces `normalize_action_and_proprio` for
    `NormalizationType.BOUNDS`,
  * action chunking reproduces `chunk_act_obs` with `window_size=1`,
    `future_action_window_size=NUM_ACTIONS_CHUNK-1` (anchors t in [0, T-K], no end padding),
  * image augmentation mirrors the `image_aug=True` RLDS kwargs (random_resized_crop 0.9 ->
    brightness -> contrast -> saturation -> hue).

Frames are read from the pre-extracted 256x256 JPEG tree written by `scripts/oft_extract_frames.py`
(the same 480x640 -> 256x256 BICUBIC downsize the official ALOHA preprocessing does), then resized
to the model's 224x224 input, matching the RLDS `resize_size` step.
"""

import os
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NormalizationType,
)

NERO_PROMPT = "pick the pink sponge and place it in the blue bucket"
DATASET_NAME = "nero_b2"
CAMS = ("cam_high", "cam_wrist")


# ---------------------------------------------------------------------------------------------
# Batch transform: verbatim copy of prismatic.vla.datasets.RLDSBatchTransform (which cannot be
# imported here because its module pulls in the TensorFlow RLDS pipeline).
# ---------------------------------------------------------------------------------------------
@dataclass
class NeroBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: Any
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        prompt_builder = self.prompt_builder_fn("openvla")

        future_actions = rlds_batch["action"][1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions))

        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] Only take the loss on the predicted action tokens.
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions
        )

        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    all_wrist_pixels.append(self.image_transform(img_wrist))
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            return_dict["proprio"] = rlds_batch["observation"]["proprio"]

        return return_dict


# ---------------------------------------------------------------------------------------------
# Normalization statistics (computed on the TRAIN episodes only -- never touch val episodes)
# ---------------------------------------------------------------------------------------------
def compute_dataset_statistics(traj: Dict[str, np.ndarray], episodes: Sequence[int]) -> Dict[str, Dict]:
    """Reproduce the fields of the RLDS `dataset_statistics` dict for our episode subset."""
    acts = np.concatenate([traj[f"ep{e:03d}_action"] for e in episodes], axis=0)
    props = np.concatenate([traj[f"ep{e:03d}_state"] for e in episodes], axis=0)

    def _stats(x):
        return {
            "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(),
            "max": x.max(0).tolist(),
            "min": x.min(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }

    return {
        DATASET_NAME: {
            "action": _stats(acts),
            "proprio": _stats(props),
            "num_transitions": int(len(acts)),
            "num_trajectories": int(len(episodes)),
        }
    }


def _bounds(stats: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        return np.array(stats["min"], np.float32), np.array(stats["max"], np.float32)
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        return np.array(stats["q01"], np.float32), np.array(stats["q99"], np.float32)
    raise ValueError(f"Unsupported normalization {ACTION_PROPRIO_NORMALIZATION_TYPE}")


def normalize(x: np.ndarray, stats: Dict[str, List[float]]) -> np.ndarray:
    """Forward of `normalize_action_and_proprio` for BOUNDS / BOUNDS_Q99 (incl. min==max -> 0)."""
    low, high = _bounds(stats)
    out = np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1)
    return np.where(low == high, 0.0, out).astype(np.float32)


def unnormalize(x: np.ndarray, stats: Dict[str, List[float]]) -> np.ndarray:
    """Inverse of the above -- what `OpenVLAForActionPrediction._unnormalize_actions` does."""
    low, high = _bounds(stats)
    return (0.5 * (x + 1) * (high - low + 1e-8) + low).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------------------------
def _augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Mirror of the RLDS `image_augment_kwargs` used when `image_aug=True`."""
    w, h = img.size
    # random_resized_crop(scale=[0.9, 0.9], ratio=[1.0, 1.0]) -> fixed 90% area, square, random offset
    side = int(round((0.9 ** 0.5) * min(w, h)))
    top = int(rng.integers(0, h - side + 1))
    left = int(rng.integers(0, w - side + 1))
    img = img.crop((left, top, left + side, top + side)).resize((w, h), Image.BILINEAR)

    x = TF.to_tensor(img)
    x = torch.clamp(x + float(rng.uniform(-0.2, 0.2)), 0, 1)                       # random_brightness [0.2]
    mean = x.mean(dim=(1, 2), keepdim=True)
    x = torch.clamp((x - mean) * float(rng.uniform(0.8, 1.2)) + mean, 0, 1)        # random_contrast
    x = torch.clamp(TF.adjust_saturation(x, float(rng.uniform(0.8, 1.2))), 0, 1)   # random_saturation
    x = torch.clamp(TF.adjust_hue(x, float(rng.uniform(-0.05, 0.05))), 0, 1)       # random_hue
    return TF.to_pil_image(x)


def load_frame(frames_dir: pathlib.Path, ep: int, cam: str, t: int, size: int) -> Image.Image:
    img = Image.open(frames_dir / f"ep{ep:03d}" / cam / f"{t:05d}.jpg").convert("RGB")
    return img.resize((size, size), Image.BILINEAR) if img.size != (size, size) else img


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------
class NeroChunkDataset(Dataset):
    """One sample per (episode, start timestep) anchor over the given episodes."""

    def __init__(
        self,
        data_root: str,
        episodes: Sequence[int],
        batch_transform: NeroBatchTransform,
        dataset_statistics: Optional[Dict[str, Dict]] = None,
        image_size: int = 224,
        image_aug: bool = True,
        seed: int = 42,
    ) -> None:
        self.root = pathlib.Path(data_root)
        self.frames_dir = self.root / "frames"
        self.image_size = image_size
        self.image_aug = image_aug
        self.batch_transform = batch_transform
        self.seed = seed

        traj = np.load(self.root / "traj.npz")
        self.traj = {k: traj[k] for k in traj.files}
        self.episodes = list(episodes)
        self.dataset_statistics = dataset_statistics or compute_dataset_statistics(self.traj, self.episodes)
        self.stats = self.dataset_statistics[DATASET_NAME]

        # chunk_act_obs: effective_traj_len = T - (NUM_ACTIONS_CHUNK - 1)
        self.index: List[Tuple[int, int]] = []
        for e in self.episodes:
            T = len(self.traj[f"ep{e:03d}_action"])
            for t in range(T - NUM_ACTIONS_CHUNK + 1):
                self.index.append((e, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ep, t = self.index[i]
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (2**31)))

        action = self.traj[f"ep{ep:03d}_action"][t : t + NUM_ACTIONS_CHUNK]
        proprio = self.traj[f"ep{ep:03d}_state"][t]
        assert action.shape == (NUM_ACTIONS_CHUNK, ACTION_DIM), action.shape
        assert proprio.shape == (PROPRIO_DIM,), proprio.shape

        obs = {}
        for cam, key in zip(CAMS, ("image_primary", "image_wrist")):
            img = load_frame(self.frames_dir, ep, cam, t, self.image_size)
            if self.image_aug:
                img = _augment(img, rng)
            obs[key] = np.asarray(img)[None]  # (1, H, W, 3), window_size=1
        obs["proprio"] = normalize(proprio, self.stats["proprio"])[None]  # (1, D)

        rlds_batch = {
            "dataset_name": DATASET_NAME,
            "action": normalize(action, self.stats["action"]),
            "observation": obs,
            "task": {"language_instruction": NERO_PROMPT.encode()},
        }
        return self.batch_transform(rlds_batch)


_DEFAULT_SPLIT = str(pathlib.Path(os.environ.get("NERO_ROOT", ".")) / "bench/data/split.json")


def load_split(split_json: str = _DEFAULT_SPLIT) -> Dict[str, List[int]]:
    return json.loads(pathlib.Path(split_json).read_text())
