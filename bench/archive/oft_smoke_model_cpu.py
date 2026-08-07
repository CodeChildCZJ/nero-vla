#!/usr/bin/env python3
"""CPU end-to-end smoke: build the exact finetune_nero model stack and run one fwd+bwd.

Verifies everything that ACTION_DIM=8 / chunk=8 / 2-camera / proprio touches (NUM_PATCHES,
hidden-state slicing, action-head shapes, L1 loss) without occupying a GPU. Slow (7B on CPU),
batch size 1.
"""
import os
import pathlib
import sys
import time

os.environ.setdefault("ROBOT_PLATFORM", "NERO")
BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH.parent / "third_party" / "openvla-oft"))

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets.nero_dataset import NeroBatchTransform, NeroChunkDataset, load_split

from experiments.robot.openvla_utils import check_model_logic_mismatch, update_auto_map

# Writable copy of the base model (weights symlinked into the HF blob store); the auto_map /
# modeling-code sync below rewrites small files in place, which must not happen inside the cache.
base = str(BENCH / "models" / "openvla-7b")
os.chdir(BENCH.parent / "third_party" / "openvla-oft")  # check_model_logic_mismatch walks ./prismatic/
update_auto_map(base)
check_model_logic_mismatch(base)

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

t0 = time.time()
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(base, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
print(f"loaded VLA in {time.time() - t0:.0f}s; llm_dim={vla.llm_dim}; attn={vla.config._attn_implementation}")

vla.vision_backbone.set_num_images_in_input(2)
vla = get_peft_model(vla, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian"))
vla.print_trainable_parameters()

proprio_projector = ProprioProjector(llm_dim=vla.module.llm_dim if hasattr(vla, "module") else vla.base_model.model.llm_dim, proprio_dim=PROPRIO_DIM)
llm_dim = proprio_projector.llm_dim if hasattr(proprio_projector, "llm_dim") else vla.base_model.model.llm_dim
action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)

NUM_PATCHES = vla.base_model.model.vision_backbone.get_num_patches() * vla.base_model.model.vision_backbone.get_num_images_in_input() + 1  # +1 proprio
print(f"NUM_PATCHES (2 imgs + proprio) = {NUM_PATCHES}")

split = load_split()
ds = NeroChunkDataset(
    data_root=str(BENCH / "data" / "b2_oft"),
    episodes=split["train_episodes"],
    batch_transform=NeroBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    ),
    image_size=224,
    image_aug=True,
)
batch = next(iter(DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"))))

t0 = time.time()
output = vla(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    pixel_values=batch["pixel_values"],
    labels=batch["labels"],
    output_hidden_states=True,
    proprio=batch["proprio"],
    proprio_projector=proprio_projector,
    use_film=False,
)
print(f"forward ok in {time.time() - t0:.0f}s; logits {tuple(output.logits.shape)}")

gt_ids = batch["labels"][:, 1:]
cur_mask, nxt_mask = get_current_action_mask(gt_ids), get_next_actions_mask(gt_ids)
text_hidden = output.hidden_states[-1][:, NUM_PATCHES:-1]
ah = text_hidden[cur_mask | nxt_mask].reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
print(f"actions_hidden_states {tuple(ah.shape)} (expect (1, {NUM_ACTIONS_CHUNK * ACTION_DIM}, {llm_dim}))")
assert ah.shape == (1, NUM_ACTIONS_CHUNK * ACTION_DIM, llm_dim), ah.shape

pred = action_head.predict_action(ah)
print(f"predicted_actions {tuple(pred.shape)} (expect (1, {NUM_ACTIONS_CHUNK}, {ACTION_DIM}))")
assert pred.shape == (1, NUM_ACTIONS_CHUNK, ACTION_DIM), pred.shape

loss = torch.nn.L1Loss()(batch["actions"], pred)
print(f"L1 loss = {loss.item():.4f}")
t0 = time.time()
loss.backward()
n_grad = sum(1 for p in vla.parameters() if p.requires_grad and p.grad is not None)
print(f"backward ok in {time.time() - t0:.0f}s; {n_grad} LoRA params got grads")
assert n_grad > 0
print("\nMODEL SMOKE OK (CPU)")
