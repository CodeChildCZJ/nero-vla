#!/usr/bin/env python3
"""Run GR00T N1.7's official finetune launcher with an overridable VLM backbone.

`gr00t/experiment/launch_finetune.py` hardcodes

    config.model.model_name = "nvidia/Cosmos-Reason2-2B"

which is a *gated* HF repo. It is fetched for two things: the Qwen3-VL
architecture skeleton and the tokenizer/image processor. The actual backbone
weights come from `--base-model-path` (nvidia/GR00T-N1.7-3B, ungated, whose
checkpoint contains all 494 `backbone.*` tensors) and overwrite whatever the
skeleton was initialized with.

So when the gated repo is unavailable, pointing `model_name` at an equivalent
ungated checkpoint is sound. Set NERO_BACKBONE_MODEL to override, e.g. to a
local snapshot of Qwen/Qwen3-VL-2B-Instruct -- Cosmos-Reason2-2B's own declared
`base_model`, same `Qwen3VLForConditionalGeneration`/`qwen3_vl` architecture,
same hidden size 2048, byte-identical chat template and special tokens.

NOTE: the override must contain the literal "Qwen/Qwen3-VL" or
"nvidia/Cosmos-Reason2" in its path -- `gr00t_n1d7.get_backbone_cls()` selects
the backbone class by substring-matching `model_name` and raises
"Unsupported model name" for anything else. Hence the local snapshot lives at
bench/models/Qwen/Qwen3-VL-2B-Instruct rather than a flat directory.

Everything else is the official launcher, unmodified: we patch the `run` symbol
on the experiment module *before* importing the launcher, so the launcher's
`from gr00t.experiment.experiment import run` picks up the wrapper.

Usage: identical to the official script, plus optional
    NERO_BACKBONE_MODEL=/path/to/backbone
    NERO_DRYRUN=1   build model+processor+dataset on whatever device is visible,
                    report what the real run would do, then exit before training.
"""
import os
import pathlib
import runpy
import sys

import gr00t.experiment.experiment as _exp

_orig_run = _exp.run


def _dryrun(config):
    """Everything experiment.run() does up to the Trainer, then stop.

    Used to verify a transplanted stack (e.g. after rsync'ing repo+venv to
    another host) without burning a GPU slot: the model build, the checkpoint
    weight-key check, the processor construction and the dataset scan all run
    here, and none of them need a CUDA context.
    """
    import torch
    from gr00t.model import MODEL_REGISTRY

    print("[dryrun] model_name       :", config.model.model_name)
    print("[dryrun] start_from_ckpt  :", config.training.start_from_checkpoint)
    print("[dryrun] use_relative_act :", config.model.use_relative_action)
    print("[dryrun] use_percentiles  :", config.model.use_percentiles)
    print("[dryrun] tune llm/vis/proj/diff:", config.model.tune_llm, config.model.tune_visual,
          config.model.tune_projector, config.model.tune_diffusion_model)
    print("[dryrun] batch/steps/lr   :", config.training.global_batch_size,
          config.training.max_steps, config.training.learning_rate)
    print("[dryrun] save steps/limit/only_model:", config.training.save_steps,
          config.training.save_total_limit, config.training.save_only_model)
    # The LR schedule is the classic silent-waste knob: a decay horizon that is
    # not max_steps burns the whole run at the wrong LR and still exits rc=0.
    # HF derives num_training_steps from max_steps, and warmup_steps>0 silently
    # OVERRIDES warmup_ratio (experiment.py:93), so print both.
    print("[dryrun] scheduler        :", config.training.lr_scheduler_type,
          "| warmup_ratio", config.training.warmup_ratio,
          "| warmup_steps", config.training.warmup_steps,
          "->", ("warmup_steps WINS" if config.training.warmup_steps > 0
                 else f"{int(config.training.warmup_ratio * config.training.max_steps)} warmup steps"))
    print("[dryrun] grad_accum/num_gpus/per_device:",
          config.training.gradient_accumulation_steps, config.training.num_gpus,
          config.training.global_batch_size // config.training.num_gpus)
    print("[dryrun] bf16/fp16/tf32/grad_ckpt:", config.training.bf16, config.training.fp16,
          config.training.tf32, config.training.gradient_checkpointing)
    print("[dryrun] wandb            :", config.training.use_wandb,
          config.training.wandb_project, config.training.experiment_name)
    print("[dryrun] output_dir       :", config.training.output_dir)
    print("[dryrun] resume_from_ckpt :", config.training.resume_from_checkpoint)
    # episode_sampling_rate's docstring says "fraction of episode timesteps to
    # use", but shard_dataset() splits each episode into int(1/rate) INTERLEAVED
    # subsequences and keeps every one -- no data is dropped. It sets shard
    # granularity. Verified against the "Total steps:" line the scan prints below.
    print("[dryrun] shard_size/ep_sampling_rate/shards_per_epoch:",
          config.data.shard_size, config.data.episode_sampling_rate,
          config.data.num_shards_per_epoch)
    print("[dryrun] data seed        :", config.data.seed)
    print("[dryrun] cuda available   :", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        # flash-attn has no CPU path, so a CPU rehearsal has to fall back to sdpa.
        # The real run keeps flash-attn; check_n17_stack.py proves the sm_120
        # kernels actually launch once a card is in hand. Note this has to go
        # through the from_pretrained override too -- setting it on config.model
        # is not enough, the checkpoint's own config wins (see _patch_model_build).
        config.model.use_flash_attention = False
        _MODEL_KWARG_OVERRIDES["use_flash_attention"] = False
        print("[dryrun] no CUDA -> use_flash_attention=False (sdpa) for this rehearsal only")

    save_cfg_dir = pathlib.Path(config.training.output_dir) / "_dryrun_cfg"
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()  # builds the model (asserts no missing/unexpected ckpt keys)
    model = pipeline.return_model()
    train_ds, _ = pipeline.return_dataset()
    processor = pipeline.return_processor()

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dryrun] params total {total:,} trainable {train:,} ({100 * train / total:.2f}%)")
    sap = processor.state_action_processor
    print("[dryrun] processor use_relative_action:", sap.use_relative_action,
          "| use_percentiles:", sap.use_percentiles)
    assert sap.use_relative_action, "RELATIVE arm needs use_relative_action on the processor"
    _report_training_episodes(train_ds)

    # ShardedMixtureDataset is iterable-style (no __len__/__getitem__), so pull the
    # first sample the way the dataloader would.
    print("[dryrun] train dataset:", type(train_ds).__name__)
    sample = next(iter(train_ds))
    for k, v in sample.items():
        shape = getattr(v, "shape", None)
        print(f"[dryrun]   {k}: {type(v).__name__} {shape if shape is not None else v}")

    # Collate too: this is where actions get padded to max_action_horizon and the
    # action mask is built, i.e. where a horizon mismatch with the action head
    # would actually bite.
    batch = pipeline.return_collator()([sample])
    print("[dryrun] collated batch:")
    for k, v in batch.items():
        shape = getattr(v, "shape", None)
        print(f"[dryrun]   {k}: {type(v).__name__} {shape if shape is not None else ''}")
    print("[dryrun] OK -- stopped before Trainer", flush=True)


def _report_training_episodes(train_ds):
    """Runtime leak check on the object the Trainer will actually iterate.

    Reading `--dataset-path` off the command line proves which string was typed,
    not which episodes get sampled. So walk the constructed dataset, recover the
    episode ids it will draw from, and diff them against split.json. A val episode
    surfacing here voids the leg, and it would otherwise be invisible -- training
    on 131 episodes raises nothing and just quietly scores better.
    """
    import json

    subsets = getattr(train_ds, "datasets", [train_ds])
    for i, ds in enumerate(subsets):
        loader = getattr(ds, "episode_loader", None)
        path = getattr(ds, "dataset_path", "?")
        if loader is None:
            print(f"[dryrun] subset {i}: {path} -- no episode_loader, CANNOT VERIFY")
            continue
        lengths = loader.episode_lengths
        # `episode_lengths` is positional; the real ids live in meta/episodes.jsonl.
        ep_file = pathlib.Path(str(path)) / "meta" / "episodes.jsonl"
        ids = None
        if ep_file.exists():
            ids = [json.loads(line)["episode_index"]
                   for line in ep_file.read_text().splitlines() if line.strip()]
        print(f"[dryrun] subset {i}: {path}")
        print(f"[dryrun]   episodes={len(lengths)} steps={int(sum(lengths))}")
        # 仓库自包含:bench/scripts/train/n17_launch_finetune.py -> parents[2] == bench/
        split_file = pathlib.Path(__file__).resolve().parents[2] / "data" / "split.json"
        if ids is not None and split_file.exists():
            split = json.loads(split_file.read_text())
            val = set(split["val_episodes"])
            train = set(split["train_episodes"])
            got = set(ids)
            print(f"[dryrun]   ids: n={len(got)} min={min(got)} max={max(got)}")
            print(f"[dryrun]   INTERSECT val = {sorted(got & val)}")
            print(f"[dryrun]   == split train_episodes? {got == train}")
            assert not (got & val), f"LEAK: val episodes in the training set: {sorted(got & val)}"
            assert got == train, (
                f"training set is not the 111-episode split "
                f"(missing {sorted(train - got)}, extra {sorted(got - train)})")
            print("[dryrun]   LEAK CHECK PASS")
        else:
            print("[dryrun]   ids/split unavailable -- CANNOT VERIFY")


# Extra config-attribute overrides to inject into the model build. See _patch_model_build.
_MODEL_KWARG_OVERRIDES: dict = {}


def _patch_model_build():
    """Make our overrides reach the *model* build, not just the processor.

    `Gr00tN1d7Pipeline._create_model` calls
    `AutoModel.from_pretrained(<GR00T-N1.7-3B ckpt>, tune_llm=..., ...)` and
    `Gr00tN1d7.__init__` then reads its settings off the **checkpoint's own
    config.json**, not off the Config object the launcher just assembled. So
    `config.model.model_name = ...` reaches the processor and the from-scratch
    branch only, and the build still 403s on the gated Cosmos repo. Same trap
    for `use_flash_attention`, and for anything else living on the model config.

    from_pretrained already applies extra kwargs as config-attribute overrides --
    that is exactly how setup.py injects tune_llm/tune_visual/... -- so routing
    ours through the same door is enough. A redirected `model_name` also gets
    baked into the finetuned checkpoint, which is what we want: inference stops
    depending on Cosmos access too.
    """
    import gr00t.model.gr00t_n1d7.setup as _setup

    _orig_auto_model = _setup.AutoModel

    class _AutoModelWithOverrides:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            for key, value in _MODEL_KWARG_OVERRIDES.items():
                kwargs.setdefault(key, value)
            if _MODEL_KWARG_OVERRIDES:
                print("[nero] model-build overrides:", _MODEL_KWARG_OVERRIDES, flush=True)
            return _orig_auto_model.from_pretrained(*args, **kwargs)

    _setup.AutoModel = _AutoModelWithOverrides


def _trim_modality_configs(config):
    """Keep only the embodiments this run actually trains on.

    `DataConfig.modality_configs` defaults to the whole built-in registry, and one
    of its entries -- unitree_g1_full_body_with_waist_height_nav_cmd -- declares 50
    action delta_indices while both Gr00tN1d7Config.action_horizon and the released
    GR00T-N1.7-3B config say 40. `Gr00tN1d7Processor.__init__` validates every
    registered embodiment against max_action_horizon (= the model's action_horizon,
    setup.py:170), so the stock launcher cannot even build its processor:

        ValueError: Embodiment action horizon exceeds max_action_horizon (40):
        unitree_g1_full_body_with_waist_height_nav_cmd=50

    Raising max_action_horizon to 50 would desync the action mask from the
    pretrained action head, so instead drop the embodiments we never touch. Safe
    for the embodiment ids, which come from the checkpoint's embodiment_id.json
    rather than from this dict's ordering.
    """
    used = {ds.embodiment_tag for ds in config.data.datasets}
    if not used:
        return
    keep = {k: v for k, v in config.data.modality_configs.items() if k in used}
    missing = used - keep.keys()
    assert not missing, f"no modality config registered for {sorted(missing)}"
    dropped = len(config.data.modality_configs) - len(keep)
    config.data.modality_configs = keep
    print(f"[nero] modality_configs: kept {sorted(keep)}, dropped {dropped} unused embodiments",
          flush=True)


def _run_with_backbone_override(config):
    _trim_modality_configs(config)
    override = os.environ.get("NERO_BACKBONE_MODEL")
    if override:
        print(f"[nero] backbone model_name: {config.model.model_name!r} -> {override!r}",
              flush=True)
        config.model.model_name = override
        _MODEL_KWARG_OVERRIDES["model_name"] = override
    _patch_model_build()
    if os.environ.get("NERO_DRYRUN") == "1":
        return _dryrun(config)
    return _orig_run(config)


_exp.run = _run_with_backbone_override

if __name__ == "__main__":
    sys.argv[0] = "gr00t/experiment/launch_finetune.py"
    runpy.run_module("gr00t.experiment.launch_finetune", run_name="__main__")
