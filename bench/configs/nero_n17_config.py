# GR00T N1.7 modality config for the NERO cobot_magic arm (b2 dataset).
#
# Mirrors the official `examples/SO100/so100_config.py` recipe -- same morphology
# (one arm + one gripper, two cameras), only the embodiment dimensions and camera
# names differ:
#
#   state / action : 8-D = 7 arm joints (single_arm) + 1 gripper (gripper)
#   video          : cam_high (overhead) + cam_wrist
#
# The 7/1 split and the camera->column mapping live in the dataset's
# meta/modality.json; this file only declares what the model consumes.
#
# Action space is JOINT space (ActionType.NON_EEF), NOT the relative-EEF space
# that N1.7's built-in humanoid embodiments use. The bench contract requires
# predictions in the dataset's own action units (arm ~degrees, gripper 0-100).
# `single_arm` is RELATIVE (delta from current state) per the official recipe --
# StateActionProcessor.unapply() adds the reference state back on the way out, so
# `get_action()` still returns absolute joint targets. See
# gr00t/data/state_action/state_action_processor.py (unapply, step 2).
#
# NOTE: joints are stored in DEGREES, so `sin_cos_embedding_keys` is deliberately
# left unset (that encoding assumes radians); min-max normalization is used.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Action horizon. The bench contract needs >= 8 predicted steps; 16 matches the
# official SO100 new-embodiment recipe. Changing this REQUIRES regenerating
# meta/relative_stats.json via gr00t/data/stats.py (per-step stats are shaped
# (horizon, D) and a mismatch raises IndexError during normalization).
ACTION_HORIZON = 16

nero_config = {
    # Current frame only; keys match "video" in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_wrist"],
    ),
    # Current proprioceptive reading; keys match "state" in meta/modality.json.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # 7 arm joints: RELATIVE delta from current state, joint-space.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Gripper: ABSOLUTE target -- a near-binary open/close signal that
            # does not benefit from being expressed as a delta.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(nero_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
