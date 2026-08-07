"""NERO single-arm (7DoF + gripper = 8-dim) policy transforms.

This is a 1:1 copy of the official ``libero_policy.py`` single-arm template, with the
only changes being the action/state dimension (7 -> 8) and the image source keys.
NERO is a single 7-DoF arm + 1 gripper, so the LeRobot dataset has:
  observation.state : float32[8]  = [7 joint deg, 1 gripper (normalized 0..1)]
  action            : float32[8]  = same layout
  observation.images.cam_high  : third-person view
  observation.images.cam_wrist : wrist view
The delta/gripper-absolute split (DeltaActions/AbsoluteActions with make_bool_mask(7,-1))
is applied in the DataConfig exactly like the official DROID 8-dim recipe.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_nero_example() -> dict:
    """Creates a random input example for the NERO policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick the pink sponge and place it in the blue bucket",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NeroInputs(transforms.DataTransformFn):
    """Convert NERO inputs to the model format. Used for both training and inference."""

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot stores images as float32 (C,H,W); parse to uint8 (H,W,C).
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Single wrist camera -> left_wrist slot; right_wrist padded with zeros.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class NeroOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the NERO dataset format. Inference only."""

    def __call__(self, data: dict) -> dict:
        # NERO action dimension is 8 (7 joints + 1 gripper); the rest is model padding.
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class NeroRefDegToRad(transforms.DataTransformFn):
    """SERVE-ONLY (ref): client_b sends state in deg + gripper raw-mm (the b1/B-abs contract),
    but the ref model was trained on rad + gripper[0,1] (baked into the b2_ref dataset by the
    --jointsRad/--gripOpenMm converter). Convert the incoming ``state`` (joints deg->rad,
    gripper mm->[0,1]) so Normalize (norm_stats live in rad+[0,1] space) applies correctly.

    Pipeline order (policy_config.py): runs as the LAST data_transforms.inputs entry, i.e.
    AFTER NeroInputs (which copies observation/state -> "state") and BEFORE Normalize.
    Append it via Group.push so it sees the "state" key, not "observation/state".
    """

    grip_open_mm: float = 76.0  # b2_ref converter: gripClosedMm=0, gripOpenMm=76 -> [0,1] = mm/76

    def __call__(self, data: dict) -> dict:
        if "state" not in data:
            return data
        s = np.asarray(data["state"], dtype=np.float32).copy()
        s[..., :7] = s[..., :7] * (np.pi / 180.0)                       # joints deg -> rad
        s[..., 7] = np.clip(s[..., 7] / self.grip_open_mm, 0.0, 1.0)    # gripper mm -> [0,1]
        return {**data, "state": s}


@dataclasses.dataclass(frozen=True)
class NeroRefRadToDeg(transforms.DataTransformFn):
    """SERVE-ONLY (ref): inverse of NeroRefDegToRad on the model OUTPUT. After Unnormalize +
    NeroOutputs the action chunk is in rad + gripper[0,1]; convert back to deg + mm so the
    server exposes the SAME deg+mm contract as b1/B-abs (client_b stays unchanged).

    Pipeline order: runs as the LAST data_transforms.outputs entry, i.e. AFTER Unnormalize and
    AFTER NeroOutputs (which has already sliced actions[:, :8]).
    """

    grip_open_mm: float = 76.0

    def __call__(self, data: dict) -> dict:
        a = np.asarray(data["actions"], dtype=np.float32).copy()        # [horizon, 8]
        a[..., :7] = a[..., :7] * (180.0 / np.pi)                       # joints rad -> deg
        a[..., 7] = np.clip(a[..., 7], 0.0, 1.0) * self.grip_open_mm    # gripper [0,1] -> mm
        return {**data, "actions": a}
