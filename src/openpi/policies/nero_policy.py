import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class NeroInputs(transforms.DataTransformFn):
    """Convert Nero observations and actions to the common OpenPI format.

    Expected inputs:
    - images.ego_view: [C, H, W] or [T, C, H, W]
    - images.wrist_view: physical right-wrist camera, [C, H, W] or [T, C, H, W]
    - state: [26]
    - actions: [action_horizon, 19] (training only)
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("ego_view", "wrist_view")
    STATE_DIM: ClassVar[int] = 26
    ACTION_DIM: ClassVar[int] = 19

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        unexpected_cameras = set(images) - set(self.EXPECTED_CAMERAS)
        if unexpected_cameras:
            raise ValueError(f"Expected images to contain only {self.EXPECTED_CAMERAS}, got {tuple(images)}")
        missing_cameras = set(self.EXPECTED_CAMERAS) - set(images)
        if missing_cameras:
            raise ValueError(f"Missing required Nero cameras: {tuple(sorted(missing_cameras))}")

        ego_view = _decode_image(images["ego_view"])
        wrist_view = _decode_image(images["wrist_view"])
        if ego_view.shape[0] != wrist_view.shape[0]:
            raise ValueError(
                "Nero camera histories must have the same length, "
                f"got ego_view={ego_view.shape[0]} and wrist_view={wrist_view.shape[0]}"
            )

        state = np.asarray(data["state"])
        if state.ndim not in (1, 2) or state.shape[-1] != self.STATE_DIM:
            raise ValueError(f"Expected Nero state to have shape (26,) or (T, 26), got {state.shape}")

        inputs = {
            "image": {
                "base_0_rgb": ego_view,
                # Nero has a right-wrist camera but no left-wrist camera.
                "left_wrist_0_rgb": np.zeros_like(ego_view),
                "right_wrist_0_rgb": wrist_view,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != self.ACTION_DIM:
                raise ValueError(f"Expected Nero actions to have shape (N, 19), got {actions.shape}")
            inputs["actions"] = actions

        for optional_key in ("prompt", "delay", "action_prefix"):
            if optional_key in data:
                inputs[optional_key] = data[optional_key]

        return inputs


@dataclasses.dataclass(frozen=True)
class NeroOutputs(transforms.DataTransformFn):
    """Remove OpenPI action padding and return the 19-dimensional Nero action vector."""

    ACTION_DIM: ClassVar[int] = 19

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < self.ACTION_DIM:
            raise ValueError(f"Expected actions to have shape (N, D) with D >= 19, got {actions.shape}")
        return {"actions": actions[:, : self.ACTION_DIM]}


def _decode_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)

    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(f"Expected image to have shape (C, H, W) or (T, C, H, W), got {image.shape}")
    if image.shape[1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-first Nero images, got {image.shape}")
    return einops.rearrange(image, "t c h w -> t h w c")
