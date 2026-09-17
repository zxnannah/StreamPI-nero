"""Validate one fully transformed and normalized Nero Mission 7 batch without training."""

import argparse

import numpy as np

from openpi.training import config
from openpi.training import data_loader


def main(config_name: str) -> None:
    train_config = config.get_config(config_name)
    loader = data_loader.create_data_loader(
        train_config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=False,
    )
    observation, actions_value = next(iter(loader))
    state = np.asarray(observation.state)
    actions = np.asarray(actions_value)

    print("state:", state.shape, state.dtype)
    print("actions:", actions.shape, actions.dtype)
    for key, value in observation.images.items():
        array = np.asarray(value)
        print("image:", key, array.shape, array.dtype, "range:", float(array.min()), float(array.max()))
    for key, value in observation.image_masks.items():
        print("image_mask:", key, np.asarray(value))

    if state.shape != (train_config.batch_size, 32):
        raise ValueError(f"Unexpected state shape: {state.shape}")
    expected_action_shape = (train_config.batch_size, train_config.model.action_horizon, 32)
    if actions.shape != expected_action_shape:
        raise ValueError(f"Unexpected action shape: {actions.shape}, expected {expected_action_shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Normalized state or actions contain NaN/Inf")
    if not np.allclose(state[..., 26:], 0):
        raise ValueError("State padding dimensions are not zero")
    if not np.allclose(actions[..., 19:], 0):
        raise ValueError("Action padding dimensions are not zero")

    expected_images = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    if set(observation.images) != expected_images:
        raise ValueError(f"Unexpected image keys: {set(observation.images)}")
    expected_image_shape = (train_config.batch_size, train_config.model.hist_horizon, 224, 224, 3)
    for key, value in observation.images.items():
        if np.asarray(value).shape != expected_image_shape:
            raise ValueError(f"Unexpected image shape for {key}: {np.asarray(value).shape}")

    if not np.asarray(observation.image_masks["base_0_rgb"]).all():
        raise ValueError("Base camera mask must be true")
    if np.asarray(observation.image_masks["left_wrist_0_rgb"]).any():
        raise ValueError("Synthetic left-wrist camera mask must be false")
    if not np.asarray(observation.image_masks["right_wrist_0_rgb"]).all():
        raise ValueError("Real right-wrist camera mask must be true")
    print("Mission 7 normalized batch validation: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    args = parser.parse_args()
    main(args.config_name)
