import numpy as np
import pytest

from openpi.policies import nero_policy


def test_nero_inputs_maps_right_wrist_camera_and_adds_masked_black_left_camera():
    inputs = nero_policy.NeroInputs()(
        {
            "images": {
                "ego_view": np.ones((5, 3, 8, 12), dtype=np.float32),
                "wrist_view": np.full((5, 3, 6, 10), 0.5, dtype=np.float32),
            },
            "state": np.arange(26, dtype=np.float32),
            "actions": np.ones((20, 19), dtype=np.float32),
            "prompt": "move the green-cap bottle into the white rectangle",
        }
    )

    assert inputs["image"]["base_0_rgb"].shape == (5, 8, 12, 3)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (5, 8, 12, 3)
    assert inputs["image"]["right_wrist_0_rgb"].shape == (5, 6, 10, 3)
    assert inputs["image"]["base_0_rgb"].dtype == np.uint8
    assert np.all(inputs["image"]["left_wrist_0_rgb"] == 0)
    assert inputs["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.True_,
    }
    assert inputs["state"].shape == (26,)
    assert inputs["actions"].shape == (20, 19)


@pytest.mark.parametrize(
    ("field", "shape", "message"),
    [
        ("state", (25,), "Nero state"),
        ("actions", (20, 18), "Nero actions"),
    ],
)
def test_nero_inputs_rejects_wrong_vector_dimensions(field, shape, message):
    data = {
        "images": {
            "ego_view": np.zeros((3, 8, 12), dtype=np.uint8),
            "wrist_view": np.zeros((3, 6, 10), dtype=np.uint8),
        },
        "state": np.zeros((26,), dtype=np.float32),
        "actions": np.zeros((20, 19), dtype=np.float32),
    }
    data[field] = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        nero_policy.NeroInputs()(data)


def test_nero_outputs_removes_model_padding():
    model_actions = np.arange(2 * 32).reshape(2, 32)

    output = nero_policy.NeroOutputs()({"actions": model_actions})

    assert output["actions"].shape == (2, 19)
    assert np.array_equal(output["actions"], model_actions[:, :19])
