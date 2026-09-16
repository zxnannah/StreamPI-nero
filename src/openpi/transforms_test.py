import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_temporal_jitter_samples_independent_intervals_and_keeps_cameras_synchronized(monkeypatch):
    sampled_gaps = iter([1, 2, 1, 2])
    monkeypatch.setattr(_transforms.random, "randint", lambda low, high: next(sampled_gaps))
    transform = _transforms.TemporalJitter(
        jitter_range=(-2, -1, 0, 1, 2),
        hist_interval=1,
        hist_horizon=5,
        hist_sequence_keys=("ego", "wrist"),
        enable_jitter=False,
        hist_interval_range=(1, 2),
    )
    data = {
        "ego": np.arange(9),
        "wrist": np.arange(9) + 100,
    }

    transformed = transform(data)

    assert np.array_equal(transformed["ego"], np.array([2, 4, 5, 7, 8]))
    assert np.array_equal(transformed["wrist"], np.array([102, 104, 105, 107, 108]))


def test_temporal_jitter_uses_one_offset_for_all_cameras(monkeypatch):
    choices = iter([1, -1])
    monkeypatch.setattr(_transforms.random, "choice", lambda values: next(choices))
    transform = _transforms.TemporalJitter(
        jitter_range=(-1, 0, 1),
        hist_interval=2,
        hist_horizon=3,
        hist_sequence_keys=("ego", "wrist"),
        enable_jitter=True,
    )
    data = {
        "ego": np.arange(7),
        "wrist": np.arange(7) + 100,
    }

    transformed = transform(data)

    assert np.array_equal(transformed["ego"], np.array([3, 5, 6]))
    assert np.array_equal(transformed["wrist"], np.array([103, 105, 106]))


def test_temporal_jitter_rejects_ambiguous_sampling_modes():
    with pytest.raises(ValueError, match="cannot be used together"):
        _transforms.TemporalJitter(
            jitter_range=(-1, 0, 1),
            hist_interval=1,
            hist_horizon=5,
            hist_sequence_keys=("ego", "wrist"),
            enable_jitter=True,
            hist_interval_range=(1, 2),
        )
