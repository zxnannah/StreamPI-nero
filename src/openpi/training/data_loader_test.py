import dataclasses
import types

import jax
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_local_dataset_root_and_random_history_span_are_forwarded(monkeypatch):
    calls = {}

    class TestDataset:
        def __init__(self, repo_id, *, root, delta_timestamps):
            calls["dataset"] = (repo_id, root, delta_timestamps)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise NotImplementedError

    def make_metadata(repo_id, *, root):
        calls["metadata"] = (repo_id, root)
        return types.SimpleNamespace(fps=10, tasks={})

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", make_metadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", TestDataset)
    data_config = _config.DataConfig(
        repo_id="nero/mission2",
        dataset_root="/data/nero/mission2-v2",
        action_sequence_keys=("action",),
        hist_sequence_keys=("observation.images.ego_view", "observation.images.wrist_view"),
        hist_horizon=5,
        hist_interval=1,
        hist_interval_range=(1, 2),
    )

    _data_loader.create_torch_dataset(data_config, action_horizon=20, model_config=pi0_config.Pi0Config())

    assert calls["metadata"] == ("nero/mission2", "/data/nero/mission2-v2")
    assert calls["dataset"][0:2] == ("nero/mission2", "/data/nero/mission2-v2")
    delta_timestamps = calls["dataset"][2]
    assert delta_timestamps["action"] == pytest.approx([step / 10 for step in range(20)])
    assert delta_timestamps["observation.images.ego_view"] == pytest.approx(
        [step / 10 for step in range(-8, 1)]
    )
    assert delta_timestamps["observation.images.wrist_view"] == pytest.approx(
        [step / 10 for step in range(-8, 1)]
    )
