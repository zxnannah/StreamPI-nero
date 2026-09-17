from openpi.models import pi0_config
from openpi.policies import nero_policy
from openpi.training import config as _config


def test_nero_data_config_preserves_local_root_and_absolute_actions(tmp_path):
    factory = _config.LeRobotNeroDataConfig(
        repo_id="nero/mission2",
        default_prompt="move the green-cap bottle",
        base_config=_config.DataConfig(
            dataset_root="/mnt/nero_view/mission2/smooth",
            hist_horizon=5,
            hist_interval=1,
            hist_interval_range=(1, 2),
        ),
        assets=_config.AssetsConfig(asset_id="nero_mission2"),
    )

    data_config = factory.create(tmp_path, pi0_config.Pi0Config(pi05=True, action_horizon=20, hist_horizon=5))

    assert data_config.dataset_root == "/mnt/nero_view/mission2/smooth"
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.hist_sequence_keys == (
        "observation.images.ego_view",
        "observation.images.wrist_view",
    )
    assert data_config.hist_interval_range == (1, 2)
    assert data_config.use_quantile_norm is False
    assert isinstance(data_config.data_transforms.inputs[0], nero_policy.NeroInputs)
    assert isinstance(data_config.data_transforms.outputs[0], nero_policy.NeroOutputs)


def test_mission2_train_config_uses_single_gpu_time_matched_settings():
    config = _config.get_config("pi05_nero_stream5_mission2")

    assert config.model.action_horizon == 17
    assert config.model.hist_horizon == 5
    assert config.batch_size == 1
    assert config.fsdp_devices == 1
    assert config.data.base_config.dataset_root.endswith("mission2_smooth_lerobot_v2_0")
    assert config.data.base_config.hist_interval_range == (1, 2)


def test_mission7_training_views_config_uses_two_gpu_long_horizon_settings():
    config = _config.get_config("pi05_nero_stream5_mission7_views")

    assert config.model.action_horizon == 17
    assert config.model.hist_horizon == 5
    assert config.batch_size == 2
    assert config.fsdp_devices == 2
    assert config.num_train_steps == 25_000
    assert config.save_interval == 5_000
    assert config.data.repo_id == "nero_mission7_views"
    assert config.data.assets.asset_id == "nero_mission7_views"
    assert config.data.base_config.dataset_root.endswith("mission7_training_views_lerobot_v2_0")
    assert config.data.base_config.hist_interval_range == (1, 2)
    assert config.data.base_config.prompt_from_task is True
    assert "pour its contents into the white box" in config.data.default_prompt
