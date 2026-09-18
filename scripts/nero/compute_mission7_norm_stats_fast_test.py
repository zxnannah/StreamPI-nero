import compute_mission7_norm_stats_fast as fast_norm
import numpy as np


def test_action_chunks_clamp_at_virtual_episode_end():
    actions = np.arange(4 * fast_norm.EXPECTED_ACTION_DIM, dtype=np.float32).reshape(
        4, fast_norm.EXPECTED_ACTION_DIM
    )

    chunks = fast_norm.build_action_chunks(actions, action_horizon=3)

    np.testing.assert_array_equal(chunks[0], actions[[0, 1, 2]])
    np.testing.assert_array_equal(chunks[2], actions[[2, 3, 3]])
    np.testing.assert_array_equal(chunks[3], actions[[3, 3, 3]])
