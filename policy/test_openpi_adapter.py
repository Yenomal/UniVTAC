import numpy as np
from policy._openpi import build_observation
from policy._openpi import compress_qpos
from policy._openpi import select_actions
from policy._openpi import stack_tactile_history
import pytest


def test_build_observation_uses_panda_qpos_prefix():
    observation = {
        "observation": {"head": {"rgb": np.zeros((2, 3, 3))}, "wrist": {"rgb": np.ones((2, 3, 3))}},
        "embodiment": {"joint": np.arange(9)},
    }

    result = build_observation(observation, "task", side_camera="head", wrist_camera="wrist", state_dim=9)

    assert result["state"].tolist() == list(range(9))
    assert result["images"]["cam_side"].dtype == np.uint8


def test_build_observation_adds_tactile_markers():
    markers = np.zeros((2, 63, 2), dtype=np.float32)
    observation = {
        "observation": {
            "head": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.ones((2, 3, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.arange(9)},
        "tactile": {
            "left_tactile": {"marker": markers},
            "right_tactile": {"marker": markers.copy()},
        },
    }

    result = build_observation(
        observation,
        "task",
        side_camera="head",
        wrist_camera="wrist",
        state_dim=9,
        include_tactile=True,
        marker_count=63,
    )

    assert result["left_marker"].shape == (2, 63, 2)
    assert result["right_marker"].dtype == np.float32


def test_build_observation_rejects_checkpoint_marker_count_mismatch():
    observation = {
        "observation": {
            "head": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.ones((2, 3, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.arange(9)},
        "tactile": {
            "left_tactile": {"marker": np.zeros((2, 64, 2))},
            "right_tactile": {"marker": np.zeros((2, 64, 2))},
        },
    }

    with pytest.raises(ValueError, match="63 markers"):
        build_observation(
            observation,
            "task",
            side_camera="head",
            wrist_camera="wrist",
            state_dim=9,
            include_tactile=True,
            marker_count=63,
        )


def test_stack_tactile_history_is_oldest_to_newest_and_pads_oldest():
    first = np.full((2, 63, 2), 1.0, dtype=np.float32)
    second = np.full((2, 63, 2), 2.0, dtype=np.float32)
    history = stack_tactile_history([first, second], history_size=4, marker_count=63)
    assert history.shape == (4, 2, 63, 2)
    np.testing.assert_array_equal(history[:, 0, 0, 0], [1.0, 1.0, 1.0, 2.0])


def test_stack_tactile_history_keeps_only_recent_frames():
    frames = [np.full((2, 63, 2), value, dtype=np.float32) for value in (1, 2, 3)]
    history = stack_tactile_history(frames, history_size=2, marker_count=63)
    np.testing.assert_array_equal(history[:, 0, 0, 0], [2.0, 3.0])


def test_compress_qpos_uses_first_finger_dimension():
    actions = np.asarray([[1, 2, 3, 4, 5, 6, 7, 0.02, 0.04]], dtype=np.float32)

    np.testing.assert_allclose(compress_qpos(actions), [[1, 2, 3, 4, 5, 6, 7, 0.02]])


def test_select_actions_requires_explicit_mapping_for_wrong_action_dimension():
    with pytest.raises(ValueError, match="action_indices"):
        select_actions({"actions": np.zeros((1, 18))}, action_dim=8, action_indices=None)


def test_select_actions_returns_writable_array_for_read_only_response():
    response_actions = np.zeros((1, 9), dtype=np.float32)
    response_actions.flags.writeable = False

    actions = select_actions({"actions": response_actions}, action_dim=9, action_indices=None)

    assert actions.flags.c_contiguous
    assert actions.flags.writeable
