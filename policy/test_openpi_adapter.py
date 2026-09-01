import numpy as np
from policy._openpi import build_observation
from policy._openpi import select_actions
import pytest


def test_build_observation_uses_panda_qpos_prefix():
    observation = {
        "observation": {"head": {"rgb": np.zeros((2, 3, 3))}, "wrist": {"rgb": np.ones((2, 3, 3))}},
        "embodiment": {"joint": np.arange(9)},
    }

    result = build_observation(observation, "task", side_camera="head", wrist_camera="wrist", state_dim=8)

    assert result["state"].tolist() == list(range(8))
    assert result["images"]["cam_side"].dtype == np.uint8


def test_select_actions_requires_explicit_mapping_for_wrong_action_dimension():
    with pytest.raises(ValueError, match="action_indices"):
        select_actions({"actions": np.zeros((1, 18))}, action_dim=8, action_indices=None)
