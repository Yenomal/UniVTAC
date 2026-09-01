import numpy as np

from policy.GR00T.state_action_adapter import _quat_to_rot6d, decode_qpos


def test_quaternion_to_gr00t_eef_shape():
    rot6d = _quat_to_rot6d(np.array([1, 0, 0, 0], dtype=np.float32))
    assert rot6d.shape == (6,)
    np.testing.assert_allclose(rot6d, [1, 0, 0, 0, 1, 0])


def test_decoded_droid_action_to_univtac_qpos():
    class Manager:
        def gripper_percent2qpos(self, value):
            return value * 0.039

    class Task:
        _robot_manager = Manager()

    action = {
        "eef_9d": np.zeros((40, 9), dtype=np.float32),
        "joint_position": np.ones((40, 7), dtype=np.float32),
        "gripper_position": np.full((40, 1), 0.5, dtype=np.float32),
    }
    qpos = decode_qpos(Task(), action)
    assert tuple(qpos.shape) == (8,)
    np.testing.assert_allclose(qpos[:7].numpy(), 1.0)
    np.testing.assert_allclose(qpos[7].item(), 0.0195)

