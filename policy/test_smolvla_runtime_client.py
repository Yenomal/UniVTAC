import numpy as np
import torch

from policy.SmolVLA import deploy_policy


class _Task:
    instruction = "move the object"
    device = "cpu"

    def __init__(self):
        self.action = None

    def take_action(self, action, *, action_type):
        assert action_type == "qpos"
        self.action = action
        return True, False


def _observation():
    return {
        "observation": {
            "head": {"rgb": np.zeros((12, 16, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.ones((12, 16, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.arange(9, dtype=np.float32)},
    }


def test_smolvla_runtime_maps_observation_executes_action_then_acknowledges(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            self.events = []

        def reset(self):
            self.events.append(("reset",))

        def observe(self, features, *, task, sequence):
            self.events.append(("observe", features, task, sequence))
            return {"action_id": 0, "action": np.arange(9, dtype=np.float32)}

        def acknowledge(self, action_id):
            self.events.append(("ack", action_id))

        def close(self):
            self.events.append(("close",))

    monkeypatch.setattr(deploy_policy, "RuntimeClient", _Client)
    policy = deploy_policy.Policy({})
    task = _Task()

    policy.eval(task, _observation())

    assert policy._client.events[0] == ("reset",)
    _, features, instruction, sequence = policy._client.events[1]
    assert set(features) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }
    np.testing.assert_array_equal(features["observation.state"], np.arange(9, dtype=np.float32))
    assert features["observation.images.image"].dtype == np.uint8
    assert instruction == task.instruction
    assert sequence == 0
    torch.testing.assert_close(task.action, torch.arange(8, dtype=torch.float32))
    assert policy._client.events[2] == ("ack", 0)


def test_smolvla_client_requests_observation_every_control_step():
    policy = deploy_policy.Policy.__new__(deploy_policy.Policy)
    assert policy.needs_observation(object())
