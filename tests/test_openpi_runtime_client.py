import numpy as np

from policy.openpi.clients import runtime_streaming


class _FakeClient:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        if request.get("op") == "reset_stream":
            return {"status": "reset"}
        return {"actions": np.arange(9, dtype=np.float32)}


class _Task:
    instruction = "pull out the key"
    device = "cpu"

    def __init__(self):
        self.action = None

    def take_action(self, action, *, action_type):
        self.action = action
        assert action_type == "qpos"


def test_runtime_client_uses_reset_then_sync_observation_action(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(runtime_streaming, "make_client", lambda host, port: client)
    policy = runtime_streaming.Policy({"host": "localhost", "port": 8000, "state_dim": 9})
    task = _Task()
    observation = {
        "observation": {
            "head": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.zeros(9, dtype=np.float32)},
    }

    policy.reset()
    policy.eval(task, observation)

    assert client.requests[0] == {"op": "reset_stream", "clear_prefix_cache": True}
    assert set(client.requests[1]) == {"images", "state", "prompt"}
    assert policy.needs_observation(task) is True
    np.testing.assert_array_equal(task.action.numpy(), np.arange(8, dtype=np.float32))
