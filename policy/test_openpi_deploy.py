import threading

import numpy as np
import pytest
import torch

from policy.openpi import deploy_policy as openpi_deploy
from policy.streaming_openpi import deploy_policy as streaming_deploy


class _Task:
    instruction = "test task"
    device = "cpu"

    def __init__(self, take_action_cnt: int = 0):
        self.take_action_cnt = take_action_cnt
        self.actions = []
        self.metadata = {}

    def take_action(self, action, *, action_type):
        assert action_type == "qpos"
        self.actions.append(action.clone())
        self.take_action_cnt += 1


def _observation():
    markers = np.zeros((2, 63, 2), dtype=np.float32)
    return {
        "observation": {
            "head": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.zeros(9, dtype=np.float32)},
        "tactile": {
            "left_tactile": {"marker": markers},
            "right_tactile": {"marker": markers.copy()},
        },
    }


def test_openpi_watermark_controls_replanning(monkeypatch):
    class Client:
        def __init__(self):
            self.calls = 0
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            self.calls += 1
            return {
                "actions": np.arange(27, dtype=np.float32).reshape(3, 9)
                + 100 * (self.calls - 1)
            }

    client = Client()
    monkeypatch.setattr(openpi_deploy, "make_client", lambda host, port: client)
    policy = openpi_deploy.Policy(
        {"watermark": 2, "state_dim": 9, "use_tactile": True, "marker_count": 63}
    )
    task = _Task()

    policy.eval(task, _observation())
    policy.eval(task, _observation())
    policy.eval(task, _observation())

    assert client.calls == 2
    assert client.requests[0]["left_marker"].shape == (2, 63, 2)
    assert client.requests[0]["right_marker"].shape == (2, 63, 2)
    torch.testing.assert_close(task.actions[0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(task.actions[1], torch.arange(9, 17, dtype=torch.float32))
    torch.testing.assert_close(task.actions[2], torch.arange(100, 108, dtype=torch.float32))

    policy.reset()
    policy.eval(task, _observation())
    assert client.calls == 3


def test_openpi_watermark_must_be_positive(monkeypatch):
    monkeypatch.setattr(openpi_deploy, "make_client", lambda host, port: object())
    with pytest.raises(ValueError, match="watermark"):
        openpi_deploy.Policy({"watermark": 0})


def test_streaming_watermark_means_remaining_horizon():
    policy = object.__new__(streaming_deploy.Policy)
    policy._side_camera = "head"
    policy._wrist_camera = "wrist"
    policy._state_dim = 9
    policy._session_id = "session"
    policy._cache_lock = threading.Lock()
    policy._cache_version = "cache-1"
    policy._watermark = 30
    policy._refresh_after = 20
    policy._last_refresh_execution_id = 0
    policy._generation = 0
    policy._latest_vlm_request = {"state": np.zeros(9, dtype=np.float32)}
    policy._action_buffer = np.zeros((5, 8), dtype=np.float32)

    assert not policy.needs_observation(_Task(take_action_cnt=19))
    assert policy.needs_observation(_Task(take_action_cnt=20))

    policy._action_buffer = np.empty((0, 8), dtype=np.float32)
    assert not policy.needs_observation(_Task(take_action_cnt=5))


def test_streaming_validates_and_buffers_fm_chunk():
    policy = object.__new__(streaming_deploy.Policy)
    policy._side_camera = "head"
    policy._wrist_camera = "wrist"
    policy._state_dim = 9
    policy._action_indices = None
    policy._use_tactile = True
    policy._left_tactile = "left_tactile"
    policy._right_tactile = "right_tactile"
    policy._marker_count = 63
    policy._session_id = "session"
    policy._cache_lock = threading.Lock()
    policy._cache_version = "cache-1"
    policy._watermark = 30
    policy._refresh_after = 20
    policy._chunk_size = 5
    policy._last_refresh_execution_id = 0
    policy._generation = 0
    policy._latest_vlm_request = streaming_deploy.build_observation(
        _observation(),
        "test task",
        side_camera="head",
        wrist_camera="wrist",
        state_dim=9,
        include_tactile=True,
        left_tactile="left_tactile",
        right_tactile="right_tactile",
        marker_count=63,
    )
    policy._action_buffer = np.empty((0, 8), dtype=np.float32)
    policy._metrics_lock = threading.Lock()
    policy._timing = {}
    policy._publish_refresh = lambda observation: pytest.fail("VLM refresh is not due")
    stream_calls = []

    def stream_infer(observation, execution_id):
        stream_calls.append((observation, execution_id))
        return {
            "actions": np.zeros((5, 9), dtype=np.float32),
            "streaming": {
                "session_id": "session",
                "execution_id": 5,
                "cache_version": "cache-1",
                "action_count": 5,
            },
        }

    policy._stream_infer = stream_infer
    task = _Task(take_action_cnt=5)

    assert not policy.needs_observation(task)
    policy.eval(task, None)

    assert len(stream_calls) == 1
    assert len(policy._action_buffer) == 4
    assert policy._latest_vlm_request["state"].shape == (9,)


def test_openpi_skips_observation_while_sync_chunk_is_buffered(monkeypatch):
    class Client:
        def infer(self, request):
            del request
            return {"actions": np.zeros((2, 9), dtype=np.float32)}

    monkeypatch.setattr(openpi_deploy, "make_client", lambda host, port: Client())
    policy = openpi_deploy.Policy({"watermark": 2, "state_dim": 9})
    task = _Task()

    policy.eval(task, _observation())
    assert not policy.needs_observation(task)
    policy.eval(task, None)

    assert len(task.actions) == 2


def test_streaming_initialises_tactile_cache_and_executes_one_fm_chunk(monkeypatch):
    class FmClient:
        def __init__(self):
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            if request["op"] == "reset_stream":
                return {"prefix_cache_cleared": True}
            if request["op"] == "stream_infer":
                return {
                    "actions": np.zeros((5, 9), dtype=np.float32),
                    "streaming": {
                        "session_id": request["session_id"],
                        "execution_id": request["executed_action_id"],
                        "cache_version": "model:1",
                        "action_count": 5,
                    },
                }
            raise AssertionError(request)

    class RefreshClient:
        def __init__(self):
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {
                "cache_version": request["cache_version"],
                "active_cache_version": request["cache_version"],
            }

    class VlmClient:
        def __init__(self):
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {"cache_id": 1, "cache_version": "model:1"}

    fm = FmClient()
    refresh = RefreshClient()
    vlm = VlmClient()
    clients = iter((fm, refresh, vlm))
    monkeypatch.setattr(streaming_deploy, "make_client", lambda host, port: next(clients))
    policy = streaming_deploy.Policy(
        {
            "state_dim": 9,
            "horizon": 50,
            "watermark": 30,
            "streaming_chunk_size": 5,
            "use_tactile": True,
            "marker_count": 63,
        }
    )
    task = _Task()
    try:
        policy.reset()
        policy.eval(task, _observation())

        assert [request["op"] for request in fm.requests] == ["reset_stream", "stream_infer"]
        assert vlm.requests[0]["observation"]["left_marker"].shape == (2, 63, 2)
        assert refresh.requests[0]["cache_version"] == "model:1"
        assert len(policy._action_buffer) == 4
        assert task.metadata["policy_timing"]["fm_stream_infer"]["count"] == 1
    finally:
        policy.close()


def test_sync_openpi_reconnects_and_retries_the_same_request(monkeypatch):
    class BrokenClient:
        def infer(self, request):
            del request
            raise ConnectionError("closed")

    class HealthyClient:
        def __init__(self):
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {"actions": np.zeros((1, 9), dtype=np.float32)}

    healthy = HealthyClient()
    clients = iter((BrokenClient(), healthy))
    monkeypatch.setattr(openpi_deploy, "make_client", lambda host, port: next(clients))
    policy = openpi_deploy.Policy(
        {"watermark": 1, "state_dim": 9, "reconnect_attempts": 1}
    )
    task = _Task()

    policy.eval(task, _observation())

    assert len(healthy.requests) == 1
    assert len(task.actions) == 1
