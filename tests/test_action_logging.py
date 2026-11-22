import importlib
import sys
import types
from queue import Queue
from threading import Event

import numpy as np
import pyarrow.parquet as pq
import pytest


def test_action_log_keeps_every_chunk_value_and_separates_episodes(tmp_path):
    from policy._action_logger import ActionLogger

    logger = ActionLogger(tmp_path / "actions")
    logger.start_episode(42)
    chunk = np.arange(2700, dtype=np.float32).reshape(300, 9)
    logger.record(chunk, step=0)
    chunk[:] = -1  # A reused response buffer must not change recorded values.
    logger.record(np.arange(9, dtype=np.float32), step=1, action_id=7)
    logger.start_episode(43)
    logger.record(np.full(8, 0.5, dtype=np.float32), step=0)
    logger.close()
    logger.close()

    first = pq.read_table(tmp_path / "actions" / "42.parquet").to_pydict()
    assert first["seed"] == [42] * 301
    assert first["request_index"] == [0] * 300 + [1]
    assert first["chunk_index"] == list(range(300)) + [0]
    assert first["step"] == [0] * 300 + [1]
    assert first["action_id"] == [None] * 300 + [7]
    assert all(timestamp > 0 for timestamp in first["timestamp"])
    np.testing.assert_array_equal(first["action"][:300], np.arange(2700).reshape(300, 9))
    assert first["action"][-1] == list(range(9))
    second = pq.read_table(tmp_path / "actions" / "43.parquet").to_pydict()
    assert second["action"] == [[0.5] * 8]
    assert second["request_index"] == [0]


def test_empty_episode_still_writes_a_readable_parquet_file(tmp_path):
    from policy._action_logger import ActionLogger

    logger = ActionLogger(tmp_path)
    logger.start_episode(42)
    logger.close()

    table = pq.read_table(tmp_path / "42.parquet")
    assert table.num_rows == 0
    assert "action" in table.column_names


def test_action_log_preserves_float64_response_precision(tmp_path):
    from policy._action_logger import ActionLogger

    logger = ActionLogger(tmp_path)
    logger.start_episode(42)
    logger.record(np.array([0.12345678901234567, 1.0000000000000002]), step=0)
    logger.close()

    rows = pq.read_table(tmp_path / "42.parquet").to_pylist()
    assert rows[0]["action"] == [0.12345678901234567, 1.0000000000000002]


@pytest.mark.parametrize("policy_name", ["ACT", "ViTAL", "GR00T", "Ablation"])
def test_other_policies_do_not_create_an_action_logger(tmp_path, policy_name):
    from policy._action_logger import make_action_logger

    assert make_action_logger(policy_name, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


class _Task:
    instruction = "move the object"
    device = "cpu"
    take_action_cnt = 0

    def take_action(self, action, *, action_type):
        self.action = action.clone()
        self.take_action_cnt += 1
        return True, False


def _observation():
    return {
        "observation": {
            "head": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.zeros(9, dtype=np.float32)},
    }


@pytest.mark.parametrize("backend_name", ["runtime", "runtime_streaming"])
def test_openpi_logs_raw_action_before_mapping_and_execution(tmp_path, monkeypatch, backend_name):
    from policy._action_logger import make_action_logger
    from policy.openpi import Policy

    backend = importlib.import_module(f"policy.openpi.clients.{backend_name}")

    class _Client:
        def infer(self, request):
            if request.get("op") in {"reset", "reset_stream"}:
                return {"status": "reset"}
            return {"actions": np.arange(12, dtype=np.float32), "action_id": 0}

    monkeypatch.setattr(backend, "make_client", lambda host, port: _Client())
    logger = make_action_logger("openpi", tmp_path)
    logger.start_episode(42)
    policy = Policy({
        "client_type": backend_name,
        "action_indices": list(range(3, 12)),
        "action_logger": logger,
    })
    task = _Task()
    try:
        policy.eval(task, _observation())
    finally:
        logger.close()
        policy.close()

    rows = pq.read_table(tmp_path / "actions" / "42.parquet").to_pylist()
    assert rows[0]["action"] == list(range(12))
    assert rows[0]["action_id"] == 0
    np.testing.assert_array_equal(task.action.numpy(), list(range(3, 11)))


def test_smolvla_logs_raw_action_even_when_execution_fails(tmp_path, monkeypatch):
    from policy._action_logger import make_action_logger
    from policy.SmolVLA import deploy_policy

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def observe(self, features, *, task, sequence):
            return {"action_id": 5, "action": np.arange(12, dtype=np.float32)}

        def close(self):
            pass

    class _FailingTask(_Task):
        def take_action(self, action, *, action_type):
            super().take_action(action, action_type=action_type)
            raise RuntimeError("execution failed")

    monkeypatch.setattr(deploy_policy, "RuntimeClient", _Client)
    logger = make_action_logger("SmolVLA", tmp_path)
    logger.start_episode(42)
    policy = deploy_policy.Policy({
        "action_logger": logger,
        "action_indices": list(range(3, 12)),
    })
    task = _FailingTask()
    try:
        with pytest.raises(RuntimeError, match="execution failed"):
            policy.eval(task, _observation())
    finally:
        logger.close()
        policy.close()

    rows = pq.read_table(tmp_path / "actions" / "42.parquet").to_pylist()
    assert rows[0]["action"] == list(range(12))
    assert rows[0]["action_id"] == 5
    np.testing.assert_array_equal(task.action.numpy(), list(range(3, 11)))


@pytest.fixture
def eval_app(monkeypatch):
    # Exercise eval code without starting Isaac Sim or parsing pytest's argv.
    class _AppLauncher:
        app = types.SimpleNamespace(close=lambda: None)

        def __init__(self, args):
            pass

        @staticmethod
        def add_app_launcher_args(parser):
            pass

    fake_app = types.ModuleType("isaaclab.app")
    fake_app.AppLauncher = _AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["eval_policy.py", "task", "config", "deploy"])


@pytest.mark.parametrize("fail", [False, True])
def test_eval_finalizes_action_log_on_success_or_error(tmp_path, monkeypatch, eval_app, fail):
    from policy._action_logger import ActionLogger

    monkeypatch.setitem(sys.modules, "scripts.eval_policy", None)
    del sys.modules["scripts.eval_policy"]
    module = importlib.import_module("scripts.eval_policy")
    monkeypatch.setattr(module, "log", lambda message: None)

    class _EvalTask(_Task):
        save_root = tmp_path / "run"
        cfg = types.SimpleNamespace(step_lim=1)
        step_count = 0
        eval_success = True

        def reset(self, **kwargs):
            self.take_action_cnt = 0

        def get_policy_observation(self):
            return _observation()

        def clean_cache(self, *, result):
            self.result = result

    logger = ActionLogger(tmp_path / "run" / "actions")

    class _Policy:
        def reset(self):
            pass

        def eval(self, task, observation):
            logger.record(np.arange(9, dtype=np.float32), step=task.take_action_cnt)
            if fail:
                raise RuntimeError("inference failed after returning action")
            task.take_action_cnt += 1

    task = _EvalTask()
    module.eval_policy(task, _Policy(), False, 42, 42, 1, {"seen": ["test"]}, action_logger=logger)

    assert task.result == ("error" if fail else "success")
    rows = pq.read_table(tmp_path / "run" / "actions" / "42.parquet").to_pylist()
    assert rows[0]["action"] == list(range(9))


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_eval_injects_logger_and_finalizes_worker_file(tmp_path, monkeypatch, eval_app, fail):
    from scripts import parallel_eval_policy

    class _EvalTask(_Task):
        step_count = 0
        eval_success = True

        def __init__(self, cfg, mode):
            self.cfg = cfg

        def _step_callback(self, status):
            pass

        def reset(self, **kwargs):
            self.take_action_cnt = 0

        def _get_observations(self):
            return _observation()

        def clean_cache(self, **kwargs):
            pass

        def close(self):
            pass

    class _Policy:
        def __init__(self, args):
            self.logger = args["action_logger"]

        def reset(self):
            pass

        def eval(self, task, observation):
            self.logger.record(np.arange(9, dtype=np.float32), step=task.take_action_cnt)
            if fail:
                raise RuntimeError("execution failed")
            task.take_action_cnt += 1

        def close(self):
            pass

    task_module = types.ModuleType("envs.action_logging_test")
    task_module.Task = _EvalTask
    task_module.TaskCfg = lambda: types.SimpleNamespace(
        decimation=1, save_frequency=0, video_frequency=0,
        scene=types.SimpleNamespace(num_envs=1), step_lim=1,
    )
    policy_module = types.ModuleType("policy.SmolVLA")
    policy_module.Policy = _Policy
    monkeypatch.setitem(sys.modules, "envs.action_logging_test", task_module)
    monkeypatch.setitem(sys.modules, "policy.SmolVLA", policy_module)
    monkeypatch.setattr(parallel_eval_policy.os, "dup2", lambda *args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        parallel_eval_policy, "current_process", lambda: types.SimpleNamespace(name="Process-2")
    )
    seeds, results = Queue(), Queue()
    seeds.put(42)
    seeds.put(None)
    progress = {"done": 0, "succ": 0, "errors": 0}
    deploy_config = {"policy_name": "SmolVLA"}

    parallel_eval_policy.worker_run(
        None, deploy_config, {}, "action_logging_test", "SmolVLA", {"seen": ["test"]},
        tmp_path, seeds, progress, Event(), tmp_path / "log.log", [], {}, results,
    )

    assert results.get_nowait()["result"] == ("error" if fail else "success")
    assert deploy_config == {"policy_name": "SmolVLA"}
    rows = pq.read_table(tmp_path / "2" / "actions" / "42.parquet").to_pylist()
    assert rows[0]["action"] == list(range(9))
