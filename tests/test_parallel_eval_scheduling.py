import json
import multiprocessing as mp
import sys
import threading
import time
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("total", "workers", "error_seeds", "cleanup_failure"),
    [
        (50, 4, [], False), (2, 4, [], False),
        (8, 4, [1000000, 1000002], False), (0, 4, [], False),
        (2, 2, [], True),
    ],
)
def test_parallel_eval_stops_at_exact_total_and_replaces_only_errors(
    tmp_path, monkeypatch, capsys, total, workers, error_seeds, cleanup_failure,
):
    from scripts import parallel_eval_policy

    # Keep real multiprocessing, queues, scheduling, and worker evaluation loops.
    # Replace Isaac Sim and inference so the concurrency checks need no GPU/server.
    assert mp.get_start_method() == "fork"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.yml").write_text("{}")
    (tmp_path / "deploy.yml").write_text(
        "policy_name: scheduling_test\ninstruction_file: null\n"
    )

    class _AppLauncher:
        app = types.SimpleNamespace(close=lambda: None)

        def __init__(self, args):
            pass

        @staticmethod
        def add_app_launcher_args(parser):
            pass

    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = _AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    class _Task:
        def __init__(self, cfg, mode):
            self.cfg = cfg
            self.save_root = cfg.save_dir
            self.save_root.mkdir(parents=True, exist_ok=True)

        def _step_callback(self, status):
            pass

        def reset(self, *, seed, instructions):
            self.seed = seed
            self.take_action_cnt = 0
            self.step_count = 0
            self.eval_success = False
            time.sleep(0.01)
            if seed in error_seeds:
                raise RuntimeError("controlled episode failure")

        def _get_observations(self):
            return None

        def check_early_stop(self):
            return True

        def clean_cache(self, *, result):
            if cleanup_failure and self.seed == 1000000:
                raise RuntimeError("controlled cleanup failure")
            with (self.save_root / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps({"seed": self.seed, "result": result}) + "\n")

        def close(self):
            pass

    class _Policy:
        def __init__(self, args):
            pass

        def reset(self):
            pass

        def eval(self, task, observation):
            task.take_action_cnt += 1
            task.step_count += 1
            task.eval_success = task.seed % 2 == 0

        def close(self):
            pass

    task_module = types.ModuleType("envs.scheduling_test")
    task_module.Task = _Task
    task_module.TaskCfg = lambda: types.SimpleNamespace(
        decimation=1, save_frequency=0, video_frequency=0,
        scene=types.SimpleNamespace(num_envs=1), step_lim=1,
        sim=types.SimpleNamespace(enable_scene_query_support=False),
    )
    policy_module = types.ModuleType("policy.scheduling_test")
    policy_module.Policy = _Policy
    monkeypatch.setitem(sys.modules, "envs.scheduling_test", task_module)
    monkeypatch.setitem(sys.modules, "policy.scheduling_test", policy_module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [
        "parallel_eval_policy.py", "scheduling_test", "task.yml", "deploy.yml",
        "--workers", str(workers), "--total_num", str(total), "--run_name", "test_run",
    ])
    stop_event = parallel_eval_policy.Event()
    monkeypatch.setattr(parallel_eval_policy, "Event", lambda: stop_event)
    # A broken dispatcher must also finish the test instead of leaving workers idle forever.
    watchdog = threading.Timer(5, stop_event.set)
    watchdog.start()
    try:
        if cleanup_failure:
            with pytest.raises(RuntimeError, match="Worker-.*exited"):
                parallel_eval_policy.main()
        else:
            parallel_eval_policy.main()
    finally:
        watchdog.cancel()

    root = tmp_path / "eval_result" / "scheduling_test" / "scheduling_test" / "deploy" / "test_run"
    if cleanup_failure:
        assert "exited" in (root / "log.log").read_text()
        return
    attempts = [
        json.loads(line)
        for path in root.glob("*/attempts.jsonl")
        for line in path.read_text().splitlines()
    ]
    completed = [row for row in attempts if row["result"] != "error"]
    errors = [row for row in attempts if row["result"] == "error"]
    assert len(completed) == total
    assert {row["seed"] for row in errors} == set(error_seeds)
    assert sorted(row["seed"] for row in attempts) == list(
        range(1000000, 1000000 + total + len(error_seeds))
    )
    successes = sum(row["result"] == "success" for row in completed)
    log = (root / "log.log").read_text()
    assert f"[Final] {successes}/{total} " in log
    assert f"errors {len(error_seeds)};" in log
    assert log.count(" Seed ") == total + len(error_seeds)
    capsys.readouterr()
