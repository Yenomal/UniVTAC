import runpy
import sys
import types
from pathlib import Path

import pytest


class _EvaluationInitializationReached(Exception):
    pass


@pytest.fixture(params=["single", "parallel"])
def run_eval(request, tmp_path, monkeypatch):
    script_name = "parallel_eval_policy.py" if request.param == "parallel" else "eval_policy.py"
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))

    class _AppLauncher:
        app = None

        def __init__(self, args):
            pass

        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default=None)

    fake_app = types.ModuleType("isaaclab.app")
    fake_app.AppLauncher = _AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    task_module = types.ModuleType("envs.directory_test")
    task_module.TaskCfg = lambda: types.SimpleNamespace(
        decimation=1, save_frequency=0, video_frequency=0, render_frequency=0,
        scene=types.SimpleNamespace(num_envs=1),
        sim=types.SimpleNamespace(device="cpu"),
    )
    monkeypatch.setitem(sys.modules, "envs.directory_test", task_module)

    def stop_before_policy_initialization(args):
        raise _EvaluationInitializationReached

    policy_module = types.ModuleType("policy.directory_test")
    policy_module.Policy = stop_before_policy_initialization
    monkeypatch.setitem(sys.modules, "policy.directory_test", policy_module)

    (tmp_path / "task.yml").write_text("{}")
    (tmp_path / "deploy.yml").write_text(
        "policy_name: directory_test\ninstruction_file: null\n"
    )

    if request.param == "parallel":
        from scripts import parallel_eval_policy

        def stop_before_workers_start():
            raise _EvaluationInitializationReached

        monkeypatch.setattr(parallel_eval_policy, "Manager", stop_before_workers_start)

    def run(*extra_args):
        monkeypatch.setattr(sys, "argv", [
            str(script), "directory_test", "task.yml", "deploy.yml", *extra_args,
        ])
        if request.param == "parallel":
            main = parallel_eval_policy.main
        else:
            main = runpy.run_path(str(script))["main"]
        with pytest.raises(_EvaluationInitializationReached):
            main()

    return run


@pytest.mark.parametrize("overwrite", [False, True])
def test_named_run_only_clears_existing_results_with_overwrite(tmp_path, run_eval, overwrite):
    base_dir = tmp_path / "eval_result" / "directory_test" / "directory_test" / "deploy"
    run_dir = base_dir / "my_run"
    old_video = run_dir / "video" / "42.mp4"
    old_video.parent.mkdir(parents=True)
    old_video.write_bytes(b"old video")
    old_log = run_dir / "log.log"
    old_log.write_text("old log\n")
    shared_seeds = base_dir / "seeds.json"
    shared_seeds.write_text('{"42": true}')
    other_run = base_dir / "other_run"
    other_run.mkdir()
    (other_run / "log.log").write_text("other run\n")

    run_eval("--run_name", "my_run", *(["--overwrite"] if overwrite else []))

    assert run_dir.is_dir()
    if overwrite:
        assert not old_video.exists()
        assert not old_log.exists() or "old log" not in old_log.read_text()
    else:
        assert old_log.read_text() == "old log\n"
        assert old_video.read_bytes() == b"old video"
    assert shared_seeds.read_text() == '{"42": true}'
    assert (other_run / "log.log").read_text() == "other run\n"


def test_overwrite_creates_a_missing_named_run(tmp_path, run_eval):
    run_eval("--overwrite", "--run_name", "new_run")

    run_dir = tmp_path / "eval_result" / "directory_test" / "directory_test" / "deploy" / "new_run"
    assert run_dir.is_dir()


def test_overwrite_without_a_run_name_preserves_existing_runs(tmp_path, run_eval):
    run_dir = tmp_path / "eval_result" / "directory_test" / "directory_test" / "deploy" / "my_run"
    run_dir.mkdir(parents=True)
    old_log = run_dir / "log.log"
    old_log.write_text("old log\n")

    run_eval("--overwrite")

    assert old_log.read_text() == "old log\n"


@pytest.mark.parametrize("run_name", [".", "..", "../../outside", "absolute", "linked"])
def test_overwrite_rejects_paths_outside_the_run_directory(tmp_path, run_eval, run_name):
    base_dir = tmp_path / "eval_result" / "directory_test" / "directory_test" / "deploy"
    base_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "log.log"
    sentinel.write_text("keep this\n")
    shared_seeds = base_dir / "seeds.json"
    shared_seeds.write_text("{}")
    if run_name == "absolute":
        run_name = str(outside)
    elif run_name == "linked":
        (base_dir / run_name).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="--run_name"):
        run_eval("--overwrite", "--run_name", run_name)

    assert sentinel.read_text() == "keep this\n"
    assert shared_seeds.read_text() == "{}"
