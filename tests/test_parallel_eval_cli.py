import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_parallel_resume_cannot_clear_results_outside_eval_root(tmp_path, monkeypatch):
    from scripts import parallel_eval_policy

    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.yml").write_text("{}")
    (tmp_path / "deploy.yml").write_text(
        "policy_name: directory_test\ninstruction_file: null\n"
    )
    outside = tmp_path / "outside"
    run_dir = outside / "deploy" / "my_run"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "log.log"
    sentinel.write_text("keep this\n")
    monkeypatch.setattr(sys, "argv", [
        "parallel_eval_policy.py", str(outside), "task.yml", "deploy.yml",
        "--run_name", "my_run", "--resume",
    ])

    def stop_before_workers_start():
        pytest.fail("An invalid result root must be rejected before starting workers")

    monkeypatch.setattr(parallel_eval_policy, "Manager", stop_before_workers_start)

    with pytest.raises(ValueError):
        parallel_eval_policy.main()

    assert sentinel.read_text() == "keep this\n"


@pytest.mark.parametrize(
    ("extra_args", "workers", "total_num", "forwarded_args"),
    [
        ([], "2", "100", []),
        (["3"], "3", "100", []),
        (["3", "50"], "3", "50", []),
        (["", "50", "--run_name", "my_run"], "2", "50", ["--run_name", "my_run"]),
        (["3", "", "--resume"], "3", "100", ["--resume"]),
        (["--run_name", "my run", "--resume"], "2", "100", ["--run_name", "my run", "--resume"]),
        (["3", "--run_name", "my_run"], "3", "100", ["--run_name", "my_run"]),
        (["3", "50", "--run_name", "my_run", "--resume"], "3", "50", ["--run_name", "my_run", "--resume"]),
    ],
)
def test_parallel_eval_shell_forwards_flags_and_preserves_optional_counts(
    tmp_path, extra_args, workers, total_num, forwarded_args,
):
    # Capture the actual command without starting evaluation or subprocess workers.
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
    )
    python.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ.get('PATH', '')}")

    result = subprocess.run(
        ["bash", str(root / "parallel_eval.sh"), "task", "task config", "policy/deploy", "0,1", *extra_args],
        cwd=root, env=env, check=True, capture_output=True, text=True,
    )

    assert json.loads(result.stdout) == {
        "argv": [
            "scripts/parallel_eval_policy.py", "task", "task config", "policy/deploy",
            "--total_num", total_num, "--workers", workers, *forwarded_args,
        ],
        "gpu": "0,1",
    }
