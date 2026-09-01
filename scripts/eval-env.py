#!/usr/bin/env python3
"""Validate the local UniVTAC installation without changing it.

Run after activating the ``UniVTAC`` Conda environment:

    conda activate UniVTAC
    python scripts/eval-env.py
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable


ENVIRONMENT_NAME = "UniVTAC"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    warning: bool = False


def check_command(name: str, label: str | None = None) -> CheckResult:
    path = shutil.which(name)
    return CheckResult(label or name, path is not None, path or "not found on PATH")


def check_distribution(name: str, expected_prefix: str | None = None) -> CheckResult:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return CheckResult(name, False, "not installed")
    if expected_prefix and not version.startswith(expected_prefix):
        return CheckResult(
            name,
            True,
            f"{version}; expected {expected_prefix}.*",
            warning=True,
        )
    return CheckResult(name, True, version)


def check_nvidia_smi() -> list[CheckResult]:
    command = check_command("nvidia-smi")
    if not command.ok:
        return [command]
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode:
        return [CheckResult("nvidia-smi", False, completed.stderr.strip() or "command failed")]
    gpus = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return [CheckResult("nvidia-smi", bool(gpus), f"{len(gpus)} GPU(s) visible")]


def check_nvcc_version() -> CheckResult:
    path = shutil.which("nvcc")
    if path is None:
        return CheckResult("CUDA nvcc", False, "not found on PATH")
    completed = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    version = next((line.strip() for line in output.splitlines() if "release" in line), "version unknown")
    return CheckResult("CUDA nvcc", completed.returncode == 0, version)


def check_torch_cuda() -> list[CheckResult]:
    try:
        import torch

        available = torch.cuda.is_available()
        results = [
            CheckResult("PyTorch CUDA", available, f"available={available}, torch_cuda={torch.version.cuda}"),
            CheckResult("GPU count", torch.cuda.device_count() > 0, str(torch.cuda.device_count())),
        ]
        if available:
            torch.zeros(1, device="cuda").add_(1)
            results.append(CheckResult("CUDA tensor smoke test", True, torch.cuda.get_device_name(0)))
        return results
    except Exception as exc:
        return [CheckResult("PyTorch CUDA", False, f"{type(exc).__name__}: {exc}")]


def collect_checks() -> list[CheckResult]:
    environment = os.environ.get("CONDA_DEFAULT_ENV")
    checks = [
        CheckResult("Python", sys.version.startswith("3.10."), sys.version.split()[0]),
        CheckResult(
            "Conda environment",
            environment == ENVIRONMENT_NAME,
            environment or "not activated; run: conda activate UniVTAC",
        ),
        CheckResult("CUDA_HOME", bool(os.environ.get("CUDA_HOME")), os.environ.get("CUDA_HOME", "unset")),
    ]
    checks.extend(check_nvidia_smi())
    checks.extend(
        [
            check_nvcc_version(),
            check_command("cmake", "CMake"),
            check_command("gcc", "GCC"),
            check_command("git", "Git"),
            check_command("git-lfs", "Git LFS"),
            check_command("sudo", "sudo"),
        ]
    )
    checks.extend(check_torch_cuda())
    for name, expected in (
        ("torch", "2.5.1"),
        ("torchvision", "0.20.1"),
        ("isaacsim", "4.5.0"),
        ("isaaclab", None),
        ("nvidia-curobo", None),
        ("tacex", None),
        # libuipc's Python binding is distributed as the ``pyuipc`` package.
        ("pyuipc", None),
        ("transforms3d", None),
        ("trimesh", None),
        ("tetgen", None),
    ):
        checks.append(check_distribution(name, expected))

    try:
        import pyuipc  # noqa: F401

        checks.append(CheckResult("pyuipc import", True, "import OK"))
    except Exception as exc:
        checks.append(CheckResult("pyuipc import", False, f"{type(exc).__name__}: {exc}"))

    vcpkg_toolchain = Path.home() / "Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
    checks.append(CheckResult("vcpkg toolchain", vcpkg_toolchain.is_file(), str(vcpkg_toolchain)))
    configured_toolchain = os.environ.get("CMAKE_TOOLCHAIN_FILE")
    checks.append(
        CheckResult(
            "CMAKE_TOOLCHAIN_FILE",
            bool(configured_toolchain and Path(configured_toolchain).is_file()),
            configured_toolchain or "unset",
            warning=True,
        )
    )
    return checks


def main() -> int:
    print("UniVTAC environment check")
    print(f"Repository: {REPOSITORY_ROOT}")
    print(f"Python executable: {sys.executable}")
    failed = 0
    for result in collect_checks():
        marker = "WARN" if result.warning else ("OK" if result.ok else "FAIL")
        print(f"[{marker:4}] {result.name}: {result.detail}")
        failed += not result.ok and not result.warning
    print()
    if failed:
        print(f"Environment check failed: {failed} check(s) need attention.")
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
