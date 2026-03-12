"""Pytest configuration for k-search tests."""

import os
import shutil
import subprocess

import pytest


def _cuda_available_without_init() -> bool:
    """Check CUDA availability without initializing a CUDA context.

    torch.cuda.is_available() initializes CUDA, which in exclusive compute mode
    blocks subprocesses from accessing the GPU. Use nvidia-smi as a proxy.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: requires CUDA GPU")
    # Tests that spawn CUDA subprocesses must run before tests that initialize
    # CUDA in the main pytest process. In exclusive compute mode, the main process
    # claiming the GPU blocks subprocesses from accessing it. By running subprocess
    # tests first, the subprocess can use and release the GPU before in-process
    # tests claim it.
    config.addinivalue_line(
        "markers",
        "cuda_subprocess: spawns CUDA subprocess (must run before in-process CUDA tests)",
    )
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not _cuda_available_without_init():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)

    # Reorder: cuda_subprocess tests must run before in-process CUDA tests
    subprocess_tests = [item for item in items if "cuda_subprocess" in item.keywords]
    other_tests = [item for item in items if "cuda_subprocess" not in item.keywords]
    items[:] = subprocess_tests + other_tests
