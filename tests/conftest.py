import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def run_train():
    def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "train.py"] + args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env if env is not None else os.environ,
        )

    return _run


@pytest.fixture
def run_python():
    def _run(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env if env is not None else os.environ,
        )

    return _run


_LIGHTBEAM_VARS = ("BRAINAUDIO_ROOT", "LIGHTBEAM_KENLM_PATH", "LIGHTBEAM_ADAPTER_PATH")


@pytest.fixture
def env_with_paths():
    return {**os.environ, **{k: "/tmp" for k in _LIGHTBEAM_VARS}}


@pytest.fixture
def env_without_paths():
    return {k: v for k, v in os.environ.items() if k not in _LIGHTBEAM_VARS}
