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


@pytest.fixture
def env_with_paths():
    return {
        **os.environ,
        "CARD_ROOT": "/tmp",
        "KUNZ_ROOT": "/tmp",
        "WILLETT_ROOT": "/tmp",
        "RNN_ARGS": "/tmp/rnn_args.yaml",
    }


@pytest.fixture
def env_without_paths():
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("CARD_ROOT", "KUNZ_ROOT", "WILLET_CODE", "RNN_ARGS")
    }
