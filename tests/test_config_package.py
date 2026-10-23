"""Sanity check for package scaffolding and config."""


def test_help_exits_zero(run_train):
    result = run_train(["--help"])
    assert result.returncode == 0, result.stderr
    for key in (
        "model",
        "data",
        "training",
        "whisper_name",
        "batch_size",
        "max_epochs",
    ):
        assert key in result.stdout, f"{key!r} missing from --help output"


def test_missing_field_raises(run_train):
    result = run_train(["model.whisper_name=???"])
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "MissingMandatoryValue" in combined, combined


def test_config_schema_imports(run_python, env_with_paths):
    result = run_python("import brainwhisperer.config.schema", env=env_with_paths)
    assert result.returncode == 0, result.stderr


def test_paths_imports_without_env_vars(run_python, env_without_paths):
    # Nothing is required at import: train.py and eval.py never need paths.
    result = run_python("import brainwhisperer.config.paths", env=env_without_paths)
    assert result.returncode == 0, result.stderr


def test_lightbeam_path_reads_env_var(run_python, env_with_paths):
    result = run_python(
        "from brainwhisperer.config.paths import BRAINAUDIO_ROOT; print(BRAINAUDIO_ROOT)",
        env=env_with_paths,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp"


def test_lightbeam_path_raises_on_missing_env_var(run_python, env_without_paths):
    result = run_python(
        "from brainwhisperer.config.paths import BRAINAUDIO_ROOT",
        env=env_without_paths,
    )
    assert result.returncode != 0
    assert "BRAINAUDIO_ROOT" in result.stderr, result.stderr
