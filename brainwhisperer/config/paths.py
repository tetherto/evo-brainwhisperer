import os


def _require(var: str) -> str:
    val = os.environ.get(var)
    if val is None:
        raise EnvironmentError(
            f"Required environment variable {var!r} is not set. "
            f"Add it to your shell profile before running BrainWhisperer."
        )
    return val


# Required paths for cascaded CTC+LLM decoding eval only,
# read on first access so other entry points never need them.
_LIGHTBEAM_ONLY_VARS = {
    "BRAINAUDIO_ROOT",
    "LIGHTBEAM_KENLM_PATH",
    "LIGHTBEAM_ADAPTER_PATH",
}


def __getattr__(name: str) -> str:
    if name in _LIGHTBEAM_ONLY_VARS:
        return _require(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
