import os


def _require(var: str) -> str:
    val = os.environ.get(var)
    if val is None:
        raise EnvironmentError(
            f"Required environment variable {var!r} is not set. "
            f"Add it to your shell profile before running BrainWhisperer."
        )
    return val


CARD_ROOT = _require("CARD_ROOT")
KUNZ_ROOT = _require("KUNZ_ROOT")
WILLETT_ROOT = _require("WILLETT_ROOT")
RNN_ARGS = _require("RNN_ARGS")
JUDE_PATH = _require("JUDE_PATH")
WAIRAGKAR_ROOT = _require("WAIRAGKAR_ROOT")

# Required paths for cascaded CTC+LLM decoding eval only,
# not the training pipeline or E2E
_LIGHTBEAM_ONLY_VARS = {
    "BRAINAUDIO_ROOT",
    "LIGHTBEAM_KENLM_PATH",
    "LIGHTBEAM_ADAPTER_PATH",
}


def __getattr__(name: str) -> str:
    if name in _LIGHTBEAM_ONLY_VARS:
        return _require(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
