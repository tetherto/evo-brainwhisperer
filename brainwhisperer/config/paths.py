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
