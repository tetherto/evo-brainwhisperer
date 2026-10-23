"""Fill narrow datasets up to the model's shared 512-channel input.

Each dataset recipe declares its rule as ``CHANNEL_FILL``:
Willet duplicates its 256 channels; Jude and Kunz T12 zero-pad;
the datasets that already publish 512 channels pass through.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

FILL_DUPLICATE = "duplicate"
FILL_ZEROS = "zeros"
FILL_NONE = "none"

FILL_MODES = (FILL_DUPLICATE, FILL_ZEROS, FILL_NONE)


def fill_to_num_features(x: Tensor, mode: str, num_features: int) -> Tensor:
    """Widen the channel axis of ``x`` to ``num_features`` channels.

    Accepts a ``(B, C, T)`` batch or a single ``(C, T)`` sample: in both
    layouts channels sit at ``dim=-2``, which is the axis this fills.
    Already-wide input passes through unchanged (the same object is
    returned, not a copy).
    """
    if mode not in FILL_MODES:
        raise ValueError(
            f"unknown channel fill mode {mode!r}; expected one of {FILL_MODES}"
        )
    c = x.shape[-2]
    if c > num_features:
        raise ValueError(
            f"Input has {c} channels but the model input layer is "
            f"{num_features} wide; fill cannot shrink channels"
        )
    if c == num_features:
        return x
    if mode == FILL_NONE:
        raise ValueError(
            f"Input has {c} channels, the model input layer is "
            f"{num_features} wide, and mode is 'none'; this dataset needs "
            f"a CHANNEL_FILL rule ('duplicate' or 'zeros')"
        )
    if mode == FILL_DUPLICATE:
        if 2 * c != num_features:
            raise ValueError(
                f"Duplicate fill needs exactly half-width input "
                f"(got {c} channels, target {num_features})"
            )
        return torch.cat([x, x], dim=-2)
    # FILL_ZEROS: pad the channel axis (second-to-last) at the end.
    return F.pad(x, (0, 0, 0, num_features - c))
