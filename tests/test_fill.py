"""Unit tests for the fill-to-512 helper (brainwhisperer/utils/fill.py).

Synthetic tensors only: no datasets, no checkpoints. Expectations are
hand-written formulas (torch.cat / F.pad against a fixed tensor), never calls
into `fill_to_num_features` itself.

Run: pytest tests/test_fill.py -q
"""

import pytest
import torch
import torch.nn.functional as F

from brainwhisperer.utils.fill import (
    FILL_DUPLICATE,
    FILL_NONE,
    FILL_ZEROS,
    fill_to_num_features,
)

_B, _T = 3, 7


def _x(c: int) -> torch.Tensor:
    """Fixed (non-random) (B, C, T) batch so failures are readable."""
    return torch.arange(_B * c * _T, dtype=torch.float32).reshape(_B, c, _T)


# --- happy paths -------------------------------------------------------------


def test_duplicate_is_exactly_a_channel_dim_cat():
    x = _x(256)
    got = fill_to_num_features(x, FILL_DUPLICATE, 512)
    assert torch.equal(got, torch.cat([x, x], dim=1))
    assert got.shape == (_B, 512, _T)
    # Second half must mirror the first channel-for-channel.
    assert torch.equal(got[:, 256:], got[:, :256])


def test_zeros_appends_zero_channels():
    x = _x(256)
    got = fill_to_num_features(x, FILL_ZEROS, 512)
    assert torch.equal(got, F.pad(x, (0, 0, 0, 256)))
    assert got.shape == (_B, 512, _T)
    assert torch.equal(got[:, :256], x)
    assert torch.count_nonzero(got[:, 256:]) == 0


@pytest.mark.parametrize("mode", [FILL_DUPLICATE, FILL_ZEROS, FILL_NONE])
def test_already_wide_passes_through_unchanged(mode):
    x = _x(512)
    assert fill_to_num_features(x, mode, 512) is x


# --- guards ------------------------------------------------------------------


@pytest.mark.parametrize("mode", [FILL_DUPLICATE, FILL_ZEROS, FILL_NONE])
def test_too_wide_raises(mode):
    with pytest.raises(ValueError, match="cannot shrink"):
        fill_to_num_features(_x(600), mode, 512)


def test_none_on_narrow_signals_raises():
    # A narrow dataset without a CHANNEL_FILL rule must not silently reach
    # the 512-wide conv1.
    with pytest.raises(ValueError, match="CHANNEL_FILL"):
        fill_to_num_features(_x(256), FILL_NONE, 512)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown channel fill mode"):
        fill_to_num_features(_x(256), "repeat_three_times", 512)


def test_duplicate_needs_exactly_half_width():
    # 128 * 3 = 384 != 512: a triple-stack is not what the checkpoint saw.
    with pytest.raises(ValueError, match="half-width"):
        fill_to_num_features(_x(128), FILL_DUPLICATE, 384)


# --- filling rows first or the collated batch gives the same tensor ----------


def test_row_fill_matches_batch_fill():
    """Filling (T, C) rows and then permuting to (B, C, T) must equal
    filling the already-collated (B, C, T) batch, which is what the model
    does at eval time.
    """
    torch.manual_seed(0)
    c, t = 256, 11
    rows = [torch.randn(t, c) for _ in range(_B)]
    collated = torch.stack(rows).permute(0, 2, 1)  # (B, C, T), collator layout

    # Willet rule: cat on the channel dim of each (T, C) row.
    row_willet = torch.stack([torch.cat([r, r], dim=1) for r in rows]).permute(
        0, 2, 1
    )
    assert torch.equal(
        fill_to_num_features(collated, FILL_DUPLICATE, 512), row_willet
    )

    # Jude rule: F.pad pads the LAST dim, which is channels on (T, C) rows.
    row_jude = torch.stack([F.pad(r, (0, 256)) for r in rows]).permute(0, 2, 1)
    assert torch.equal(fill_to_num_features(collated, FILL_ZEROS, 512), row_jude)
