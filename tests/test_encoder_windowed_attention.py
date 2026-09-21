"""
Regression test — BWEncoder windowed attention boundary.

Bug #1 (fixed): the boundary was coded as `i < last_phoneme_layer` (exclusive),
which left the critical last_phoneme_layer without a windowed mask and silently
changed which hidden state fed the CTC head.  The correct code is
`i <= last_phoneme_layer` (inclusive).

These tests verify the inclusive boundary holds by monkey-patching each
WhisperEncoderLayer.forward to record what attention_mask it received.
"""

import pytest
import torch

from brainwhisperer.config.schema import BWEncoderConfig
from brainwhisperer.models.encoder import BWEncoder

_D_MODEL = 384
_N_HEADS = 6
_N_LAYERS = 6
_SEQ = 50  # short sequence for speed


def _encoder_cfg(last_phoneme_layer: int) -> BWEncoderConfig:
    return BWEncoderConfig(
        n_layers=_N_LAYERS,
        n_heads=_N_HEADS,
        d_model=_D_MODEL,
        d_ff=1536,
        dropout=0.0,
        last_phoneme_layer=last_phoneme_layer,
        attn_window_size=11,
        rope=False,
        max_source_pos=1500,
        temporal_pe_std=0.02,
    )


def _capture_masks(encoder: BWEncoder, x: torch.Tensor) -> dict[int, bool]:
    """Run a forward pass and return {layer_idx: got_windowed_mask}."""
    received = {}
    original_fwds = {i: layer.forward for i, layer in enumerate(encoder.layers)}

    def _make_hook(idx: int):
        orig = original_fwds[idx]

        def _hook(*args, **kwargs):
            received[idx] = kwargs.get("attention_mask") is not None
            return orig(*args, **kwargs)

        return _hook

    for i, layer in enumerate(encoder.layers):
        layer.forward = _make_hook(i)

    with torch.no_grad():
        encoder(x)

    return received


@pytest.mark.parametrize("last_phoneme_layer", [0, 2, 3, 5])
def test_windowed_mask_applied_inclusive(last_phoneme_layer: int):
    """Layers 0..last_phoneme_layer receive a windowed mask; layers above do not."""
    encoder = BWEncoder(_encoder_cfg(last_phoneme_layer))
    encoder.eval()
    x = torch.randn(1, _SEQ, _D_MODEL)
    masks = _capture_masks(encoder, x)

    for i in range(_N_LAYERS):
        expected = i <= last_phoneme_layer
        assert masks[i] == expected, (
            f"Layer {i}: expected got_mask={expected} "
            f"(last_phoneme_layer={last_phoneme_layer}), got {masks[i]}"
        )


def test_windowed_mask_applied_to_last_phoneme_layer_specifically():
    """last_phoneme_layer=3: layer 3 must get the windowed mask, layer 4 must not."""
    encoder = BWEncoder(_encoder_cfg(last_phoneme_layer=3))
    encoder.eval()
    x = torch.randn(1, _SEQ, _D_MODEL)
    masks = _capture_masks(encoder, x)

    assert masks[3] is True, (
        "Layer 3 (last_phoneme_layer) must receive the windowed mask"
    )
    assert masks[4] is False, "Layer 4 must NOT receive the windowed mask"


def test_ctc_hidden_state_index_matches_last_phoneme_layer():
    """hidden_states[last_phoneme_layer + 1] is the output of layer last_phoneme_layer."""
    last_phoneme_layer = 3
    encoder = BWEncoder(_encoder_cfg(last_phoneme_layer))
    encoder.eval()
    x = torch.randn(1, _SEQ, _D_MODEL)
    with torch.no_grad():
        out = encoder(x)

    # hidden_states has n_layers + 1 entries: index 0 = encoder input,
    # index k = output of layer k-1 (with layer_norm applied to the last one).
    assert len(out.hidden_states) == _N_LAYERS + 1
    # The CTC head reads hidden_states[last_phoneme_layer + 1] in model.py.
    # Verify it exists and has the right shape.
    ctc_hidden = out.hidden_states[last_phoneme_layer + 1]
    assert ctc_hidden.shape == (1, _SEQ, _D_MODEL)
