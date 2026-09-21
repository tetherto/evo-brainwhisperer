"""
Unit tests for training/losses.py — written before implementation per Phase 3 protocol.

Each function is tested against a known input/output so that the implementation
can be verified independently of the rest of the training stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from brainwhisperer.training.losses import ctc_loss


# ---------------------------------------------------------------------------
# ctc_loss
# ---------------------------------------------------------------------------

class TestCTCLoss:
    def _make_inputs(self, B=2, T=20, C=10, S=4):
        """Returns (logits, targets, input_lengths, target_lengths) with valid values."""
        torch.manual_seed(1)
        logits = torch.randn(B, T, C)
        # targets must be in [1, C-1] (0 is blank)
        targets = torch.randint(1, C, (B, S))
        input_lengths = torch.tensor([T, T - 4])
        target_lengths = torch.tensor([S, S - 1])
        return logits, targets, input_lengths, target_lengths

    def test_output_is_scalar(self):
        out = ctc_loss(*self._make_inputs())
        assert out.shape == ()

    def test_output_finite_nonneg(self):
        out = ctc_loss(*self._make_inputs())
        assert torch.isfinite(out)
        assert out >= 0.0

    def test_matches_pytorch_ctc_directly(self):
        # Our function must produce the exact same result as calling
        # nn.CTCLoss(blank=0, zero_infinity=True) with the same permute+log_softmax.
        logits, targets, input_lengths, target_lengths = self._make_inputs()

        ref_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=-1)  # (T, B, C)
        expected = ref_fn(log_probs, targets, input_lengths, target_lengths)

        assert torch.isclose(ctc_loss(logits, targets, input_lengths, target_lengths), expected)

    def test_zero_infinity_when_target_longer_than_input(self):
        # If target_length > input_length the path is impossible.
        # zero_infinity=True must return 0 rather than inf/nan.
        B, T, C = 2, 5, 6
        logits = torch.randn(B, T, C)
        targets = torch.ones(B, T + 2, dtype=torch.long)  # target longer than input
        input_lengths = torch.tensor([T, T])
        target_lengths = torch.tensor([T + 2, T + 2])
        out = ctc_loss(logits, targets, input_lengths, target_lengths)
        assert torch.isfinite(out)
        assert torch.isclose(out, torch.tensor(0.0), atol=1e-6)

    def test_blank_is_zero(self):
        # Construct logits where class 0 dominates; CTC should be able to decode
        # blank-only paths and produce a finite non-inf loss.
        B, T, C = 2, 10, 5
        logits = torch.full((B, T, C), -10.0)
        logits[:, :, 0] = 10.0  # strongly predict blank (class 0)
        targets = torch.ones(B, 2, dtype=torch.long)  # target: [1, 1]
        input_lengths = torch.full((B,), T)
        target_lengths = torch.full((B,), 2)
        out = ctc_loss(logits, targets, input_lengths, target_lengths)
        assert torch.isfinite(out)
