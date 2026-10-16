"""
Pure loss functions for BrainWhisperer training.

CE loss is computed internally by WhisperForConditionalGeneration and returned
as `out["ce_loss"]` from `BrainWhisperer.forward()`, no implementation needed here.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

_ctc = nn.CTCLoss(blank=0, zero_infinity=True)


def ctc_loss(
    logits: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    """CTC loss wrapping nn.CTCLoss(blank=0, zero_infinity=True).
    Applies log_softmax and the (T, B, n_classes) permutation required by PyTorch's CTCLoss.

    Args:
        logits:         (B, T_ctc, n_classes) — pre-softmax scores from phone_head;
                        n_classes is 41 for the CMU phoneme set.
        targets:        (B, S) — target token sequences; class 0 is reserved for blank.
        input_lengths:  (B,)  — valid CTC frames per sample (out["x_len"]).
        target_lengths: (B,)  — actual target sequence lengths.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits.permute(1, 0, 2), dim=-1)  # (T, B, n_classes)
    return _ctc(log_probs, targets, input_lengths, target_lengths)
