"""
Unit tests for models/group_beam_search.py.

The decoding loop itself is covered end to end by
test_model_forward.py::TestBrainWhisperer::test_generate_grouped_beams; these
check the diversity penalty against hand-computed scores.
"""

import torch

from brainwhisperer.models.group_beam_search import HammingDiversityLogitsProcessor

VOCAB = 8
PENALTY = 0.25


def _processor() -> HammingDiversityLogitsProcessor:
    # 4 beams in 2 groups of 2.
    return HammingDiversityLogitsProcessor(PENALTY, num_beams=4, num_beam_groups=2)


class TestHammingDiversity:
    def test_first_group_unchanged(self):
        scores = torch.randn(2, VOCAB)
        current = torch.tensor([3, 5, 0, 0])
        out = _processor()(None, scores, current_tokens=current, beam_group_idx=0)
        assert torch.equal(out, scores)

    def test_penalises_tokens_of_earlier_groups(self):
        # Group 0 picked tokens 3 and 3: group 1 loses 2 * PENALTY on token 3.
        scores = torch.zeros(2, VOCAB)
        current = torch.tensor([3, 3, 0, 0])
        out = _processor()(None, scores, current_tokens=current, beam_group_idx=1)
        expected = torch.zeros(2, VOCAB)
        expected[:, 3] = -2 * PENALTY
        assert torch.allclose(out, expected)

    def test_penalty_is_per_batch_item(self):
        # Two batch items: each group-1 row is penalised by its own item's group 0.
        scores = torch.zeros(4, VOCAB)
        current = torch.tensor([1, 2, 0, 0, 6, 6, 0, 0])
        out = _processor()(None, scores, current_tokens=current, beam_group_idx=1)
        expected = torch.zeros(4, VOCAB)
        expected[0:2, [1, 2]] = -PENALTY
        expected[2:4, 6] = -2 * PENALTY
        assert torch.allclose(out, expected)
