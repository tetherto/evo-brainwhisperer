"""Unit tests for the shared inference chain (prediction.py, NeuralBatch.to_device).

These check the wiring only: the same batch in must give the same hypotheses
as a hand-written eval loop. Scores themselves are checked by running eval.py
on a trained checkpoint.
"""

import torch

from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.prediction import predict_batch, reference_texts


class _StubTokenizer:
    """Minimal stand-in: batch_decode stringifies token ids."""

    pad_token_id = 1

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["+".join(str(int(t)) for t in row) for row in ids]


def _make_batch():
    torch.manual_seed(0)
    return NeuralBatch(
        signals=torch.randn(2, 4, 64),
        labels=torch.tensor([[5, 6, -100], [7, -100, -100]]),
        datasets=["card", "willet"],
        day_idx=torch.tensor([0, 1]),
        lengths=torch.tensor([64, 40]),
    )


class _StubModel:
    def __init__(self):
        self.forward_calls = 0
        self.generate_batch = None

    def generate(self, batch, **gen_kwargs):
        self.generate_batch = batch
        return torch.full((batch.signals.shape[0], 2), 7, dtype=torch.long)

    def forward(self, batch, encoder_only=False):
        self.forward_calls += 1
        assert encoder_only
        logits = torch.zeros(2, 8, 10)
        # Row 0: argmax 1,1,2,2 inside x_len=4, then 9s outside (must be cut).
        logits[0, 0:2, 1] = 1.0
        logits[0, 2:4, 2] = 1.0
        logits[0, 4:8, 9] = 1.0
        # Row 1: argmax 3,blank inside x_len=2, then 9s outside.
        logits[1, 0, 3] = 1.0
        logits[1, 2:8, 9] = 1.0
        return {"phone_logits": logits, "x_len": torch.tensor([4, 2])}

    # nn.Module instances are callable; predict_batch calls model(...) exactly
    # as the original eval.py loop did, so the stub must be too.
    def __call__(self, batch, encoder_only=False):
        return self.forward(batch, encoder_only=encoder_only)


def test_to_device_keeps_none_fields_and_shape():
    batch = _make_batch()
    moved = batch.to_device(torch.device("cpu"))
    assert moved is not batch
    assert moved.datasets == batch.datasets
    assert moved.phone_seq is None and moved.phone_seq_len is None
    assert torch.equal(moved.signals, batch.signals)
    assert moved.signals.shape == batch.signals.shape


def test_reference_texts_maps_ignore_index_to_pad():
    tokenizer = _StubTokenizer()
    refs = reference_texts(_make_batch().labels, tokenizer)
    # -100 positions become pad_token_id=1 (compare: raw ids would show -100).
    assert refs == ["5+6+1", "7+1+1"]


def test_predict_batch_wer_path():
    model = _StubModel()
    batch = _make_batch()
    result = predict_batch(
        model,
        batch,
        tokenizer=_StubTokenizer(),
        gen_kwargs={"num_beams": 4},
        aug_kwargs={"smooth_kernel_size": 5, "smooth_sigma": 1.0},
        measure_per=False,
    )
    assert result.hyps == ["7+7", "7+7"]
    assert result.refs == ["5+6+1", "7+1+1"]
    assert result.phone_hyps == [] and result.phone_refs == []
    assert model.forward_calls == 0  # PER path skipped entirely
    # generate() saw a smoothed val batch (lengths preserved, signals changed).
    assert model.generate_batch is not None
    assert torch.equal(model.generate_batch.lengths, batch.lengths)
    assert not torch.equal(model.generate_batch.signals, batch.signals)


def test_predict_batch_per_path_decodes_and_truncates():
    model = _StubModel()
    batch = _make_batch()
    batch.phone_seq = torch.tensor([[1, 2, 5, 5], [9, 0, 0, 0]])
    batch.phone_seq_len = torch.tensor([2, 1])
    result = predict_batch(
        model,
        batch,
        tokenizer=_StubTokenizer(),
        gen_kwargs={"num_beams": 4},
        aug_kwargs={"smooth_kernel_size": 5, "smooth_sigma": 1.0},
        measure_per=True,
    )
    # Hypotheses: greedy CTC decode truncated at x_len (the 9s outside stay
    # out); blanks stripped. Refs: phone_seq truncated at phone_seq_len.
    assert result.phone_hyps == [[1, 2], [3]]
    assert result.phone_refs == [[1, 2], [9]]
