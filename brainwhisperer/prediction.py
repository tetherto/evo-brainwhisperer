"""Prediction utils, inferring text and phonemes from a NeuralBatch.
Scoring (Whisper normalization, WER/CER/PER) is deliberately kept out;
scoring functions are in `brainwhisperer.utils.metrics`.
"""

from dataclasses import dataclass, field

import torch
from transformers import WhisperTokenizer

from brainwhisperer.data.augmentations import transform_data
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.models.model import BrainWhisperer


@dataclass
class PredictionResult:
    """Hypotheses for one batch. Text lists align index-by-index."""

    hyps: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    phone_hyps: list[list[int]] = field(default_factory=list)
    phone_refs: list[list[int]] = field(default_factory=list)


def reference_texts(labels: torch.Tensor, tokenizer: WhisperTokenizer) -> list[str]:
    """Decode target token ids to text, mapping the -100 ignore index to pad."""
    label_ids = labels.clone()
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(label_ids, skip_special_tokens=True)


def _ctc_decode_phones(
    phone_logits: torch.Tensor, x_len: torch.Tensor
) -> list[list[int]]:
    """Greedy CTC decode: argmax → collapse consecutive → strip blanks (class 0)."""
    result = []
    for i in range(phone_logits.shape[0]):
        decoded = torch.argmax(phone_logits[i, : x_len[i], :].detach(), dim=-1)
        decoded = torch.unique_consecutive(decoded).cpu().numpy()
        result.append([int(p) for p in decoded if p != 0])
    return result


def predict_batch(
    model: BrainWhisperer,
    batch: NeuralBatch,
    *,
    tokenizer: WhisperTokenizer,
    gen_kwargs: dict,
    aug_kwargs: dict,
    measure_per: bool,
) -> PredictionResult:
    """Run the evaluation prediction chain on one batch:
    smoothing → diverse beam generate → decode → (optional) CTC
    forward for phoneme hypotheses.

    Call inside `torch.no_grad()` as `eval.py` did; no autograd config here.
    """
    # Val-mode preprocessing: gaussian smoothing only (aug_kwargs from
    # cfg.training.augmentation carries kernel size and sigma).
    x_val, len_val = transform_data(
        batch.signals, batch.lengths, mode="val", **aug_kwargs
    )
    val_batch = NeuralBatch(
        signals=x_val,
        labels=batch.labels,
        datasets=batch.datasets,
        day_idx=batch.day_idx,
        lengths=len_val,
        phone_seq=batch.phone_seq,
        phone_seq_len=batch.phone_seq_len,
    )

    result = PredictionResult()

    # WER path: free generation with beam search.
    generated_ids = model.generate(val_batch, **gen_kwargs)
    result.hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if batch.labels is not None:  # None on competition sets
        result.refs = reference_texts(batch.labels, tokenizer)

    # PER path: encoder-only forward to obtain phone logits.
    if measure_per and batch.phone_seq is not None:
        out = model(val_batch, encoder_only=True)
        result.phone_hyps = _ctc_decode_phones(out["phone_logits"], out["x_len"])
        for i in range(batch.phone_seq.shape[0]):
            seq_len = int(batch.phone_seq_len[i])
            result.phone_refs.append(batch.phone_seq[i, :seq_len].cpu().tolist())

    return result
