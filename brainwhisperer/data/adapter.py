"""Adapts a neuralset ragged batch into the BrainWhisperer's `NeuralBatch`."""

from collections.abc import Sequence

import torch

from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.data.collection import MEMBER
from brainwhisperer.data.fields import DAY, NEURAL, PHONEMES, WHISPER


def _adapt(
    batch,
    datasets: Sequence[str],
    *,
    neural: str,
    whisper: str,
    phonemes: str,
    day: str,
) -> NeuralBatch:
    """Build the dataclass from a ragged batch and one dataset name per row.

    Label fields absent from the batch (competition sets) become None.
    """
    signals = batch.data[neural]
    datasets = list(datasets)
    if len(datasets) != signals.shape[0]:
        raise ValueError(
            f"{len(datasets)} dataset names for {signals.shape[0]} batch rows"
        )
    return NeuralBatch(
        signals=signals,
        labels=batch.data.get(whisper),
        datasets=datasets,
        day_idx=batch.data[day].to(torch.long).reshape(-1),
        lengths=batch.lengths[neural],
        phone_seq=batch.data.get(phonemes),
        phone_seq_len=batch.lengths.get(phonemes),
    )


def to_neural_batch(
    batch,
    dataset_name: str,
    *,
    neural: str = NEURAL,
    whisper: str = WHISPER,
    phonemes: str = PHONEMES,
    day: str = DAY,
) -> NeuralBatch:
    """Map an adapted ragged batch onto BrainWhisperer's batch dataclass.

    `dataset_name` is the registered dataset name routing every row of
    the batch to the model's name-keyed embedder.
    """
    rows = batch.data[neural].shape[0]
    return _adapt(
        batch,
        [dataset_name] * rows,
        neural=neural,
        whisper=whisper,
        phonemes=phonemes,
        day=day,
    )


def to_mixed_neural_batch(
    batch,
    *,
    neural: str = NEURAL,
    whisper: str = WHISPER,
    phonemes: str = PHONEMES,
    day: str = DAY,
    member: str = MEMBER,
) -> NeuralBatch:
    """Map a mixed-member ragged batch onto BrainWhisperer's batch dataclass."""
    return _adapt(
        batch,
        batch.metadata[member],
        neural=neural,
        whisper=whisper,
        phonemes=phonemes,
        day=day,
    )
