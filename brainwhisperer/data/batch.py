from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NeuralBatch:
    signals: Tensor  # (B, C, T)  raw neural signals, channels-first
    labels: Tensor  # (B, L)     Whisper token ids
    datasets: list[
        str
    ]  # (B,)  registered dataset name per sample; routes each to its BWEmbedder
    day_idx: Tensor  # (B,)       selects day/session encoding
    lengths: Tensor  # (B,)       valid signal length per sample in frames
    phone_seq: Tensor | None = None  # (B, S) CTC phoneme targets; 0 = blank
    phone_seq_len: Tensor | None = None  # (B,)  actual phone sequence lengths

    def to_device(self, device: torch.device) -> "NeuralBatch":
        """Return a copy with every tensor on `device`"""
        return NeuralBatch(
            signals=self.signals.to(device),
            labels=self.labels.to(device),
            datasets=self.datasets,
            day_idx=self.day_idx.to(device),
            lengths=self.lengths.to(device),
            phone_seq=None if self.phone_seq is None else self.phone_seq.to(device),
            phone_seq_len=(
                None if self.phone_seq_len is None else self.phone_seq_len.to(device)
            ),
        )
