from dataclasses import dataclass

from torch import Tensor


@dataclass
class NeuralBatch:
    signals: Tensor  # (B, C, T)  raw neural signals, channels-first
    labels: Tensor  # (B, L)     Whisper token ids
    sbj_idx: Tensor  # (B,)       routes each sample to the right BWEmbedder
    day_idx: Tensor  # (B,)       selects day/session encoding
    lengths: Tensor  # (B,)       valid signal length per sample in frames
    phone_seq: Tensor | None = None  # (B, S) CTC phoneme targets; 0 = blank
    phone_seq_len: Tensor | None = None  # (B,)  actual phone sequence lengths
