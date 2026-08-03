import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal encoding table. Given integer indices of any shape (...,),
    returns the corresponding rows (..., d_model // 2).

    Each token's vector is split into two disjoint halves: the first half
    is used for session encoding, the second for temporal encoding.
    This class produces vectors of length d_model // 2, used for either half.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        half = d_model // 2
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float) * (-math.log(10000.0) / half)
        )
        pe = torch.zeros(max_len, half)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: half // 2])
        self.register_buffer("pe", pe)  # (max_len, half)

    def forward(self, idx: Tensor) -> Tensor:
        return self.pe[idx]
