import torch
import torch.nn.functional as F
from torch import Tensor, nn

from brainwhisperer.config.schema import BWEmbedderConfig
from brainwhisperer.models.positional_encoding import PositionalEncoding

_MAX_SOURCE_POSITIONS = 1500  # Whisper's fixed encoder sequence length
_SESSIONS_PER_MONTH = 4  # approximate; used only for low-rank month grouping


class BWEmbedder(nn.Module):
    """
    Maps raw neural signals (B, C, T) + session day indices (B,) to
    Whisper-compatible token sequences (B, 1500, d_model) [conv mode]
    or (B, n_patches, d_model) [patch mode].

    Forward order (conv mode):
        1. Month/day projection in raw feature space (C × C)
        2. Convolutional tokenisation: conv1 + conv2 → (B, 1500, d_model)
        3. Session encoding: sinusoidal or learnable vector added to first d_model//2 dims

    One instance per subject group; conv weights are shared across instances
    by BrainWhisperer after construction (when all groups have the same C).

    Args:
        cfg: embedder hyperparameters.
        num_days: number of distinct recording sessions for this subject group.
        session_to_month: optional mapping from session index → month bucket.
            If None, approximated as session_idx // _SESSIONS_PER_MONTH.
    """

    def __init__(
        self,
        cfg: BWEmbedderConfig,
        num_days: int,
        session_to_month: list[int] | None = None,
        session_to_calendar_day: list[int] | None = None,
    ):
        super().__init__()
        self.embed_type = cfg.embed_type
        self.d_model = cfg.d_model
        self.num_features = cfg.num_features
        self.stride_2 = cfg.stride_2
        self.day_projections = cfg.day_projections
        self.day_proj_rank = cfg.day_proj_rank
        self.num_days = num_days

        # Day projections
        if cfg.day_projections:
            C = cfg.num_features
            r = cfg.day_proj_rank

            # Resolve session→month mapping before allocating parameters so
            # that num_months is known whether an explicit mapping is provided or not.
            if session_to_month is None:
                mapping = [d // _SESSIONS_PER_MONTH for d in range(num_days)]
                num_months = (num_days + _SESSIONS_PER_MONTH - 1) // _SESSIONS_PER_MONTH
            else:
                mapping = list(session_to_month)
                num_months = max(mapping) + 1
            self.register_buffer(
                "session_to_month", torch.tensor(mapping, dtype=torch.long)
            )

            if r == 0:
                # full-rank: one (C, C) weight + (C,) bias per session
                self.proj_W = nn.Parameter(torch.zeros(num_days, C, C))
                for i in range(num_days):
                    nn.init.eye_(self.proj_W[i])
                self.proj_b_day = nn.Parameter(torch.zeros(num_days, C))
            else:
                # low-rank: month-level (C, C) + (C,) bias shared within month;
                # per-day (C, r) and (r, C) delta + (C,) bias
                self.proj_W_month = nn.Parameter(torch.zeros(num_months, C, C))
                for i in range(num_months):
                    nn.init.eye_(self.proj_W_month[i])
                self.proj_b_month = nn.Parameter(torch.zeros(num_months, C))
                self.proj_A = nn.Parameter(torch.randn(num_days, C, r) * 0.01)
                self.proj_B = nn.Parameter(torch.zeros(num_days, r, C))
                self.proj_b_day = nn.Parameter(torch.zeros(num_days, C))

        self.proj_dropout = nn.Dropout(cfg.dropout)

        # Buffer for sinusoidal day encoding so that PE amplitude
        # reflects actual temporal spacing between recordings.
        self.register_buffer(
            "session_to_calendar_day",
            torch.tensor(session_to_calendar_day, dtype=torch.long)
            if session_to_calendar_day is not None
            else None,
        )

        # Signal tokenisation
        if cfg.embed_type == "conv":
            self.conv1 = nn.Conv1d(
                cfg.num_features,
                cfg.d_model,
                kernel_size=cfg.kernel_size_1,
                padding=(cfg.kernel_size_1 - 1) // 2,
            )
            self.conv2 = nn.Conv1d(
                cfg.d_model,
                cfg.d_model,
                kernel_size=3,
                stride=cfg.stride_2,
                padding=1,
            )
        else:  # patch
            patch_in = cfg.num_features * cfg.patch_size
            self.patch_norm1 = nn.LayerNorm(patch_in)
            self.patch_proj = nn.Linear(patch_in, cfg.d_model)
            self.patch_norm2 = nn.LayerNorm(cfg.d_model)
            self.patch_size = cfg.patch_size

        # Day encoding
        if cfg.encodings == "sin":
            self.day_encoding = PositionalEncoding(cfg.d_model)
        else:
            self.day_encoding = nn.Embedding(num_days, cfg.d_model // 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, signals: Tensor, day_idx: Tensor) -> Tensor:
        """
        signals:  (B, C, T)
        day_idx:  (B,)  session indices in [0, num_days)
        returns:  (B, 1500, d_model) for conv mode
                  (B, n_patches, d_model) for patch mode
        """
        if self.day_projections:
            signals = self._apply_day_proj(signals, day_idx)

        signals = self.proj_dropout(signals)

        if self.embed_type == "conv":
            x = self._embed_conv(signals)
        else:
            x = self._embed_patch(signals)

        x = self._add_day_encoding(x, day_idx)
        return x

    # ------------------------------------------------------------------
    # Day projections
    # ------------------------------------------------------------------

    def _apply_day_proj(self, signals: Tensor, day_idx: Tensor) -> Tensor:
        """
        Applies softsign( (W_mth + A_day @ B_day) @ x_t + b_mth + b_day )
        independently to each time frame, in the raw feature space (C dims).

        signals: (B, C, T)  →  returns (B, C, T)
        """
        # einsum expects (B, T, C); we work sample-by-sample to select per-day weights
        x = signals.permute(0, 2, 1)  # (B, T, C)
        parts = []
        for i in range(x.shape[0]):
            d = int(day_idx[i].item())
            xi = x[i]  # (T, C)

            if self.day_proj_rank == 0:
                W = self.proj_W[d]  # (C, C)
                b = self.proj_b_day[d]  # (C,)
            else:
                m = int(self.session_to_month[d].item())
                W = self.proj_W_month[m] + self.proj_A[d] @ self.proj_B[d]
                b = self.proj_b_month[m] + self.proj_b_day[d]

            parts.append(F.softsign(xi @ W + b))

        x = torch.stack(parts)  # (B, T, C)
        return x.permute(0, 2, 1)  # (B, C, T)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _embed_conv(self, signals: Tensor) -> Tensor:
        target_len = _MAX_SOURCE_POSITIONS * self.stride_2  # 3000 for stride_2=2
        B, C, T = signals.shape
        if T < target_len:
            signals = F.pad(signals, (0, target_len - T))
        else:
            signals = signals[:, :, :target_len]

        x = F.gelu(self.conv1(signals))  # (B, d_model, target_len)
        x = F.gelu(self.conv2(x))  # (B, d_model, 1500)
        return x.transpose(1, 2)  # (B, 1500, d_model)

    def _embed_patch(self, signals: Tensor) -> Tensor:
        B, C, T = signals.shape
        T_trunc = (T // self.patch_size) * self.patch_size
        x = signals[:, :, :T_trunc]
        n = T_trunc // self.patch_size
        x = x.reshape(B, C, n, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, n, C * self.patch_size)
        x = self.patch_norm1(x)
        x = self.patch_proj(x)
        x = self.patch_norm2(x)
        return x

    # ------------------------------------------------------------------
    # Day encoding  (added to last d_model//2 dims after tokenisation)
    # ------------------------------------------------------------------

    def _add_day_encoding(self, x: Tensor, day_idx: Tensor) -> Tensor:
        positions = (
            self.session_to_calendar_day[day_idx]
            if self.session_to_calendar_day is not None
            else day_idx
        )
        day_enc = self.day_encoding(positions)  # (B, d_model//2)
        half = self.d_model // 2
        left = x[:, :, :half] + day_enc.unsqueeze(1)  # session encoding on first half
        right = x[:, :, half:]  # the encoder applies temporal encoding on this half
        return torch.cat([left, right], dim=-1)
